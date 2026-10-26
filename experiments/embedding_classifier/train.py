"""Train subjectivity/aspect/polarity classifiers with sentence-embedding
features, using the exact same weak labels and train/dev split logic as the
production eight-aspect baseline (src/classification_pipeline.py).

Two feature modes, selected with ``--features``:

- ``embedding`` (default): replace ``TfidfVectorizer`` with a frozen
  ``all-MiniLM-L6-v2`` sentence encoder. Nothing else changes.
- ``combined``: concatenate the TF-IDF sparse matrix and the MiniLM dense
  embedding into one feature matrix (``scipy.sparse.hstack``), then train
  the same ``LogisticRegression`` on the combined space. The model learns
  its own weighting between the two feature blocks; there is no manual
  mixing coefficient.

Label generation, eligibility filtering, and the train/dev group split are
untouched and reused directly from the production module so the comparison
isolates the feature extractor as the only variable.

Motivation: a confusion-matrix audit against an independent human annotation found the
TF-IDF bag-of-words baseline conflates aspects that differ only by context
(the two most frequent nouns describing airport-ground staff and cabin
staff share the word "crew"/"staff"), and almost never predicts "objective"
for a sentence that mentions an aspect keyword, even when the sentence is a
plain factual statement. A sentence embedding encodes word order and
context, which a bag-of-words model cannot. See README.md for the full
diagnosis and for why the weak-label dev split's "objective" recall is not
a reliable signal on its own.

Usage:
    .venv/bin/python -m experiments.embedding_classifier.train \
        --output-dir outputs/embedding_classifier_run1 --features embedding

    .venv/bin/python -m experiments.embedding_classifier.train \
        --output-dir outputs/combined_classifier_run1 --features combined
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.classification_pipeline import (
    CHUNKS,
    DATA_EVAL,
    TASKS,
    build_labels,
    checked_input,
    score,
    split_labels,
)
from src.training_guards import check_fit_boundary

ENCODER_NAME = "all-MiniLM-L6-v2"
VERSIONS = {"embedding": "embedding-lr-v1", "combined": "combined-lr-v1"}


def encode_texts(encoder, texts: list[str], batch_size: int = 128) -> np.ndarray:
    return np.asarray(
        encoder.encode(texts, batch_size=batch_size, show_progress_bar=False,
                       normalize_embeddings=True),
        dtype=np.float32,
    )


def _fit_tasks(x_train, x_dev, train: pd.DataFrame, dev: pd.DataFrame, *, seed: int):
    models, metrics = {}, {}
    for task, classes in TASKS.items():
        use, val = train[task].notna().to_numpy(), dev[task].notna().to_numpy()
        print(f"  {task}: 训练 {int(use.sum()):,} / 验证 {int(val.sum()):,}", flush=True)
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        model.fit(x_train[use], train.loc[use, task])
        pred = model.predict(x_dev[val])
        metrics[task] = score(dev.loc[val, task], pred, classes)
        dummy = DummyClassifier(strategy="most_frequent").fit(x_train[use], train.loc[use, task])
        metrics[task]["majority_reference"] = score(
            dev.loc[val, task], dummy.predict(x_dev[val]), classes)
        models[task] = model
    return models, metrics


def fit_models_embedding(train: pd.DataFrame, dev: pd.DataFrame, *, seed: int = 42):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(ENCODER_NAME, device="cpu")
    t0 = time.perf_counter()
    x_train = encode_texts(encoder, train.text.tolist())
    x_dev = encode_texts(encoder, dev.text.tolist())
    encode_seconds = time.perf_counter() - t0

    models, metrics = _fit_tasks(x_train, x_dev, train, dev, seed=seed)
    bundle = {"version": VERSIONS["embedding"], "tasks": TASKS, "encoder": ENCODER_NAME, "models": models}
    return bundle, metrics, encode_seconds


def fit_models_combined(train: pd.DataFrame, dev: pd.DataFrame, *, seed: int = 42,
                        max_features: int = 20000):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(ENCODER_NAME, device="cpu")
    t0 = time.perf_counter()
    e_train = encode_texts(encoder, train.text.tolist())
    e_dev = encode_texts(encoder, dev.text.tolist())
    encode_seconds = time.perf_counter() - t0

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, stop_words=None)
    t_train = vectorizer.fit_transform(train.text)
    t_dev = vectorizer.transform(dev.text)

    x_train = sparse.hstack([t_train, sparse.csr_matrix(e_train)]).tocsr()
    x_dev = sparse.hstack([t_dev, sparse.csr_matrix(e_dev)]).tocsr()

    models, metrics = _fit_tasks(x_train, x_dev, train, dev, seed=seed)
    bundle = {"version": VERSIONS["combined"], "tasks": TASKS, "encoder": ENCODER_NAME,
              "vectorizer": vectorizer, "models": models}
    return bundle, metrics, encode_seconds


def predict_texts_embedding(bundle: dict, texts: list[str]) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(bundle["encoder"], device="cpu")
    x = encode_texts(encoder, texts)
    out = pd.DataFrame({"text": texts})
    for task in bundle["tasks"]:
        out[task] = bundle["models"][task].predict(x)
    return out


def predict_texts_combined(bundle: dict, texts: list[str]) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(bundle["encoder"], device="cpu")
    e = encode_texts(encoder, texts)
    t = bundle["vectorizer"].transform(texts)
    x = sparse.hstack([t, sparse.csr_matrix(e)]).tocsr()
    out = pd.DataFrame({"text": texts})
    for task in bundle["tasks"]:
        out[task] = bundle["models"][task].predict(x)
    return out


def predict_texts(bundle: dict, texts: list[str]) -> pd.DataFrame:
    """Dispatch on the bundle's own version tag; callers do not need to know
    which feature mode produced a given model file."""
    if bundle["version"] == VERSIONS["combined"]:
        return predict_texts_combined(bundle, texts)
    if bundle["version"] == VERSIONS["embedding"]:
        return predict_texts_embedding(bundle, texts)
    raise ValueError(f"未知的模型版本：{bundle['version']}")


def train_command(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    dest = args.output_dir.resolve()
    if dest.exists():
        raise ValueError(f"输出目录已存在：{dest}；请用新目录，禁止覆盖旧实验")
    paths = {"input": CHUNKS, "kb": CHUNKS,
             "eval_meta": DATA_EVAL / "to_label_meta.csv", "eval_text": DATA_EVAL / "to_label.csv"}
    print("[1/3] 核验数据，生成跟生产基线相同的弱标签...", flush=True)
    frame, reviews, texts, _ = checked_input(paths["input"])
    labels, generation = build_labels(frame, reviews, texts)
    if labels.is_eval.any() or labels.review_id.isin(reviews).any():
        raise ValueError("评测数据泄漏，拒绝训练")
    train, dev = split_labels(labels, args.seed)
    check_fit_boundary(train, dev, eval_reviews=reviews, eval_texts=texts)
    fit_fn = fit_models_combined if args.features == "combined" else fit_models_embedding
    print(f"[2/3] 用 {args.features} 特征训练三个模型...", flush=True)
    bundle, metrics, encode_seconds = fit_fn(train, dev, seed=args.seed)
    dest.mkdir(parents=True)
    joblib.dump(bundle, dest / "model.joblib")
    report = {
        "version": bundle["version"], "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_kind": "weak_label_dev_split_not_independent_annotation",
        "feature_mode": args.features, "baseline_feature_extractor": "TfidfVectorizer",
        "generation": generation, "metrics": metrics,
        "encode_seconds": round(encode_seconds, 1),
        "elapsed_seconds": round(time.perf_counter() - start, 1),
    }
    (dest / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("[3/3] 完成（以下仅为弱标签验证分数，不是独立人工标注分数）：", flush=True)
    for task, m in metrics.items():
        print(f"  {task}: macro-F1={m['macro_f1']:.4f}", flush=True)
    print(dest, flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", choices=["embedding", "combined"], default="embedding")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    return train_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
