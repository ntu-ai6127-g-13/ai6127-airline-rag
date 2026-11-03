"""TF-IDF + 冻结 MiniLM 句向量拼接的快速三任务分类器（训练、预测共用）。

特征做法沿用 feat/embedding-aspect-polarity-classifier（Sunsirmiemie）的 combined 方案；
本目录只换监督来源（弱标签 / r36 教师标签 / 两者混合），不改特征。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.classification_pipeline import TASKS

ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ENCODER_PATH = (Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
                / "snapshots" / ENCODER_REVISION)
VERSION = "distilled-combined-v1"


def load_encoder(device: str = "mps"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(ENCODER_PATH), device=device)


def encode(encoder, texts: list[str], batch_size: int = 256) -> np.ndarray:
    return np.asarray(encoder.encode(texts, batch_size=batch_size, show_progress_bar=False,
                                     normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)


def features(vectorizer, texts, embeddings: np.ndarray):
    return sparse.hstack([vectorizer.transform(texts), sparse.csr_matrix(embeddings)]).tocsr()


def fit(name: str, texts: pd.Series, embeddings: np.ndarray, labels: pd.DataFrame,
        weights: dict[str, np.ndarray] | None = None, *, seed: int = 42, max_features: int = 20000) -> dict:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
    vectorizer.fit(texts)
    x = features(vectorizer, texts, embeddings)
    models, fit_info = {}, {}
    for task in TASKS:
        use = labels[task].notna().to_numpy()
        model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
        model.fit(x[use], labels.loc[use, task], sample_weight=None if weights is None else weights[task][use])
        models[task] = model
        fit_info[task] = {"rows": int(use.sum()), "iterations": model.n_iter_.tolist(),
                          "classes": model.classes_.tolist()}
    return {"version": VERSION, "candidate": name, "tasks": TASKS, "vectorizer": vectorizer,
            "models": models, "encoder": {"id": ENCODER_ID, "revision": ENCODER_REVISION},
            "fit_info": fit_info}


def predict(bundle: dict, texts: list[str], embeddings: np.ndarray) -> pd.DataFrame:
    if bundle.get("version") != VERSION or bundle.get("tasks") != TASKS:
        raise ValueError("不是 distilled-combined-v1 模型")
    if len(texts) != len(embeddings):
        raise ValueError("文本与句向量数量不一致")
    x = features(bundle["vectorizer"], texts, embeddings)
    out = pd.DataFrame({"aspect": bundle["models"]["aspect"].predict(x),
                        "subjectivity": bundle["models"]["subjectivity"].predict(x)})
    out["polarity"] = "NA"
    subjective = out.subjectivity.eq("subjective").to_numpy()
    if subjective.any():
        out.loc[subjective, "polarity"] = bundle["models"]["polarity"].predict(x[subjective])
    return out
