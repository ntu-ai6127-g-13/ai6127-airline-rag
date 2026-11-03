"""抽取 r36 教师标注用的非评测句子，按父评论分 pseudo_train / pseudo_dev。

只从 v4.1 弱标签 dev 父评论里抽，保证任何已有快速模型都没见过这些父评论；
同时排除 1,100 条评测句、其父评论和规范化重复文本，以及 v4.1 训练侧的规范化重复文本。
整条父评论的检索池句子全部入样，分布与全库标签索引一致。不读取任何人工答案。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import CHUNKS, DATA_EVAL
from src.training_isolation import install_answer_read_guard
from src.weak_labels import load_test_exclusions, normalize_text, sha256

WEAK_SPLIT = ROOT / "outputs/weak_rebuild_20261019/split.parquet"
RETRIEVAL_IDS = ROOT / "index/chunk_ids.npy"


def role_of(review_id: str, seed: int, dev_fraction: float) -> str:
    value = int(hashlib.sha256(f"{seed}:{review_id}".encode()).hexdigest()[:8], 16) / 2**32
    return "pseudo_dev" if value < dev_fraction else "pseudo_train"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sentences", type=int, default=10000)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args(argv)
    dest = args.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    denied = install_answer_read_guard()

    kb = pd.read_parquet(CHUNKS, columns=["chunk_id", "review_id", "text", "is_eval"])
    eval_reviews, eval_texts, eval_info = load_test_exclusions(
        kb, DATA_EVAL / "to_label_meta.csv", DATA_EVAL / "to_label.csv")
    pool_ids = set(np.load(RETRIEVAL_IDS, allow_pickle=True).tolist())
    split = pd.read_parquet(WEAK_SPLIT)
    dev_parents = set(split.loc[split.split.eq("dev"), "review_id"])
    train_norms = set(kb.loc[kb.chunk_id.isin(split.loc[split.split.eq("train"), "chunk_id"]), "text"]
                      .map(normalize_text))

    kb["norm"] = kb.text.map(normalize_text)
    candidates = kb.loc[kb.chunk_id.isin(pool_ids) & kb.review_id.isin(dev_parents)
                        & ~kb.review_id.isin(eval_reviews) & ~kb.norm.isin(eval_texts | train_norms)]
    candidates = candidates.drop_duplicates("norm")

    rng = np.random.RandomState(args.seed)
    parents = np.array(sorted(candidates.review_id.unique()))
    rng.shuffle(parents)
    sizes = candidates.groupby("review_id").size()
    taken, count = [], 0
    for parent in parents:
        if count >= args.sentences:
            break
        taken.append(parent)
        count += int(sizes[parent])
    sample = candidates.loc[candidates.review_id.isin(set(taken))].copy()
    sample["role"] = sample.review_id.map(lambda r: role_of(r, args.seed, args.dev_fraction))
    sample = sample.sort_values("chunk_id").reset_index(drop=True)

    if sample.chunk_id.duplicated().any() or sample.norm.duplicated().any():
        raise AssertionError("样本内 chunk 或规范化文本重复")
    if set(sample.review_id) & eval_reviews or set(sample.norm) & (eval_texts | train_norms):
        raise AssertionError("样本与评测集或 v4.1 训练侧重叠")
    if set(sample.loc[sample.role.eq("pseudo_dev"), "review_id"]) & set(sample.loc[sample.role.eq("pseudo_train"), "review_id"]):
        raise AssertionError("父评论跨越 pseudo_train / pseudo_dev")

    sample[["chunk_id", "review_id", "text", "role"]].to_parquet(dest / "sample.parquet", index=False)
    with (dest / "texts.jsonl").open("w") as handle:
        for row in sample.itertuples():
            handle.write(json.dumps({"id": row.chunk_id, "text": row.text}, ensure_ascii=False) + "\n")
    report = {
        "seed": args.seed, "requested_sentences": args.sentences, "sentences": len(sample),
        "parent_reviews": int(sample.review_id.nunique()),
        "roles": sample.role.value_counts().to_dict(),
        "role_parents": sample.groupby("role").review_id.nunique().to_dict(),
        "source_parents": "v4.1 weak-label dev split only (outputs/weak_rebuild_20261019/split.parquet)",
        "excluded": {"eval": eval_info, "v41_train_normalized_texts": len(train_norms)},
        "inputs": {str(p): sha256(p) for p in [CHUNKS, WEAK_SPLIT, RETRIEVAL_IDS]},
        "gold_labels_read": False, "denied_reads": denied,
    }
    (dest / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["sentences", "parent_reviews", "roles"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
