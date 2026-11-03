"""用 gold 终评通过的学生模型重建全库标签索引，输出格式与 src.label_index 一致。

只在 gold_audit.json 里 adopt_for_label_index=true 时运行；输出到新目录，不覆盖旧索引。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import CHUNKS
from src.label_index import _load_retrieval_pool, validate_labels, write_json
from src.training_isolation import install_answer_read_guard
from src.weak_labels import sha256
from experiments.distilled_label_index import model as student
from experiments.distilled_label_index.encode import load_cache, lookup

RETRIEVAL_IDS = ROOT / "index/chunk_ids.npy"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50000)
    args = parser.parse_args(argv)
    audit = json.loads(args.audit.read_text())
    selection = json.loads(args.selection.read_text())
    if not audit["adopt_for_label_index"] or audit["selection_sha256"] != sha256(args.selection):
        raise ValueError("gold 终评未通过采用规则，或 selection 与终评不对应")
    model_path = Path(selection["selected_model"])
    if sha256(model_path) != selection["selected_sha256"]:
        raise ValueError("选中模型在挑选后被改动")
    dest = args.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    denied = install_answer_read_guard()
    started = time.monotonic()

    frame, retrieval_ids = _load_retrieval_pool(CHUNKS, RETRIEVAL_IDS)
    bundle = joblib.load(model_path)
    cache = load_cache(args.embeddings)
    parts = []
    for start in range(0, len(frame), args.batch_size):
        batch = frame.iloc[start:start + args.batch_size].reset_index(drop=True)
        predicted = student.predict(bundle, batch.text.tolist(), lookup(cache, batch.chunk_id))
        parts.append(pd.concat([batch[["chunk_id", "review_id"]], predicted], axis=1))
        print(f"{min(start + args.batch_size, len(frame))}/{len(frame)}", flush=True)
    labels = pd.concat(parts, ignore_index=True)
    labels["label_source"] = "fast_classifier"
    labels["model_version"] = f"{bundle['version']}:{bundle['candidate']}"
    validate_labels(labels, retrieval_ids)
    labels.to_parquet(dest / "labels.parquet", index=False, compression="zstd")
    write_json(dest / "report.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(labels), "model_version": labels.model_version.iloc[0],
        "label_source": "fast_classifier",
        "inputs": {"model": {"path": str(model_path), "sha256": sha256(model_path)},
                   "chunks": {"path": str(CHUNKS), "sha256": sha256(CHUNKS)},
                   "retrieval_ids": {"path": str(RETRIEVAL_IDS), "sha256": sha256(RETRIEVAL_IDS)},
                   "embeddings": {"path": str(args.embeddings), "sha256": sha256(args.embeddings / "embeddings.f16.npy")}},
        "gold_audit": str(args.audit), "gold_labels_read": False, "denied_reads": denied,
        "distribution": {c: labels[c].value_counts().to_dict() for c in ["aspect", "subjectivity", "polarity"]},
        "labels_sha256": sha256(dest / "labels.parquet"),
        "seconds": round(time.monotonic() - started, 1),
    })
    print(dest / "labels.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
