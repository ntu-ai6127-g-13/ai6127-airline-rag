"""一次性把知识库全部句子编码成 MiniLM 句向量（float16 缓存），训练与全库索引共用。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import CHUNKS
from src.weak_labels import sha256
from experiments.distilled_label_index.model import ENCODER_ID, ENCODER_REVISION, encode, load_encoder


def load_cache(cache_dir: Path) -> tuple[pd.Index, np.ndarray]:
    ids = pd.Index(np.load(cache_dir / "chunk_ids.npy", allow_pickle=True))
    return ids, np.load(cache_dir / "embeddings.f16.npy", mmap_mode="r")


def lookup(cache, chunk_ids) -> np.ndarray:
    ids, matrix = cache
    position = ids.get_indexer(list(chunk_ids))
    if (position < 0).any():
        raise KeyError("句向量缓存缺少部分 chunk_id")
    return np.asarray(matrix[position], dtype=np.float32)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--chunk", type=int, default=20000)
    args = parser.parse_args(argv)
    dest = args.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    kb = pd.read_parquet(CHUNKS, columns=["chunk_id", "text"])
    encoder = load_encoder(args.device)
    parts = dest / "parts"
    parts.mkdir(exist_ok=True)
    started = time.monotonic()
    for number, start in enumerate(range(0, len(kb), args.chunk)):
        target = parts / f"part-{number:04d}.npy"
        if target.exists():
            continue
        block = encode(encoder, kb.text.iloc[start:start + args.chunk].tolist())
        np.save(parts / f"tmp-{number:04d}.npy", block.astype(np.float16))
        (parts / f"tmp-{number:04d}.npy").replace(target)
        print(f"{min(start + args.chunk, len(kb))}/{len(kb)} {time.monotonic() - started:.0f}s", flush=True)
    matrix = np.concatenate([np.load(p) for p in sorted(parts.glob("part-*.npy"))])
    if len(matrix) != len(kb):
        raise RuntimeError("句向量行数与知识库不一致")
    np.save(dest / "embeddings.f16.npy", matrix)
    np.save(dest / "chunk_ids.npy", kb.chunk_id.to_numpy(dtype=object), allow_pickle=True)
    (dest / "report.json").write_text(json.dumps({
        "encoder": ENCODER_ID, "revision": ENCODER_REVISION, "device": args.device,
        "rows": len(kb), "dim": int(matrix.shape[1]), "dtype": "float16", "normalized": True,
        "chunks_sha256": sha256(CHUNKS), "seconds": round(time.monotonic() - started, 1)},
        indent=2) + "\n")
    for p in parts.glob("part-*.npy"):
        p.unlink()
    parts.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
