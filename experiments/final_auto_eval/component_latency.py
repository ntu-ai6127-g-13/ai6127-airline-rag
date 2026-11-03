#!/usr/bin/env python3
"""Time the retrieval components alone over the full pool (report Section E2).

Uses the 30 query texts of the retrieval benchmark, without metadata filters:
BM25 top-20, FAISS top-20, hybrid (BM25 and FAISS top-100 each, then RRF) and
cross-encoder reranking of the 100 hybrid candidates to 20. Every query is run
once to warm caches, then timed ``--repeats`` times; medians are reported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.final_auto_eval.run import CASES, LABELS, retrieval_units  # noqa: E402
from src.config import CHUNKS  # noqa: E402
from src.rerank import rerank  # noqa: E402
from src.retriever import rrf_fuse, search_bm25, search_dense  # noqa: E402


def timed(fn) -> tuple[object, float]:
    started = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    chunks = pd.read_parquet(CHUNKS, columns=["chunk_id", "text", "airline", "cabin",
                                              "traveller_type", "route_from", "route_to", "date"])
    labels = pd.read_parquet(LABELS, columns=["chunk_id", "aspect", "subjectivity", "polarity"])
    frame = chunks.merge(labels, on="chunk_id", how="inner", validate="one_to_one")
    queries = [unit["query"] for unit in retrieval_units(json.loads(CASES.read_text()), frame)]

    def hybrid(query: str):
        return rrf_fuse(search_bm25(query, 100), search_dense(query, 100), top_k=100)

    steps = {
        "bm25_top20": lambda q: search_bm25(q, 20),
        "faiss_top20": lambda q: search_dense(q, 20),
        "hybrid_rrf_top100": hybrid,
    }
    for query in queries:
        rerank(query, hybrid(query), top_k=20)
    rows = []
    for query in queries:
        candidates = hybrid(query)
        for _ in range(args.repeats):
            for name, step in steps.items():
                rows.append({"query": query, "step": name, "seconds": timed(lambda: step(query))[1]})
            rows.append({"query": query, "step": "rerank_100_to_20",
                         "seconds": timed(lambda: rerank(query, candidates, top_k=20))[1]})
    data = pd.DataFrame(rows)
    summary = {step: {"median_ms": round(1000 * statistics.median(group.seconds), 1),
                      "p95_ms": round(1000 * group.seconds.quantile(0.95), 1)}
               for step, group in data.groupby("step", sort=False)}
    args.output_dir.mkdir(parents=True)
    data.to_csv(args.output_dir / "per_query.csv", index=False)
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "machine": platform.platform(),
              "processor": platform.processor(), "queries": len(queries), "repeats": args.repeats,
              "summary": summary}
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
