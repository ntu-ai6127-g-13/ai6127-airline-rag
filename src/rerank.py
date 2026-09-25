"""Cross-encoder reranking for fused retrieval candidates. See 手册 §22.

负责人：检索组

Only ever call ``rerank`` on the ~100-candidate output of
``src.retriever.search_hybrid`` — a cross-encoder scores one
``(query, sentence)`` pair per forward pass, so running it over the full
corpus is not viable.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.config import CHUNKS, RERANK_MODEL, TOP_K_CANDIDATES, TOP_K_FINAL
from src.retriever import load_chunk_lookup, search_hybrid


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(
    query: str,
    candidates: Sequence[tuple[str, float]],
    top_k: int = TOP_K_FINAL,
    *,
    chunks_path: str | Path = CHUNKS,
    model_name: str = RERANK_MODEL,
    model: Any | None = None,
) -> list[dict[str, Any]]:
    """Rerank ``candidates`` with a cross-encoder and attach display fields.

    Returns dicts (not the ``(chunk_id, score)`` pairs the retrievers use)
    because callers need the sentence text to show evidence and the
    ``dup_count`` to show how many reviews repeat it
    (``docs/数据使用指南.md`` §6.2) — both travel through here rather than
    requiring a second lookup downstream.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    query = query.strip()

    lookup = load_chunk_lookup(chunks_path)
    pairs: list[tuple[str, str]] = []
    resolved: list[tuple[str, str, int]] = []
    for chunk_id, _ in candidates:
        entry = lookup.get(str(chunk_id))
        if entry is None:
            continue
        text, dup_count = entry
        pairs.append((query, text))
        resolved.append((str(chunk_id), text, dup_count))

    if not pairs:
        return []

    if model is None:
        model = _load_cross_encoder(model_name)
    scores = np.asarray(model.predict(pairs), dtype=np.float64)
    if scores.shape[0] != len(pairs):
        raise ValueError(
            "Cross-encoder returned "
            f"{scores.shape[0]} scores for {len(pairs)} pairs"
        )

    order = sorted(
        range(len(pairs)), key=lambda i: (-float(scores[i]), resolved[i][0])
    )[:top_k]

    return [
        {
            "chunk_id": resolved[i][0],
            "score": float(scores[i]),
            "text": resolved[i][1],
            "dup_count": resolved[i][2],
        }
        for i in order
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=TOP_K_FINAL)
    parser.add_argument("--candidate-k", type=int, default=TOP_K_CANDIDATES)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the full hybrid-retrieve-then-rerank pipeline for one query."""

    args = _build_parser().parse_args(argv)
    candidates = search_hybrid(args.query, top_k=args.candidate_k)
    for item in rerank(args.query, candidates, top_k=args.top_k):
        print(
            f"{item['score']:.4f}\tdup={item['dup_count']}\t"
            f"{item['chunk_id']}\t{item['text']}"
        )


if __name__ == "__main__":
    main()
