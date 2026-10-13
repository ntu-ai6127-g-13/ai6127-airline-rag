"""Cross-encoder reranking for fused retrieval candidates. See 手册 §22.

负责人：检索组

Only ever call ``rerank`` on the ~100-candidate output of
``src.retriever.search_hybrid`` — a cross-encoder scores one
``(query, sentence)`` pair per forward pass, so running it over the full
corpus is not viable.
"""

from __future__ import annotations

import argparse
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.config import CHUNKS, RERANK_MODEL, TOP_K_CANDIDATES, TOP_K_FINAL
from src.retriever import load_chunk_lookup, search_hybrid


_NEGATIVE_QUERY = re.compile(
    r"\b(?:bad|poor|terrible|awful|horrible|rude|unfriendly|unhelpful|"
    r"uncomfortable|cramped|dirty|filthy|broken|delay(?:ed)?|cancel(?:led|ed))\b",
    re.I,
)
_POSITIVE_QUERY = re.compile(
    r"\b(?:good|great|excellent|wonderful|amazing|pleasant|comfortable|"
    r"friendly|helpful|delicious|tasty|spacious|recommend(?:ed)?)\b",
    re.I,
)
_POSITIVE_TEXT = re.compile(
    r"\b(?:good|great|excellent|wonderful|amazing|pleasant|comfortable|"
    r"friendly|helpful|delicious|tasty|spacious|recommend(?:ed)?)\b",
    re.I,
)
_NEGATED_POSITIVE = re.compile(
    r"\b(?:not|n't|never|no)\s+(?:very\s+|particularly\s+|especially\s+)?"
    r"(?:good|great|excellent|wonderful|comfortable|friendly|helpful|"
    r"delicious|tasty|spacious|recommend(?:ed)?)\b",
    re.I,
)
_NEGATED_NEGATIVE = re.compile(
    r"\b(?:not|n't|never|no)\s+(?:very\s+|particularly\s+|especially\s+)?"
    r"(?:bad|poor|terrible|awful|horrible|rude|unfriendly|unhelpful|"
    r"uncomfortable|cramped|dirty|filthy|broken|problems?|issues?)\b",
    re.I,
)
_NEGATIVE_CUE = re.compile(
    r"\b(?:bad|poor|terrible|awful|horrible|rude|unfriendly|unhelpful|"
    r"uncomfortable|cramped|dirty|filthy|broken|delay(?:ed)?|cancel(?:led|ed))\b",
    re.I,
)


def polarity_adjustment(query: str, text: str) -> float:
    """Penalize an opposite-polarity candidate for an explicitly polar query.

    The cross-encoder is relevance-only and can rank ``not bad`` above a
    genuinely negative sentence for a query such as ``bad food``.  Keep the
    raw model score in the output, but use this small deterministic tie-break
    for ordering.  Neutral queries are unchanged.
    """
    if _NEGATIVE_QUERY.search(query) and (
        _NEGATED_NEGATIVE.search(text)
        or (_POSITIVE_TEXT.search(text) and not _NEGATED_POSITIVE.search(text))
    ):
        return -1.0
    if _POSITIVE_QUERY.search(query) and (
        _NEGATED_POSITIVE.search(text)
        or (_NEGATIVE_CUE.search(text) and not _NEGATED_NEGATIVE.search(text))
    ):
        return -1.0
    return 0.0


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
        range(len(pairs)),
        key=lambda i: (
            -(float(scores[i]) + polarity_adjustment(query, resolved[i][1])),
            resolved[i][0],
        ),
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
