"""BM25 and dense retrieval for the airline-review sentence corpus.

The public search functions return ``[(chunk_id, score), ...]`` sorted from
most to least relevant. Index construction is deliberately separate from
querying because the full dense index takes several minutes to build.

Both builders exclude the two sentence-level noise classes documented in
``docs/数据使用指南.md``: route-only metadata and clearly non-English text.
Search results are deduplicated by sentence text by default so repeated stock
phrases do not consume the whole top-k list.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rank_bm25 import BM25Okapi

from src.config import (
    BM25_INDEX,
    CHUNKS,
    CHUNK_IDS,
    CHUNK_TEXTS,
    EMBED_MODEL,
    FAISS_INDEX,
    RRF_K,
    TOP_K_CANDIDATES,
    TOP_K_FINAL,
)

_INDEX_VERSION = 1
_TOKEN_RE = re.compile(r"[\w']+", flags=re.UNICODE)
_REQUIRED_COLUMNS = ("chunk_id", "text")
_NOISE_COLUMNS = ("is_route_only", "is_non_english")


def _tokenize(text: str) -> list[str]:
    """Lowercase and strip punctuation while keeping contractions intact."""

    return _TOKEN_RE.findall(text.casefold())


def _load_index_pool(chunks_path: str | Path) -> pd.DataFrame:
    """Load the minimal index data and remove documented noise rows.

    ``is_eval`` rows deliberately remain searchable: BM25/FAISS index building
    is not supervised model training and never reads the human labels. Any
    downstream classifier training must instead use ``df[~df.is_eval]`` as
    required by ``docs/数据使用指南.md``.
    """

    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk data not found: {path}. Run `python -m src.build_kb` first."
        )

    available = set(pq.read_schema(path).names)
    missing = set(_REQUIRED_COLUMNS) - available
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    columns = list(_REQUIRED_COLUMNS) + [
        column for column in _NOISE_COLUMNS if column in available
    ]
    frame = pd.read_parquet(path, columns=columns)

    if frame["chunk_id"].isna().any() or frame["text"].isna().any():
        raise ValueError("chunk_id and text must not contain null values")
    if frame["chunk_id"].duplicated().any():
        raise ValueError("chunk_id must be unique before building a retrieval index")

    keep = pd.Series(True, index=frame.index)
    for column in _NOISE_COLUMNS:
        if column in frame:
            keep &= ~frame[column].fillna(False).astype(bool)

    pool = frame.loc[keep, list(_REQUIRED_COLUMNS)].reset_index(drop=True)
    if pool.empty:
        raise ValueError("No indexable sentences remain after noise filtering")
    return pool


def _validate_search(query: str, top_k: int) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return query.strip()


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _atomic_pickle(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_numpy(value: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _top_positions(scores: np.ndarray, limit: int) -> np.ndarray:
    """Return up to ``limit`` positions sorted by score, then row order."""

    limit = min(limit, len(scores))
    if limit == 0:
        return np.empty(0, dtype=np.int64)
    if limit == len(scores):
        candidates = np.arange(len(scores))
    else:
        candidates = np.argpartition(scores, -limit)[-limit:]
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order]


def _deduplicate_ranked(
    positions: Sequence[int],
    scores: Sequence[float],
    chunk_ids: Sequence[Any],
    texts: Sequence[Any],
    top_k: int,
    *,
    positive_only: bool = False,
) -> list[tuple[str, float]]:
    results: list[tuple[str, float]] = []
    seen_texts: set[str] = set()
    for position, score in zip(positions, scores):
        position = int(position)
        score = float(score)
        if position < 0:
            continue
        if positive_only and score <= 0:
            continue
        text_key = str(texts[position]).strip().casefold()
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        results.append((str(chunk_ids[position]), score))
        if len(results) == top_k:
            break
    return results


def _stable_rank(positions: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Order candidates by score desc, then by position asc.

    FAISS's own tie-break for identical scores is not guaranteed to match
    this convention (verified: ``IndexFlatIP`` can return a higher row index
    before a lower one for tied cosine scores). BM25's ``_top_positions``
    already sorts this way, so dense results are re-sorted here too — the
    two indices must agree on which duplicate-text row "wins" a tie, or
    ``rrf_fuse`` ends up splitting one sentence's evidence across two
    different chunk_ids.
    """

    return np.lexsort((positions, -scores))


def _source_row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def build_bm25(
    chunks_path: str | Path = CHUNKS,
    index_path: str | Path = BM25_INDEX,
) -> dict[str, int]:
    """Build and persist a BM25 index."""

    chunks_path = Path(chunks_path)
    index_path = Path(index_path)
    pool = _load_index_pool(chunks_path)
    texts = pool["text"].astype(str).tolist()
    tokenized = [_tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized)

    payload = {
        "version": _INDEX_VERSION,
        "bm25": bm25,
        "chunk_ids": pool["chunk_id"].astype(str).to_numpy(dtype=object),
        "texts": np.asarray(texts, dtype=object),
        "source": str(chunks_path.resolve()),
    }
    _atomic_pickle(payload, index_path)
    _load_bm25_cached.cache_clear()
    return {"indexed": len(pool), "excluded": _source_row_count(chunks_path) - len(pool)}


@lru_cache(maxsize=4)
def _load_bm25_cached(
    path_text: str, modified_ns: int, size: int
) -> tuple[BM25Okapi, np.ndarray, np.ndarray]:
    del modified_ns, size
    path = Path(path_text)
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict) or payload.get("version") != _INDEX_VERSION:
        raise RuntimeError(f"Unsupported BM25 index format: {path}; rebuild the index")
    try:
        bm25 = payload["bm25"]
        chunk_ids = np.asarray(payload["chunk_ids"], dtype=object)
        texts = np.asarray(payload["texts"], dtype=object)
    except KeyError as error:
        raise RuntimeError(f"Incomplete BM25 index: missing {error.args[0]}") from error
    if len(chunk_ids) != len(texts) or len(chunk_ids) != len(bm25.doc_len):
        raise RuntimeError(f"BM25 index metadata is misaligned: {path}")
    return bm25, chunk_ids, texts


def _load_bm25(index_path: str | Path) -> tuple[BM25Okapi, np.ndarray, np.ndarray]:
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {path}. Run `python -m src.retriever build-bm25`."
        )
    stat = path.stat()
    return _load_bm25_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def search_bm25(
    query: str,
    top_k: int = 100,
    *,
    index_path: str | Path = BM25_INDEX,
    deduplicate: bool = True,
) -> list[tuple[str, float]]:
    """Search the BM25 index and return positive-scoring sentence matches."""

    query = _validate_search(query, top_k)
    bm25, chunk_ids, texts = _load_bm25(index_path)
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float64)
    if not np.any(scores > 0):
        return []

    if not deduplicate:
        positions = _top_positions(scores, top_k)
        return [
            (str(chunk_ids[position]), float(scores[position]))
            for position in positions
            if scores[position] > 0
        ]

    candidate_count = min(len(scores), max(100, top_k * 5))
    while True:
        positions = _top_positions(scores, candidate_count)
        results = _deduplicate_ranked(
            positions,
            scores[positions],
            chunk_ids,
            texts,
            top_k,
            positive_only=True,
        )
        if len(results) >= top_k or candidate_count == len(scores):
            return results
        candidate_count = min(len(scores), candidate_count * 2)


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _encode(
    model: Any,
    texts: Sequence[str],
    *,
    batch_size: int,
    show_progress_bar: bool,
) -> np.ndarray:
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError(
            "Embedding model returned an unexpected shape: "
            f"{embeddings.shape} for {len(texts)} texts"
        )
    if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
        raise ValueError("Embedding model returned empty or non-finite vectors")
    faiss.normalize_L2(embeddings)
    return embeddings


def build_faiss(
    chunks_path: str | Path = CHUNKS,
    index_path: str | Path = FAISS_INDEX,
    chunk_ids_path: str | Path = CHUNK_IDS,
    chunk_texts_path: str | Path = CHUNK_TEXTS,
    *,
    model_name: str = EMBED_MODEL,
    model: Any | None = None,
    batch_size: int = 64,
    show_progress_bar: bool = True,
) -> dict[str, int]:
    """Encode the corpus and persist a normalized FAISS ``IndexFlatIP``."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    chunks_path = Path(chunks_path)
    index_path = Path(index_path)
    chunk_ids_path = Path(chunk_ids_path)
    chunk_texts_path = Path(chunk_texts_path)
    pool = _load_index_pool(chunks_path)
    texts = pool["text"].astype(str).tolist()
    if model is None:
        model = _load_embedding_model(model_name)
    embeddings = _encode(
        model,
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(index_path)
    try:
        faiss.write_index(index, str(temporary))
        os.replace(temporary, index_path)
    finally:
        temporary.unlink(missing_ok=True)

    _atomic_numpy(pool["chunk_id"].astype(str).to_numpy(dtype=object), chunk_ids_path)
    _atomic_numpy(np.asarray(texts, dtype=object), chunk_texts_path)
    _load_dense_cached.cache_clear()
    return {
        "indexed": len(pool),
        "excluded": _source_row_count(chunks_path) - len(pool),
        "dimensions": embeddings.shape[1],
    }


@lru_cache(maxsize=4)
def _load_dense_cached(
    index_path_text: str,
    index_modified_ns: int,
    ids_path_text: str,
    ids_modified_ns: int,
    texts_path_text: str,
    texts_modified_ns: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
    del index_modified_ns, ids_modified_ns, texts_modified_ns
    index = faiss.read_index(index_path_text)
    chunk_ids = np.load(ids_path_text, allow_pickle=True)
    texts = np.load(texts_path_text, allow_pickle=True)
    if index.ntotal != len(chunk_ids) or index.ntotal != len(texts):
        raise RuntimeError(
            "FAISS index metadata is misaligned; rebuild with "
            "`python -m src.retriever build-dense`"
        )
    return index, chunk_ids, texts


def _load_dense(
    index_path: str | Path,
    chunk_ids_path: str | Path,
    chunk_texts_path: str | Path,
) -> tuple[Any, np.ndarray, np.ndarray]:
    paths = (Path(index_path), Path(chunk_ids_path), Path(chunk_texts_path))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dense index files not found: "
            f"{missing}. Run `python -m src.retriever build-dense`."
        )
    stats = [path.stat() for path in paths]
    return _load_dense_cached(
        str(paths[0].resolve()),
        stats[0].st_mtime_ns,
        str(paths[1].resolve()),
        stats[1].st_mtime_ns,
        str(paths[2].resolve()),
        stats[2].st_mtime_ns,
    )


def search_dense(
    query: str,
    top_k: int = 100,
    *,
    index_path: str | Path = FAISS_INDEX,
    chunk_ids_path: str | Path = CHUNK_IDS,
    chunk_texts_path: str | Path = CHUNK_TEXTS,
    model_name: str = EMBED_MODEL,
    model: Any | None = None,
    deduplicate: bool = True,
) -> list[tuple[str, float]]:
    """Search normalized sentence embeddings by cosine similarity."""

    query = _validate_search(query, top_k)
    index, chunk_ids, texts = _load_dense(
        index_path, chunk_ids_path, chunk_texts_path
    )
    if model is None:
        model = _load_embedding_model(model_name)
    query_vector = _encode(
        model, [query], batch_size=1, show_progress_bar=False
    )

    requested = min(top_k, index.ntotal)
    if requested == 0:
        return []
    if not deduplicate:
        scores, positions = index.search(query_vector, requested)
        order = _stable_rank(positions[0], scores[0])
        return [
            (str(chunk_ids[positions[0][i]]), float(scores[0][i]))
            for i in order
            if positions[0][i] >= 0
        ]

    candidate_count = min(index.ntotal, max(100, top_k * 5))
    while True:
        scores, positions = index.search(query_vector, candidate_count)
        order = _stable_rank(positions[0], scores[0])
        results = _deduplicate_ranked(
            positions[0][order], scores[0][order], chunk_ids, texts, top_k
        )
        if len(results) >= requested or candidate_count == index.ntotal:
            return results
        candidate_count = min(index.ntotal, candidate_count * 2)


def rrf_fuse(
    list_a: Sequence[tuple[str, float]],
    list_b: Sequence[tuple[str, float]],
    *,
    k: int = RRF_K,
    top_k: int = TOP_K_FINAL,
) -> list[tuple[str, float]]:
    """Combine two ranked result lists by reciprocal rank, not raw score.

    BM25 scores are unbounded and FAISS cosine scores are bounded to
    [-1, 1] — the two are not on a comparable scale, so summing them
    directly would need per-query normalization that is unstable across
    queries with very different score spreads. RRF sidesteps that by only
    using each candidate's *rank* within its own list, per
    ``docs/项目执行手册.md`` §20.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    scores: dict[str, float] = {}
    for ranked_list in (list_a, list_b):
        for rank, (chunk_id, _) in enumerate(ranked_list, start=1):
            chunk_id = str(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    # Tie-break on chunk_id so output order is deterministic even when two
    # candidates land on the exact same RRF score.
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:top_k]


def search_hybrid(
    query: str,
    top_k: int = TOP_K_FINAL,
    *,
    candidate_k: int = TOP_K_CANDIDATES,
    rrf_k: int = RRF_K,
    bm25_index_path: str | Path = BM25_INDEX,
    faiss_index_path: str | Path = FAISS_INDEX,
    chunk_ids_path: str | Path = CHUNK_IDS,
    chunk_texts_path: str | Path = CHUNK_TEXTS,
    model_name: str = EMBED_MODEL,
    model: Any | None = None,
) -> list[tuple[str, float]]:
    """Fuse BM25 and dense candidates with RRF (``docs/项目执行手册.md`` §20).

    Each retriever is queried for ``candidate_k`` results (both already
    deduplicated by sentence text) before fusion, then the fused list is cut
    to ``top_k``. Widen ``candidate_k`` for aggressive metadata filtering
    applied after this call, since filtering only shrinks the pool.
    """

    bm25_hits = search_bm25(query, candidate_k, index_path=bm25_index_path)
    dense_hits = search_dense(
        query,
        candidate_k,
        index_path=faiss_index_path,
        chunk_ids_path=chunk_ids_path,
        chunk_texts_path=chunk_texts_path,
        model_name=model_name,
        model=model,
    )
    return rrf_fuse(bm25_hits, dense_hits, k=rrf_k, top_k=top_k)


@lru_cache(maxsize=2)
def _load_chunk_lookup_cached(
    path_text: str, modified_ns: int, size: int
) -> dict[str, tuple[str, int]]:
    del modified_ns, size
    frame = pd.read_parquet(path_text, columns=["chunk_id", "text", "dup_count"])
    lookup: dict[str, tuple[str, int]] = {}
    for chunk_id, text, dup_count in frame.itertuples(index=False, name=None):
        dup_count = int(dup_count) if pd.notna(dup_count) else 1
        lookup[str(chunk_id)] = (str(text), dup_count)
    return lookup


def load_chunk_lookup(chunks_path: str | Path = CHUNKS) -> dict[str, tuple[str, int]]:
    """Map ``chunk_id`` to ``(text, dup_count)`` for display/reranking.

    ``dup_count`` is how many times this exact sentence recurs in the
    corpus — per ``docs/数据使用指南.md`` §6.2, that count is aggregate
    evidence ("34 reviews said this") and should travel with search results
    even though the retrievers themselves dedup down to one representative
    row.
    """

    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk data not found: {path}. Run `python -m src.build_kb` first."
        )
    stat = path.stat()
    return _load_chunk_lookup_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("build-bm25", "build-dense"):
        build = subparsers.add_parser(command)
        build.add_argument("--chunks", type=Path, default=CHUNKS)
        if command == "build-dense":
            build.add_argument("--batch-size", type=int, default=64)

    search = subparsers.add_parser("search")
    search.add_argument("method", choices=("bm25", "dense", "hybrid"))
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "build-bm25":
        print(build_bm25(args.chunks))
    elif args.command == "build-dense":
        print(build_faiss(args.chunks, batch_size=args.batch_size))
    else:
        search = {
            "bm25": search_bm25,
            "dense": search_dense,
            "hybrid": search_hybrid,
        }[args.method]
        for chunk_id, score in search(args.query, top_k=args.top_k):
            print(f"{score:.6f}\t{chunk_id}")


if __name__ == "__main__":
    main()
