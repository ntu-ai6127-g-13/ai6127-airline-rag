"""Hybrid retrieval, optional experimental labels, and traceable review evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd

from src.config import CHUNKS
from src.retriever import search_with_filters
from src.rerank import rerank
from src.classify import classify_texts
from src.generate import evidence_answer
from src.label_index import count_label_matches, labels_for_chunks

ASPECT_QUERIES = {
    "seat": "passenger seat comfort legroom recline",
    "staff": "cabin crew flight attendants service attitude",
    "food": "onboard food meals drinks quality",
    "entertainment": "inflight entertainment movies screen Wi-Fi",
    "ground": "airport check-in boarding luggage flight delays punctuality",
    "value": "ticket price value for money refund customer service",
    "overall": "overall airline travel experience",
    "other": "cabin cleanliness toilets aircraft amenities",
}


def rank_evidence(candidates: list[dict], predictions: list[dict] | None = None,
                  *, aspect: str | None = None, polarity: str | None = None,
                  top_k: int = 10, max_per_review: int = 2) -> list[dict]:
    """Labels only adjust ranks. Objective/uncertain predictions never delete evidence."""
    if predictions is not None and len(predictions) != len(candidates):
        raise ValueError("Prediction count differs from candidate count")
    if top_k < 1 or max_per_review < 1: raise ValueError("Limits must be positive")
    rows = []
    for index, item in enumerate(candidates):
        row = dict(item)
        # Rank-based base score avoids treating uncalibrated logits as probabilities.
        score = 1. / (10 + index)
        if predictions is not None:
            prediction = predictions[index]
            row.update({"pred_"+key: prediction[key] for key in ["aspect","subjectivity","polarity"]})
            if aspect is not None and prediction["aspect"] == aspect: score += .025
            if polarity is not None and prediction["polarity"] == polarity: score += .015
        row["query_ranking_score"] = score; rows.append(row)
    rows.sort(key=lambda row: (-row["query_ranking_score"], str(row["chunk_id"])))
    selected = []; counts = {}; seen = set()
    for row in rows:
        parent = str(row.get("review_id", row["chunk_id"]))
        key = row["text"].strip().casefold()
        if key in seen or counts.get(parent,0) >= max_per_review: continue
        selected.append(row); seen.add(key); counts[parent] = counts.get(parent,0)+1
        if len(selected) == top_k: break
    return selected


def query_reviews(query: str = "", *, aspect: str | None = None, polarity: str | None = None,
                  filters: dict | None = None, top_k: int = 10, candidate_k: int = 60,
                  classifier_manifest: str | Path | None = None, chunks_path: str | Path = CHUNKS,
                  label_index_path: str | Path | None = None,
                  include_candidates: bool = False) -> dict:
    if aspect is not None and aspect not in ASPECT_QUERIES: raise ValueError("Unknown aspect")
    if polarity not in {None,"positive","negative"}: raise ValueError("Unknown polarity")
    if not isinstance(top_k,int) or isinstance(top_k,bool) or top_k < 1: raise ValueError("top_k must be positive")
    if not isinstance(candidate_k,int) or candidate_k < top_k: raise ValueError("candidate_k must be >= top_k")
    phrase = " ".join(part for part in [query.strip(), ASPECT_QUERIES.get(aspect,""),
        {"positive":"positive praise satisfied", "negative":"negative complaints dissatisfied"}.get(polarity,"")] if part)
    if not phrase: raise ValueError("Provide a query or choose an aspect")
    started = time.monotonic()
    hits = search_with_filters(phrase, filters=filters, top_k=candidate_k, chunks_path=chunks_path)
    scored = rerank(phrase, hits, top_k=min(max(30, top_k),len(hits)), chunks_path=chunks_path) if hits else []
    retrieval_seconds = time.monotonic()-started
    if scored:
        metadata = pd.read_parquet(chunks_path, columns=["chunk_id","review_id","review_text","airline","cabin","date","route"],
            filters=[("chunk_id","in",[row["chunk_id"] for row in scored])]).set_index("chunk_id")
        for row in scored:
            for key, value in metadata.loc[row["chunk_id"]].items():
                row[key] = None if pd.isna(value) else (value.isoformat() if hasattr(value,"isoformat") else str(value))
    baseline = rank_evidence(scored, aspect=aspect, polarity=polarity, top_k=top_k)
    # Classify the final diverse pool once. It is the same pool used by paired query evaluation.
    predictions = None
    label_lookup_mode = "none"
    if label_index_path and baseline:
        predictions = labels_for_chunks([row["chunk_id"] for row in baseline], label_index_path)
        label_lookup_mode = "offline_index"
    elif classifier_manifest and baseline:
        predictions = classify_texts([row["text"] for row in baseline], classifier_manifest)
        label_lookup_mode = "realtime_classifier"
    selected = rank_evidence(baseline, predictions, aspect=aspect, polarity=polarity, top_k=top_k)
    answer = evidence_answer(selected, aspect=aspect)
    corpus_label_matches = count_label_matches(label_index_path, aspect=aspect, polarity=polarity,
        filters=filters, chunks_path=chunks_path) if label_index_path and aspect is not None else None
    result = {"query": query, "retrieval_query": phrase, "aspect": aspect, "polarity": polarity,
            "filters": filters or {}, "evidence": selected, "baseline_evidence": baseline,
            "answer": answer, "baseline_answer": evidence_answer(baseline, aspect=aspect),
            "retrieval_seconds": retrieval_seconds,
            "total_seconds": time.monotonic()-started, "classifier_used": predictions is not None,
            "classifier_is_experimental": predictions is not None, "hard_label_filter": False,
            "label_lookup_mode": label_lookup_mode,
            "corpus_label_matches": corpus_label_matches,
            "acceptance_evaluated": False}
    if include_candidates: result["candidate_evidence"] = baseline
    return result


def compare_airlines(airlines: list[str], *, aspect: str, query: str = "", polarity: str | None = None,
                     filters: dict | None = None, classifier_manifest: str | Path | None = None,
                     label_index_path: str | Path | None = None, top_k: int = 10) -> dict:
    """Compare evidence under identical constraints without claiming corpus-wide rankings."""
    if not 2 <= len(set(airlines)) <= 3 or len(set(airlines)) != len(airlines):
        raise ValueError("Choose two or three distinct airlines")
    common = dict(filters or {}); common.pop("airline", None)
    results = {airline: query_reviews(query, aspect=aspect, polarity=polarity,
               filters={**common,"airline":airline}, classifier_manifest=classifier_manifest,
               label_index_path=label_index_path, top_k=top_k,
               candidate_k=max(60, top_k)) for airline in airlines}
    missing = [name for name,result in results.items() if result["answer"]["status"]=="insufficient_evidence"]
    return {"type":"comparison", "aspect":aspect, "results":results,
            "status":"insufficient_evidence" if missing else "evidence_available",
            "text":("部分航司的证据不足，暂时无法可靠比较："+"、".join(missing)) if missing else
                    "以下按同一方面和筛选条件展示各航司的评论证据。检索样本不代表全体旅客，不能据此直接给出总体航司排名。"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--aspect", choices=list(ASPECT_QUERIES)); parser.add_argument("--airline")
    parser.add_argument("--polarity", choices=["positive","negative"])
    parser.add_argument("--classifier-manifest", type=Path); parser.add_argument("--label-index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = query_reviews(args.query, aspect=args.aspect, polarity=args.polarity,
        filters={"airline":args.airline} if args.airline else None, classifier_manifest=args.classifier_manifest,
        label_index_path=args.label_index)
    value = json.dumps(result,ensure_ascii=False,indent=2)
    if args.output: args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(value)
    else: print(value)
