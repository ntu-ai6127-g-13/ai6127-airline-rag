"""Hybrid retrieval, optional experimental labels, and traceable review evidence."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import wraps
import json
from pathlib import Path
import re
import threading
import time

import pandas as pd

from src.config import CHUNKS
from src.retriever import search_with_filters
from src.rerank import rerank
from src.classify import classify_texts
from src.generate import evidence_answer
from src.label_index import count_label_matches, labels_for_chunks
from src.rating_summary import summarize_ratings

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

# These terms are used only to prefer evidence which explicitly mentions the
# requested attribute. Semantic retrieval still supplies recall and can
# backfill when reviewers describe the same idea with unexpected wording.
SUBTOPIC_ANCHORS = {
    "quality": ("quality", "taste", "tasted", "tasty", "delicious", "bland", "inedible", "fresh", "flavour", "flavor"),
    "portion": ("portion", "quantity", "serving size", "amount of food", "amount of meal", "meal size", "enough food"),
    "variety": ("variety", "selection", "choice", "choices", "menu", "options"),
    "drinks": ("drink", "beverage", "wine", "coffee", "tea", "juice"),
    "special_meals": ("special meal", "vegetarian", "vegan", "child meal", "halal", "dietary", "allergy", "allergic", "allergies"),
    "temperature": ("served hot", "served cold", "cold meal", "hot meal", "meal temperature", "reheated"),
    "content": ("movie", "film", "program", "programme", "selection", "content", "channel"),
    "screen": ("screen", "display", "touchscreen", "touch screen"),
    "wifi": ("wi-fi", "wifi", "internet", "connection", "connectivity"),
    "audio": ("headphone", "headset", "earphone", "audio", "sound quality"),
    "usability": ("interface", "remote control", "remote", "system froze", "responsive", "touch screen"),
    "legroom": ("legroom", "leg room", "knee room", "seat pitch", "leg space"),
    "width": ("seat width", "wide seat", "narrow seat", "shoulder room"),
    "recline": ("recline", "reclined", "reclining", "backrest", "seat back", "lie flat", "lie-flat", "upright position"),
    "comfort": ("comfortable", "uncomfortable", "comfort", "seat cushion", "padding", "hard seat", "sore"),
    "condition": ("broken seat", "damaged seat", "worn seat", "seat was broken", "seat did not work"),
    "seat_cleanliness": ("clean seat", "clean seats", "dirty seat", "dirty seats",
                         "seat stain", "stained seat", "stain on the seat"),
    "power": ("power socket", "power outlet", "usb", "charging", "charger", "electrical outlet"),
    "attitude": ("friendly", "polite", "rude", "attitude", "courteous", "smile", "helpful"),
    "efficiency": ("efficient", "prompt", "quickly", "slow service", "response time", "responsive"),
    "professionalism": ("professional", "competent", "trained", "knowledgeable", "inexperienced"),
    "availability": ("call button", "crew available", "crew disappeared", "nowhere to be seen", "attentive", "unattentive"),
    "punctuality": ("on time", "punctual", "arrived early", "departed late", "flight was delayed"),
    "delay_handling": ("rebook", "rebooked", "rebooking", "compensation", "hotel", "delay handling", "cancellation", "cancelled", "disruption", "missed connection"),
    "baggage": ("baggage", "luggage", "bag drop", "lost bag", "damaged bag", "suitcase"),
    "checkin": ("check-in", "check in", "checkin", "check-in desk", "check-in counter", "kiosk"),
    "gate_service": ("gate agent", "gate staff", "boarding", "boarding gate", "boarding call", "boarding group", "priority boarding"),
    "lounge": ("lounge", "club room"),
    "price": ("price", "fare", "expensive", "cheap", "cost", "value for money"),
    "refund": ("refund", "reimburse", "money back", "cancellation fee", "change fee", "rebooking fee"),
    "support": ("customer service", "call centre", "call center", "hotline", "complaint", "complaints", "phone support"),
    "booking": ("booking", "reservation", "website", "mobile app", "online ticket"),
    "cleanliness": ("clean", "cleanliness", "dirty", "filthy", "stain", "stained", "stains", "hygiene"),
    "toilet": ("toilet", "lavatory", "washroom", "restroom"),
    "cabin_temperature": ("cabin temperature", "too cold", "too hot", "air conditioning", "freezing cabin"),
}

SUBTOPIC_CONTEXT_PATTERNS = {
    "seat_cleanliness": (
        r"\b(?:seat|seats|seating area|seating areas|tray table|seat pocket)\b.{0,45}"
        r"\b(?:clean|cleaning|dirty|filthy|stain|stained|stains|crumb|crumbs|mess|sticky|hygiene)\b",
        r"\b(?:clean|cleaning|dirty|filthy|stain|stained|stains|crumb|crumbs|mess|sticky|hygiene)\b"
        r".{0,45}\b(?:seat|seats|seating area|seating areas|tray table|seat pocket)\b",
        r"\bdeep clean\b",
    ),
}


def _direct_subtopic_match(text: str, topic: str) -> bool:
    folded = text.casefold()
    if topic == "seat_cleanliness":
        folded = re.sub(r"\b(?:toilet|lavatory|restroom) seat\b", "toilet_fixture", folded)
    if any(re.search(pattern, folded) for pattern in SUBTOPIC_CONTEXT_PATTERNS.get(topic, ())):
        return True
    return any(re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", folded)
               for term in SUBTOPIC_ANCHORS.get(topic, ()))


class QueryBusyError(RuntimeError):
    """Raised instead of loading a second copy of the retrieval models."""


_QUERY_RUNTIME_LOCK = threading.RLock()


@contextmanager
def query_session(wait_seconds: float = 1.0):
    """Allow one complete local query at a time on memory-constrained hosts."""
    acquired = _QUERY_RUNTIME_LOCK.acquire(timeout=wait_seconds)
    if not acquired:
        raise QueryBusyError("已有一个问题正在处理，请等待当前回答完成后再提交。")
    try:
        yield
    finally:
        _QUERY_RUNTIME_LOCK.release()


def _single_query(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        with query_session():
            return func(*args, **kwargs)
    return wrapped


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


def merge_subtopic_rankings(rankings: dict[str, list[dict]], *, limit: int) -> list[dict]:
    """Merge per-subtopic rankings while guaranteeing early coverage of every facet.

    A simple global sort can fill the first page with one easy-to-retrieve facet.
    Round-robin selection makes a multi-part question auditable: when evidence is
    available, each requested facet appears before a second item from any facet.
    Reciprocal-rank scores then order the remaining candidates without treating
    reranker scores from different queries as directly comparable.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    merged: dict[str, dict] = {}
    order: list[str] = []
    seen_in_order: set[str] = set()
    max_depth = max((len(rows) for rows in rankings.values()), default=0)
    for topic, rows in rankings.items():
        for rank, item in enumerate(rows, 1):
            key = str(item["chunk_id"])
            if key not in merged:
                merged[key] = dict(item)
                merged[key]["matched_subtopics"] = []
                merged[key]["direct_subtopic_matches"] = []
                merged[key]["subtopic_retrieval_score"] = 0.0
            if topic not in merged[key]["matched_subtopics"]:
                merged[key]["matched_subtopics"].append(topic)
            if item.get("_direct_subtopic_match") and topic not in merged[key]["direct_subtopic_matches"]:
                merged[key]["direct_subtopic_matches"].append(topic)
            merged[key]["subtopic_retrieval_score"] += 1.0 / (10 + rank)
    # First preserve facet coverage in the visible part of the result.
    for rank in range(max_depth):
        for rows in rankings.values():
            if rank >= len(rows):
                continue
            key = str(rows[rank]["chunk_id"])
            if key not in seen_in_order:
                order.append(key)
                seen_in_order.add(key)
    # The second ordering key is deterministic for repeatable evaluations.
    remaining = sorted((key for key in merged if key not in seen_in_order),
                       key=lambda key: (-merged[key]["subtopic_retrieval_score"], key))
    order.extend(remaining)
    answer = [merged[key] for key in order[:limit]]
    for item in answer:
        item.pop("_direct_subtopic_match", None)
    return answer


@_single_query
def query_reviews(query: str = "", *, aspect: str | None = None, polarity: str | None = None,
                  filters: dict | None = None, top_k: int = 10, candidate_k: int = 60,
                  classifier_manifest: str | Path | None = None, chunks_path: str | Path = CHUNKS,
                  label_index_path: str | Path | None = None,
                  subtopics: dict[str, str] | None = None,
                  include_candidates: bool = False) -> dict:
    if aspect is not None and aspect not in ASPECT_QUERIES: raise ValueError("Unknown aspect")
    if polarity not in {None,"positive","negative"}: raise ValueError("Unknown polarity")
    if not isinstance(top_k,int) or isinstance(top_k,bool) or top_k < 1: raise ValueError("top_k must be positive")
    if not isinstance(candidate_k,int) or candidate_k < top_k: raise ValueError("candidate_k must be >= top_k")
    subtopics = dict(subtopics or {})
    polarity_terms = {"positive":"positive praise satisfied", "negative":"negative complaints dissatisfied"}.get(polarity,"")
    phrase = " ".join(part for part in [query.strip(), ASPECT_QUERIES.get(aspect,""), polarity_terms] if part)
    if not phrase: raise ValueError("Provide a query or choose an aspect")
    started = time.monotonic()
    if subtopics:
        rankings = {}
        per_topic_candidates = max(40, candidate_k // max(1, len(subtopics)))
        per_topic_limit = max(12, (top_k + len(subtopics) - 1) // len(subtopics) * 3)
        for topic, expansion in subtopics.items():
            topic_phrase = " ".join(part for part in [query.strip(), expansion, polarity_terms] if part)
            hits = search_with_filters(topic_phrase, filters=filters, top_k=per_topic_candidates,
                                       chunks_path=chunks_path)
            ranked = (rerank(topic_phrase, hits, top_k=min(max(24, per_topic_limit), len(hits)),
                             chunks_path=chunks_path) if hits else [])
            ranked = [{**row, "_direct_subtopic_match": _direct_subtopic_match(row.get("text", ""), topic)}
                      for row in ranked]
            ranked.sort(key=lambda row: not row["_direct_subtopic_match"])
            direct = [row for row in ranked if row["_direct_subtopic_match"]]
            if len(direct) >= 2:
                ranked = direct
            rankings[topic] = ranked[:per_topic_limit]
        scored = merge_subtopic_rankings(rankings, limit=max(30, top_k))
        phrase = " | ".join("%s: %s" % (topic, expansion) for topic, expansion in subtopics.items())
    else:
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
    rating_summary = summarize_ratings(aspect, filters=filters, chunks_path=chunks_path) if aspect else None
    subtopic_evidence_counts = {topic: sum(topic in row.get("matched_subtopics", []) for row in selected)
                                for topic in subtopics}
    direct_subtopic_evidence_counts = {
        topic: sum(topic in row.get("direct_subtopic_matches", []) for row in selected)
        for topic in subtopics}
    result = {"query": query, "retrieval_query": phrase, "aspect": aspect, "polarity": polarity,
            "filters": filters or {}, "evidence": selected, "baseline_evidence": baseline,
            "subtopics": subtopics, "subtopic_evidence_counts": subtopic_evidence_counts,
            "direct_subtopic_evidence_counts": direct_subtopic_evidence_counts,
            "answer": answer, "baseline_answer": evidence_answer(baseline, aspect=aspect),
            "retrieval_seconds": retrieval_seconds,
            "total_seconds": time.monotonic()-started, "classifier_used": predictions is not None,
            "classifier_is_experimental": predictions is not None, "hard_label_filter": False,
            "label_lookup_mode": label_lookup_mode,
            "corpus_label_matches": corpus_label_matches,
            "rating_summary": rating_summary,
            "acceptance_evaluated": False}
    if include_candidates: result["candidate_evidence"] = baseline
    return result


@_single_query
def compare_airlines(airlines: list[str], *, aspect: str, query: str = "", polarity: str | None = None,
                     filters: dict | None = None, classifier_manifest: str | Path | None = None,
                     label_index_path: str | Path | None = None, top_k: int = 10,
                     subtopics: dict[str, str] | None = None) -> dict:
    """Compare evidence under identical constraints without claiming corpus-wide rankings."""
    if not 2 <= len(set(airlines)) <= 3 or len(set(airlines)) != len(airlines):
        raise ValueError("Choose two or three distinct airlines")
    common = dict(filters or {}); common.pop("airline", None)
    results = {airline: query_reviews(query, aspect=aspect, polarity=polarity,
               filters={**common,"airline":airline}, classifier_manifest=classifier_manifest,
               label_index_path=label_index_path, subtopics=subtopics, top_k=top_k,
               candidate_k=max(60, top_k)) for airline in airlines}
    missing = [name for name,result in results.items() if result["answer"]["status"]=="insufficient_evidence"]
    rating_summary = summarize_ratings(aspect, filters=common, airlines=airlines)
    return {"type":"comparison", "query":query, "aspect":aspect, "polarity":polarity,
            "subtopics":dict(subtopics or {}),
            "filters":common, "airlines":airlines, "results":results, "rating_summary":rating_summary,
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
