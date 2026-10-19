#!/usr/bin/env python3
"""Run the fixed 40-question retrieval and grounded-answer smoke suite.

The suite deliberately uses the deterministic program answer.  It tests the
parser, structured constraints, full direct-match aggregates, evidence
selection, citations, and positive/negative example coverage without making 40
slow and nondeterministic generation calls to the local language model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHUNKS  # noqa: E402
from src.grounded_qa import (  # noqa: E402
    _SUBTOPIC_SUBJECTS,
    _single_subtopic_direction,
    answer_question,
    collect_evidence,
)
from src.question_parser import SUBTOPIC_NAMES, parse_question  # noqa: E402
from src.query_service import (  # noqa: E402
    compare_airlines,
    compare_available_airlines,
    query_aspects,
    query_reviews,
)

DEFAULT_CASES = ROOT / "data/eval/query_smoke_40_20261019.json"
DEFAULT_LABELS = ROOT / "outputs/label_index_20261019/labels.parquet"
DEFAULT_REPORT = ROOT / "outputs/query_smoke_40_20261019/report.json"
ENGLISH_SUBTOPIC_NAMES = {
    "quality": "food quality", "portion": "meal portion", "variety": "meal variety",
    "drinks": "onboard drinks", "content": "entertainment content", "screen": "screen",
    "wifi": "Wi-Fi", "usability": "system usability", "legroom": "legroom",
    "width": "seat width", "recline": "recline", "comfort": "seat comfort",
    "seat_cleanliness": "seat cleanliness", "attitude": "crew attitude",
    "delay_handling": "delay", "baggage": "baggage", "checkin": "check-in",
    "gate_service": "gate", "punctuality": "punctuality", "price": "price",
    "refund": "refund", "support": "support", "booking": "booking",
}


def _parts(result: dict) -> list[dict]:
    return list(result.get("results", {}).values()) or [result]


def _dispatch(parsed, *, labels: Path, top_k: int) -> dict:
    common = dict(
        query=parsed.retrieval_keywords,
        polarity=parsed.polarity,
        filters=parsed.filters,
        label_index_path=labels,
        top_k=top_k,
    )
    if len(parsed.airlines) >= 2:
        return compare_airlines(parsed.airlines, aspect=parsed.aspect,
                                subtopics=parsed.subtopics, **common)
    if parsed.comparison and "airline" not in parsed.filters:
        return compare_available_airlines(aspect=parsed.aspect,
                                          subtopics=parsed.subtopics, **common)
    if len(parsed.aspects) > 1:
        return query_aspects(parsed.aspects,
                             subtopics_by_aspect=parsed.subtopics_by_aspect,
                             candidate_k=max(60, top_k), **common)
    return query_reviews(aspect=parsed.aspect, subtopics=parsed.subtopics,
                         candidate_k=max(60, top_k), **common)


def _audit(case: dict, parsed, result: dict, answer: dict) -> tuple[list[str], dict]:
    failures = []
    if parsed.aspect != case["aspect"]:
        failures.append(f"aspect:{parsed.aspect}")
    if "aspects" in case and parsed.aspects != case["aspects"]:
        failures.append(f"aspects:{parsed.aspects}")
    if "subtopics" in case and list(parsed.subtopics) != case["subtopics"]:
        failures.append(f"subtopics:{list(parsed.subtopics)}")
    if case.get("comparison") is not None and parsed.comparison != case["comparison"]:
        failures.append(f"comparison:{parsed.comparison}")

    evidence = collect_evidence(result)
    valid_ids = {str(row["citation_id"]) for row in evidence}
    cited_ids = set(re.findall(r"\[([A-Z]?\d+)\]", answer["text"]))
    if not cited_ids.issubset(valid_ids):
        failures.append("unknown_citation")
    if evidence and not cited_ids:
        failures.append("missing_citation")
    if case["language"] == "en" and re.search(r"[\u4e00-\u9fff]", answer["text"]):
        failures.append("english_answer_contains_chinese")

    topics = case.get("subtopics", [])
    for topic in topics:
        label = (ENGLISH_SUBTOPIC_NAMES if case["language"] == "en"
                 else _SUBTOPIC_SUBJECTS).get(topic, SUBTOPIC_NAMES.get(topic, topic))
        if label.casefold() not in answer["text"].casefold():
            failures.append(f"missing_answer_topic:{topic}")

    direction_rows = []
    for part in _parts(result):
        for topic, analysis in part.get("subtopic_analysis", {}).items():
            summary = analysis.get("polarity_summary") or {}
            direction = _single_subtopic_direction(part, topic)
            polarities = [row.get("pred_polarity") for row in part.get("evidence", [])
                          if topic in row.get("direct_subtopic_matches", [])]
            positive = int(summary.get("positive_review_count") or 0)
            negative = int(summary.get("negative_review_count") or 0)
            # If the full index contains both views, both should survive visible
            # evidence truncation so the answer can give a representative view
            # and a counterexample.
            if positive and negative and not {"positive", "negative"}.issubset(set(polarities)):
                failures.append(f"unbalanced_examples:{topic}")
            direction_rows.append({
                "airline": (part.get("filters") or {}).get("airline"),
                "topic": topic,
                "direction": direction.get("status"),
                "direct_reviews": analysis.get("direct_review_count"),
                "labeled_reviews": summary.get("labeled_review_count"),
                "positive": positive,
                "negative": negative,
                "positive_share": summary.get("positive_share"),
                "displayed_positive": polarities.count("positive"),
                "displayed_negative": polarities.count("negative"),
            })
    return failures, {
        "evidence_count": len(evidence),
        "citations": sorted(cited_ids),
        "answer_mode": answer["mode"],
        "directions": direction_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text())
    if args.limit:
        cases = cases[:args.limit]
    frame = pd.read_parquet(CHUNKS, columns=["airline", "route_from", "route_to"])
    parser_args = {
        "known_airlines": sorted(frame.airline.dropna().astype(str).unique()),
        "origins": sorted(frame.route_from.dropna().astype(str).unique()),
        "destinations": sorted(frame.route_to.dropna().astype(str).unique()),
    }
    started = time.monotonic()
    rows = []
    for number, case in enumerate(cases, 1):
        item_started = time.monotonic()
        print(f"[{number:02d}/{len(cases):02d}] {case['id']} {case['question']}", flush=True)
        try:
            parsed = parse_question(case["question"], **parser_args)
            result = _dispatch(parsed, labels=args.labels, top_k=args.top_k)
            if case["language"] == "en":
                result["language"] = "en"
            answer = answer_question(case["question"], result, use_model=False)
            failures, metrics = _audit(case, parsed, result, answer)
            row = {
                "id": case["id"], "question": case["question"],
                "status": "pass" if not failures else "fail", "failures": failures,
                "seconds": round(time.monotonic() - item_started, 3),
                "parsed": parsed.to_dict(), "metrics": metrics, "answer": answer["text"],
            }
        except Exception as exc:  # keep the suite running to expose every failure
            row = {
                "id": case["id"], "question": case["question"], "status": "error",
                "failures": [f"{type(exc).__name__}: {exc}"],
                "seconds": round(time.monotonic() - item_started, 3),
            }
        rows.append(row)
        print(f"  {row['status'].upper()} {row['seconds']:.1f}s"
              + (" | " + "; ".join(row["failures"]) if row["failures"] else ""), flush=True)

    passed = sum(row["status"] == "pass" for row in rows)
    report = {
        "suite": "query_smoke_40_20261019", "generated_at": pd.Timestamp.now().isoformat(),
        "mode": "offline_labels_and_program_answer", "case_count": len(rows),
        "passed": passed, "failed": len(rows) - passed,
        "seconds": round(time.monotonic() - started, 3), "cases": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"SUMMARY {passed}/{len(rows)} passed in {report['seconds']:.1f}s", flush=True)
    print(f"REPORT {args.report}", flush=True)
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
