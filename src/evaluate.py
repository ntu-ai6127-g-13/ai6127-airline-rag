"""Query acceptance from explicit relevance and answer judgments, never model guesses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.quality import QUERY_TARGETS


def evaluate_queries(records: list[dict]) -> dict:
    if not records: raise ValueError("Cannot evaluate an empty query set")
    if len({r["query_id"] for r in records}) != len(records): raise ValueError("Query IDs must be unique")
    values = {key: [] for key in QUERY_TARGETS}; missing = []; claim_total = 0; claim_supported = 0
    for row in records:
        qid = row["query_id"]; expected = row.get("expected_answerable")
        if type(expected) is not bool:
            missing.append({"query_id":qid,"field":"expected_answerable"}); continue
        returned = row.get("returned_ids", [])
        if len(set(returned)) != len(returned): raise ValueError("Duplicate returned IDs: " + qid)
        if expected:
            judgments = row.get("relevance", {})
            if all(type(judgments.get(cid)) is bool for cid in returned[:10]):
                values["precision_at_5"].append(sum(judgments[c] for c in returned[:5])/5.)
                values["success_at_10"].append(float(any(judgments[c] for c in returned[:10])))
            else: missing.append({"query_id":qid,"field":"relevance_for_top_10"})
            success = row.get("answer_successful")
            if type(success) is bool: values["answer_success_rate"].append(float(success))
            else: missing.append({"query_id":qid,"field":"answer_successful"})
            total = row.get("total_claims"); supported = row.get("supported_claims")
            if type(total) is int and type(supported) is int and 0 <= supported <= total:
                claim_total += total; claim_supported += supported
            else: missing.append({"query_id":qid,"field":"claim_support"})
        else:
            correct = row.get("correct_abstention")
            if type(correct) is bool: values["correct_abstention_rate"].append(float(correct))
            else: missing.append({"query_id":qid,"field":"correct_abstention"})
    metrics = {key: (sum(items)/len(items) if items else None) for key,items in values.items()}
    metrics["supported_claim_rate"] = claim_supported/claim_total if claim_total else None
    failures = {key:{"actual":metrics[key],"target":goal} for key,goal in QUERY_TARGETS.items()
                if metrics[key] is None or metrics[key] < goal}
    point_passed = not missing and not failures
    return {"query_count":len(records), "metrics":metrics, "targets":QUERY_TARGETS,
            "missing_judgments":missing, "target_failures":failures,
            "evaluation_complete":not missing and all(v is not None for v in metrics.values()),
            "passed":point_passed,
            "independent_acceptance_passed": point_passed and all(
                r.get("independent_test") is True and r.get("human_reviewed") is True for r in records),
            "judgment_provenance":[r.get("judgment_provenance","unspecified") for r in records]}


def evaluate_query_modes(records: list[dict]) -> dict:
    """Keep paired baseline/label scores separate; never average competing systems."""
    if not records: raise ValueError("Cannot evaluate an empty query set")
    modes = {row.get("mode", "default") for row in records}
    if len(modes) == 1: return evaluate_queries(records)
    grouped = {mode: [row for row in records if row.get("mode", "default") == mode]
               for mode in sorted(modes)}
    query_sets = {}
    for mode, rows in grouped.items():
        ids = [row.get("base_query_id") for row in rows]
        if None in ids or len(set(ids)) != len(ids):
            raise ValueError("Paired modes need unique base_query_id values: " + mode)
        query_sets[mode] = set(ids)
    reports = {mode: evaluate_queries(rows) for mode, rows in grouped.items()}
    paired = all(ids == next(iter(query_sets.values())) for ids in query_sets.values())
    deltas = None
    if paired and {"baseline", "labels"}.issubset(reports) and all(
            reports[mode]["evaluation_complete"] for mode in ["baseline", "labels"]):
        left = {row["base_query_id"]: row for row in grouped["baseline"]}
        right = {row["base_query_id"]: row for row in grouped["labels"]}
        if any(left[q]["expected_answerable"] != right[q]["expected_answerable"] for q in left):
            raise ValueError("Answerability must agree across paired modes")
        deltas = {key: reports["labels"]["metrics"][key] - reports["baseline"]["metrics"][key]
                  for key in QUERY_TARGETS}
    return {"record_count": len(records), "by_mode": reports,
            "paired_query_sets_match": paired, "label_minus_baseline": deltas}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("judgments",type=Path)
    parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    rows=[json.loads(line) for line in args.judgments.read_text().splitlines() if line.strip()]
    report=evaluate_query_modes(rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    summaries = report.get("by_mode", {"single": report})
    print(json.dumps({mode: {key: part[key] for key in ["passed", "evaluation_complete", "metrics"]}
                      for mode, part in summaries.items()}))
