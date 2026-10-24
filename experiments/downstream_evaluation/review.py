"""Validate explicit human annotations; missing judgments never count as passes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.downstream_evaluation.common import read, write


def validate_review(row):
    if row.get("review_complete") is not True:
        raise ValueError("Review is not marked complete")
    if not str(row.get("reviewer_id") or "").strip():
        raise ValueError("Reviewer ID is required")
    if type(row.get("answerable_from_evidence")) is not bool:
        raise ValueError("Answerability needs explicit human judgment")
    if type(row.get("answer_relevance_0_to_4")) is not int or not 0 <= row["answer_relevance_0_to_4"] <= 4:
        raise ValueError("Answer relevance must be an integer from 0 to 4")
    ids = {str(e["citation_id"]) for e in row["evidence"]}
    if set(row["evidence_relevance"]) != ids or any(type(v) is not bool for v in row["evidence_relevance"].values()):
        raise ValueError("Every displayed evidence passage needs a relevance judgment")
    if not row["claims"] and row.get("no_checkable_claims") is not True:
        raise ValueError("Enter atomic claims or explicitly mark no checkable claims")
    for claim in row["claims"]:
        if not str(claim.get("claim") or "").strip() or type(claim.get("supported")) is not bool:
            raise ValueError("Each claim needs text and an explicit support judgment")
        if not str(claim.get("reason") or "").strip():
            raise ValueError("Each claim needs an evidence/aggregate reason")
    if not row["answerable_from_evidence"] and type(row.get("correct_abstention")) is not bool:
        raise ValueError("Unanswerable cases require an abstention judgment")


def summarize(packet):
    records = packet["records"]
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate annotation IDs")
    complete, incomplete = [], []
    for row in records:
        try:
            validate_review(row)
            complete.append(row)
        except ValueError as error:
            incomplete.append({"id": row["id"], "reason": str(error)})
    total = sum(len(r["claims"]) for r in complete)
    supported = sum(c["supported"] for r in complete for c in r["claims"])
    def mean(values):
        return sum(values)/len(values) if values else None
    return {"records": len(records), "reviewed": len(complete), "pending": incomplete,
            "evaluation_complete": len(complete) == len(records),
            "independent_acceptance_passed": False,
            "provenance": "human_explicit_judgments; independence not certified",
            "reviewed_subset_metrics": {
                "faithfulness_micro": supported/total if total else None,
                "answer_relevance": mean([r["answer_relevance_0_to_4"]/4 for r in complete]),
                "context_precision_macro": mean([sum(r["evidence_relevance"].values())/len(r["evidence_relevance"])
                                                  for r in complete if r["evidence_relevance"]]),
                "correct_abstention_rate": mean([float(r["correct_abstention"]) for r in complete
                                                   if not r["answerable_from_evidence"]])}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(read(args.packet))
    write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
