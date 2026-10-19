#!/usr/bin/env python3
"""Re-score saved answers while exposing trusted program aggregates to the judge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run import DEFAULT_ENDPOINT, collect_evidence, judge, render_readme, summarize_full


def aggregate_facts(result: dict) -> dict:
    return {
        "rating_summary": result.get("rating_summary"),
        "rating_summaries": result.get("rating_summaries"),
        "corpus_label_matches": result.get("corpus_label_matches"),
        "label_distributions": result.get("label_distributions"),
        "subtopic_analysis": result.get("subtopic_analysis"),
        "comparison_subtopic_analysis": {
            airline: part.get("subtopic_analysis")
            for airline, part in result.get("results", {}).items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args()
    artifacts = [json.loads(line) for line in
                 (args.output_dir / "answers.jsonl").read_text().splitlines() if line.strip()]
    metrics = pd.read_csv(args.output_dir / "per_question_metrics.csv").set_index("case_id")
    rewritten = []
    for number, item in enumerate(artifacts, 1):
        case = item["case"]
        print(f"[rejudge {number:02d}/{len(artifacts):02d}] {case['id']}", flush=True)
        result = item["result"]
        value = judge(case["question"], item["answer"]["text"], collect_evidence(result),
                      endpoint=args.endpoint, aggregate_facts=aggregate_facts(result))
        item["judge_without_aggregate_facts"] = item.get("judge")
        item["judge"] = value
        rewritten.append(item)
        for key, metric_value in value.items():
            metrics.loc[case["id"], key] = json.dumps(metric_value, ensure_ascii=False) \
                if isinstance(metric_value, list) else metric_value
    metrics.reset_index().to_csv(args.output_dir / "per_question_metrics.csv", index=False)
    with (args.output_dir / "answers.jsonl").open("w") as handle:
        for item in rewritten:
            handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    rows = metrics.reset_index().to_dict("records")
    summary = summarize_full(rows)
    (args.output_dir / "full_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    results_path = args.output_dir / "results.json"
    if results_path.exists():
        result = json.loads(results_path.read_text())
        result["full_system"] = summary
        result["judge_received_program_aggregate_facts"] = True
        results_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        (args.output_dir / "README.md").write_text(render_readme(
            summary, result.get("common_metric_ablation", {}),
            result.get("run_invocation_elapsed_seconds", result.get("elapsed_seconds", 0.0))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
