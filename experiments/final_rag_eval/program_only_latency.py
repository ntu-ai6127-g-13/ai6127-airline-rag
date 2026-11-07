"""Time the program-written answer path (LLM off) on the evaluation questions (E2, E8)."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiments.final_rag_eval.run import CHUNKS, LABELS, SELECTED_IDS, build_result  # noqa: E402
from src.grounded_qa import answer_question  # noqa: E402
from src.question_parser import parse_question  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    frame = pd.read_parquet(CHUNKS, columns=["airline", "route_from", "route_to"])
    known = sorted(frame.airline.dropna().astype(str).unique())
    origins = sorted(frame.route_from.dropna().astype(str).unique())
    destinations = sorted(frame.route_to.dropna().astype(str).unique())
    rows = []
    for case in json.loads(args.cases.read_text()):
        if case["id"] not in SELECTED_IDS:
            continue
        started = time.monotonic()
        parsed = parse_question(case["question"], known_airlines=known,
                                origins=origins, destinations=destinations)
        result = build_result(parsed, top_k=12, labels=LABELS)
        result["question"] = case["question"]
        result["language"] = case.get("language", "zh")
        result["parsed_question"] = parsed.to_dict()
        answer = answer_question(case["question"], result, use_model=False)
        rows.append({"case_id": case["id"], "seconds": round(time.monotonic() - started, 3),
                     "answer_mode": answer["mode"]})
        print(rows[-1], flush=True)
    seconds = [row["seconds"] for row in rows]
    summary = {"questions": len(rows), "mean_seconds": round(statistics.mean(seconds), 3),
               "mean_seconds_warm": round(statistics.mean(seconds[1:]), 3),
               "median_seconds": round(statistics.median(seconds), 3), "rows": rows}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
