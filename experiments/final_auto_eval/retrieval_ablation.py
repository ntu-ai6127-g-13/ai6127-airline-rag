#!/usr/bin/env python3
"""Ablate the retrieval innovations of report Section 4.1 on the E1 units (report E1b).

Configurations, each ranked by hybrid BM25 + MiniLM retrieval with RRF:
  v0_raw_question     the question as the user typed it, whole corpus (first version)
  v1_keywords         English "airline cabin attribute" keywords, whole corpus
  v2_constraints      + R1: airline, cabin and other parsed fields as hard filters
  v3_expansion        + R2: sub-attribute expansion terms as the query (deployed retriever)
  v3_minus_r1         v3 without the filters (expansion terms plus airline and cabin words)

Relevance is judged independently of the expansion vocabulary: a hit counts only if
it satisfies the parsed constraints (checked from metadata) and the local LLM, given
only the attribute's English name, answers yes to "is this sentence about it?".
Judgements are cached so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.final_auto_eval.run import (  # noqa: E402
    CASES, LABELS, filtered_hits, metadata_mask, qrels_for, retrieval_units,
)
from src.config import CHUNKS  # noqa: E402
from src.retriever import rrf_fuse, search_bm25, search_dense  # noqa: E402

ENDPOINT = "http://127.0.0.1:8765"
CONFIGS = ("v0_raw_question", "v1_keywords", "v2_constraints", "v3_expansion", "v3_minus_r1")
STEPS = (("v0_raw_question", "v1_keywords"), ("v1_keywords", "v2_constraints"),
         ("v2_constraints", "v3_expansion"), ("v3_minus_r1", "v3_expansion"))
# English names as shown in the interface (app/streamlit_app.py), for the topics of E1.
TOPIC_NAMES = {
    "legroom": "seat legroom", "width": "seat width", "recline": "seat recline and lie-flat capability",
    "seat_cleanliness": "seat cleanliness", "variety": "meal variety and choice",
    "portion": "meal portion size", "quality": "food quality and taste", "drinks": "onboard drinks",
    "content": "entertainment content", "wifi": "onboard Wi-Fi", "attitude": "cabin crew attitude",
    "baggage": "baggage service", "delay_handling": "delay and cancellation handling",
    "refund": "refunds and changes", "support": "customer support", "checkin": "check-in service",
    "punctuality": "punctuality",
}
JUDGE_PROMPT = (
    "A traveller searched airline reviews for: {topic}.\n"
    "Sentence: \"{text}\"\n"
    "Does this sentence give information or an opinion about {topic} specifically? Answer yes or no.")


def hybrid(query: str, allowed: set[str] | None) -> list[str]:
    if allowed is None:
        bm25, dense = search_bm25(query, 100), search_dense(query, 100)
    else:
        bm25, _ = filtered_hits("bm25", query, allowed)
        dense, _ = filtered_hits("dense", query, allowed)
    return [key for key, _ in rrf_fuse(bm25, dense, top_k=100)][:10]


def judge(topic: str, text: str) -> int:
    prompt = JUDGE_PROMPT.format(topic=TOPIC_NAMES[topic], text=text)
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": 4,
                       "temperature": 0.0}).encode("utf-8")
    request = urllib.request.Request(ENDPOINT + "/generate", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        reply = json.loads(response.read().decode("utf-8"))["text"]
    answer = reply.strip().lower()
    if not answer.startswith(("yes", "no")):
        raise RuntimeError(f"unparseable judge reply: {reply!r}")
    return int(answer.startswith("yes"))


def metrics(ids: list[str], grades: dict[str, int]) -> dict:
    # Binary gains; the ideal ranking is over every relevant hit pooled from all configurations.
    relevant = [grades.get(key, 0) == 1 for key in ids[:10]]
    ideal = min(10, sum(grades.values()))
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal + 1))
    dcg = sum(1 / math.log2(rank + 1) for rank, hit in enumerate(relevant, 1) if hit)
    return {"precision_at_5": sum(relevant[:5]) / 5, "precision_at_10": sum(relevant) / 10,
            "mrr": next((1 / rank for rank, hit in enumerate(relevant, 1) if hit), 0.0),
            "ndcg_at_10": dcg / idcg if idcg else 0.0}


def bootstrap(data: pd.DataFrame, left: str, right: str, column: str, repeats: int = 10000) -> dict:
    pivot = data.pivot(index="unit_id", columns="config", values=column)
    values = (pivot[right] - pivot[left]).to_numpy(dtype=float)
    samples = np.random.default_rng(6127).choice(values, size=(repeats, len(values))).mean(axis=1)
    return {"mean_delta": round(float(values.mean()), 4),
            "ci95": [round(float(np.percentile(samples, 2.5)), 4),
                     round(float(np.percentile(samples, 97.5)), 4)]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    if (out / "results.json").exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "judgements.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            row = json.loads(line)
            cache[(row["topic"], row["chunk_id"])] = row["grade"]

    chunks = pd.read_parquet(CHUNKS, columns=["chunk_id", "text", "airline", "cabin",
                                              "traveller_type", "route_from", "route_to", "date"])
    labels = pd.read_parquet(LABELS, columns=["chunk_id", "aspect", "subjectivity", "polarity"])
    frame = chunks.merge(labels, on="chunk_id", how="inner", validate="one_to_one")
    text_of = dict(zip(frame.chunk_id.astype(str), frame.text.astype(str)))
    units = retrieval_units(json.loads(CASES.read_text()), frame)

    rows, runs = [], []
    with cache_path.open("a") as cache_file:
        for number, unit in enumerate(units, 1):
            allowed = set(frame.loc[metadata_mask(frame, unit["filters"]), "chunk_id"].astype(str))
            cabin = unit["filters"].get("cabin", "")
            keywords = f"{unit['airline']} {cabin} {TOPIC_NAMES[unit['topic']]}".strip()
            ranked = {
                "v0_raw_question": hybrid(unit["question"], None),
                "v1_keywords": hybrid(keywords, None),
                "v2_constraints": hybrid(TOPIC_NAMES[unit["topic"]], allowed),
                "v3_expansion": hybrid(unit["query"], allowed),
                "v3_minus_r1": hybrid(f"{unit['airline']} {cabin} {unit['query']}", None),
            }
            pooled = {key for ids in ranked.values() for key in ids if key in allowed}
            for key in sorted(pooled):
                if (unit["topic"], key) not in cache:
                    cache[(unit["topic"], key)] = judge(unit["topic"], text_of[key])
                    cache_file.write(json.dumps({"topic": unit["topic"], "chunk_id": key,
                                                 "grade": cache[(unit["topic"], key)]}) + "\n")
                    cache_file.flush()
            # Hits outside the parsed constraints are irrelevant whatever their text says.
            grades = {key: cache[(unit["topic"], key)] for key in pooled}
            proxy = qrels_for(unit, frame)
            print(f"[{number:02d}/{len(units)}] {unit['unit_id']} pooled={len(pooled)}", flush=True)
            for config, ids in ranked.items():
                rows.append({"unit_id": unit["unit_id"], "case_id": unit["case_id"],
                             "topic": unit["topic"], "config": config,
                             "in_constraints_at_10": sum(key in allowed for key in ids) / max(len(ids), 1),
                             "proxy_direct_at_10": sum(key in proxy for key in ids) / max(len(ids), 1),
                             **metrics(ids, grades)})
                runs += [{"unit_id": unit["unit_id"], "config": config, "rank": rank, "chunk_id": key,
                          "in_constraints": key in allowed, "grade": grades.get(key, 0),
                          "text": text_of[key]} for rank, key in enumerate(ids, 1)]

    data = pd.DataFrame(rows)
    columns = ["in_constraints_at_10", "precision_at_5", "precision_at_10", "mrr", "ndcg_at_10",
               "proxy_direct_at_10"]
    summary = {config: {column: round(float(group[column].mean()), 4) for column in columns}
               for config, group in data.groupby("config", sort=False)}
    deltas = {f"{right}_minus_{left}": {column: bootstrap(data, left, right, column)
                                        for column in ("precision_at_5", "ndcg_at_10")}
              for left, right in STEPS}
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "units": len(units),
              "judge": "Qwen3-4B-Instruct-2507 (local MLX server), temperature 0, yes/no, attribute name only",
              "judged_sentences": len(cache), "summary": summary, "paired_deltas": deltas}
    data.to_csv(out / "per_unit.csv", index=False)
    pd.DataFrame(runs).to_csv(out / "runs_top10.csv", index=False)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
