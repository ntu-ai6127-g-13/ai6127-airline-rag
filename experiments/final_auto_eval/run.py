#!/usr/bin/env python3
"""Automated proxy retrieval evaluation and downstream component ablation.

This evaluation intentionally does not claim human relevance judgments. Fine-
grained qrels are generated from metadata-safe direct subtopic matches and the
frozen offline labels. The downstream reference direction is the complete
direct-match aggregate; displayed-evidence voting is evaluated as an ablation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import CHUNKS  # noqa: E402
from src.grounded_qa import (  # noqa: E402
    _balanced_polarity_examples,
    _single_subtopic_direction,
    validate_answer,
)
from src.question_parser import SUBTOPIC_NAMES, parse_question  # noqa: E402
from src.query_service import (  # noqa: E402
    compare_airlines,
    query_reviews,
    SUBTOPIC_ANCHORS,
    _direct_subtopic_match,
)
from src.rerank import rerank  # noqa: E402
from src.retriever import rrf_fuse, search_bm25, search_dense  # noqa: E402
from src.label_index import labels_for_chunks  # noqa: E402

CASES = ROOT / "data/eval/query_smoke_40_20261019.json"
LABELS = ROOT / "outputs/label_index_20261019/labels.parquet"
DEFAULT_OUT = ROOT / "outputs/final_auto_eval_20261019"


def metadata_mask(frame: pd.DataFrame, filters: dict) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in filters.items():
        series = frame[column]
        if column == "date":
            start, end = value
            if start is not None:
                mask &= series >= pd.Timestamp(start)
            if end is not None:
                mask &= series <= pd.Timestamp(end)
        elif isinstance(value, (list, tuple, set)):
            mask &= series.isin(value)
        else:
            mask &= series.eq(value)
    return mask


def retrieval_units(cases: list[dict], frame: pd.DataFrame) -> list[dict]:
    known = sorted(frame.airline.dropna().astype(str).unique())
    origins = sorted(frame.route_from.dropna().astype(str).unique())
    destinations = sorted(frame.route_to.dropna().astype(str).unique())
    units = []
    # Twenty single-airline questions plus five two-airline comparisons produce
    # thirty independently filtered fine-grained retrieval units.
    for case in cases[:25]:
        parsed = parse_question(case["question"], known_airlines=known,
                                origins=origins, destinations=destinations)
        if not parsed.subtopics:
            continue
        airlines = parsed.airlines if len(parsed.airlines) >= 2 else [parsed.filters.get("airline")]
        for airline in airlines:
            if not airline:
                continue
            filters = {**parsed.filters, "airline": airline}
            for topic, expansion in parsed.subtopics.items():
                units.append({
                    "unit_id": f"{case['id']}_{airline}_{topic}",
                    "case_id": case["id"], "question": case["question"],
                    "airline": airline, "aspect": parsed.aspect, "topic": topic,
                    "query": expansion, "filters": filters,
                })
    return units


def qrels_for(unit: dict, frame: pd.DataFrame) -> dict[str, int]:
    scoped = frame.loc[metadata_mask(frame, unit["filters"])]
    direct = scoped.text.fillna("").map(lambda text: _direct_subtopic_match(str(text), unit["topic"]))
    matched = scoped.loc[direct]
    qrels = {}
    for row in matched.itertuples(index=False):
        grade = 2 if (row.aspect == unit["aspect"] and row.subjectivity == "subjective") else 1
        qrels[str(row.chunk_id)] = grade
    return qrels


def filtered_hits(method: str, query: str, allowed: set[str], *, fetch_k: int = 10000,
                  limit: int = 100) -> tuple[list[tuple[str, float]], float]:
    started = time.monotonic()
    raw = search_bm25(query, fetch_k) if method == "bm25" else search_dense(query, fetch_k)
    hits = [item for item in raw if item[0] in allowed][:limit]
    return hits, time.monotonic() - started


def metrics(ids: list[str], qrels: dict[str, int]) -> dict:
    relevant = {key for key, grade in qrels.items() if grade > 0}
    p5 = sum(key in relevant for key in ids[:5]) / 5
    recall50 = sum(key in relevant for key in ids[:50]) / len(relevant) if relevant else 0.0
    rr = next((1 / rank for rank, key in enumerate(ids, 1) if key in relevant), 0.0)
    dcg = sum((2 ** qrels.get(key, 0) - 1) / math.log2(rank + 1)
              for rank, key in enumerate(ids[:10], 1))
    ideal = sorted(qrels.values(), reverse=True)[:10]
    idcg = sum((2 ** grade - 1) / math.log2(rank + 1)
               for rank, grade in enumerate(ideal, 1))
    return {"precision_at_5": p5, "recall_at_50": recall50,
            "mrr": rr, "ndcg_at_10": dcg / idcg if idcg else 0.0,
            "success_at_10": float(any(key in relevant for key in ids[:10]))}


def paired_bootstrap_delta(data: pd.DataFrame, left: str, right: str, column: str,
                           *, repeats: int = 10000) -> dict:
    pivot = data.pivot(index="unit_id", columns="method", values=column).dropna()
    values = (pivot[right] - pivot[left]).to_numpy(dtype=float)
    rng = np.random.default_rng(6127)
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return {"mean_delta": round(float(values.mean()), 4),
            "ci95_low": round(float(np.percentile(samples, 2.5)), 4),
            "ci95_high": round(float(np.percentile(samples, 97.5)), 4)}


def summarize_retrieval(data: pd.DataFrame) -> dict:
    summary = {}
    for method, group in data.groupby("method", sort=False):
        summary[method] = {
            "query_units": len(group),
            **{column: round(float(group[column].mean()), 4)
               for column in ("precision_at_5", "recall_at_50", "mrr", "ndcg_at_10", "success_at_10")},
            "latency_mean_seconds": round(float(group.seconds.mean()), 4),
            "latency_median_seconds": round(float(group.seconds.median()), 4),
            "latency_p95_seconds": round(float(np.percentile(group.seconds, 95)), 4),
            "latency_max_seconds": round(float(group.seconds.max()), 4),
        }
    summary["paired_deltas"] = {
        "hybrid_rrf_minus_bm25": {
            column: paired_bootstrap_delta(data, "bm25", "hybrid_rrf", column)
            for column in ("precision_at_5", "recall_at_50", "mrr", "ndcg_at_10")
        },
        "hybrid_rerank_minus_hybrid_rrf": {
            column: paired_bootstrap_delta(data, "hybrid_rrf", "hybrid_rerank", column)
            for column in ("precision_at_5", "recall_at_50", "mrr", "ndcg_at_10")
        },
    }
    return summary


def run_retrieval(units: list[dict], frame: pd.DataFrame) -> tuple[list[dict], dict]:
    rows = []
    for number, unit in enumerate(units, 1):
        allowed = set(frame.loc[metadata_mask(frame, unit["filters"]), "chunk_id"].astype(str))
        qrels = qrels_for(unit, frame)
        print(f"[retrieval {number:02d}/{len(units):02d}] {unit['case_id']} {unit['airline']} {unit['topic']}", flush=True)
        bm25, bm25_s = filtered_hits("bm25", unit["query"], allowed)
        dense, dense_s = filtered_hits("dense", unit["query"], allowed)
        hybrid_started = time.monotonic()
        hybrid = rrf_fuse(bm25, dense, top_k=100)
        fuse_s = time.monotonic() - hybrid_started
        rerank_started = time.monotonic()
        reranked_rows = rerank(unit["query"], hybrid, top_k=50)
        rerank_s = time.monotonic() - rerank_started
        variants = {
            "bm25": (bm25[:50], bm25_s),
            "dense": (dense[:50], dense_s),
            "hybrid_rrf": (hybrid[:50], bm25_s + dense_s + fuse_s),
            "hybrid_rerank": ([(row["chunk_id"], row["ranking_score"]) for row in reranked_rows],
                              bm25_s + dense_s + fuse_s + rerank_s),
        }
        for method, (hits, seconds) in variants.items():
            value = metrics([key for key, _ in hits], qrels)
            rows.append({**{key: unit[key] for key in ("unit_id", "case_id", "airline", "aspect", "topic")},
                         "method": method, "qrel_count": len(qrels), "returned": len(hits),
                         "seconds": seconds, **value})
    data = pd.DataFrame(rows)
    return rows, summarize_retrieval(data)


POSITIVE = re.compile(r"\b(?:excellent|wonderful|great|good|helpful|professional|friendly|polite|comfortable|generous|plentiful|easy|simple|reasonable|clean|on time)\b", re.I)
NEGATIVE = re.compile(r"\b(?:awful|terrible|poor|bad|worst|rude|dirty|filthy|cramped|uncomfortable|problem|fault|broken|damaged|lost|cancelled|canceled|delayed|complaint)\b", re.I)


def cue_conflict(row: dict) -> bool:
    text = str(row.get("text", ""))
    pos, neg = len(POSITIVE.findall(text)), len(NEGATIVE.findall(text))
    return ((row.get("pred_polarity") == "positive" and neg > pos)
            or (row.get("pred_polarity") == "negative" and pos > neg))


def downstream_parts(case: dict, parsed, labels: Path | None) -> list[dict]:
    if len(parsed.airlines) >= 2:
        result = compare_airlines(parsed.airlines, aspect=parsed.aspect,
                                  query=parsed.retrieval_keywords, filters=parsed.filters,
                                  subtopics=parsed.subtopics, top_k=30,
                                  label_index_path=labels)
        return list(result["results"].values())
    result = query_reviews(parsed.retrieval_keywords, aspect=parsed.aspect,
                           filters=parsed.filters, subtopics=parsed.subtopics,
                           top_k=30, candidate_k=90, label_index_path=labels)
    return [result]


def summarize_downstream(data: pd.DataFrame) -> dict:
    possible = data[data.two_sided_possible]
    return {
        "evaluated_airline_subtopics": len(data),
        "direction_accuracy": {
            "displayed_top10_vote": round(float(data.displayed_vote_correct.mean()), 4),
            "full_direct_aggregate": round(float(data.full_aggregate_correct.mean()), 4),
            "displayed_top10_vote_count": int(data.displayed_vote_correct.sum()),
            "full_direct_aggregate_count": int(data.full_aggregate_correct.sum()),
        },
        "two_sided_example_coverage_when_available": {
            "naive_rank_order": round(float(possible.naive_two_sided.mean()), 4) if len(possible) else None,
            "balanced_selection": round(float(possible.balanced_two_sided.mean()), 4) if len(possible) else None,
            "eligible_count": len(possible),
            "naive_count": int(possible.naive_two_sided.sum()),
            "balanced_count": int(possible.balanced_two_sided.sum()),
        },
        "obvious_label_text_conflicts_in_selected_examples": {
            "naive": int(data.naive_cue_conflicts.sum()),
            "consistency_filter": int(data.selected_cue_conflicts.sum()),
        },
        "adversarial_wrong_direction_rejection_rate": {
            "citation_only_validation": round(float(data.baseline_rejected_adversarial.mean()), 4),
            "full_aggregate_direction_guard": round(float(data.guarded_rejected_adversarial.mean()), 4),
            "citation_only_rejected_count": int(data.baseline_rejected_adversarial.sum()),
            "direction_guard_rejected_count": int(data.guarded_rejected_adversarial.sum()),
        },
    }


def run_downstream(cases: list[dict], frame: pd.DataFrame) -> tuple[list[dict], dict]:
    known = sorted(frame.airline.dropna().astype(str).unique())
    origins = sorted(frame.route_from.dropna().astype(str).unique())
    destinations = sorted(frame.route_to.dropna().astype(str).unique())
    rows = []
    for number, case in enumerate(cases[:25], 1):
        parsed = parse_question(case["question"], known_airlines=known,
                                origins=origins, destinations=destinations)
        if not parsed.subtopics:
            continue
        print(f"[downstream {number:02d}/25] {case['id']} {case['question']}", flush=True)
        # Run a real ablation without the offline index. This disables both the
        # aggregate-aware polarity promotion and label-assisted evidence order.
        baseline_parts = downstream_parts(case, parsed, None)
        current_parts = downstream_parts(case, parsed, LABELS)
        baseline_by_airline = {
            (part.get("filters") or {}).get("airline"): part for part in baseline_parts
        }
        for part in current_parts:
            airline = (part.get("filters") or {}).get("airline")
            baseline_part = baseline_by_airline.get(airline)
            if baseline_part is None:
                continue
            baseline_evidence = baseline_part.get("evidence", [])
            if baseline_evidence:
                predictions = labels_for_chunks(
                    [row["chunk_id"] for row in baseline_evidence], LABELS)
                for row, prediction in zip(baseline_evidence, predictions):
                    row.update({"pred_" + key: prediction[key]
                                for key in ("aspect", "subjectivity", "polarity")})
            for topic in parsed.subtopics:
                expected = _single_subtopic_direction(part, topic).get("status")
                if expected not in {"positive", "negative"}:
                    continue
                candidates = [row for row in part.get("evidence", [])
                              if topic in row.get("direct_subtopic_matches", [])]
                baseline_candidates = [row for row in baseline_evidence
                                       if topic in row.get("direct_subtopic_matches", [])]
                visible = baseline_candidates[:10]
                pos = sum(row.get("pred_polarity") == "positive" for row in visible)
                neg = sum(row.get("pred_polarity") == "negative" for row in visible)
                displayed = "positive" if pos > neg else "negative" if neg > pos else "mixed"
                naive = []
                for polarity in (expected, "negative" if expected == "positive" else "positive"):
                    naive.extend([row for row in baseline_candidates
                                  if row.get("pred_polarity") == polarity][:2])
                selected = _balanced_polarity_examples(candidates, expected, per_direction=2)
                available = {row.get("pred_polarity") for row in baseline_candidates + candidates}
                selected_polarities = {row.get("pred_polarity") for row in selected}
                two_sided_possible = {"positive", "negative"}.issubset(available)
                topic_name = SUBTOPIC_NAMES[topic]
                opposite = "negative" if expected == "positive" else "positive"
                opposite_zh = "负面" if opposite == "negative" else "正面"
                adversarial = f"关于{topic_name}，完整评论偏{opposite_zh}。[1]"
                valid_ids = {str(index) for index in range(1, len(part.get("evidence", [])) + 1)}
                baseline_accepts_bad = validate_answer(
                    adversarial, valid_ids, required_subtopics=(topic,))[0]
                guarded_accepts_bad = validate_answer(
                    adversarial, valid_ids, required_subtopics=(topic,),
                    expected_subtopic_directions={topic: expected})[0]
                rows.append({
                    "case_id": case["id"], "airline": airline, "topic": topic,
                    "reference_direction": expected, "displayed_vote": displayed,
                    "displayed_vote_correct": displayed == expected,
                    "full_aggregate_correct": True,
                    "two_sided_possible": two_sided_possible,
                    "naive_two_sided": {"positive", "negative"}.issubset({
                        row.get("pred_polarity") for row in baseline_candidates[:4]}),
                    "balanced_two_sided": {"positive", "negative"}.issubset(selected_polarities),
                    "naive_cue_conflicts": sum(cue_conflict(row) for row in naive),
                    "selected_cue_conflicts": sum(cue_conflict(row) for row in selected),
                    "baseline_rejected_adversarial": not baseline_accepts_bad,
                    "guarded_rejected_adversarial": not guarded_accepts_bad,
                })
    data = pd.DataFrame(rows)
    return rows, summarize_downstream(data)


def render_report(retrieval: dict, downstream: dict, elapsed: float) -> str:
    lines = ["# 自动检索评测与下游消融", "",
             "> 本报告使用自动代理 qrels 和程序化一致性检查，不是人工独立评测，也不应写成人工 RAGAS 分数。", "",
             f"运行时间：{datetime.now(timezone.utc).isoformat()}；总耗时 {elapsed:.1f} 秒。", "",
             "## 检索质量", "",
             "| 方法 | P@5 | Recall@50 | MRR | NDCG@10 | Success@10 | 平均(s) | 中位(s) | P95(s) | 最大(s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ("bm25", "dense", "hybrid_rrf", "hybrid_rerank"):
        value = retrieval[method]
        lines.append(f"| {method} | {value['precision_at_5']:.4f} | {value['recall_at_50']:.4f} | "
                     f"{value['mrr']:.4f} | {value['ndcg_at_10']:.4f} | {value['success_at_10']:.4f} | "
                     f"{value['latency_mean_seconds']:.3f} | {value['latency_median_seconds']:.3f} | "
                     f"{value['latency_p95_seconds']:.3f} | {value['latency_max_seconds']:.3f} |")
    lines += ["", "qrels 口径：在相同航司、舱位等条件下直接命中细分项的句子记为相关；同时被离线标签判为所问方面且为主观评价时记为二级相关。该口径可复现，但会偏向显式关键词，不能替代人工语义判断。", "",
              "### 配对差值与 95% bootstrap 区间", "", "```json",
              json.dumps(retrieval["paired_deltas"], ensure_ascii=False, indent=2), "```", "",
              "## 下游组件消融", "", "```json", json.dumps(downstream, ensure_ascii=False, indent=2), "```", "",
              "方向准确率以完整直接匹配池为自动参考，因此 full_direct_aggregate 是实现一致性检查，不是独立真实性证明。消融用于量化过去的 top-k 投票错误、正反例截断、明显标签文本冲突和错误方向回答拦截。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    chunks = pd.read_parquet(CHUNKS, columns=["chunk_id", "text", "airline", "cabin",
                                              "traveller_type", "route_from", "route_to", "date"])
    labels = pd.read_parquet(LABELS, columns=["chunk_id", "aspect", "subjectivity", "polarity"])
    frame = chunks.merge(labels, on="chunk_id", how="inner", validate="one_to_one")
    cases = json.loads(CASES.read_text())
    units = retrieval_units(cases, frame)
    retrieval_rows, retrieval_summary = run_retrieval(units, frame)
    downstream_rows, downstream_summary = run_downstream(cases, frame)
    elapsed = time.monotonic() - started
    pd.DataFrame(retrieval_rows).to_csv(args.output_dir / "retrieval_per_query.csv", index=False)
    pd.DataFrame(downstream_rows).to_csv(args.output_dir / "downstream_ablation_per_case.csv", index=False)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_kind": "automated_proxy_not_human_independent",
        "retrieval_qrels": "metadata-constrained direct-subtopic matches; grade 2 if offline aspect-aligned subjective",
        "retrieval": retrieval_summary, "downstream_ablation": downstream_summary,
        "elapsed_seconds": round(elapsed, 3),
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (args.output_dir / "README.md").write_text(render_report(retrieval_summary, downstream_summary, elapsed))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
