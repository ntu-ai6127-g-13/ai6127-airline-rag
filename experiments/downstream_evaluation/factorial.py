#!/usr/bin/env python3
"""E6: full 2^3 factorial ablation of the three answer-side components.

  X  full-pool statistics: the direct-match label tally and the overall direction it implies are
     given to generation, validation and fallback; off, they are hidden from all three.
  Y  balanced, label-consistent examples: two majority-direction examples and up to two
     counter-examples whose wording agrees with their label; off, the top four candidates.
  Z  validation, one repair and the program-written fallback; off, the draft is returned unchecked.

Each question has one frozen pool of up to 30 candidates ranked without labels. The draft of an
X/Y setting is shared by its Z-off and Z-on variants. The local Qwen3-4B both generates and judges
(E5 rubric). Every model call is cached, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import statistics
import sys
import textwrap
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.final_rag_eval.run import LABELS, build_result, inject_labels, judge  # noqa: E402
from src.config import CHUNKS  # noqa: E402
from src.grounded_qa import (  # noqa: E402
    _CITATION_RE, _SUBTOPIC_NAMES_EN, DEFAULT_ENDPOINT, _balanced_polarity_examples, _generate,
    _prompt, _single_subtopic_direction, collect_evidence, fallback_answer, validate_answer,
)
from src.question_parser import parse_question  # noqa: E402

CASES = ROOT / "data/eval/query_smoke_40_20261019.json"
CASE_IDS = [f"Q{number:02d}" for number in range(1, 21)]
SEED, REPEATS, POOL, BUDGET = 6127, 2000, 30, 4
CONFIGS = ["".join(bits) for bits in itertools.product("01", repeat=3)]
NAMES = {"000": "Baseline", "100": "X only", "010": "Y only", "001": "Z only", "110": "X + Y",
         "101": "X + Z", "011": "Y + Z", "111": "X + Y + Z (deployed)"}
METRICS = ("faithfulness", "answer_relevance", "context_precision")
# Attribute wording of the question; the parser must map it back to the case's sub-attribute.
WORDING = {"legroom": "legroom", "width": "seat width", "recline": "seat recline",
           "seat_cleanliness": "seat cleanliness", "variety": "meal variety", "portion": "meal portion size",
           "quality": "food quality", "drinks": "coffee, tea and other drinks", "content": "entertainment content",
           "wifi": "Wi-Fi", "attitude": "cabin crew attitude", "baggage": "baggage handling",
           "delay_handling": "delay handling", "refund": "refund and rebooking",
           "support": "customer service", "checkin": "check-in", "punctuality": "flights running on time"}
X_OFF_RULE = ("6. No statistics over all matching reviews are supplied. Describe what the displayed evidence "
              "says about each requested subtopic, citing it; do not state counts, percentages or an overall "
              "direction for the airline.")


def questions(frame: pd.DataFrame) -> list[dict]:
    known = sorted(frame.airline.dropna().astype(str).unique())
    cases = {case["id"]: case for case in json.loads(CASES.read_text())}
    result = []
    for ident in CASE_IDS:
        case = cases[ident]
        topic = case["subtopics"][0]
        source = parse_question(case["question"], known_airlines=known, origins=[], destinations=[])
        airline, cabin = source.filters["airline"], source.filters["cabin"]
        question = f"What do historical reviews say about {WORDING[topic]} on {airline} in {cabin}?"
        parsed = parse_question(question, known_airlines=known, origins=[], destinations=[])
        if (set(parsed.subtopics) != {topic} or parsed.filters.get("airline") != airline
                or parsed.filters.get("cabin") != cabin):
            raise ValueError(f"{ident}: English question parsed as {parsed.to_dict()}")
        result.append({"id": ident, "source_question": case["question"], "question": question,
                       "airline": airline, "cabin": cabin, "topic": topic, "parsed": parsed})
    return result


def select(pool: list[dict], topic: str, balanced: bool) -> list[dict]:
    if not balanced:
        return pool[:BUDGET]
    matches = [row for row in pool if topic in row.get("matched_subtopics", [])] or pool
    # The majority label of the candidates, not the full-pool tally, so Y does not depend on X.
    votes = Counter(row.get("pred_polarity") for row in matches)
    preferred = "negative" if votes["negative"] > votes["positive"] else "positive"
    return _balanced_polarity_examples(matches, preferred, per_direction=2)[:BUDGET]


def setting(item: dict, rows: list[dict], stats: dict, with_x: bool) -> dict:
    parsed = item["parsed"]
    return {"type": "single", "language": "en", "answer_intent": "evaluation", "question": item["question"],
            "aspect": parsed.aspect, "filters": parsed.filters, "subtopics": parsed.subtopics,
            "evidence": [dict(row) for row in rows], "subtopic_analysis": stats if with_x else {}}


def messages_for(item: dict, result: dict, with_x: bool) -> list[dict]:
    messages = _prompt(item["question"], result, collect_evidence(result))
    if with_x:
        return messages
    system = re.sub(r"^6\. .*$", X_OFF_RULE, messages[0]["content"], count=1, flags=re.M)
    payload = json.loads(messages[1]["content"])
    for key in ("subtopic_coverage", "single_subtopic_directions", "full_label_distributions"):
        payload.pop(key, None)
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def validation_kwargs(result: dict, rows: list[dict], with_x: bool) -> dict:
    numbers = {token.replace(",", "") for row in rows
               for token in re.findall(r"\d[\d,]*(?:\.\d+)?", str(row.get("text", "")))}
    expected = {}
    if with_x:
        # The figures the generator was given, exactly as answer_question admits them.
        for topic, analysis in result["subtopic_analysis"].items():
            numbers.update(str(value) for value in analysis.values() if type(value) is int)
            summary = analysis.get("polarity_summary") or {}
            for value in summary.values():
                if type(value) is int:
                    numbers.add(str(value))
                elif isinstance(value, float):
                    numbers.update({f"{value:.4f}", f"{value*100:.1f}"})
            total = int(summary.get("labeled_review_count") or 0)
            for key in ("positive_review_count", "negative_review_count"):
                if total:
                    numbers.add(f"{int(summary.get(key) or 0) / total * 100:.1f}")
            expected[topic] = _single_subtopic_direction(result, topic)["status"]
    return {"rating_summary": None, "require_rating_winner": False,
            "required_subtopics": tuple(result["subtopics"]), "evidence_numbers": numbers,
            "allow_aggregate_counts": with_x, "language": "en", "expected_subtopic_directions": expected}


def fallback(item: dict, result: dict, rows: list[dict], with_x: bool) -> str:
    if with_x:
        return fallback_answer(item["question"], result, rows)
    # Without X the program fallback may only quote what is displayed.
    topic = item["topic"]
    shown = [row for row in rows if topic in row.get("matched_subtopics", [])] or rows
    examples = "; ".join(
        f'a passenger wrote, “{textwrap.shorten(str(row["text"]).strip(), width=155, placeholder="…")}”'
        f'[{row["citation_id"]}]' for row in shown)
    return (f"For {_SUBTOPIC_NAMES_EN[topic]} on {item['airline']} in {item['cabin']}, the displayed reviews "
            f"give these examples: {examples}.\n\nThese are individual historical experiences and do not "
            "represent every flight.")


def repair_instruction(errors: list[str], with_x: bool) -> str:
    text = ("The previous answer failed validation (" + "; ".join(errors) + "). Rewrite it in English only. "
            "Keep valid citations, remove unsupported numbers and scope claims, and put every requested subtopic "
            "or aspect in its own paragraph. ")
    if with_x:
        text += "The overall subtopic direction must follow the full aggregate, not the displayed examples. "
    return text + "Output only the corrected answer."


class Cache:
    """Append-only JSONL cache keyed by a hash of the call's inputs."""

    def __init__(self, path: Path):
        self.path = path
        self.items = {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                self.items[row["key"]] = row["value"]

    def get(self, parts, compute):
        key = hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key not in self.items:
            self.items[key] = compute()
            with self.path.open("a") as handle:
                handle.write(json.dumps({"key": key, "value": self.items[key]}, ensure_ascii=False) + "\n")
        return self.items[key]


def generate(cache: Cache, messages: list[dict], endpoint: str) -> dict:
    def call():
        payload = _generate(messages, endpoint=endpoint, timeout=180)
        return {"text": str(payload.get("text", "")).strip(), "seconds": float(payload.get("seconds", 0.0))}
    return cache.get(["generate", messages], call)


def interval(values: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    samples = rng.choice(values, size=(REPEATS, len(values))).mean(axis=1)
    return {"mean": round(float(values.mean()), 4),
            "ci95": [round(float(np.percentile(samples, 2.5)), 4), round(float(np.percentile(samples, 97.5)), 4)]}


def effects(pivot: pd.DataFrame) -> dict:
    """Per-question effects as defined in the report: a main effect is on minus off averaged over the
    other factors, a two-way interaction the change in one factor's effect when the other is switched
    on (averaged over the third), and XYZ the change of the XY interaction when Z is switched on."""
    sign = {"X": 0, "Y": 1, "Z": 2}
    result = {}
    for name in ("X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"):
        weights = np.array([np.prod([1 if config[sign[f]] == "1" else -1 for f in name]) for config in CONFIGS])
        result[name] = (pivot[CONFIGS].to_numpy(float) * weights).sum(axis=1) / 2 ** (3 - len(name))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args()
    out = args.output_dir
    if (out / "results.json").exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.endpoint + "/health", timeout=5) as response:
        health = json.loads(response.read())

    frame = pd.read_parquet(CHUNKS, columns=["airline"])
    items = questions(frame)
    protocol = {"questions": [{key: item[key] for key in ("id", "source_question", "question", "airline",
                                                           "cabin", "topic")} for item in items],
                "configs": NAMES, "pool_size": POOL, "citation_budget": BUDGET, "labels": str(LABELS.relative_to(ROOT)),
                "model": health, "temperature": 0.0, "max_tokens": 480, "x_off_rule": X_OFF_RULE,
                "judge": "experiments/final_rag_eval/run.py judge(), same aggregate facts for every configuration",
                "bootstrap": {"repeats": REPEATS, "seed": SEED, "unit": "question"}}
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text())["questions"] != protocol["questions"]:
        raise SystemExit("Questions differ from the frozen protocol")
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")

    calls, judgements = Cache(out / "model_calls.jsonl"), Cache(out / "judgements.jsonl")
    rows, answers, pools = [], [], []
    for number, item in enumerate(items, 1):
        print(f"[{number:02d}/{len(items)}] {item['id']} {item['question']}", flush=True)
        parsed = item["parsed"]
        stats = build_result(parsed, top_k=POOL, labels=LABELS)["subtopic_analysis"]
        pool = build_result(parsed, top_k=POOL, labels=None)["evidence"][:POOL]
        inject_labels(pool)
        pools.append({"id": item["id"], "subtopic_analysis": stats,
                      "pool": [{key: row.get(key) for key in ("chunk_id", "text", "matched_subtopics",
                                                              "direct_subtopic_matches", "pred_polarity")}
                               for row in pool]})
        facts = {"subtopic_analysis": stats}
        for x, y in itertools.product((0, 1), repeat=2):
            rows_xy = select(pool, item["topic"], bool(y))
            result = setting(item, rows_xy, stats, bool(x))
            evidence = collect_evidence(result)
            messages = messages_for(item, result, bool(x))
            draft = generate(calls, messages, args.endpoint)
            for z in (0, 1):
                config = f"{x}{y}{z}"
                record = {"id": item["id"], "config": config, "draft": draft["text"], "errors": [],
                          "repair": None, "mode": "draft_unchecked", "generation_seconds": draft["seconds"]}
                text = draft["text"]
                if z:
                    kwargs = validation_kwargs(result, evidence, bool(x))
                    valid, errors = validate_answer(text, {row["citation_id"] for row in evidence}, **kwargs)
                    record.update(mode="draft", errors=errors)
                    if not valid:
                        repair = generate(calls, messages + [
                            {"role": "assistant", "content": text},
                            {"role": "user", "content": repair_instruction(errors, bool(x))}], args.endpoint)
                        record["repair"] = repair["text"]
                        record["generation_seconds"] += repair["seconds"]
                        repaired_valid, repaired_errors = validate_answer(
                            repair["text"], {row["citation_id"] for row in evidence}, **kwargs)
                        record["repair_errors"] = repaired_errors
                        if repaired_valid:
                            text, record["mode"] = repair["text"], "repair"
                        else:
                            text, record["mode"] = fallback(item, result, evidence, bool(x)), "fallback"
                ids = [row["citation_id"] for row in evidence]
                scored = judgements.get(
                    ["judge", item["question"], text, [[row["citation_id"], row["text"]] for row in evidence]],
                    lambda: judge(item["question"], text, evidence, endpoint=args.endpoint, aggregate_facts=facts))
                record.update(answer=text, evidence_ids=ids, evidence_chunks=[row["chunk_id"] for row in evidence],
                              judge=scored)
                answers.append(record)
                rows.append({"id": item["id"], "config": config, "mode": record["mode"],
                             "evidence_count": len(ids),
                             "citation_count": len(set(_CITATION_RE.findall(text)) & set(ids)),
                             "generation_seconds": record["generation_seconds"],
                             **{metric: scored[metric] for metric in METRICS},
                             "total_claims": scored["total_claims"], "supported_claims": scored["supported_claims"],
                             "unsupported_claim_count": scored["unsupported_claim_count"]})

    data = pd.DataFrame(rows)
    summary = {}
    for config in CONFIGS:
        group = data.loc[data.config.eq(config)]
        summary[config] = {"name": NAMES[config], **{metric: round(float(group[metric].mean()), 4) for metric in METRICS},
                           "mean_evidence": round(float(group.evidence_count.mean()), 2),
                           "mean_citations": round(float(group.citation_count.mean()), 2),
                           "median_generation_seconds": round(float(group.generation_seconds.median()), 2),
                           "modes": dict(Counter(group["mode"]))}
    deltas, main_effects = {}, {}
    for metric in METRICS:
        pivot = data.pivot(index="id", columns="config", values=metric)
        deltas[metric] = {config: interval((pivot[config] - pivot["000"]).to_numpy(float))
                          for config in CONFIGS if config != "000"}
        main_effects[metric] = {name: interval(values) for name, values in effects(pivot).items()}
    z_on = data.loc[data.config.str.endswith("1")]
    audit = [answer["judge"] for answer in answers]
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(), "questions": len(items), "model": health,
        "drafts": len({(a["id"], a["config"][:2]) for a in answers}),
        "repair_attempts": int(z_on["mode"].isin(["repair", "fallback"]).sum()),
        "final_answers": len(answers), "judged_unique_answers": len(judgements.items),
        "z_on_modes": dict(Counter(z_on["mode"])),
        "z_on_modes_by_x": {f"X={x}": dict(Counter(z_on.loc[z_on.config.str[0].eq(x), "mode"])) for x in "01"},
        "summary": summary, "delta_vs_baseline": deltas, "effects": main_effects,
        "judge_audit": {"scorings": len(audit), "unsupported_count_inconsistent": sum(
            j["unsupported_claim_count"] != j["total_claims"] - j["supported_claims"] for j in audit)},
        "median_generation_seconds_overall": round(statistics.median(data.generation_seconds), 2),
    }
    data.to_csv(out / "per_answer.csv", index=False)
    (out / "pools.json").write_text(json.dumps(pools, ensure_ascii=False, indent=1, default=str) + "\n")
    with (out / "answers.jsonl").open("w") as handle:
        for answer in answers:
            handle.write(json.dumps(answer, ensure_ascii=False, default=str) + "\n")
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("drafts", "repair_attempts", "z_on_modes", "summary")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
