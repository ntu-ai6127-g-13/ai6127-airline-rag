#!/usr/bin/env python3
"""Create the submission eval.xls and agreement report from the frozen gold."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from itertools import combinations

import pandas as pd
import xlwt

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data/eval/gold_human_frozen_1000_20261019_v1.csv"
ANNOTATIONS = (ROOT / "outputs/gold_frozen_1000_20261019_v1/sources/"
               "original_annotations_snapshot.csv")
DEFAULT_XLS = ROOT / "data/eval/eval.xls"
DEFAULT_REPORT = ROOT / "experiments/final_rag_eval/agreement_and_eval.json"


def normalized(value) -> str:
    if pd.isna(value) or not str(value).strip():
        return "NA"
    return str(value).strip()


def fleiss_kappa(frame: pd.DataFrame, field: str) -> tuple[float, float, int]:
    groups = []
    pair_matches = pair_total = 0
    for _, group in frame.groupby("sentence_id", sort=False):
        labels = [normalized(value) for value in group[field]]
        if len(labels) != 3:
            raise ValueError(f"Expected three annotations, got {len(labels)}")
        groups.append(labels)
        for left, right in combinations(labels, 2):
            pair_matches += int(left == right)
            pair_total += 1
    categories = sorted({label for labels in groups for label in labels})
    n = 3
    per_item = []
    totals = {category: 0 for category in categories}
    for labels in groups:
        counts = {category: labels.count(category) for category in categories}
        per_item.append(sum(value * (value - 1) for value in counts.values()) / (n * (n - 1)))
        for category, value in counts.items():
            totals[category] += value
    observed = sum(per_item) / len(per_item)
    proportions = {category: value / (len(groups) * n) for category, value in totals.items()}
    expected = sum(value * value for value in proportions.values())
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return kappa, pair_matches / pair_total, len(groups)


def write_xls(frame: pd.DataFrame, path: Path) -> None:
    workbook = xlwt.Workbook(encoding="utf-8")
    sheet = workbook.add_sheet("eval")
    for column, name in enumerate(("text", "label")):
        sheet.write(0, column, name)
    for row_number, row in enumerate(frame[["text", "label"]].itertuples(index=False), 1):
        sheet.write(row_number, 0, str(row.text))
        sheet.write(row_number, 1, str(row.label))
    sheet.col(0).width = 256 * 100
    sheet.col(1).width = 256 * 14
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(str(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_XLS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    gold = pd.read_csv(GOLD)
    if len(gold) != 1000 or gold.sentence_id.nunique() != 1000:
        raise ValueError("Frozen gold must contain exactly 1,000 unique sentence IDs")
    if gold.text.astype(str).str.strip().str.casefold().nunique() != 1000:
        raise ValueError("Frozen gold contains normalized duplicate texts")
    if list(gold.columns[:2]) != ["text", "label"]:
        raise ValueError("Frozen gold first columns must be text,label")
    annotations = pd.read_csv(ANNOTATIONS, encoding="utf-8-sig")
    annotations = annotations[annotations.sentence_id.isin(set(gold.sentence_id))].copy()
    counts = annotations.groupby("sentence_id").size()
    if len(counts) != 1000 or not counts.eq(3).all():
        raise ValueError("Every frozen sentence must have exactly three original annotations")
    annotations["joint"] = annotations.apply(
        lambda row: "|".join(normalized(row[field])
                             for field in ("subjectivity", "polarity", "aspect")), axis=1)
    agreement = {}
    for field in ("subjectivity", "polarity", "aspect", "joint"):
        kappa, pairwise, records = fleiss_kappa(annotations, field)
        agreement[field] = {
            "records": records,
            "original_pairwise_agreement": round(pairwise, 6),
            "fleiss_kappa": round(kappa, 6),
        }
    write_xls(gold, args.output)
    report = {
        "generated_from": str(GOLD.relative_to(ROOT)),
        "records": len(gold),
        "unique_normalized_texts": gold.text.astype(str).str.strip().str.casefold().nunique(),
        "first_two_columns": ["text", "label"],
        "label_distribution": gold.label.value_counts().to_dict(),
        "agreement_scope": "three original annotators on the frozen 1,000-sentence subset",
        "agreement": agreement,
        "eval_xls": str(args.output.relative_to(ROOT)),
        "eval_xls_sha256": sha256(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
