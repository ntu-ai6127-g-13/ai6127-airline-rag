"""挑选冻结后，在人工 gold v1 上一次性计分；核对 selection.json 中的模型哈希。

采用规则（运行前固定）：选中学生在 gold 上的 proxy_rank 与联合正确率都高于 v4.1 生产模型，
才用它重建全库标签索引。其余候选和 r36 的分数只作事后对照，不改选。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.classification_pipeline import predict_texts
from src.gold_release import align_cached_predictions, check_training_overlap, load_frozen_gold
from src.quality import CLASSIFIER_TARGETS, classification_metrics, proxy_rank
from src.weak_labels import sha256
from experiments.distilled_label_index import model as student

AUDIT = ROOT / "outputs/gold_release_audit_20261019"
V41 = ROOT / "outputs/weak_rebuild_20261019/model.joblib"
KEYS = ["subjective_aspect_f1", "subjective_polarity_f1", "subjective_joint_accuracy",
        "subjective_recall", "objective_recall"]


def joint_bootstrap(gold, a, b, seed=20261029, rounds=2000):
    """父评论分组 bootstrap：a 减 b 的联合正确率差值。"""
    mask = gold.subjectivity.eq("subjective").to_numpy()
    correct = lambda p: (gold.aspect.eq(p.aspect) & gold.polarity.eq(p.polarity)).to_numpy()
    diff = pd.DataFrame({"review_id": gold.review_id[mask],
                         "d": correct(a).astype(float)[mask] - correct(b).astype(float)[mask]})
    groups = diff.groupby("review_id").d.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(rounds):
        pick = groups.iloc[rng.integers(0, len(groups), len(groups))]
        values.append(pick["sum"].sum() / pick["count"].sum())
    return {"mean_delta": float(diff.d.mean()), "ci95": np.quantile(values, [.025, .975]).tolist()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args(argv)
    selection = json.loads(args.selection.read_text())
    dest = args.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    base = args.selection.resolve().parent

    gold, release = load_frozen_gold()
    texts = gold.text.tolist()
    embeddings = student.encode(student.load_encoder(args.device), texts)
    predictions, report = {}, {}
    for name, info in selection["candidates"].items():
        path = base / name / "model.joblib"
        if sha256(path) != info["model_sha256"]:
            raise ValueError(f"{name} 权重在挑选后被改动")
        predictions[name] = student.predict(joblib.load(path), texts, embeddings)
    if sha256(V41) != selection["reference_v41_production"]["model_sha256"]:
        raise ValueError("v4.1 权重被改动")
    predictions["v41_production"] = predict_texts(joblib.load(V41), texts)
    cached = pd.read_json(AUDIT / "r36_choice_subjective_consensus_gold_predictions.jsonl", lines=True, dtype=False)
    predictions["r36_teacher"] = align_cached_predictions(gold, cached)

    for name, pred in predictions.items():
        metrics = classification_metrics(gold, pred)
        report[name] = {"proxy_rank": proxy_rank(metrics), "metrics": metrics}
        pred.assign(sentence_id=gold.sentence_id.to_numpy()).to_json(
            dest / f"{name}_gold_predictions.jsonl", orient="records", lines=True, force_ascii=False)

    selected = selection["selected"]
    adopt = (report[selected]["proxy_rank"] > report["v41_production"]["proxy_rank"]
             and report[selected]["metrics"]["subjective_joint_accuracy"]
             > report["v41_production"]["metrics"]["subjective_joint_accuracy"])
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(), "gold_release": release.get("version", release.get("data_file")),
        "gold_n": len(gold), "targets": CLASSIFIER_TARGETS,
        "selected": selected, "adopt_for_label_index": bool(adopt),
        "adoption_rule": "selected proxy_rank and joint accuracy both above v41_production on gold",
        "selected_minus_v41_joint": joint_bootstrap(gold, predictions[selected], predictions["v41_production"]),
        "selected_minus_r36_joint": joint_bootstrap(gold, predictions[selected], predictions["r36_teacher"]),
        "results": report,
        "post_hoc_note": "Non-selected candidates and r36 are reported for context only; selection was fixed before gold was read.",
        "selection_sha256": sha256(args.selection),
    }
    teacher_sample = ROOT / "outputs/distill_sample_20261029/sample.parquet"
    if teacher_sample.exists():
        result["gold_overlap_with_teacher_sample"] = check_training_overlap(gold, pd.read_parquet(teacher_sample))
    (dest / "gold_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
    print(f"{'模型':<18}" + "".join(f"{k.replace('subjective_', ''):>16}" for k in KEYS) + f"{'proxy':>8}")
    for name, item in report.items():
        print(f"{name:<18}" + "".join(f"{(item['metrics'][k] or 0) * 100:>15.1f}%" for k in KEYS)
              + f"{item['proxy_rank']:>8.3f}")
    print("选中：", selected, "采用：", adopt)
    print("选中−v4.1 联合差值：", result["selected_minus_v41_joint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
