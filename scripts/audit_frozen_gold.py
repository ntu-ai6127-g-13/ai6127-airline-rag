"""Rescore already-frozen predictions on the final human labels without model selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gold_release import DEFAULT_RELEASE, load_frozen_gold, align_cached_predictions, check_training_overlap, sha256
from src.quality import classification_metrics, CLASSIFIER_TARGETS


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--previous-run", type=Path, default=ROOT / "outputs/query_quality_20261018")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists(): raise ValueError("Preserve existing audit; choose a new directory")
    gold, release = load_frozen_gold(args.release)
    previous = json.loads((args.previous_run / "audit.json").read_text())
    freeze = args.previous_run / "frozen_candidates.json"
    if sha256(freeze) != previous["frozen_sha256"]: raise ValueError("Prior model freeze changed")
    frozen = json.loads(freeze.read_text())
    for section in ["source_hashes", "input_hashes", "model_weight_hashes"]:
        for path, digest in frozen[section].items():
            if sha256(path) != digest: raise ValueError("Frozen artifact changed: " + path)
    old_path = ROOT / "outputs/c_gold_smoke_20261018/inputs/gold_1073.csv"
    old = pd.read_csv(old_path, keep_default_na=False)
    if sha256(old_path) != previous["gold_sha256"]: raise ValueError("Historical evaluation labels changed")
    old_subset = old.set_index("sentence_id").loc[gold.sentence_id].reset_index()
    if old_subset.text.tolist() != gold.text.tolist(): raise ValueError("The new release changed sentence texts")
    changed = gold[["sentence_id", "text", "aspect", "subjectivity", "polarity"]].copy()
    difference = np.zeros(len(gold), dtype=bool)
    change_counts = {}
    for field in ["aspect", "subjectivity", "polarity"]:
        changed["previous_" + field] = old_subset[field]
        mask = gold[field].ne(old_subset[field]); difference |= mask.to_numpy()
        change_counts[field] = int(mask.sum())
    weak = ROOT / "outputs/c_gold_smoke_20261018/reproduced_current_git_seed42/weak_labels.parquet"
    split = weak.with_name("split.parquet")
    data = pd.read_parquet(weak).merge(pd.read_parquet(split), on=["chunk_id", "review_id"], validate="one_to_one")
    overlap = {role: check_training_overlap(gold, data.loc[data.split.eq(role)]) for role in ["train", "dev"]}
    if any(value for counts in overlap.values() for value in counts.values()):
        raise ValueError("Final gold overlaps the existing weak training/development data")
    paths = {"r6_previous_retained": ROOT / "outputs/c_tuning_20261018/r6_semantic_C0.1/predictions.jsonl"}
    paths.update({name: args.previous_run / name / "audit_predictions.jsonl" for name in frozen["candidates"]})
    reports = {}; prediction_frames = {}
    for name, path in paths.items():
        cache = pd.read_json(path, lines=True)
        pred = align_cached_predictions(gold, cache)
        prediction_frames[name] = pred
        metrics = classification_metrics(gold, pred)
        metrics["three_way_all_rows"] = classification_report(gold.label, pred.label,
            labels=["objective", "negative", "positive"], output_dict=True, zero_division=0)
        reports[name] = {"final_gold": metrics, "same_rows_previous_labels": classification_metrics(old_subset, pred),
                         "prediction_cache_sha256": sha256(path), "prediction_cache": str(path.relative_to(ROOT))}
    eligible = gold.subjectivity.eq("subjective").to_numpy()
    groups, group_ids = pd.factorize(gold.review_id)
    count = np.bincount(groups, weights=eligible)
    rng = np.random.RandomState(20261019)
    samples = rng.randint(0, len(group_ids), size=(2000, len(group_ids)))
    for name in ["r6_previous_retained", "r34_qwen_closed_choice", "r36_choice_subjective_consensus"]:
        pred = prediction_frames[name]
        correct = eligible & gold.aspect.eq(pred.aspect).to_numpy() & gold.polarity.eq(pred.polarity).to_numpy()
        hits = np.bincount(groups, weights=correct)
        boot = hits[samples].sum(axis=1) / count[samples].sum(axis=1)
        reports[name]["joint_parent_bootstrap_95"] = np.quantile(boot, [.025, .975]).tolist()
    args.output_dir.mkdir(parents=True)
    for name, pred in prediction_frames.items():
        gold.add_prefix("gold_").join(pred.add_prefix("pred_")).to_json(
            args.output_dir / (name + "_gold_predictions.jsonl"), orient="records", lines=True, force_ascii=False)
    changed.loc[difference].to_csv(args.output_dir / "gold_label_changes.csv", index=False)
    report = {"status":"completed", "created_at":datetime.now(timezone.utc).isoformat(),
        "release_id":release["release_id"], "gold_sha256":release["gold_sha256"], "gold_n":len(gold),
        "subjective_n":int(eligible.sum()), "dataset_scope":release["dataset_scope"],
        "prior_gold_n":len(old), "excluded_from_prior":len(old)-len(gold), "changed_sentence_count":int(difference.sum()),
        "changed_fields":change_counts, "targets":CLASSIFIER_TARGETS, "models":reports,
        "frozen_sha256":sha256(freeze), "prediction_mode":"reuse frozen predictions after exact ID/text/metadata verification",
        "gold_used_for_training":False, "gold_used_for_model_selection":False, "independent_test":False,
        "model_promoted":False, "weak_training_overlap":overlap,
        "weak_data_sha256":sha256(weak), "weak_split_sha256":sha256(split),
        "limitations":["All 1000 sentences were in prior development audits.",
            "The subset was selected post hoc using aspect agreement; it is not a random test sample.",
            "Human approval completion is not inter-annotator agreement.",
            "Labels are final, but this evaluation does not establish independent generalization."]}
    assert sha256(ROOT / release["data_file"]) == release["gold_sha256"]
    write_json(args.output_dir / "audit.json", report)
    passed = [name for name, item in reports.items() if item["final_gold"]["point_targets_passed"]]
    write_json(args.output_dir / "release_gate.json", {"gold_validated":True,"candidates_passing_all_targets":passed,
        "independent_query_test_passed":False,"production_ready":False,"previous_model_retained":True})
    lines = ["# 人工定稿 gold 接入与冻结模型复评", "",
        f"数据：`{release['release_id']}`，{len(gold)} 句，其中 {int(eligible.sum())} 句主观评价。",
        f"相对之前 1073 句：保留 1000 句，移出 73 句；重叠句中 {int(difference.sum())} 句标签变化，原文未变。",
        "指标变化同时含样本组成与标签修订的影响；audit.json 另存同样 1000 句在旧标签下的分数。", "",
        "gold 只读，未进训练或模型选择。原 1100 句及父评论的训练排除范围保留；固定弱训练、开发集与本版 gold 的句子、父评论、规范化文本交集均为零。",
        "本版是人工审核定稿，但存在历史开发曝光及事后子集筛选，不能称为全新独立测试。", "",
        "| 模型 | 方面 F1 | 极性 F1 | 联合正确率 | 主观召回 | 全部过线 |",
        "|---|---:|---:|---:|---:|---|", "| 目标 | 85.0% | 90.0% | 80.0% | 97.0% | — |"]
    keys = list(CLASSIFIER_TARGETS)
    for name, item in reports.items():
        m=item["final_gold"]
        lines.append("| "+name+" | "+" | ".join(f"{m[k]:.1%}" for k in keys)+f" | {m['point_targets_passed']} |")
    lines += ["", "全部模型复用先前已冻结、逐 ID 和原文核验的预测，不重新拟合参数。r36 是事后比较候选；原预选与保留模型指针没有更改。",
              "查询相关性及答案质量仍需独立判定；分类点估计不等于查询系统验收。"]
    (args.output_dir / "README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"gold_n":len(gold),"changed":int(difference.sum()),"passed":passed,"output":str(args.output_dir)}))


if __name__ == "__main__": main()
