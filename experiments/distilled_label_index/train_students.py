"""训练预先规定的学生候选，只按与 r36 教师在 pseudo_dev 上的一致度挑选，全程不读 gold。

候选（运行前固定，不按结果增删）：
- combined_weak：TF-IDF+MiniLM，只用 v4.1 弱标签训练侧（复现 combined 方案）
- combined_teacher：TF-IDF+MiniLM，只用 r36 教师标签 pseudo_train
- combined_mixed：两者合并；每个任务里教师样本总权重与弱标签总权重相等
v4.1 生产模型（含规则层）只作对照，不参与挑选。
挑选指标为 src.quality.proxy_rank（参照为教师标签）；选中后写 selection.json 冻结哈希。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.classification_pipeline import TASKS, predict_texts
from src.quality import classification_metrics, proxy_rank
from src.training_isolation import install_answer_read_guard
from src.weak_labels import normalize_text, sha256
from experiments.distilled_label_index import model as student
from experiments.distilled_label_index.encode import load_cache, lookup

WEAK = ROOT / "outputs/weak_rebuild_20261019"
CANDIDATES = ["combined_weak", "combined_teacher", "combined_mixed"]


def teacher_frame(sample_dir: Path) -> pd.DataFrame:
    sample = pd.read_parquet(sample_dir / "sample.parquet")
    labels = pd.read_parquet(sample_dir / "teacher_labels.parquet")
    if labels.id.tolist() != sample.chunk_id.tolist():
        raise ValueError("教师标签与样本不对齐")
    frame = sample.assign(aspect=labels.aspect.to_numpy(), subjectivity=labels.subjectivity.to_numpy(),
                          polarity=labels.polarity.to_numpy())
    ok = frame.aspect.isin(TASKS["aspect"]) & frame.subjectivity.isin(TASKS["subjectivity"])
    ok &= frame.polarity.isin(TASKS["polarity"]) | frame.subjectivity.eq("objective")
    frame = frame.loc[ok].reset_index(drop=True)
    frame["polarity"] = frame.polarity.where(frame.subjectivity.eq("subjective"))
    return frame, int((~ok).sum())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefit", action="store_true", help="教师标签就绪前，只预训练 combined_weak")
    args = parser.parse_args(argv)
    dest = args.output_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "selection.json").exists():
        raise ValueError("已完成挑选；拒绝覆盖")
    denied = install_answer_read_guard()
    started = time.monotonic()
    cache = load_cache(args.embeddings)

    weak = pd.read_parquet(WEAK / "weak_labels.parquet")
    split = pd.read_parquet(WEAK / "split.parquet")
    w_train = weak.loc[weak.chunk_id.isin(split.loc[split.split.eq("train"), "chunk_id"])].reset_index(drop=True)
    for column in TASKS:
        w_train[column] = w_train[column].astype(object).where(w_train[column].notna())
    columns = ["chunk_id", "review_id", "text", *TASKS]
    sample_parents = set(pd.read_parquet(args.sample_dir / "sample.parquet").review_id)
    if set(w_train.review_id) & sample_parents:
        raise AssertionError("弱标签训练侧与教师样本父评论重叠")
    if args.prefit:
        target = dest / "combined_weak" / "model.joblib"
        if target.exists():
            raise ValueError("combined_weak 已存在")
        t0 = time.monotonic()
        bundle = student.fit("combined_weak", w_train.text, lookup(cache, w_train.chunk_id), w_train[columns])
        bundle["fit_seconds"] = round(time.monotonic() - t0, 1)
        bundle["teacher_weight"] = None
        target.parent.mkdir(exist_ok=True)
        joblib.dump(bundle, target)
        print("combined_weak 预训练完成", bundle["fit_seconds"], "秒", bundle["fit_info"])
        return 0

    teacher, dropped = teacher_frame(args.sample_dir)
    t_train = teacher.loc[teacher.role.eq("pseudo_train")].reset_index(drop=True)
    t_dev = teacher.loc[teacher.role.eq("pseudo_dev")].reset_index(drop=True)

    # 学生与对照模型都不能见过 pseudo_dev 的父评论或规范化文本。
    dev_norms = set(t_dev.text.map(normalize_text))
    leaks = {"weak_train_parent_overlap": len(set(w_train.review_id) & set(t_dev.review_id)),
             "weak_train_text_overlap": len(set(w_train.text.map(normalize_text)) & dev_norms),
             "teacher_train_parent_overlap": len(set(t_train.review_id) & set(t_dev.review_id)),
             "teacher_train_text_overlap": len(set(t_train.text.map(normalize_text)) & dev_norms)}
    if any(leaks.values()):
        raise AssertionError(f"pseudo_dev 泄漏：{leaks}")

    sources = {"combined_weak": (w_train[columns], None),
               "combined_teacher": (t_train[columns], None)}
    mixed = pd.concat([w_train[columns], t_train[columns]], ignore_index=True)
    is_teacher = np.r_[np.zeros(len(w_train), bool), np.ones(len(t_train), bool)]
    sources["combined_mixed"] = (mixed, is_teacher)

    dev_x = lookup(cache, t_dev.chunk_id)
    results = {}
    for name in CANDIDATES:
        target = dest / name / "model.joblib"
        if target.exists():
            bundle = joblib.load(target)
        else:
            frame, teacher_mask = sources[name]
            weights, ratios = None, None
            if teacher_mask is not None:
                # 每个任务分别让教师样本总权重等于弱标签总权重。
                weights, ratios = {}, {}
                for task in TASKS:
                    has = frame[task].notna().to_numpy()
                    ratios[task] = float((has & ~teacher_mask).sum() / max((has & teacher_mask).sum(), 1))
                    weights[task] = np.where(teacher_mask, ratios[task], 1.0)
            print(f"[{name}] 训练 {len(frame):,} 行", flush=True)
            t0 = time.monotonic()
            bundle = student.fit(name, frame.text, lookup(cache, frame.chunk_id), frame, weights)
            bundle["fit_seconds"] = round(time.monotonic() - t0, 1)
            bundle["teacher_weight"] = ratios
            target.parent.mkdir(exist_ok=True)
            joblib.dump(bundle, target)
        pred = student.predict(bundle, t_dev.text.tolist(), dev_x)
        metrics = classification_metrics(t_dev, pred)
        results[name] = {"proxy_rank": proxy_rank(metrics), "metrics_vs_teacher": metrics,
                         "fit_info": bundle["fit_info"], "fit_seconds": bundle.get("fit_seconds"),
                         "teacher_weight": bundle.get("teacher_weight"),
                         "model_sha256": sha256(target)}
        print(name, json.dumps({k: round(metrics[k], 4) for k in
                                ["subjective_aspect_f1", "subjective_polarity_f1",
                                 "subjective_joint_accuracy", "subjective_recall", "objective_recall"]}), flush=True)

    v41 = joblib.load(WEAK / "model.joblib")
    v41_metrics = classification_metrics(t_dev, predict_texts(v41, t_dev.text.tolist()))
    selected = max(CANDIDATES, key=lambda n: results[n]["proxy_rank"])
    selection = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule": "max src.quality.proxy_rank against r36 teacher labels on pseudo_dev; gold never read",
        "selected": selected, "selected_model": str(dest / selected / "model.joblib"),
        "selected_sha256": results[selected]["model_sha256"],
        "candidates": results,
        "reference_v41_production": {"proxy_rank": proxy_rank(v41_metrics), "metrics_vs_teacher": v41_metrics,
                                     "model_sha256": sha256(WEAK / "model.joblib")},
        "data": {"teacher_rows_dropped_invalid": dropped, "pseudo_train": len(t_train), "pseudo_dev": len(t_dev),
                 "pseudo_dev_parents": int(t_dev.review_id.nunique()), "weak_train": len(w_train),
                 "isolation": leaks},
        "inputs": {str(p): sha256(p) for p in [args.sample_dir / "sample.parquet",
                                               args.sample_dir / "teacher_labels.parquet",
                                               WEAK / "weak_labels.parquet", WEAK / "split.parquet",
                                               args.embeddings / "embeddings.f16.npy"]},
        "gold_labels_read": False, "denied_reads": denied,
        "seconds": round(time.monotonic() - started, 1),
    }
    (dest / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2, default=str) + "\n")
    print("v41_production", json.dumps({k: round(v41_metrics[k], 4) for k in
                                        ["subjective_aspect_f1", "subjective_polarity_f1",
                                         "subjective_joint_accuracy", "subjective_recall", "objective_recall"]}))
    print("选中：", selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
