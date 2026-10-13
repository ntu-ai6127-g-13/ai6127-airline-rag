"""复用既有 v3 预测，生成 C 到 B 的本地候选交接包；不训练、不预测。

运行：.venv/bin/python -B scripts/prepare_c_handoff.py
评测 CSV 由 load_test_exclusions 仅解析 id/text，绝不解析人工答案列。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import zipfile

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classification_pipeline import eligible_rows
from src.weak_labels import load_test_exclusions, normalize_text, sha256, validate_ids

VERSION = "eight-aspect-subjectivity-polarity-v3"
GUIDE_PATH = "docs/C组当前使用与交接指南.md"
HISTORICAL_PACKAGING_PATHS = frozenset({
    "scripts/prepare_c_handoff.py", "scripts/prepare_c_b_sync.py",
    "scripts/check_c_b_sync.py", "scripts/supplement_c_b_predictions.py",
    "scripts/check_c_b_extended.py", "docs/C到B分类标签交接说明.md",
    "docs/C到B同步操作说明.md", "docs/C到B补预测交接说明.md",
})
ASPECTS = ["seat", "staff", "food", "entertainment", "ground", "value", "overall", "other"]
KB_COLUMNS = ["chunk_id", "review_id", "text", "is_eval", "is_route_only", "is_non_english", "is_fragment"]
LABEL_COLUMNS = ["chunk_id", "review_id", "aspect", "subjectivity", "polarity", "model_version",
                 "source", "aspect_source", "subjectivity_source", "polarity_source", "parent_split"]
SKIP_REASONS = ["eval_sentence", "eval_parent", "eval_duplicate_text", "empty", "route_only", "non_english", "fragment"]
METADATA_COLUMNS = LABEL_COLUMNS[2:]
LIMITS = [
    "候选本地交接草稿：未人工验收，未做人工金标准评估，未选择为最终版本。",
    "未取得 B 组实际读取 schema 或联调证据；文档接口仅为可审查建议。",
    "复用既有预测，未调用训练、fit、predict 或 transform；当前源码变更不追溯重写旧预测。",
    "带噪声规则与评论评分弱监督的开发成绩不等于人工金标准准确率。",
    "未预测句保留空标签和跳过理由；不得把空值解释成 negative、other 或 objective。",
    "预测中的 other 是显式类别；其情感依赖跨主题泛化，尚未验证。",
    "复杂多方面、否定、转折及严格客观模板覆盖仍有局限。",
]


def _source_relative_path(name):
    """来源键与 ZIP 成员必须是规范 POSIX 相对文件路径。"""
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or ":" in name):
        raise ValueError(f"来源路径非法：{name!r}")
    path = PurePosixPath(name)
    if (not path.parts or path.is_absolute() or path.as_posix() != name
            or any(part in {".", ".."} for part in path.parts)):
        raise ValueError(f"来源路径必须是规范相对路径：{name!r}")
    return name


def verify_packaging_sources(recorded, root=ROOT, historical_sources=None):
    """严格核对当前来源；显式 ZIP 仅证明八项旧包装来源的原字节。

    ZIP 内容只作为哈希证据读取，不执行、不导入、不解压。分类、规则和
    读取器不在历史回退白名单中，即使提供 ZIP 也必须与当前文件一致。
    """
    if not isinstance(recorded, dict):
        raise ValueError("来源 hash 记录必须是字典")
    root = Path(root).resolve()
    for name, expected in recorded.items():
        _source_relative_path(name)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"来源 SHA-256 非法：{name}")
        if not (root / name).resolve().is_relative_to(root):
            raise ValueError(f"来源路径越出根目录：{name}")

    archived_hashes = {}
    if historical_sources is not None:
        try:
            with zipfile.ZipFile(Path(historical_sources)) as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if len(names) != len(set(names)):
                    raise ValueError("历史 ZIP 含重复成员")
                for info in infos:
                    name = _source_relative_path(info.filename)
                    if info.is_dir() or name not in HISTORICAL_PACKAGING_PATHS:
                        raise ValueError(f"历史 ZIP 成员不在包装来源白名单：{name}")
                for info in infos:
                    archived_hashes[info.filename] = hashlib.sha256(archive.read(info)).hexdigest()
        except (OSError, zipfile.BadZipFile, RuntimeError) as error:
            raise ValueError(f"无法读取历史包装来源 ZIP：{error}") from error

    result = {"current": [], "archived": []}
    for name, expected in sorted(recorded.items()):
        path = root / name
        if path.is_file() and sha256(path) == expected:
            result["current"].append(name)
        elif (historical_sources is not None and name in HISTORICAL_PACKAGING_PATHS
              and archived_hashes.get(name) == expected):
            result["archived"].append(name)
        else:
            raise ValueError(f"来源 hash 变化或文件缺失：{name}")
    return result


def validate_kb(kb: pd.DataFrame) -> None:
    missing = set(KB_COLUMNS) - set(kb.columns)
    if missing:
        raise ValueError(f"KB 缺少列：{sorted(missing)}")
    validate_ids(kb, "KB")
    if not kb.text.map(lambda value: isinstance(value, str)).all():
        raise ValueError("KB text 含空值或非文本")
    for column in KB_COLUMNS[3:]:
        if not pd.api.types.is_bool_dtype(kb[column]) or kb[column].isna().any():
            raise ValueError(f"KB {column} 必须是无空值布尔列")


def skip_reasons(kb: pd.DataFrame, eval_reviews: set[str], eval_texts: set[str]) -> pd.Series:
    """按先命中理由互斥计数；选中集合必须与既有 eligible_rows 一致。"""
    validate_kb(kb)
    reason = pd.Series(pd.NA, index=kb.index, dtype="string", name="skip_reason")
    masks = [kb.is_eval, kb.review_id.isin(eval_reviews),
             kb.text.map(normalize_text).isin(eval_texts), kb.text.str.strip().eq(""),
             kb.is_route_only, kb.is_non_english, kb.is_fragment]
    for name, mask in zip(SKIP_REASONS, masks):
        reason.loc[reason.isna() & mask] = name
    selected = set(kb.loc[reason.isna(), "chunk_id"])
    if selected != set(eligible_rows(kb, eval_reviews, eval_texts).chunk_id):
        raise ValueError("互斥覆盖理由与现有 eligible_rows 不一致，拒绝交接")
    return reason


def validate_labels(labels: pd.DataFrame, version: str = VERSION) -> None:
    if list(labels.columns) != LABEL_COLUMNS:
        raise ValueError("thin labels schema 与候选接口不一致")
    validate_ids(labels, "thin labels")
    for column in METADATA_COLUMNS:
        if not labels[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"预测 {column} 有空值或非文本；不得补成 other")
    if not labels.aspect.isin(ASPECTS).all():
        raise ValueError("预测方面标签非法")
    if not labels.subjectivity.isin(["objective", "subjective"]).all():
        raise ValueError("预测主客观标签非法")
    objective = labels.subjectivity.eq("objective")
    if not labels.loc[objective, "polarity"].eq("NA").all():
        raise ValueError("objective 的 polarity 必须是字面值 NA")
    if not labels.loc[~objective, "polarity"].isin(["negative", "positive"]).all():
        raise ValueError("subjective 的 polarity 必须为二元标签")
    if not labels.model_version.eq(version).all() or not labels.source.eq("hybrid_prediction").all():
        raise ValueError("预测的模型版本或来源不一致")
    if not labels.parent_split.isin(["training_parent", "development_parent", "unlabelled_parent"]).all():
        raise ValueError("预测 parent_split 非法")


def validate_handoff(kb, labels, coverage, eval_reviews, eval_texts, version=VERSION) -> None:
    validate_kb(kb)
    validate_labels(labels, version)
    validate_ids(coverage, "coverage")
    expected_columns = ["chunk_id", "review_id", "is_eval", "prediction_status", "skip_reason", *METADATA_COLUMNS]
    if list(coverage.columns) != expected_columns:
        raise ValueError("coverage schema 与候选接口不一致")
    if set(coverage.chunk_id) != set(kb.chunk_id) or len(coverage) != len(kb):
        raise ValueError("coverage 没有完整覆盖全部 KB ID")
    aligned = coverage.set_index("chunk_id").loc[kb.chunk_id].reset_index()
    if not aligned.review_id.eq(kb.review_id.to_numpy()).all():
        raise ValueError("coverage 父评论与 KB 不一致")
    if (not pd.api.types.is_bool_dtype(aligned.is_eval) or aligned.is_eval.isna().any()
            or not aligned.is_eval.eq(kb.is_eval.to_numpy()).all()):
        raise ValueError("coverage is_eval 与 KB 不一致")
    expected_reason = skip_reasons(kb, eval_reviews, eval_texts).reset_index(drop=True)
    if not aligned.skip_reason.fillna("").eq(expected_reason.fillna("")).all():
        raise ValueError("coverage 跳过理由不一致")
    predicted = expected_reason.isna()
    expected_status = predicted.map({True: "predicted", False: "skipped"})
    if not aligned.prediction_status.eq(expected_status).all():
        raise ValueError("coverage prediction_status 与理由不一致")
    if not aligned.loc[~predicted, METADATA_COLUMNS].isna().all().all():
        raise ValueError("跳过句必须保留空标签和空模型元数据，不得填 other")
    selected_ids = set(kb.loc[predicted.to_numpy(), "chunk_id"])
    if set(labels.chunk_id) != selected_ids:
        raise ValueError("候选 coverage 与实际 prediction ID 集不精确一致")
    actual = aligned.loc[predicted, LABEL_COLUMNS].sort_values("chunk_id").reset_index(drop=True)
    expected = labels.sort_values("chunk_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def make_tables(kb, predictions, eval_reviews, eval_texts, version=VERSION, bundle=None):
    kb = kb.reset_index(drop=True)
    validate_kb(kb)
    required = set(LABEL_COLUMNS) | {"text", "is_eval"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"原预测缺少列：{sorted(required - set(predictions.columns))}")
    validate_ids(predictions, "原预测")
    if (not pd.api.types.is_bool_dtype(predictions.is_eval) or predictions.is_eval.isna().any()
            or predictions.is_eval.any()):
        raise ValueError("原预测包含评测句或非法 is_eval")
    joined = predictions[["chunk_id", "review_id", "text"]].merge(
        kb[["chunk_id", "review_id", "text"]], on="chunk_id", how="left",
        suffixes=("_prediction", "_kb"), validate="one_to_one", indicator=True)
    if (not joined._merge.eq("both").all()
            or not joined.review_id_prediction.eq(joined.review_id_kb).all()
            or not joined.text_prediction.eq(joined.text_kb).all()):
        raise ValueError("原预测 ID、父评论或 text 与当前 KB 不一致")
    labels = predictions[LABEL_COLUMNS].copy().reset_index(drop=True)
    for column in LABEL_COLUMNS:
        labels[column] = labels[column].astype("string")
    validate_labels(labels, version)
    if bundle is not None:
        train, dev = set(bundle["train_review_ids"]), set(bundle["dev_review_ids"])
        if train & dev:
            raise ValueError("模型训练/开发父评论重叠")
        parent = pd.Series("unlabelled_parent", index=labels.index, dtype="string")
        parent.loc[labels.review_id.isin(train)] = "training_parent"
        parent.loc[labels.review_id.isin(dev)] = "development_parent"
        if not parent.eq(labels.parent_split).all():
            raise ValueError("预测父评论来源与模型切分不一致")
    reason = skip_reasons(kb, eval_reviews, eval_texts)
    coverage = kb[["chunk_id", "review_id", "is_eval"]].copy()
    coverage["prediction_status"] = reason.isna().map({True: "predicted", False: "skipped"}).astype("string")
    coverage["skip_reason"] = reason
    coverage = coverage.merge(labels, on=["chunk_id", "review_id"], how="left", validate="one_to_one")
    for column in ["chunk_id", "review_id", *METADATA_COLUMNS]:
        coverage[column] = coverage[column].astype("string")
    validate_handoff(kb, labels, coverage, eval_reviews, eval_texts, version)
    return labels, coverage


def verify_provenance(report, bundle, file_hashes):
    if report.get("model_version") != VERSION or bundle.get("version") != VERSION:
        raise ValueError("报告/模型不是目标 v3 版本")
    expected_tasks = {"aspect": ASPECTS, "subjectivity": ["objective", "subjective"], "polarity": ["negative", "positive"]}
    if bundle.get("tasks") != expected_tasks:
        raise ValueError("模型任务标签字典不一致")
    for key in ["model", "input", "kb", "eval_meta", "eval_text"]:
        if report.get("inputs", {}).get(key, {}).get("sha256") != file_hashes[key]:
            raise ValueError(f"原预测报告 {key} hash 与当前文件不一致")
    if report.get("output_sha256") != file_hashes["predictions"]:
        raise ValueError("原预测文件 hash 与原报告不一致")
    for key in ["input", "kb", "eval_meta", "eval_text"]:
        if bundle.get("training_inputs", {}).get(key, {}).get("sha256") != file_hashes[key]:
            raise ValueError(f"模型 training_inputs/{key} 与当前文件不一致")
    required_checks = ["unique_ids", "row_count_matches", "zero_eval_parent_text_overlap", "legal_aspects",
                       "legal_subjectivity", "objective_is_NA", "subjective_has_binary_polarity", "inputs_unchanged", "roundtrip_identical"]
    if any(report.get("checks", {}).get(key) is not True for key in required_checks):
        raise ValueError("原预测报告核验记录不完整或未通过")


def prepare(args):
    destination = args.output_dir.resolve()
    if destination.exists():
        raise ValueError(f"输出目录已存在，拒绝覆盖：{destination}")
    paths = {"model": args.model.resolve(), "input": args.kb.resolve(), "kb": args.kb.resolve(),
             "predictions": args.predictions.resolve(), "prediction_report": args.prediction_report.resolve(),
             "eval_meta": args.eval_meta.resolve(), "eval_text": args.eval_text.resolve(),
             "training_report": args.training_report.resolve()}
    hashes = {key: sha256(path) for key, path in paths.items()}
    report = json.loads(paths["prediction_report"].read_text(encoding="utf-8"))
    bundle = joblib.load(paths["model"])
    verify_provenance(report, bundle, hashes)
    training = json.loads(paths["training_report"].read_text(encoding="utf-8"))
    if training.get("version") != VERSION or training.get("inputs") != bundle["training_inputs"]:
        raise ValueError("训练报告与现有模型 provenance 不一致")
    kb = pd.read_parquet(paths["kb"], columns=KB_COLUMNS)
    validate_kb(kb)
    reviews, texts, eval_info = load_test_exclusions(kb, paths["eval_meta"], paths["eval_text"])
    if eval_info != report.get("evaluation_exclusion"):
        raise ValueError("当前评测映射计数/标记与原预测报告不一致")
    predictions = pd.read_parquet(paths["predictions"], columns=[*LABEL_COLUMNS, "text", "is_eval"])
    if (len(kb) != report.get("input_rows") or len(predictions) != report.get("output_rows")
            or len(kb) - len(predictions) != report.get("excluded_rows")):
        raise ValueError("当前输入/预测行数与原报告不一致")
    counts = {task: predictions[task].value_counts().to_dict() for task in ["aspect", "subjectivity", "polarity"]}
    if counts != report.get("counts"):
        raise ValueError("预测标签计数与原报告不一致")
    labels, coverage = make_tables(kb, predictions, reviews, texts, bundle=bundle)
    dropped = {name: int(coverage.skip_reason.eq(name).sum()) for name in SKIP_REASONS}
    source_at_training = training.get("implementation_sha256", {})
    source_now = {}
    for relative in sorted(set(source_at_training) | {"scripts/prepare_c_handoff.py", "tests/test_c_handoff.py", GUIDE_PATH}):
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"源码审计路径无效：{relative}")
        source_now[relative] = sha256(path)
    source_changes = [name for name, value in source_at_training.items() if source_now[name] != value]
    manifest = {
        "status": "candidate", "schema_version": "c-b-handoff-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": VERSION, "prediction_created_at": report.get("created_at"),
        "purpose": "local reviewable candidate handoff; not human gold, acceptance, final selection or completed B integration",
        "acceptance": {"human_accepted": False, "human_gold_evaluated": False, "final_version_selected": False, "b_integration_verified": False},
        "inputs": {key: {"path": str(path), "sha256": hashes[key]} for key, path in paths.items()},
        "training_inputs_recorded_in_model": bundle["training_inputs"],
        "hyperparameters_recorded_at_training": training.get("hyperparameters", {}),
        "prediction_parameters": {"kind": "reuse_existing_predictions", "retraining": False, "new_inference": False,
                                  "eligibility_policy": "existing eligible_rows", "first_skip_reason_priority": SKIP_REASONS},
        "implementation_sha256_recorded_at_training": source_at_training,
        "implementation_sha256_recorded_at_export": report.get("implementation_sha256", {}),
        "implementation_sha256_at_preparation": source_now,
        "source_differences_since_training": source_changes,
        "source_hash_policy": "Record historical/current hashes; differences are allowed for packaging existing predictions. No inference equivalence claimed.",
        "evaluation_exclusion": eval_info,
        "coverage": {"kb_rows": len(kb), "eval_sentence_rows": int(kb.is_eval.sum()),
                     "only_not_is_eval_rows": int((~kb.is_eval).sum()), "predicted_rows": len(labels),
                     "skipped_rows": len(coverage) - len(labels), "excluded_by_first_reason": dropped,
                     "additional_skips_after_not_is_eval": int((~kb.is_eval).sum()) - len(labels)},
        "prediction_counts": counts,
        "parent_split_counts": labels.parent_split.value_counts().to_dict(),
        "schemas": {"labels.parquet": {column: str(labels[column].dtype) for column in labels},
                    "coverage.parquet": {column: str(coverage[column].dtype) for column in coverage}},
        "null_policy": {"predicted": "all label/source fields non-null; objective polarity is literal NA",
                        "skipped": "all label/source fields null; status skipped; one explicit skip_reason",
                        "missing_external_join": "unknown, never negative/other/objective"},
        "checks": {"source_report_hashes_match": True, "model_training_inputs_match": True, "all_kb_ids_once": True,
                   "eligibility_exactly_matches_prediction_ids": True, "prediction_text_parent_match_kb": True,
                   "zero_prediction_eval_overlap": True, "no_empty_prediction_labels": True, "skipped_labels_null": True,
                   "human_label_columns_parsed": False, "original_comment_or_rating_columns_copied": False,
                   "model_training_or_prediction_called": False},
        "limits": LIMITS + training.get("limits", []),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".c_b_handoff_", dir=destination.parent))
    try:
        labels.to_parquet(temporary / "labels.parquet", index=False)
        coverage.to_parquet(temporary / "coverage.parquet", index=False)
        restored_labels = pd.read_parquet(temporary / "labels.parquet")
        restored_coverage = pd.read_parquet(temporary / "coverage.parquet")
        pd.testing.assert_frame_equal(labels, restored_labels)
        pd.testing.assert_frame_equal(coverage, restored_coverage)
        validate_handoff(kb, restored_labels, restored_coverage, reviews, texts)
        package_note = (
            "# 本包范围说明\n\n"
            f"本包接口为 `c-b-handoff-v1`，模型版本为 `{VERSION}`，"
            f"实际提供 {len(labels):,} 条预测，覆盖 {len(coverage):,} 条 KB 记录。\n\n"
            "以下统一指南介绍当前交接流程；其中新 v2 包的数量不替代本包数量。"
            "读取本包必须以本目录 manifest.json 的版本、覆盖范围和字段为准，"
            "不能把本历史接口当作补预测后的 v2 接口。\n\n---\n\n"
        )
        (temporary / "README.md").write_text(
            package_note + (ROOT / GUIDE_PATH).read_text(encoding="utf-8"), encoding="utf-8")
        manifest["checks"]["parquet_roundtrip_identical"] = True
        manifest["checks"]["inputs_unchanged"] = all(sha256(path) == hashes[key] for key, path in paths.items())
        if not manifest["checks"]["inputs_unchanged"]:
            raise ValueError("交接期间输入发生变动，拒绝保存")
        manifest["artifacts_sha256"] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if destination.exists():
            raise ValueError("交接期间输出目录出现，拒绝覆盖")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps({"output_dir": str(destination), "status": "candidate", "coverage": manifest["coverage"]}, ensure_ascii=False, indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", type=Path, default=ROOT / "data/processed/chunks.parquet")
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/classification_eight_full_v3/model.joblib")
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/processed/chunk_predictions_eight_v3.parquet")
    parser.add_argument("--prediction-report", type=Path, default=ROOT / "data/processed/chunk_predictions_eight_v3.report.json")
    parser.add_argument("--training-report", type=Path, default=ROOT / "outputs/classification_eight_full_v3/metrics.json")
    parser.add_argument("--eval-meta", type=Path, default=ROOT / "data/eval/to_label_meta.csv")
    parser.add_argument("--eval-text", type=Path, default=ROOT / "data/eval/to_label.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/c_b_handoff_v1")
    prepare(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
