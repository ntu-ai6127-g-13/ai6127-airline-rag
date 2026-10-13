"""Read the v2 C→B sync contract, including held-out parent inference provenance."""
from pathlib import Path
import hashlib
import json

import pandas as pd


SCHEMA_VERSION = "c-b-sync-v2"
VERSION = "eight-aspect-subjectivity-polarity-v4.1"
BASE_METADATA_COLUMNS = ["aspect", "subjectivity", "polarity", "model_version", "source",
                         "aspect_source", "subjectivity_source", "polarity_source", "parent_split"]
METADATA_COLUMNS = [*BASE_METADATA_COLUMNS, "prediction_origin"]
LABEL_COLUMNS = ["chunk_id", "review_id", "is_eval_parent", *METADATA_COLUMNS]
COVERAGE_COLUMNS = ["chunk_id", "review_id", "is_eval", "is_eval_parent", "prediction_status",
                    "skip_reason", *METADATA_COLUMNS]
SKIP_REASONS = ["eval_sentence", "eval_parent", "eval_duplicate_text", "empty", "route_only", "non_english", "fragment"]
ASPECTS = ["seat", "staff", "food", "entertainment", "ground", "value", "overall", "other"]
INFERENCE_ORIGINS = ["existing_v4_1", "supplemental_eval_parent"]
EXISTING_PARENT_ROLES = ["training_parent", "development_parent", "unlabelled_parent"]
HELDOUT_PARENT_ROLE = "heldout_parent_prediction_only"


def sha256(path):
    """Hash one file without loading models or parsing its contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _ordered_ids_sha256(identifiers):
    return hashlib.sha256(("\n".join(identifiers) + "\n").encode("utf-8")).hexdigest()


def _nonempty_strings(series):
    return series.map(lambda value: isinstance(value, str) and bool(value.strip())).all()


def _validate_ids(frame, name):
    for column in ["chunk_id", "review_id"]:
        if column not in frame or not _nonempty_strings(frame[column]):
            raise ValueError(f"{name} {column} 必须是无空值的文本编号")
    if not frame.chunk_id.is_unique:
        raise ValueError(f"{name} chunk_id 重复")


def _validate_table(frame):
    if list(frame.columns) != COVERAGE_COLUMNS:
        raise ValueError("b_coverage 列及顺序与 c-b-sync-v2 不一致")
    _validate_ids(frame, "b_coverage")
    for column in ["is_eval", "is_eval_parent"]:
        if not pd.api.types.is_bool_dtype(frame[column]) or frame[column].isna().any():
            raise ValueError(f"b_coverage {column} 必须是无空值布尔列")
    if not frame.prediction_status.isin(["predicted", "skipped"]).all():
        raise ValueError("b_coverage prediction_status 非法或为空")
    predicted = frame.prediction_status.eq("predicted")
    skipped = ~predicted
    if not frame.loc[predicted, "skip_reason"].isna().all():
        raise ValueError("predicted 的 skip_reason 必须为空值")
    if not frame.loc[skipped, "skip_reason"].isin(SKIP_REASONS).all():
        raise ValueError("skipped 必须具有合法 skip_reason")
    if not frame.loc[skipped, METADATA_COLUMNS].isna().all().all():
        raise ValueError("skipped 的所有标签、来源及 prediction_origin 必须为空值")
    for column in METADATA_COLUMNS:
        if not _nonempty_strings(frame.loc[predicted, column]):
            raise ValueError(f"predicted 的 {column} 为空或不是文本")
    if (frame.is_eval & predicted).any():
        raise ValueError("金标准评测句不得具有预测标签")
    if not frame.is_eval.eq(frame.skip_reason.eq("eval_sentence").fillna(False)).all():
        raise ValueError("is_eval 与 eval_sentence 跳过理由不一致")
    if (frame.is_eval & ~frame.is_eval_parent).any():
        raise ValueError("金标准评测句必须属于 is_eval_parent")
    if (frame.skip_reason.eq("eval_parent").fillna(False) & ~frame.is_eval_parent).any():
        raise ValueError("eval_parent 跳过理由必须属于 is_eval_parent")
    labels = frame.loc[predicted]
    if not labels.model_version.eq(VERSION).all():
        raise ValueError("预测 model_version 不是支持的 v4.1 版本")
    if not labels.source.eq("hybrid_prediction").all():
        raise ValueError("预测 source 不是 hybrid_prediction")
    if not labels.aspect.isin(ASPECTS).all():
        raise ValueError("预测 aspect 不属于八方面")
    if not labels.subjectivity.isin(["objective", "subjective"]).all():
        raise ValueError("预测 subjectivity 非法")
    objective = labels.subjectivity.eq("objective")
    if not labels.loc[objective, "polarity"].eq("NA").all():
        raise ValueError("objective 的 polarity 必须是字面字符串 NA")
    if not labels.loc[~objective, "polarity"].isin(["negative", "positive"]).all():
        raise ValueError("subjective 的 polarity 必须为正负二元标签")
    if not labels.prediction_origin.isin(INFERENCE_ORIGINS).all():
        raise ValueError("prediction_origin 非法")
    heldout = labels.is_eval_parent
    if (not labels.loc[heldout, "prediction_origin"].eq("supplemental_eval_parent").all()
            or not labels.loc[heldout, "parent_split"].eq(HELDOUT_PARENT_ROLE).all()):
        raise ValueError("评测父评论预测必须标为 supplemental_eval_parent / heldout_parent_prediction_only")
    if (not labels.loc[~heldout, "prediction_origin"].eq("existing_v4_1").all()
            or not labels.loc[~heldout, "parent_split"].isin(EXISTING_PARENT_ROLES).all()):
        raise ValueError("非评测父评论预测必须保留 existing_v4_1 及原父评论角色")


def _validate_manifest_counts(manifest, frame):
    pool = manifest.get("b_pool")
    if not isinstance(pool, dict):
        raise ValueError("manifest 缺少 b_pool 记录")
    counts = {"rows": len(frame),
              "predicted_rows": int(frame.prediction_status.eq("predicted").sum()),
              "skipped_rows": int(frame.prediction_status.eq("skipped").sum())}
    for name, actual in counts.items():
        if type(pool.get(name)) is not int or pool[name] != actual:
            raise ValueError(f"manifest b_pool/{name} 与实际行数不一致")
    reasons = pool.get("excluded_by_first_reason")
    if not isinstance(reasons, dict) or set(reasons) != set(SKIP_REASONS):
        raise ValueError("manifest b_pool 跳过理由记录不完整")
    for reason in SKIP_REASONS:
        if type(reasons[reason]) is not int or reasons[reason] != int(frame.skip_reason.eq(reason).sum()):
            raise ValueError(f"manifest b_pool 跳过计数不一致：{reason}")
    if pool.get("ordered_chunk_ids_sha256") != _ordered_ids_sha256(frame.chunk_id.tolist()):
        raise ValueError("manifest B 池句子编号顺序哈希不一致")
    origins = manifest.get("inference_origin_counts")
    if not isinstance(origins, dict) or set(origins) != set(INFERENCE_ORIGINS):
        raise ValueError("manifest inference_origin_counts 记录不完整")
    for origin in INFERENCE_ORIGINS:
        if type(origins[origin]) is not int or origins[origin] != int(frame.prediction_origin.eq(origin).sum()):
            raise ValueError(f"manifest 推理来源计数不一致：{origin}")


def load_b_labels(package_dir, b_pool=None):
    """Read only manifest.json and b_coverage.parquet; validate the v2 contract.

    An optional B pool must contain the exact same sentence IDs and parents.
    Its optional is_eval and is_eval_parent flags are also checked by ID. The
    result follows B-pool order, retaining all skipped rows and their null
    metadata. Supplemental parent predictions are inference-only records,
    not training labels, development references or final evaluation results.
    """
    package = Path(package_dir).resolve()
    manifest_path = package / "manifest.json"
    table_path = package / "b_coverage.parquet"
    if any(path.resolve().parent != package for path in [manifest_path, table_path]):
        raise ValueError("同步包读取文件必须位于包目录内")
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or (manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("model_version") != VERSION or manifest.get("status") != "candidate"):
        raise ValueError("不是支持的 c-b-sync-v2 / v4.1 候选包")
    artifacts = manifest.get("artifacts_sha256")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("b_coverage.parquet"), str):
        raise ValueError("manifest 缺少 b_coverage.parquet 哈希")
    table_hash = artifacts["b_coverage.parquet"]
    if sha256(table_path) != table_hash:
        raise ValueError("b_coverage.parquet 哈希与 manifest 不一致")
    frame = pd.read_parquet(table_path)
    _validate_table(frame)
    schemas = manifest.get("schemas")
    if (not isinstance(schemas, dict)
            or schemas.get("b_coverage.parquet") != {column: str(frame[column].dtype) for column in frame}):
        raise ValueError("b_coverage schema 与 manifest 不一致")
    _validate_manifest_counts(manifest, frame)
    if sha256(table_path) != table_hash or sha256(manifest_path) != manifest_hash:
        raise ValueError("检查期间同步包发生变化")
    if b_pool is None:
        return frame
    if not isinstance(b_pool, pd.DataFrame) or not b_pool.columns.is_unique:
        raise ValueError("b_pool 必须是列名唯一的 DataFrame")
    _validate_ids(b_pool, "b_pool")
    if set(b_pool.chunk_id) != set(frame.chunk_id):
        raise ValueError("B 池 ID 集合与同步包不精确一致；不能静默丢行")
    aligned = frame.set_index("chunk_id").loc[b_pool.chunk_id.tolist()].reset_index()
    if not aligned.review_id.eq(b_pool.review_id.to_numpy()).all():
        raise ValueError("B 池父评论编号与同步包不一致")
    for column in ["is_eval", "is_eval_parent"]:
        if column in b_pool:
            if not pd.api.types.is_bool_dtype(b_pool[column]) or b_pool[column].isna().any():
                raise ValueError(f"B 池 {column} 必须是无空值布尔列")
            if not aligned[column].eq(b_pool[column].to_numpy()).all():
                raise ValueError(f"B 池 {column} 与同步包不一致")
    return aligned[COVERAGE_COLUMNS]
