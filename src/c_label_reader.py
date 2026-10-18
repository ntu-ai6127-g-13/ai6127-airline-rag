"""Read a C→B candidate sync table without loading models, text or training data."""
from pathlib import Path
import hashlib
import json

import pandas as pd


VERSION = "eight-aspect-subjectivity-polarity-v4.1"
LABEL_COLUMNS = ["chunk_id", "review_id", "aspect", "subjectivity", "polarity", "model_version",
                 "source", "aspect_source", "subjectivity_source", "polarity_source", "parent_split"]
METADATA_COLUMNS = LABEL_COLUMNS[2:]
COVERAGE_COLUMNS = ["chunk_id", "review_id", "is_eval", "prediction_status", "skip_reason", *METADATA_COLUMNS]
SKIP_REASONS = ["eval_sentence", "eval_parent", "eval_duplicate_text", "empty", "route_only", "non_english", "fragment"]
ASPECTS = ["seat", "staff", "food", "entertainment", "ground", "value", "overall", "other"]


def sha256(path):
    """Hash a file without loading its contents into memory."""
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
        raise ValueError("b_coverage 列及顺序与 c-b-sync-v1 不一致")
    _validate_ids(frame, "b_coverage")
    if not pd.api.types.is_bool_dtype(frame.is_eval) or frame.is_eval.isna().any():
        raise ValueError("b_coverage is_eval 必须是无空值布尔列")
    if not frame.prediction_status.isin(["predicted", "skipped"]).all():
        raise ValueError("b_coverage prediction_status 非法或为空")
    predicted = frame.prediction_status.eq("predicted")
    skipped = ~predicted
    if not frame.loc[predicted, "skip_reason"].isna().all():
        raise ValueError("predicted 的 skip_reason 必须为空值")
    if not frame.loc[skipped, "skip_reason"].isin(SKIP_REASONS).all():
        raise ValueError("skipped 必须具有合法 skip_reason")
    if not frame.loc[skipped, METADATA_COLUMNS].isna().all().all():
        raise ValueError("skipped 的所有标签及来源必须为空值")
    for column in METADATA_COLUMNS:
        if not _nonempty_strings(frame.loc[predicted, column]):
            raise ValueError(f"predicted 的 {column} 为空或不是文本")
    if (frame.is_eval & predicted).any():
        raise ValueError("评测句不得具有预测标签")
    if not frame.is_eval.eq(frame.skip_reason.eq("eval_sentence").fillna(False)).all():
        raise ValueError("is_eval 与 eval_sentence 跳过理由不一致")
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
    if not labels.parent_split.isin(["training_parent", "development_parent", "unlabelled_parent"]).all():
        raise ValueError("预测 parent_split 非法")


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


def load_b_labels(package_dir, b_pool=None):
    """Return all B-pool coverage rows, checking versions, hashes and optional B IDs.

    Only manifest.json and b_coverage.parquet are read. With b_pool, require the
    exact same chunk IDs and parent IDs, then return rows in b_pool order. Extra
    B columns are ignored; an optional is_eval column is checked by chunk ID.
    This checks the reader contract, not human acceptance or model quality.
    """
    package = Path(package_dir).resolve()
    manifest_path = package / "manifest.json"
    table_path = package / "b_coverage.parquet"
    if any(path.resolve().parent != package for path in [manifest_path, table_path]):
        raise ValueError("同步包读取文件必须位于包目录内")
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or (manifest.get("schema_version") != "c-b-sync-v1"
            or manifest.get("model_version") != VERSION or manifest.get("status") != "candidate"):
        raise ValueError("不是支持的 c-b-sync-v1 / v4.1 候选包")
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
    if "is_eval" in b_pool:
        if not pd.api.types.is_bool_dtype(b_pool.is_eval) or b_pool.is_eval.isna().any():
            raise ValueError("B 池 is_eval 必须是无空值布尔列")
        if not aligned.is_eval.eq(b_pool.is_eval.to_numpy()).all():
            raise ValueError("B 池 is_eval 与同步包不一致")
    return aligned[COVERAGE_COLUMNS]
