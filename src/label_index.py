"""Build and read a complete offline label index aligned to retrieval IDs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from src.config import CHUNKS, CHUNK_IDS, ROOT


ASPECTS = {"seat", "staff", "food", "entertainment", "ground", "value", "overall", "other"}
SUBJECTIVITY = {"subjective", "objective"}
POLARITY = {"positive", "negative", "NA"}
REQUIRED_COLUMNS = ["chunk_id", "review_id", "aspect", "subjectivity", "polarity"]


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def validate_labels(frame: pd.DataFrame, expected_ids: list[str] | None = None) -> None:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"标签索引缺少字段：{sorted(missing)}")
    if not frame.chunk_id.is_unique:
        raise ValueError("标签索引 chunk_id 重复")
    if expected_ids is not None and frame.chunk_id.astype(str).tolist() != expected_ids:
        raise ValueError("标签索引与检索 ID 的数量、顺序或内容不一致")
    if not frame.aspect.isin(ASPECTS).all():
        raise ValueError("标签索引含非法方面标签")
    if not frame.subjectivity.isin(SUBJECTIVITY).all():
        raise ValueError("标签索引含非法主客观标签")
    if not frame.polarity.isin(POLARITY).all():
        raise ValueError("标签索引含非法极性标签")
    if not frame.loc[frame.subjectivity.eq("objective"), "polarity"].eq("NA").all():
        raise ValueError("客观句必须使用 polarity=NA")
    if not frame.loc[frame.subjectivity.eq("subjective"), "polarity"].isin({"positive", "negative"}).all():
        raise ValueError("主观句必须具有正面或负面极性")


@lru_cache(maxsize=2)
def _load_label_lookup_cached(path_text: str, modified_ns: int, size: int) -> dict[str, dict]:
    del modified_ns, size
    frame = pd.read_parquet(path_text, columns=REQUIRED_COLUMNS)
    validate_labels(frame)
    return {
        str(row.chunk_id): {
            "aspect": str(row.aspect),
            "subjectivity": str(row.subjectivity),
            "polarity": str(row.polarity),
        }
        for row in frame.itertuples(index=False)
    }


def load_label_lookup(path: str | Path) -> dict[str, dict]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError("离线标签索引不存在：" + str(resolved))
    stat = resolved.stat()
    return _load_label_lookup_cached(str(resolved), stat.st_mtime_ns, stat.st_size)


def labels_for_chunks(chunk_ids: list[str], path: str | Path) -> list[dict]:
    lookup = load_label_lookup(path)
    missing = [str(chunk_id) for chunk_id in chunk_ids if str(chunk_id) not in lookup]
    if missing:
        raise RuntimeError(f"离线标签索引缺少 {len(missing)} 个检索结果；首个缺失 ID：{missing[0]}")
    return [lookup[str(chunk_id)] for chunk_id in chunk_ids]


def _load_retrieval_pool(chunks_path: Path, ids_path: Path) -> tuple[pd.DataFrame, list[str]]:
    retrieval_ids = [str(value) for value in np.load(ids_path, allow_pickle=True).tolist()]
    if len(retrieval_ids) != len(set(retrieval_ids)):
        raise ValueError("检索 ID 存在重复")
    frame = pd.read_parquet(chunks_path, columns=["chunk_id", "review_id", "text"])
    if not frame.chunk_id.is_unique:
        raise ValueError("知识库 chunk_id 存在重复")
    frame["chunk_id"] = frame.chunk_id.astype(str)
    indexed = frame.set_index("chunk_id").reindex(retrieval_ids)
    if indexed[["review_id", "text"]].isna().any().any():
        missing = indexed.index[indexed.review_id.isna()].tolist()
        raise ValueError(f"检索 ID 无法在知识库中找到：{missing[:3]}")
    indexed = indexed.reset_index()
    return indexed, retrieval_ids


def build(args: argparse.Namespace) -> int:
    from src.classification_pipeline import predict_texts
    from src.training_isolation import install_answer_read_guard
    import joblib

    started = time.monotonic()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(exist_ok=True)
    labels_path = output_dir / "labels.parquet"
    report_path = output_dir / "report.json"
    monitor_path = output_dir / "monitor.json"
    plan_path = output_dir / "plan.json"
    if labels_path.exists() or report_path.exists():
        raise ValueError("标签索引已经完成；拒绝覆盖")
    if args.batch_size < 1:
        raise ValueError("batch-size 必须大于零")

    denied = install_answer_read_guard()
    inputs = {
        "model": {"path": str(args.model.resolve()), "sha256": sha256(args.model)},
        "chunks": {"path": str(args.chunks.resolve()), "sha256": sha256(args.chunks)},
        "retrieval_ids": {"path": str(args.retrieval_ids.resolve()), "sha256": sha256(args.retrieval_ids)},
    }
    plan = {
        "version": 1,
        "inputs": inputs,
        "batch_size": args.batch_size,
        "output_contract": "one prediction for every retrieval chunk_id in identical order",
        "gold_labels_read": False,
    }
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("续跑参数或输入哈希与既有 plan 不一致")
    else:
        write_json(plan_path, plan)

    frame, retrieval_ids = _load_retrieval_pool(args.chunks, args.retrieval_ids)
    bundle = joblib.load(args.model)
    total = len(frame)
    shard_count = (total + args.batch_size - 1) // args.batch_size
    monitor = {
        "status": "running",
        "completed": 0,
        "total": total,
        "shards_completed": 0,
        "shards_total": shard_count,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "gold_labels_read": False,
    }
    write_json(monitor_path, monitor)

    completed = 0
    for number, start in enumerate(range(0, total, args.batch_size)):
        stop_path = output_dir / "STOP"
        if stop_path.exists():
            monitor.update(status="stopped", completed=completed, elapsed_seconds=time.monotonic() - started)
            write_json(monitor_path, monitor)
            return 2
        batch = frame.iloc[start:start + args.batch_size].reset_index(drop=True)
        shard_path = shards_dir / f"part-{number:05d}.parquet"
        expected = batch.chunk_id.tolist()
        if shard_path.exists():
            shard = pd.read_parquet(shard_path)
            validate_labels(shard, expected)
        else:
            predicted = predict_texts(bundle, batch.text.tolist())
            shard = pd.concat([
                batch[["chunk_id", "review_id"]],
                predicted.drop(columns=["text"], errors="ignore"),
            ], axis=1)
            shard["label_source"] = "fast_classifier"
            shard["model_version"] = bundle["version"]
            validate_labels(shard, expected)
            with tempfile.NamedTemporaryFile(dir=shards_dir, suffix=".parquet", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                shard.to_parquet(temporary, index=False, compression="zstd")
                pd.testing.assert_frame_equal(shard, pd.read_parquet(temporary))
                temporary.replace(shard_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        completed += len(batch)
        elapsed = time.monotonic() - started
        monitor.update(
            completed=completed,
            shards_completed=number + 1,
            elapsed_seconds=round(elapsed, 2),
            rows_per_second=round(completed / elapsed, 2) if elapsed else None,
            estimated_remaining_seconds=round((total - completed) * elapsed / completed, 2) if completed else None,
            heartbeat_at=datetime.now(timezone.utc).isoformat(),
        )
        write_json(monitor_path, monitor)
        print(f"标签 {completed:,}/{total:,}", flush=True)

    parts = [pd.read_parquet(shards_dir / f"part-{number:05d}.parquet") for number in range(shard_count)]
    labels = pd.concat(parts, ignore_index=True)
    validate_labels(labels, retrieval_ids)
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        labels.to_parquet(temporary, index=False, compression="zstd")
        restored = pd.read_parquet(temporary)
        pd.testing.assert_frame_equal(labels, restored)
        temporary.replace(labels_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if len(denied) != 1:
        raise RuntimeError("gold 读取守卫自检记录异常")
    elapsed = time.monotonic() - started
    report = {
        "status": "completed",
        "rows": len(labels),
        "unique_chunk_ids": int(labels.chunk_id.nunique()),
        "retrieval_alignment_exact": True,
        "gold_labels_read": False,
        "model_version": bundle["version"],
        "label_source": "fast_classifier",
        "elapsed_seconds": round(elapsed, 2),
        "counts": {
            "aspect": labels.aspect.value_counts().to_dict(),
            "subjectivity": labels.subjectivity.value_counts().to_dict(),
            "polarity": labels.polarity.value_counts().to_dict(),
        },
        "inputs": inputs,
        "output_sha256": sha256(labels_path),
        "limitations": [
            "Base labels are for soft ranking and descriptive drafts, not hard evidence deletion.",
            "Qwen review is a later incremental layer.",
        ],
    }
    write_json(report_path, report)
    monitor.update(status="completed", completed=total, elapsed_seconds=round(elapsed, 2), output_sha256=report["output_sha256"])
    write_json(monitor_path, monitor)
    print(json.dumps({"status": "completed", "rows": total, "elapsed_seconds": round(elapsed, 2)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/weak_rebuild_20261019/model.joblib")
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--retrieval-ids", type=Path, default=CHUNK_IDS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/label_index_20261019")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args(argv)
    try:
        return build(args)
    except (ValueError, OSError, KeyError) as error:
        print("错误：" + str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
