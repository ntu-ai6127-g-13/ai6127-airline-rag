"""用冻结的 r36 manifest 分批给样本打教师标签；可 STOP、可续跑，每批独立校验 ID。

每批调用 experiments/query_quality/predict_cli.py，与查询页面实时回退走同一条推理链路。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "outputs/query_r36_experiment_20261019/manifest.json"


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None, help="只标前 N 句，用于测速")
    args = parser.parse_args(argv)
    sample_dir = args.sample_dir.resolve()
    rows = [json.loads(line) for line in (sample_dir / "texts.jsonl").read_text().splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    parts = sample_dir / ("teacher_parts" if not args.limit else f"teacher_speed_{args.limit}")
    parts.mkdir(exist_ok=True)
    monitor_path = parts / "monitor.json"
    started = time.monotonic()
    batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
    monitor = {"status": "running", "manifest": str(MANIFEST), "batches_total": len(batches),
               "sentences_total": len(rows), "started_at": datetime.now(timezone.utc).isoformat()}
    for number, batch in enumerate(batches):
        if (parts / "STOP").exists():
            monitor.update(status="stopped"); write_json(monitor_path, monitor); return 2
        target = parts / f"part-{number:05d}.jsonl"
        if not target.exists():
            source = parts / f"input-{number:05d}.jsonl"
            source.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in batch))
            pending = parts / f"pending-{number:05d}.jsonl"
            pending.unlink(missing_ok=True)
            batch_started = time.monotonic()
            result = subprocess.run(
                [sys.executable, str(ROOT / "experiments/query_quality/predict_cli.py"),
                 "--manifest", str(MANIFEST), "--input", str(source), "--output", str(pending)],
                cwd=ROOT, capture_output=True, text=True)
            if result.returncode:
                monitor.update(status="failed", failed_batch=number, stderr=result.stderr[-3000:])
                write_json(monitor_path, monitor)
                return 1
            got = [json.loads(line) for line in pending.read_text().splitlines()]
            if [str(r["id"]) for r in got] != [r["id"] for r in batch]:
                raise RuntimeError(f"第 {number} 批 ID 不一致")
            pending.replace(target)
            source.unlink()
            monitor[f"batch_{number:05d}_seconds"] = round(time.monotonic() - batch_started, 1)
        done = min((number + 1) * args.batch_size, len(rows))
        monitor.update(completed=done, batches_completed=number + 1,
                       elapsed_seconds=round(time.monotonic() - started, 1))
        write_json(monitor_path, monitor)
        print(f"{done}/{len(rows)}", flush=True)
    frames = [pd.read_json(p, lines=True, dtype={"id": str}) for p in sorted(parts.glob("part-*.jsonl"))]
    labels = pd.concat(frames, ignore_index=True)
    if labels.id.tolist() != [r["id"] for r in rows]:
        raise RuntimeError("合并后的教师标签与样本顺序不一致")
    labels.to_parquet(parts.parent / ("teacher_labels.parquet" if not args.limit else f"teacher_speed_{args.limit}.parquet"), index=False)
    monitor.update(status="completed"); write_json(monitor_path, monitor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
