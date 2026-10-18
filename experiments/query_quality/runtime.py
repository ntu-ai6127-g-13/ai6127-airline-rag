"""Shared local inference orchestration, provenance and fail-closed gold guard."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT / "outputs/c_no_gold_optimization_20261018"
DEFAULT_OUT = ROOT / "outputs/query_quality_20261018"
MLX_PYTHON = ROOT / "outputs/query_quality_runtime/venv311/bin/python"
MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def guard_gold():
    denied = []
    blocked_dirs = [ROOT / "data/eval", ROOT / "annotation", ROOT / "outputs/c_tuning_20261018",
                    ROOT / "outputs/c_tuning_quality_20261018", ROOT / "outputs/c_gold_smoke_20261018/inputs"]
    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, Path)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        relative = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else ""
        blocked = any(p.is_relative_to(d) for d in blocked_dirs)
        blocked |= relative.startswith("outputs/") and (
            p.name.startswith("gold_") or p.name in {"selected_errors.jsonl","audit.json","audit_predictions.jsonl","audit_metrics.json"} or
            any(term in relative for term in ["adjudicated", "human_gold", "final_gold", "majority_merge", "agreement_selected"]))
        if blocked:
            denied.append(str(p))
            raise PermissionError("Gold answers and historical gold-tuned artifacts are forbidden during selection")
    sys.addaudithook(guard)
    try:
        open(ROOT / "outputs/c_gold_smoke_20261018/inputs/gold_1073.csv")
    except PermissionError:
        pass
    else:
        raise AssertionError("Gold guard did not intercept the self-test")
    return denied


def llm_predict(rows, variant, directory, progress=None):
    """Launch MLX in its own environment; every emitted record has an aligned ID."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    inputs = directory / "texts.jsonl"
    outputs = directory / "predictions.jsonl"
    with inputs.open("w") as handle:
        for row in rows:
            handle.write(json.dumps({"id": str(row["id"]), "text": str(row["text"])}, ensure_ascii=False) + "\n")
    worker = "mlx_choice_worker.py" if variant == "closed_choice" else "mlx_worker.py"
    command = [str(MLX_PYTHON), str(HERE / worker), "--input", str(inputs),
               "--output", str(outputs), "--model", str(MODEL_PATH)]
    if variant != "closed_choice": command.extend(["--variant", variant])
    with (directory / "inference.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            while process.poll() is None:
                if progress:
                    completed = sum(1 for _ in outputs.open()) if outputs.exists() else 0
                    if variant == "closed_choice" and not completed:
                        try:
                            lines = (directory/"inference.log").read_text().splitlines()
                            last = next(json.loads(line) for line in reversed(lines) if line.startswith('{') and '"task"' in line)
                            completed = (last["completed"] + (len(rows) if last["task"]=="sentiment" else 0))//2
                        except (StopIteration, json.JSONDecodeError): pass
                    progress(completed, len(rows))
                time.sleep(2)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
    if process.returncode:
        raise RuntimeError(f"Local inference failed; see {directory / 'inference.log'}")
    result = [json.loads(line) for line in outputs.read_text().splitlines()]
    if [r["id"] for r in result] != [str(r["id"]) for r in rows]:
        raise AssertionError("Local inference IDs/rows differ from the request")
    return result
