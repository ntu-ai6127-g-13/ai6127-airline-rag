"""Bridge to an explicitly selected frozen experimental classifier.

Inference runs in a subprocess so historical C-group modules cannot shadow
the current application's src package. This does not build a global label index.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from src.config import ROOT


def classify_texts(texts: list[str], manifest: str | Path, *, timeout_seconds: int = 600) -> list[dict]:
    path = Path(manifest).resolve()
    if not path.is_file(): raise FileNotFoundError("Frozen classifier manifest is missing: " + str(path))
    if not texts: return []
    with tempfile.TemporaryDirectory(prefix="airline_classifier_") as directory:
        source = Path(directory)/"texts.jsonl"; target = Path(directory)/"predictions.jsonl"
        source.write_text("".join(json.dumps({"id": str(i), "text": text}, ensure_ascii=False)+"\n" for i,text in enumerate(texts)))
        process = subprocess.Popen([sys.executable, str(ROOT/"experiments/query_quality/predict_cli.py"),
            "--manifest", str(path), "--input", str(source), "--output", str(target)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            _, stderr = process.communicate(timeout=timeout_seconds)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise
        if process.returncode:
            raise RuntimeError("Local classifier failed: " + stderr[-2000:])
        rows = [json.loads(line) for line in target.read_text().splitlines()]
    if [str(row["id"]) for row in rows] != [str(i) for i in range(len(texts))]:
        raise RuntimeError("Classifier prediction IDs are misaligned")
    return rows
