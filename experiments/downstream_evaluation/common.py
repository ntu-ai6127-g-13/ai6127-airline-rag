"""Shared immutable protocol and artifact helpers for C乙 experiments."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 6127


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def frozen(protocol_path):
    protocol = read(protocol_path)
    for name, expected in protocol["input_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen input changed: " + name)
    for name, expected in protocol["source_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    return protocol


def new_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def now():
    return datetime.now(timezone.utc).isoformat()
