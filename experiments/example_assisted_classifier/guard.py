"""Keep human answers and historical gold audit reports out of experiment inference."""
from pathlib import Path
import os
import sys


def install_guard(root, metadata_paths=()):
    root = Path(root).resolve()
    allowed = {Path(p).resolve() for p in metadata_paths}
    denied = []

    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, Path)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path in allowed or path == root or not path.is_relative_to(root):
            return
        relative = path.relative_to(root)
        blocked = relative.parts[0] in {"annotation", "data"} and (
            relative.parts[0] == "annotation" or path.is_relative_to(root / "data/eval"))
        blocked |= relative.parts[0] == "outputs" and (
            any(term in str(relative) for term in [
                "gold_frozen", "gold_release_audit", "human_gold", "adjudicated",
                "final_gold", "majority_merge", "agreement_selected", "c_tuning",
                "c_gold_smoke_20261018/inputs"])
            or path.name.startswith("gold_")
            or path.name in {"audit.json", "audit_predictions.jsonl", "audit_metrics.json"})
        if blocked:
            denied.append(str(path))
            raise PermissionError("Human gold answers and historical gold audits are forbidden")

    sys.addaudithook(guard)
    try:
        open(root / "data/eval/__answer_read_guard_probe__.csv")
    except PermissionError:
        pass
    else:
        raise RuntimeError("Answer guard failed")
    return denied
