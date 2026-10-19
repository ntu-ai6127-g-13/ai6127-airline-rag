"""Deny answer/report reads during CLI training while allowing original exclusion metadata."""
from pathlib import Path
import os
import sys

from src.config import ROOT


def install_answer_read_guard():
    denied = []
    allowed = {ROOT / "data/eval/to_label.csv", ROOT / "data/eval/to_label_meta.csv"}
    blocked_roots = [ROOT / "annotation", ROOT / "outputs/c_tuning_20261018",
                     ROOT / "outputs/c_tuning_quality_20261018", ROOT / "outputs/c_gold_smoke_20261018/inputs"]
    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, Path)): return
        path = Path(os.fsdecode(args[0])).resolve()
        if path in allowed: return
        if not path.is_relative_to(ROOT): return
        relative = str(path.relative_to(ROOT))
        blocked = path.is_relative_to(ROOT / "data/eval") or any(path.is_relative_to(p) for p in blocked_roots)
        blocked |= relative.startswith("outputs/") and (
            any(term in relative for term in ["gold_frozen", "gold_release_audit", "human_gold", "adjudicated", "final_gold", "majority_merge", "agreement_selected"])
            or path.name.startswith("gold_") or path.name in {"audit.json","audit_predictions.jsonl","audit_metrics.json"})
        if blocked:
            denied.append(str(path))
            raise PermissionError("Human gold answers and historical audit outputs are forbidden in training")
    sys.addaudithook(guard)
    try: open(ROOT / "data/eval/gold_human_frozen_1000_20261019_v1.csv")
    except PermissionError: pass
    else: raise RuntimeError("Gold answer-read guard did not intercept the probe")
    return denied
