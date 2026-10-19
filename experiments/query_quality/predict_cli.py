"""Load a frozen experimental candidate and predict new texts locally."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import joblib
import pandas as pd

from runtime import ROOT, HERE, PRIOR, sha, llm_predict
sys.path.insert(0, str(ROOT))
from run_monitor import combine_predictions


def predict_frozen(manifest, rows, progress=None):
    frozen = json.loads(Path(manifest).read_text())
    for file, digest in frozen["source_hashes"].items():
        if sha(file) != digest: raise RuntimeError("Frozen candidate source changed: " + file)
    for file, digest in frozen.get("model_weight_hashes", {}).items():
        if sha(file) != digest: raise RuntimeError("Frozen local-model weights changed: " + file)
    spec = frozen["selected_spec"]
    required = {spec[k] for k in ["source", "aspect_source", "sentiment_source", "other_source"] if k in spec}
    components = {}
    texts = [row["text"] for row in rows]
    rule_func = None
    if spec["mode"] == "clause_rules" or any(name not in {"taxonomy","evidence_first","closed_choice"} for name in required):
        legacy = ROOT / "outputs/c_gold_smoke_20261018/source/src"
        for module in ["weak_label_rules", "language_rules"]:
            module_spec = importlib.util.spec_from_file_location("src." + module, legacy / (module + ".py"))
            loaded = importlib.util.module_from_spec(module_spec); sys.modules["src." + module] = loaded; module_spec.loader.exec_module(loaded)
        rule_func = sys.modules["src.language_rules"].choose_focus
    with tempfile.TemporaryDirectory(prefix="query_quality_") as temporary:
        for name in sorted(required):
            if name in {"taxonomy", "evidence_first", "closed_choice"}:
                components[name] = pd.DataFrame(llm_predict(rows, name, Path(temporary)/name, progress))
            else:
                path = PRIOR / name / "model.joblib"
                if sha(path) != frozen["input_hashes"][str(path)]: raise RuntimeError("Component weights changed")
                sys.path.insert(0, str(ROOT / "experiments/no_gold_optimization"))
                from inference import predict
                components[name] = predict(joblib.load(path), texts)
        result = combine_predictions(spec, components, texts, rule_func)
    result["id"] = [row["id"] for row in rows]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise RuntimeError("Output already exists")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if any(set(row) != {"id", "text"} for row in rows): raise ValueError("Only IDs/text are accepted")
    result = predict_frozen(args.manifest, rows)
    result.to_json(args.output, orient="records", lines=True, force_ascii=False)
