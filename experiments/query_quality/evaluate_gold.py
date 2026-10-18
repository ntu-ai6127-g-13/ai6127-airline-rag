"""One post-freeze development audit. No candidate or prompt is changed by gold."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from runtime import ROOT, HERE, PRIOR, DEFAULT_OUT, sha, write_json, llm_predict
sys.path.insert(0, str(ROOT))
from src.quality import classification_metrics, CLASSIFIER_TARGETS
from run_monitor import combine_predictions


def load_previous_predictions(path, gold):
    saved = pd.read_json(path, lines=True)
    assert saved.gold_sentence_id.tolist() == gold.sentence_id.tolist()
    assert saved.gold_text.tolist() == gold.text.tolist()
    fields = ["aspect", "subjectivity", "polarity", "label"]
    result = saved[["pred_"+c for c in fields]].copy(); result.columns = fields
    return result


def main(out):
    if (out / "audit.json").exists() or (out / "audit_monitor.json").exists():
        raise RuntimeError("This audit was already started; preserve its outputs")
    frozen_path = out / "frozen_candidates.json"
    frozen = json.loads(frozen_path.read_text()); frozen_hash = sha(frozen_path)
    for section in ["source_hashes", "input_hashes", "model_weight_hashes"]:
        for path, digest in frozen[section].items():
            if sha(path) != digest: raise RuntimeError("Frozen artifact changed: " + path)
    state = {"status": "running", "phase": "reading gold after freeze", "completed": []}
    def phase(value):
        state.update(phase=value, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(out / "audit_monitor.json", state)
    phase(state["phase"])
    gold_path = ROOT / "outputs/c_gold_smoke_20261018/inputs/gold_1073.csv"
    gold = pd.read_csv(gold_path, keep_default_na=False); gold_hash = sha(gold_path)
    old_path = ROOT / "outputs/c_tuning_20261018/r6_semantic_C0.1/predictions.jsonl"
    old = load_previous_predictions(old_path, gold)
    components = {}
    for name in frozen["component_sources"].values():
        components[name] = load_previous_predictions(PRIOR / name / "gold_predictions.jsonl", gold)
    variants = {value for spec in frozen["candidates"].values() for key,value in spec.items()
                if key in {"source","aspect_source","sentiment_source","other_source"} and value in {"taxonomy","evidence_first","closed_choice"}}
    for variant in sorted(variants):
        rows = [{"id": str(i), "text": text} for i, text in enumerate(gold.text)]
        components[variant] = pd.DataFrame(llm_predict(rows, variant, out / ("audit_" + variant),
            lambda done, total: phase(f"{variant}: {done}/{total}")))
    import importlib.util
    legacy = ROOT / "outputs/c_gold_smoke_20261018/source/src"
    for module in ["weak_label_rules", "language_rules"]:
        spec = importlib.util.spec_from_file_location("src." + module, legacy / (module + ".py"))
        loaded = importlib.util.module_from_spec(spec); sys.modules["src."+module] = loaded; spec.loader.exec_module(loaded)
    rule_func = sys.modules["src.language_rules"].choose_focus
    predictions = {}; results = {}
    for name, spec in frozen["candidates"].items():
        pred = combine_predictions(spec, components, gold.text, rule_func)
        predictions[name] = pred
        metrics = classification_metrics(gold, pred)
        metrics["all_aspect_f1"] = float(f1_score(gold.aspect, pred.aspect, average="macro"))
        metrics["all_label_f1"] = float(f1_score(gold.label, pred.label, labels=["objective", "negative", "positive"], average="macro"))
        results[name] = metrics
        write_json(out / name / "audit_metrics.json", metrics)
        gold.add_prefix("gold_").join(pred.add_prefix("pred_")).to_json(out / name / "audit_predictions.jsonl", orient="records", lines=True, force_ascii=False)
        state["completed"].append(name); phase(name + " completed")
    selected = frozen["selected_without_gold"]; selected_pred = predictions[selected]
    # Parent-group bootstrap of the primary joint accuracy, with fixed predictions only.
    eligible = gold.subjectivity.eq("subjective")
    correct = gold.aspect.eq(selected_pred.aspect) & gold.polarity.eq(selected_pred.polarity)
    old_correct = gold.aspect.eq(old.aspect) & gold.polarity.eq(old.polarity)
    grouped = pd.DataFrame({"group": gold.review_id, "n": eligible.astype(int),
                            "new": (correct & eligible).astype(int), "old": (old_correct & eligible).astype(int)}).groupby("group").sum()
    rng = np.random.RandomState(20261019); estimates = []; deltas = []
    for _ in range(2000):
        sample = grouped.iloc[rng.randint(0, len(grouped), len(grouped))].sum()
        estimates.append(float(sample["new"]/sample["n"])); deltas.append(float((sample["new"]-sample["old"])/sample["n"]))
    assert sha(gold_path) == gold_hash and sha(frozen_path) == frozen_hash
    for section in ["source_hashes", "input_hashes", "model_weight_hashes"]:
        for path, digest in frozen[section].items(): assert sha(path) == digest
    report = {"status": "completed", "gold_n": len(gold), "gold_sha256": gold_hash,
              "frozen_sha256": frozen_hash, "selected_without_gold": selected,
              "targets": CLASSIFIER_TARGETS, "previous_retained_metrics": classification_metrics(gold, old),
              "selected_metrics": results[selected], "candidates": results,
              "selected_joint_accuracy_95": np.quantile(estimates, [.025,.975]).tolist(),
              "selected_joint_delta_vs_old_95": np.quantile(deltas, [.025,.975]).tolist(),
              "gold_used_for_training": False, "gold_used_for_selection": False,
              "independent_test": False, "independent_query_test_passed": False,
              "model_promoted": False,
              "verification": {"gold_unchanged": True, "freeze_unchanged": True, "frozen_artifacts_unchanged": True},
              "limitations": ["Historical gold exposure remains; this is development evidence.",
                              "LLM is frozen zero-shot; project gold was never included in prompts or fitting.",
                              "Weak/synthetic monitor is a proxy, not acceptance.",
                              "No system acceptance claim without judged query results."]}
    write_json(out / "audit.json", report)
    write_json(out / "release_gate.json", {"classifier_point_targets_passed": results[selected]["point_targets_passed"],
                "independent_query_test_passed": False, "production_ready": False, "previous_candidate_retained": True})
    state.update(status="completed", phase="audit complete; no gold-driven reselection"); phase(state["phase"])
    monitor = json.loads((out / "monitor.json").read_text())
    monitor.update(status="completed_audited", classifier_point_targets_passed=results[selected]["point_targets_passed"],
                   independent_query_test_passed=False, production_ready=False)
    write_json(out / "monitor.json", monitor)
    print(json.dumps({"selected": selected, "metrics": results[selected]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(); main(args.run_dir.resolve())
