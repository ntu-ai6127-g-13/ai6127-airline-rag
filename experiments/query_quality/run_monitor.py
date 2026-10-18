"""Freeze query-oriented candidates before any new project-gold evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pandas as pd

from runtime import ROOT, HERE, PRIOR, DEFAULT_OUT, MODEL_PATH, sha, write_json, guard_gold, llm_predict
sys.path.insert(0, str(ROOT))
from src.quality import CLASSIFIER_TARGETS, QUERY_TARGETS, classification_metrics, proxy_rank
from prompts import PROMPTS, MODEL_ID, MODEL_REVISION


class Monitor:
    def __init__(self, out):
        self.out = out
        if (out / "monitor.json").exists():
            raise RuntimeError("Choose a new output directory; completed and interrupted runs are preserved")
        out.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic(); self.lock = threading.RLock(); self.stop = threading.Event()
        self.state = {"status": "running", "pid": os.getpid(), "phase": "initializing", "rounds": [],
                      "started_at": datetime.now(timezone.utc).isoformat(), "classifier_targets": CLASSIFIER_TARGETS,
                      "query_targets": QUERY_TARGETS, "gold_used_for_training": False, "gold_used_for_selection": False,
                      "max_wall_seconds": 5400, "selection_is_proxy_only": True,
                      "schedule": ["qwen_taxonomy", "qwen_evidence_first", "weak_aspect_qwen_sentiment",
                                   "qwen_aspect_nli_sentiment", "qwen_subjective_consensus", "qwen_clause_rules"]}
        self.flush()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True); self.thread.start()
    def flush(self):
        with self.lock:
            self.state.update(updated_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-self.started)
            write_json(self.out / "monitor.json", self.state)
    def heartbeat(self):
        while not self.stop.wait(10): self.flush()
    def phase(self, phase):
        if (self.out / "STOP").exists() or time.monotonic()-self.started > self.state["max_wall_seconds"]:
            raise RuntimeError("Monitor stopped by STOP file or wall-time limit")
        self.state["phase"] = phase; self.flush()
    def close(self, status):
        self.stop.set(); self.thread.join(timeout=2)
        self.state.update(status=status, finished_at=datetime.now(timezone.utc).isoformat()); self.flush()


def combined(frame, pred):
    reports = {}
    for source in sorted(frame.source.unique()):
        mask = frame.source.eq(source)
        reports[source] = classification_metrics(frame.loc[mask], pred.loc[mask])
    return {"by_source": reports, "proxy_selection_score": float(np.mean([proxy_rank(m) for m in reports.values()]))}


def combine_predictions(spec, components, texts, rule_func=None):
    mode = spec["mode"]
    if mode == "direct": return components[spec["source"]].copy()
    p = components[spec["aspect_source"]].copy()
    sentiment = components[spec["sentiment_source"]]
    for col in ["subjectivity", "polarity", "label"]: p[col] = sentiment[col].to_numpy()
    if mode == "subjective_consensus":
        other = components[spec["other_source"]]
        recover = p.subjectivity.eq("objective") & other.subjectivity.eq("subjective")
        for col in ["subjectivity", "polarity", "label"]: p.loc[recover, col] = other.loc[recover, col]
    elif mode == "clause_rules":
        if rule_func is None: raise ValueError("Clause rules are required")
        for i, text in enumerate(texts):
            rule = rule_func(text)
            if rule.get("aspect") and rule["reason"].startswith("r16_"):
                p.loc[i, "aspect"] = rule["aspect"]
    return p


def main(out):
    monitor = Monitor(out)
    try:
        denied = guard_gold()
        source = PRIOR / "selection_data.jsonl"
        whole = pd.read_json(source, lines=True)
        # Predeclared stratified sample; roles and parent isolation from the original dev split are preserved.
        chosen = set(whole.index[whole.source.ne("weak_validation_proxy")])
        rng = np.random.RandomState(20261019)
        for _, group in whole[whole.source.eq("weak_validation_proxy")].groupby(["role", "aspect"], dropna=False):
            chosen.update(rng.choice(group.index, min(24, len(group)), replace=False).tolist())
        chosen.update(whole.index[whole.subjectivity.eq("objective")])
        indices = np.asarray(sorted(chosen)); frame = whole.iloc[indices].reset_index(drop=True)
        assert not set(frame[frame.role.eq("calibrate")].review_id) & set(frame[frame.role.eq("monitor")].review_id)
        frame.to_json(out / "selection_data.jsonl", lines=True, orient="records", force_ascii=False)
        inputs = {str(source): sha(source), str(out / "selection_data.jsonl"): sha(out / "selection_data.jsonl")}
        write_json(out / "prompts.json", PROMPTS)
        components = {}; results = []; specs = {}
        def assess(pred):
            return {role: combined(frame.loc[frame.role.eq(role)], pred.loc[frame.role.eq(role)]) for role in ["calibrate", "monitor"]}
        def record(name, spec, pred):
            report = {"name": name, "spec": spec, **assess(pred), "proxy_only": True}
            directory = out / name; directory.mkdir(exist_ok=False)
            write_json(directory / "candidate.json", spec)
            write_json(directory / "metrics.json", report)
            pred.to_json(directory / "selection_predictions.jsonl", orient="records", lines=True, force_ascii=False)
            components[name] = pred; specs[name] = spec; results.append(report)
            monitor.state["rounds"].append(report); monitor.flush()
            print(json.dumps({"candidate": name, "proxy_monitor": report["monitor"]["proxy_selection_score"]}), flush=True)
        # Existing components were fitted/selected without gold in the preceding run.
        old_components = {}
        for directory in sorted(PRIOR.glob("r*")):
            file = directory / "selection_predictions.jsonl"
            if not file.exists(): continue
            stored = pd.read_json(file, lines=True)
            assert stored.reference_text.tolist() == whole.text.tolist()
            pred = stored[["pred_aspect", "pred_subjectivity", "pred_polarity", "pred_label"]].iloc[indices].reset_index(drop=True)
            pred.columns = [c.removeprefix("pred_") for c in pred.columns]
            old_components[directory.name] = {"prediction": pred, "metrics": assess(pred)}
            inputs[str(file)] = sha(file)
            inputs[str(directory / "model.joblib")] = sha(directory / "model.joblib")
        weak_names = [n for n in old_components if n.startswith(("r0_", "r19_", "r20_", "r21_", "r22_"))]
        nli_names = [n for n in old_components if n.startswith(("r23_", "r24_", "r25_"))]
        weak = max(weak_names, key=lambda n: old_components[n]["metrics"]["calibrate"]["proxy_selection_score"])
        nli = max(nli_names, key=lambda n: old_components[n]["metrics"]["calibrate"]["proxy_selection_score"])
        for name in [weak, nli]: components[name] = old_components[name]["prediction"]
        monitor.state["component_sources"] = {"weak": weak, "nli": nli}
        monitor.state["selection_n"] = len(frame)
        record("r28_reweighted_baseline", {"mode": "direct", "source": weak, "existing_model": str(PRIOR / weak / "model.joblib")}, components[weak])
        for variant in PROMPTS:
            name = "r29_qwen_" + variant
            monitor.phase(name + ": local inference on non-gold diagnostics")
            rows = [{"id": str(i), "text": text} for i, text in enumerate(frame.text)]
            prediction = pd.DataFrame(llm_predict(rows, variant, out / (name + "_inference"),
                lambda done, total: monitor.phase(f"{name}: {done}/{total}")))
            components[variant] = prediction
            record(name, {"mode": "direct", "source": variant, "llm_variant": variant,
                          "model_id": MODEL_ID, "model_revision": MODEL_REVISION}, prediction)
        qwen = max([r for r in results if r["name"].startswith("r29_")], key=lambda r: r["calibrate"]["proxy_selection_score"])["spec"]["source"]
        other = next(v for v in PROMPTS if v != qwen)
        new = {
            "r30_weak_aspect_qwen_sentiment": {"mode": "combine", "aspect_source": weak, "sentiment_source": qwen},
            "r31_qwen_aspect_nli_sentiment": {"mode": "combine", "aspect_source": qwen, "sentiment_source": nli},
            "r32_qwen_subjective_consensus": {"mode": "subjective_consensus", "aspect_source": qwen, "sentiment_source": qwen, "other_source": other},
            "r33_qwen_clause_rules": {"mode": "clause_rules", "aspect_source": qwen, "sentiment_source": qwen},
        }
        sys.path.insert(0, str(ROOT / "outputs/c_gold_smoke_20261018/source"))
        # Explicit file loading avoids replacing the live project's src package.
        import importlib.util
        legacy = ROOT / "outputs/c_gold_smoke_20261018/source/src"
        for module in ["weak_label_rules", "language_rules"]:
            spec = importlib.util.spec_from_file_location("src." + module, legacy / (module + ".py"))
            loaded = importlib.util.module_from_spec(spec); sys.modules["src." + module] = loaded; spec.loader.exec_module(loaded)
            inputs[str(legacy / (module + ".py"))] = sha(legacy / (module + ".py"))
        rule_func = sys.modules["src.language_rules"].choose_focus
        for name, spec in new.items(): record(name, spec, combine_predictions(spec, components, frame.text, rule_func))
        champion = max(results, key=lambda r: r["monitor"]["proxy_selection_score"])
        weight_hashes = {str(p): sha(p) for p in MODEL_PATH.glob("*.safetensors")}
        code = [*HERE.glob("*.py"), ROOT / "src/quality.py"]
        frozen = {"frozen_at": datetime.now(timezone.utc).isoformat(), "selected_without_gold": champion["name"],
                  "selected_spec": champion["spec"], "candidates": specs, "component_sources": monitor.state["component_sources"],
                  "classifier_targets": CLASSIFIER_TARGETS, "query_targets": QUERY_TARGETS,
                  "source_hashes": {str(p): sha(p) for p in code}, "input_hashes": inputs,
                  "model_weight_hashes": weight_hashes, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                  "project_gold_used_for_training": False, "project_gold_used_for_selection": False,
                  "training_kind": "frozen pretrained model plus existing weak-trained components; no weight fine-tuning",
                  "proxy_selection_is_not_acceptance": True}
        assert len(denied) == 1
        write_json(out / "frozen_candidates.json", frozen)
        monitor.state.update(selected_without_gold=champion["name"], gold_guard_denials=denied,
                             gold_guard_only_self_test=True, frozen_sha256=sha(out / "frozen_candidates.json"))
        monitor.close("completed_frozen_pending_audit")
    except BaseException as error:
        monitor.state["error"] = repr(error); monitor.close("failed_or_stopped"); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(); main(args.output_dir.resolve())
