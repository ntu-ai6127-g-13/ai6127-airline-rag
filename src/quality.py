"""Query-oriented quality targets and explicit-denominator classification metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ASPECTS = ["entertainment", "food", "ground", "other", "overall", "seat", "staff", "value"]
CLASSIFIER_TARGETS = {
    "subjective_aspect_f1": 0.85,
    "subjective_polarity_f1": 0.90,
    "subjective_joint_accuracy": 0.80,
    "subjective_recall": 0.97,
}
QUERY_TARGETS = {"precision_at_5": 0.90, "success_at_10": 0.95,
                 "supported_claim_rate": 0.95, "answer_success_rate": 0.90,
                 "correct_abstention_rate": 0.95}


def classification_metrics(reference: pd.DataFrame, prediction: pd.DataFrame) -> dict:
    """Missing labels stay missing; wrongly gated subjective rows count as errors."""
    g = reference.reset_index(drop=True)
    p = prediction.reset_index(drop=True)
    if len(g) != len(p):
        raise ValueError("Reference and prediction row counts differ")
    subjective = g.subjectivity.eq("subjective")
    aspect_known = subjective & g.aspect.isin(ASPECTS)
    polarity_known = subjective & g.polarity.isin(["negative", "positive"])
    joint_known = aspect_known & polarity_known

    def macro(column, mask, labels):
        return float(f1_score(g.loc[mask, column], p.loc[mask, column],
                              labels=labels, average="macro", zero_division=0)) if mask.any() else None

    def fraction(correct, eligible):
        return float(correct[eligible].mean()) if eligible.any() else None

    valid_subject = g.subjectivity.isin(["objective", "subjective"])
    objective = g.subjectivity.eq("objective")
    predicted_objective = p.subjectivity.eq("objective") & valid_subject
    report = {
        "n": len(g), "subjective_n": int(subjective.sum()),
        "aspect_scored_n": int(aspect_known.sum()), "polarity_scored_n": int(polarity_known.sum()),
        "joint_scored_n": int(joint_known.sum()),
        "subjective_aspect_f1": macro("aspect", aspect_known, ASPECTS),
        "subjective_polarity_f1": macro("polarity", polarity_known, ["negative", "positive"]),
        "subjective_joint_accuracy": fraction(g.aspect.eq(p.aspect) & g.polarity.eq(p.polarity), joint_known),
        "subjective_recall": fraction(p.subjectivity.eq("subjective"), subjective),
        "subjective_false_objective_n": int((subjective & p.subjectivity.eq("objective")).sum()),
        "objective_n": int(objective.sum()),
        "objective_precision": fraction(objective, predicted_objective),
        "objective_recall": fraction(predicted_objective, objective),
        "subjective_aspect_per_class": {},
    }
    for label in ASPECTS:
        report["subjective_aspect_per_class"][label] = {
            "support": int((aspect_known & g.aspect.eq(label)).sum()),
            "f1": macro("aspect", aspect_known, [label]),
        }
    report["target_failures"] = {
        key: {"actual": report[key], "target": goal}
        for key, goal in CLASSIFIER_TARGETS.items()
        if report[key] is None or report[key] < goal
    }
    report["point_targets_passed"] = not report["target_failures"]
    return report


def proxy_rank(metrics: dict) -> float:
    """Development ranking only. Proxy scores cannot authorize deployment."""
    weights = {"subjective_aspect_f1": .35, "subjective_polarity_f1": .20,
               "subjective_joint_accuracy": .35, "subjective_recall": .10}
    score = sum(weight * (metrics[key] or 0.) for key, weight in weights.items())
    score -= 2 * max(0., .97 - (metrics["subjective_recall"] or 0.))
    # An all-subjective predictor must not get a free pass on neutral evidence.
    if metrics["objective_n"]:
        score -= .05 * max(0., .60 - (metrics["objective_recall"] or 0.))
    return float(score)
