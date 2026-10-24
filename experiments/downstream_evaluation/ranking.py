"""C10: isolated text prediction, then heldout-rating external validity scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.downstream_evaluation.common import frozen, new_dir, now, sha, write
from src.weak_labels import load_test_exclusions, normalize_text

TEXT_COLUMNS = ["chunk_id", "review_id", "text", "airline", "cabin"]


def heldout(review_id, seed, fraction):
    value = int(hashlib.sha256(f"{seed}:{review_id}".encode()).hexdigest()[:8], 16)
    return value / 2**32 < fraction


def split_material(kb, gold_parents, gold_texts, seed, fraction):
    ids = sorted(set(kb.review_id) - set(gold_parents))
    test_ids = {rid for rid in ids if heldout(rid, seed, fraction)}
    normalized = kb.text.map(normalize_text)
    test_norms = set(normalized[kb.review_id.isin(test_ids)])
    train_pool = kb.loc[~kb.review_id.isin(test_ids | set(gold_parents))
                        & ~normalized.isin(test_norms | set(gold_texts))].copy()
    quality = ~kb[["is_route_only", "is_non_english", "is_fragment"]].any(axis=1)
    test = kb.loc[kb.review_id.isin(test_ids) & quality & kb.cabin.eq("Economy Class"), TEXT_COLUMNS].copy()
    if set(train_pool.review_id) & test_ids:
        raise ValueError("Heldout parents entered training pool")
    if set(train_pool.text.map(normalize_text)) & test_norms:
        raise ValueError("Heldout normalized text entered training pool")
    return train_pool, test, test_ids, test_norms


def predict_heldout(model_path, input_path, output_path):
    """The prediction function never opens reference ratings or the full KB."""
    if set(pq.read_schema(input_path).names) != set(TEXT_COLUMNS):
        raise ValueError("Prediction input must contain only the text/ID/metadata whitelist")
    from src.classification_pipeline import predict_texts
    model = joblib.load(model_path)
    frame = pd.read_parquet(input_path, columns=TEXT_COLUMNS)
    if set(frame.review_id) & (set(model["train_review_ids"]) | set(model["dev_review_ids"])):
        raise ValueError("Prediction material overlaps fitting parents")
    if set(frame.text.map(normalize_text)) & set(model["fit_normalized_texts"]):
        raise ValueError("Prediction material overlaps fitting normalized texts")
    parts = []
    for start in range(0, len(frame), 8192):
        batch = frame.iloc[start:start + 8192].reset_index(drop=True)
        predicted = predict_texts(model, batch.text.tolist())
        parts.append(pd.concat([batch, predicted.drop(columns="text")], axis=1))
        print(f"C10 text prediction {min(start+8192, len(frame))}/{len(frame)}", flush=True)
    output = pd.concat(parts, ignore_index=True)
    output.to_parquet(output_path, index=False)
    return output


def review_predictions(predicted):
    selected = predicted.loc[predicted.aspect.eq("seat") & predicted.subjectivity.eq("subjective")].copy()
    selected["positive"] = selected.polarity.eq("positive").astype(float)
    return selected.groupby("review_id").agg(text_score=("positive", "mean"),
                                             seat_sentence_count=("positive", "size")).reset_index()


def rank_and_bootstrap(paired, settings, seed):
    table = paired.groupby("airline").agg(
        reference_count=("seat_comfort", "count"), predicted_count=("text_score", "count"),
        reference_mean=("seat_comfort", "mean"), predicted_mean=("text_score", "mean"))
    table["included"] = ((table.reference_count >= settings["min_reference_reviews"])
                         & (table.predicted_count >= settings["min_predicted_reviews"]))
    eligible = table.loc[table.included].copy()
    if len(eligible) < 3:
        raise ValueError("Too few eligible airlines for a ranking comparison")
    if eligible.predicted_mean.nunique() < 2 or eligible.reference_mean.nunique() < 2:
        raise ValueError("Constant airline scores have undefined Spearman")
    eligible["predicted_rank"] = eligible.predicted_mean.rank(ascending=False, method="average")
    eligible["reference_rank"] = eligible.reference_mean.rank(ascending=False, method="average")
    rho, pvalue = spearmanr(eligible.predicted_mean, eligible.reference_mean)
    groups = [paired.loc[paired.airline.eq(name), ["seat_comfort", "text_score"]].to_numpy(float)
              for name in eligible.index]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(2000):
        means = []
        for group in groups:
            resample = group[rng.integers(0, len(group), len(group))]
            if np.isnan(resample[:, 1]).all():
                break
            means.append(np.nanmean(resample, axis=0))
        if len(means) == len(groups):
            means = np.asarray(means)
            if np.ptp(means[:, 0]) and np.ptp(means[:, 1]):
                values.append(float(spearmanr(means[:, 0], means[:, 1]).statistic))
    result = {"airlines_included": len(eligible), "spearman": float(rho), "pvalue_descriptive": float(pvalue),
              "ci95": np.quantile(values, [.025, .975]).tolist() if values else None,
              "bootstrap_valid": len(values), "bootstrap_total": 2000,
              "bootstrap_scope": "parent-review uncertainty within fixed included airlines, not model-training uncertainty",
              "prediction_coverage": int(paired.text_score.notna().sum()), "reference_reviews": len(paired),
              "predicted_ties": int(eligible.predicted_mean.duplicated(keep=False).sum()),
              "reference_ties": int(eligible.reference_mean.duplicated(keep=False).sum())}
    return table.reset_index(), eligible.sort_values("predicted_rank").reset_index(), result


def run(protocol_path, dest):
    protocol = frozen(protocol_path)
    settings = protocol["ranking"]
    dest = new_dir(dest)
    kb = pd.read_parquet(ROOT / "data/processed/chunks.parquet")
    gold_parents, gold_texts, gold_info = load_test_exclusions(
        kb, ROOT / "data/eval/to_label_meta.csv", ROOT / "data/eval/to_label.csv")
    pool, test, test_ids, test_norms = split_material(
        kb, gold_parents, gold_texts, protocol["seed"], settings["test_fraction"])
    write(dest / "heldout_parents.json", sorted(test_ids))
    test.to_parquet(dest / "test_text.parquet", index=False)
    from src.classification_pipeline import build_labels, split_labels, fit_models
    print("Building training-side weak labels with C甲 rules...", flush=True)
    labels, label_report = build_labels(pool, gold_parents | test_ids, gold_texts | test_norms)
    train, dev = split_labels(labels, seed=protocol["seed"])
    fit_norms = set(pd.concat([train.text, dev.text]).map(normalize_text))
    if fit_norms & (test_norms | gold_texts):
        raise ValueError("Text overlap before fitting")
    model, dev_metrics = fit_models(train, dev, max_features=settings["max_features"],
                                    max_iter=settings["max_iter"], seed=protocol["seed"])
    model.update(train_review_ids=sorted(set(train.review_id)), dev_review_ids=sorted(set(dev.review_id)),
                 fit_normalized_texts=sorted(fit_norms))
    joblib.dump(model, dest / "model.joblib")
    labels.to_parquet(dest / "weak_labels.parquet", index=False)
    pd.concat([train.assign(split="train"), dev.assign(split="dev")])[
        ["chunk_id", "review_id", "split"]].to_parquet(dest / "fit_split.parquet", index=False)
    prediction_path = dest / "test_predictions.parquet"
    predict_heldout(dest / "model.joblib", dest / "test_text.parquet", prediction_path)
    # Commit the prediction hash before the scoring stage can read target ratings.
    write(dest / "predictions_frozen.json", {"created_at": now(), "sha256": sha(prediction_path),
                                            "model_sha256": sha(dest / "model.joblib"),
                                            "input_sha256": sha(dest / "test_text.parquet")})
    del kb, pool, train, dev, labels, model
    reference = pd.read_parquet(ROOT / "data/processed/reviews.parquet",
                                columns=["review_id", "airline", "cabin", "seat_comfort"])
    reference = reference.loc[reference.review_id.isin(test_ids) & reference.cabin.eq("Economy Class")].copy()
    reference["seat_comfort"] = pd.to_numeric(reference.seat_comfort, errors="coerce")
    reference = reference.loc[reference.seat_comfort.between(1, 5)]
    predicted = pd.read_parquet(prediction_path)
    paired = reference.merge(review_predictions(predicted), on="review_id", how="left", validate="one_to_one")
    paired.to_parquet(dest / "scoring_only_reference_and_predictions.parquet", index=False)
    coverage, ranking, scores = rank_and_bootstrap(paired, settings, protocol["seed"])
    coverage.to_csv(dest / "all_airline_coverage.csv", index=False)
    ranking.to_csv(dest / "airline_rankings.csv", index=False)
    write(dest / "report.json", {"created_at": now(), "protocol_sha256": sha(protocol_path),
                                  "scope": settings["scope"], "gold_exclusions": gold_info,
                                  "heldout_parents": len(test_ids), "label_report": label_report,
                                  "development_metrics_weak_not_human": dev_metrics,
                                  "isolation": {"parent_overlap": 0, "normalized_text_overlap": 0,
                                                "gold_labels_read_for_training": False,
                                                "prediction_input_columns": TEXT_COLUMNS,
                                                "reference_opened_after_predictions_frozen": True},
                                  "ranking": scores})
    print(json.dumps(scores, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output_dir)
