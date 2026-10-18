"""Read-only validation and ID-based alignment of a versioned human gold release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.config import ROOT
from src.quality import ASPECTS
from src.weak_labels import normalize_text

DEFAULT_RELEASE = ROOT / "data/eval/gold_human_frozen_1000_20261019_v1.release.json"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_frozen_gold(release_path=DEFAULT_RELEASE):
    release_path = Path(release_path)
    release = json.loads(release_path.read_text())
    path = ROOT / release["data_file"]
    if release.get("status") != "frozen" or sha256(path) != release["gold_sha256"]:
        raise ValueError("Gold release status or checksum does not match")
    gold = pd.read_csv(path, keep_default_na=False, dtype=str)
    if len(gold) != release["record_count"] or list(gold)[:2] != ["text", "label"]:
        raise ValueError("Gold size or column order differs from the release")
    for key in ["sentence_id", "chunk_id"]:
        if not gold[key].is_unique or gold[key].eq("").any():
            raise ValueError("Gold has duplicate or empty IDs: " + key)
    if not gold.text.map(normalize_text).is_unique or gold.text.str.strip().eq("").any():
        raise ValueError("Gold has duplicate or empty texts")
    if not gold.human_review_status.eq("approved").all():
        raise ValueError("Gold contains unapproved human decisions")
    valid = gold.subjectivity.eq("objective") & gold.polarity.eq("NA")
    valid |= gold.subjectivity.eq("subjective") & gold.polarity.isin(["negative", "positive"])
    expected = gold.polarity.where(gold.subjectivity.eq("subjective"), "objective")
    if not valid.all() or not gold.aspect.isin(ASPECTS).all() or not gold.label.eq(expected).all():
        raise ValueError("Gold contains inconsistent labels")
    meta = pd.read_csv(ROOT / "data/eval/to_label_meta.csv", dtype=str, keep_default_na=False).set_index("id")
    original = pd.read_csv(ROOT / "data/eval/to_label.csv", dtype=str, keep_default_na=False).set_index("id")
    if not set(gold.sentence_id).issubset(meta.index) or not set(gold.sentence_id).issubset(original.index):
        raise ValueError("Gold introduces IDs outside the original evaluation exclusion set")
    for field, source in [("chunk_id", meta), ("review_id", meta), ("text", original)]:
        if source.loc[gold.sentence_id, field].tolist() != gold[field].tolist():
            raise ValueError("Gold mapping differs from frozen evaluation inputs: " + field)
    return gold, release


def align_cached_predictions(gold, cached):
    """Never align by row number; reject renamed, changed, or missing sentences."""
    key = "gold_sentence_id"
    if key not in cached or not cached[key].is_unique:
        raise ValueError("Prediction IDs are absent or duplicated")
    indexed = cached.set_index(key)
    if not set(gold.sentence_id).issubset(indexed.index):
        raise ValueError("Predictions are missing gold IDs")
    aligned = indexed.loc[gold.sentence_id].reset_index(drop=True)
    if aligned.gold_text.tolist() != gold.text.tolist():
        raise ValueError("Prediction text differs from the gold sentence")
    for field in ["review_id", "chunk_id"]:
        if "gold_" + field in aligned and aligned["gold_" + field].tolist() != gold[field].tolist():
            raise ValueError("Prediction metadata differs: " + field)
    fields = ["aspect", "subjectivity", "polarity", "label"]
    pred = aligned[["pred_" + field for field in fields]].copy()
    pred.columns = fields
    if pred.isna().any().any():
        raise ValueError("Prediction cache contains missing outputs")
    return pred


def check_training_overlap(gold, frame):
    return {"matching_chunk_ids": len(set(gold.chunk_id) & set(frame.chunk_id)),
            "matching_review_ids": len(set(gold.review_id) & set(frame.review_id)),
            "matching_normalized_texts": len(set(gold.text.map(normalize_text)) & set(frame.text.map(normalize_text)))}
