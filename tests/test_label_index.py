from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from src.label_index import labels_for_chunks, validate_labels
from src.query_service import query_reviews


class LabelIndexTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([
            {"chunk_id": "a", "review_id": "r1", "aspect": "seat", "subjectivity": "subjective", "polarity": "negative"},
            {"chunk_id": "b", "review_id": "r2", "aspect": "other", "subjectivity": "objective", "polarity": "NA"},
        ])

    def test_reader_preserves_requested_order_and_rejects_missing_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.parquet"
            self.fixture().to_parquet(path, index=False)
            self.assertEqual([row["aspect"] for row in labels_for_chunks(["b", "a"], path)], ["other", "seat"])
            with self.assertRaisesRegex(RuntimeError, "首个缺失 ID：missing"):
                labels_for_chunks(["missing"], path)

    def test_validator_rejects_invalid_objective_polarity(self):
        frame = self.fixture()
        frame.loc[1, "polarity"] = "positive"
        with self.assertRaisesRegex(ValueError, "客观句"):
            validate_labels(frame)

    def test_query_prefers_offline_labels_without_realtime_classifier(self):
        candidates = [
            {"chunk_id": "a", "review_id": "r1", "text": "Bad seat", "score": 1.0},
            {"chunk_id": "b", "review_id": "r2", "text": "Flight number", "score": 0.5},
        ]
        metadata = pd.DataFrame([
            {"chunk_id": "a", "review_id": "r1", "review_text": "Bad seat", "airline": "A", "cabin": None, "date": None, "route": None},
            {"chunk_id": "b", "review_id": "r2", "review_text": "Flight number", "airline": "A", "cabin": None, "date": None, "route": None},
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.parquet"
            self.fixture().to_parquet(path, index=False)
            offline_predictions = labels_for_chunks(["a", "b"], path)
            with patch("src.query_service.search_with_filters", return_value=[("a", 1.0), ("b", .5)]), \
                 patch("src.query_service.rerank", return_value=candidates), \
                 patch("src.query_service.pd.read_parquet", return_value=metadata), \
                 patch("src.query_service.labels_for_chunks", return_value=offline_predictions), \
                 patch("src.query_service.classify_texts") as realtime:
                result = query_reviews(aspect="seat", label_index_path=path, classifier_manifest="unused")
            realtime.assert_not_called()
            self.assertEqual(result["label_lookup_mode"], "offline_index")
            self.assertTrue(result["classifier_used"])


if __name__ == "__main__":
    unittest.main()
