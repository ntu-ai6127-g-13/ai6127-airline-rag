from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.rating_summary import rank_airlines_by_rating, summarize_ratings


class RatingSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.reviews, self.chunks = root / "reviews.parquet", root / "chunks.parquet"
        rows = []
        chunks = []
        for airline, score in (("A Air", 4.2), ("B Air", 2.1)):
            for index in range(40):
                review_id = f"{airline}-{index}"
                rows.append({"review_id": review_id, "airline": airline, "cabin": "Economy Class",
                    "traveller_type": "Solo Leisure", "review_date": pd.Timestamp("2018-01-01"),
                    "overall": score * 2, "seat_comfort": score, "cabin_service": score,
                    "food_bev": score, "entertainment": score, "ground_service": score,
                    "value_for_money": score})
                for sentence in range(2):
                    chunks.append({"review_id": review_id, "route_from": "Singapore",
                                   "route_to": "London", "sentence": sentence})
        pd.DataFrame(rows).to_parquet(self.reviews, index=False)
        pd.DataFrame(chunks).to_parquet(self.chunks, index=False)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_review_ratings_are_not_duplicated_by_sentence_chunks(self):
        result = summarize_ratings("entertainment", airlines=["A Air", "B Air"],
            filters={"cabin": "Economy Class", "route_from": "Singapore"},
            reviews_path=self.reviews, chunks_path=self.chunks)
        self.assertEqual(result["airlines"]["A Air"]["n"], 40)
        self.assertEqual(result["airlines"]["B Air"]["n"], 40)
        self.assertEqual(result["comparison"]["status"], "clear_difference")
        self.assertEqual(result["comparison"]["winner"], "A Air")
        self.assertAlmostEqual(result["comparison"]["difference"], 2.1)

    def test_unrated_project_aspect_is_explicit(self):
        result = summarize_ratings("other", reviews_path=self.reviews, chunks_path=self.chunks)
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "no_rating_field")

    def test_open_comparison_discovers_airlines_inside_structured_slice(self):
        result = rank_airlines_by_rating(
            "seat", filters={"cabin": "Economy Class", "route_from": "Singapore"},
            reviews_path=self.reviews, chunks_path=self.chunks)
        self.assertEqual(result["airlines"], ["A Air", "B Air"])
        self.assertEqual(result["candidates"][0]["rating_count"], 40)
        self.assertEqual(result["rating_field"], "seat_comfort")


if __name__ == "__main__":
    unittest.main()
