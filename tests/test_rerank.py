from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.rerank import polarity_adjustment, rerank


class FakeCrossEncoder:
    """Scores a (query, text) pair by lowercase word-overlap count."""

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            q_words = set(query.lower().split())
            t_words = set(text.lower().replace(".", "").split())
            scores.append(float(len(q_words & t_words)))
        return scores


class RerankTest(unittest.TestCase):
    def test_polarity_adjustment_rejects_opposite_polarity(self):
        self.assertLess(polarity_adjustment("bad food", "Food was not bad."), 0)
        self.assertEqual(polarity_adjustment("bad food", "Food was bad."), 0)
        self.assertLess(polarity_adjustment("good food", "Food was terrible."), 0)
        self.assertLess(polarity_adjustment("not bad food", "Food was bad."), 0)
        self.assertEqual(polarity_adjustment("not bad food", "Food was not bad."), 0)

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.chunks = Path(self.tempdir.name) / "chunks.parquet"
        pd.DataFrame(
            [
                ("c0", "Plenty of legroom in this seat.", 1),
                ("c1", "The food and dinner were excellent.", 3),
                ("c2", "The cabin crew offered friendly service.", 1),
            ],
            columns=["chunk_id", "text", "dup_count"],
        ).to_parquet(self.chunks, index=False)
        self.model = FakeCrossEncoder()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_reorders_by_cross_encoder_score_and_attaches_dup_count(self):
        # Deliberately pass candidates in an order the cross-encoder should
        # NOT preserve, to prove rerank uses its own score, not input order.
        candidates = [("c1", 0.9), ("c2", 0.5), ("c0", 0.1)]
        results = rerank(
            "legroom seat", candidates, top_k=3,
            chunks_path=self.chunks, model=self.model,
        )
        self.assertEqual(results[0]["chunk_id"], "c0")
        self.assertEqual(results[0]["dup_count"], 1)
        self.assertEqual(results[0]["text"], "Plenty of legroom in this seat.")
        self.assertEqual(results[0]["ranking_score"], results[0]["score"])

    def test_top_k_truncates(self):
        candidates = [("c1", 0.9), ("c0", 0.1), ("c2", 0.5)]
        results = rerank(
            "legroom seat", candidates, top_k=1,
            chunks_path=self.chunks, model=self.model,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "c0")

    def test_skips_candidates_missing_from_chunks(self):
        candidates = [("missing", 0.9), ("c0", 0.1)]
        results = rerank(
            "legroom seat", candidates,
            chunks_path=self.chunks, model=self.model,
        )
        self.assertEqual([item["chunk_id"] for item in results], ["c0"])

    def test_empty_candidates_returns_empty_list(self):
        self.assertEqual(
            rerank("legroom seat", [], chunks_path=self.chunks, model=self.model),
            [],
        )

    def test_rejects_empty_query(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            rerank(" ", [("c0", 1.0)], chunks_path=self.chunks, model=self.model)


if __name__ == "__main__":
    unittest.main()
