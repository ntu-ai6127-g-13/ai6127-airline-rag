from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.retriever import (
    build_bm25,
    build_faiss,
    filter_coverage,
    search_hybrid,
    search_with_filters,
)
from tests.test_retriever import FakeEmbeddingModel


class FilterTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.chunks = self.root / "chunks.parquet"
        frame = pd.DataFrame(
            [
                (
                    "c0", "Plenty of legroom in this seat.", False, False,
                    "Cathay Pacific", "Economy Class", "Solo",
                    "Singapore", "London", "2015-06-01",
                ),
                (
                    "c1", "The seat pitch gave me knee space.", False, False,
                    "Singapore Airlines", "Economy Class", None,
                    None, None, "2018-01-01",
                ),
                (
                    "c2", "The food and dinner were excellent.", False, False,
                    "Cathay Pacific", "Business Class", "Couple",
                    "Hong Kong", "Tokyo", "2012-03-01",
                ),
                (
                    "c3", "The cabin crew offered friendly service.", False, False,
                    "Singapore Airlines", "Economy Class", "Business",
                    "Singapore", "Sydney", "2019-05-01",
                ),
            ],
            columns=[
                "chunk_id", "text", "is_route_only", "is_non_english",
                "airline", "cabin", "traveller_type",
                "route_from", "route_to", "date",
            ],
        )
        # The real chunks.parquet stores `date` as an actual date type, not a
        # string (docs/数据使用指南.md §3) — match that so comparisons behave.
        frame["date"] = pd.to_datetime(frame["date"])
        frame.to_parquet(self.chunks, index=False)
        self.bm25_path = self.root / "bm25.pkl"
        self.faiss_path = self.root / "faiss.index"
        self.ids_path = self.root / "chunk_ids.npy"
        self.texts_path = self.root / "chunk_texts.npy"
        self.model = FakeEmbeddingModel()
        build_bm25(self.chunks, self.bm25_path)
        build_faiss(
            self.chunks, self.faiss_path, self.ids_path, self.texts_path,
            model=self.model, show_progress_bar=False,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _search(self, query: str, filters=None, **kwargs):
        return search_with_filters(
            query,
            filters,
            bm25_index_path=self.bm25_path,
            faiss_index_path=self.faiss_path,
            chunk_ids_path=self.ids_path,
            chunk_texts_path=self.texts_path,
            chunks_path=self.chunks,
            model=self.model,
            **kwargs,
        )

    def test_no_filters_matches_search_hybrid(self):
        filtered = self._search("seat legroom", filters=None, top_k=3)
        unfiltered = search_hybrid(
            "seat legroom", 3,
            bm25_index_path=self.bm25_path,
            faiss_index_path=self.faiss_path,
            chunk_ids_path=self.ids_path,
            chunk_texts_path=self.texts_path,
            model=self.model,
        )
        self.assertEqual(filtered, unfiltered)

    def test_exact_match_filter_narrows_to_matching_airline(self):
        results = self._search("food dinner", filters={"airline": "Cathay Pacific"})
        ids = {chunk_id for chunk_id, _ in results}
        self.assertTrue(ids <= {"c0", "c2"})
        self.assertIn("c2", ids)

    def test_list_value_filter_is_isin(self):
        results = self._search(
            "friendly service", filters={"cabin": ["Business Class"]}
        )
        self.assertEqual([chunk_id for chunk_id, _ in results], ["c2"])

    def test_date_range_filter_excludes_out_of_range_rows(self):
        results = self._search(
            "seat legroom", filters={"date": ("2013-01-01", None)}
        )
        ids = {chunk_id for chunk_id, _ in results}
        self.assertNotIn("c2", ids)

        results = self._search(
            "food dinner", filters={"date": (None, "2013-01-01")}
        )
        self.assertEqual([chunk_id for chunk_id, _ in results], ["c2"])

    def test_filter_matching_nothing_returns_empty_list(self):
        results = self._search("seat legroom", filters={"airline": "Qantas"})
        self.assertEqual(results, [])

    def test_unsupported_column_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported filter columns"):
            self._search("seat", filters={"aircraft": "A380"})

    def test_malformed_date_filter_raises(self):
        with self.assertRaisesRegex(ValueError, "date.*pair"):
            self._search("seat", filters={"date": "2015-01-01"})

    def test_rejects_non_positive_top_k_and_multiplier(self):
        with self.assertRaises(ValueError):
            self._search("seat", top_k=0)
        with self.assertRaises(ValueError):
            self._search("seat", candidate_multiplier=0)


class FilterCoverageTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.chunks = Path(self.tempdir.name) / "chunks.parquet"
        pd.DataFrame(
            [
                ("c0", "Solo", "Singapore"),
                ("c1", None, None),
                ("c2", "Couple", "Hong Kong"),
                ("c3", "Business", "Singapore"),
            ],
            columns=["chunk_id", "traveller_type", "route_from"],
        ).assign(
            text="x", is_route_only=False, is_non_english=False,
            airline="x", cabin="x", route_to="x", date="2015-01-01",
        ).to_parquet(self.chunks, index=False)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_reports_fraction_non_null_per_filtered_column(self):
        coverage = filter_coverage(
            {"traveller_type": "Solo", "route_from": "Singapore"},
            chunks_path=self.chunks,
        )
        self.assertAlmostEqual(coverage["traveller_type"], 0.75)
        self.assertAlmostEqual(coverage["route_from"], 0.75)

    def test_no_filters_returns_empty_dict(self):
        self.assertEqual(filter_coverage(None, chunks_path=self.chunks), {})
        self.assertEqual(filter_coverage({}, chunks_path=self.chunks), {})

    def test_date_column_excluded_from_coverage_keys(self):
        coverage = filter_coverage(
            {"date": ("2015-01-01", None)}, chunks_path=self.chunks
        )
        self.assertEqual(coverage, {})


if __name__ == "__main__":
    unittest.main()
