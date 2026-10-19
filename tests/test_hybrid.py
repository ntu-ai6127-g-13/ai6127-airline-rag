from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.retriever import build_bm25, build_faiss, rrf_fuse, search_hybrid
from tests.test_retriever import FakeEmbeddingModel


class RrfFuseTest(unittest.TestCase):
    def test_overlapping_candidate_outranks_single_list_candidate(self):
        list_a = [("x", 5.0), ("y", 3.0), ("z", 1.0)]
        list_b = [("y", 0.9), ("w", 0.1)]
        fused = rrf_fuse(list_a, list_b, k=60, top_k=10)
        ids = [chunk_id for chunk_id, _ in fused]
        # "y" is in both lists (rank 2 in a, rank 1 in b) so it should beat
        # "x", which only appears in list_a despite a higher raw score.
        self.assertEqual(ids[0], "y")
        self.assertEqual(set(ids), {"x", "y", "z", "w"})

    def test_score_matches_reciprocal_rank_formula(self):
        fused = rrf_fuse([("x", 1.0)], [("x", 1.0)], k=60, top_k=1)
        self.assertEqual(fused, [("x", 2 / 61)])

    def test_top_k_truncates(self):
        list_a = [("x", 3.0), ("y", 2.0), ("z", 1.0)]
        fused = rrf_fuse(list_a, [], top_k=2)
        self.assertEqual([chunk_id for chunk_id, _ in fused], ["x", "y"])

    def test_ties_break_by_chunk_id(self):
        # "a" and "b" each rank 1st in their own list, so raw RRF scores tie.
        fused = rrf_fuse([("b", 5.0)], [("a", 5.0)], top_k=2)
        self.assertEqual([chunk_id for chunk_id, _ in fused], ["a", "b"])

    def test_rejects_non_positive_top_k_and_k(self):
        with self.assertRaises(ValueError):
            rrf_fuse([], [], top_k=0)
        with self.assertRaises(ValueError):
            rrf_fuse([], [], k=0)


class SearchHybridTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.chunks = self.root / "chunks.parquet"
        pd.DataFrame(
            [
                ("c0", "Plenty of legroom in this seat.", False, False),
                ("c1", "The seat pitch gave me knee space.", False, False),
                ("c2", "The food and dinner were excellent.", False, False),
                ("c3", "The cabin crew offered friendly service.", False, False),
            ],
            columns=["chunk_id", "text", "is_route_only", "is_non_english"],
        ).to_parquet(self.chunks, index=False)
        self.bm25_path = self.root / "bm25.pkl"
        self.faiss_path = self.root / "faiss.index"
        self.ids_path = self.root / "chunk_ids.npy"
        self.texts_path = self.root / "chunk_texts.npy"
        self.model = FakeEmbeddingModel()
        build_bm25(self.chunks, self.bm25_path)
        build_faiss(
            self.chunks,
            self.faiss_path,
            self.ids_path,
            self.texts_path,
            model=self.model,
            show_progress_bar=False,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _search(self, query: str, **kwargs):
        return search_hybrid(
            query,
            candidate_k=10,
            bm25_index_path=self.bm25_path,
            faiss_index_path=self.faiss_path,
            chunk_ids_path=self.ids_path,
            chunk_texts_path=self.texts_path,
            model=self.model,
            **kwargs,
        )

    def test_fuses_bm25_and_dense_candidates(self):
        results = self._search("seat legroom", top_k=3)
        ids = [chunk_id for chunk_id, _ in results]
        self.assertEqual(ids[0], "c0")

    def test_survives_a_query_bm25_cannot_match(self):
        # FakeEmbeddingModel treats "space" as a seat-aspect signal but BM25
        # needs an exact token match on "space", which no sentence contains
        # literally; dense-only coverage must still reach the fused output.
        results = self._search("roomy space", top_k=3)
        self.assertTrue(results)


if __name__ == "__main__":
    unittest.main()
