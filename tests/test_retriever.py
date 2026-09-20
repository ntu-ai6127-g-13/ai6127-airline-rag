from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.retriever import (
    build_bm25,
    build_faiss,
    search_bm25,
    search_dense,
)


class FakeEmbeddingModel:
    """Small deterministic encoder that avoids downloading a model in tests."""

    def encode(
        self,
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    ):
        del batch_size, show_progress_bar, normalize_embeddings
        vectors = []
        for text in texts:
            words = set(text.lower().replace(".", "").split())
            vectors.append(
                [
                    float(bool(words & {"seat", "legroom", "pitch", "space"})),
                    float(bool(words & {"food", "meal", "dinner"})),
                    float(bool(words & {"crew", "staff", "service"})),
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


class RetrieverTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.chunks = self.root / "chunks.parquet"
        pd.DataFrame(
            [
                ("c0", "Plenty of legroom in this seat.", False, False, True),
                ("c1", "The seat pitch gave me knee space.", False, False, False),
                ("c2", "The food and dinner were excellent.", False, False, False),
                ("c3", "The cabin crew offered friendly service.", False, False, False),
                ("c4", "Plenty of legroom in this seat.", False, False, False),
                ("c5", "Singapore to London.", True, False, False),
                ("c6", "Le service etait mauvais.", False, True, False),
            ],
            columns=[
                "chunk_id",
                "text",
                "is_route_only",
                "is_non_english",
                "is_eval",
            ],
        ).to_parquet(self.chunks, index=False)
        self.bm25_path = self.root / "bm25.pkl"
        self.faiss_path = self.root / "faiss.index"
        self.ids_path = self.root / "chunk_ids.npy"
        self.texts_path = self.root / "chunk_texts.npy"
        self.model = FakeEmbeddingModel()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_bm25_build_filters_noise_and_searches(self):
        stats = build_bm25(self.chunks, self.bm25_path)
        self.assertEqual(stats, {"indexed": 5, "excluded": 2})

        results = search_bm25("legroom seat", 5, index_path=self.bm25_path)
        # Gold-standard rows remain in the retrieval corpus; no labels are used.
        self.assertEqual(results[0][0], "c0")
        self.assertEqual([chunk_id for chunk_id, _ in results].count("c4"), 0)
        self.assertTrue(all(score > 0 for _, score in results))

    def test_bm25_handles_punctuation_and_unknown_terms(self):
        build_bm25(self.chunks, self.bm25_path)
        results = search_bm25("Dinner!", 3, index_path=self.bm25_path)
        self.assertEqual(results[0][0], "c2")
        self.assertEqual(
            search_bm25("zzzxxyy", 3, index_path=self.bm25_path), []
        )

    def test_dense_build_filters_noise_normalizes_and_deduplicates(self):
        stats = build_faiss(
            self.chunks,
            self.faiss_path,
            self.ids_path,
            self.texts_path,
            model=self.model,
            show_progress_bar=False,
        )
        self.assertEqual(stats, {"indexed": 5, "excluded": 2, "dimensions": 3})

        results = search_dense(
            "seat space",
            5,
            index_path=self.faiss_path,
            chunk_ids_path=self.ids_path,
            chunk_texts_path=self.texts_path,
            model=self.model,
        )
        ids = [chunk_id for chunk_id, _ in results]
        self.assertIn(ids[0], {"c0", "c1", "c4"})
        self.assertEqual(len({"c0", "c4"}.intersection(ids)), 1)
        self.assertAlmostEqual(results[0][1], 1.0, places=5)

    def test_dense_without_deduplication_keeps_duplicate_evidence(self):
        build_faiss(
            self.chunks,
            self.faiss_path,
            self.ids_path,
            self.texts_path,
            model=self.model,
            show_progress_bar=False,
        )
        results = search_dense(
            "legroom seat",
            5,
            index_path=self.faiss_path,
            chunk_ids_path=self.ids_path,
            chunk_texts_path=self.texts_path,
            model=self.model,
            deduplicate=False,
        )
        ids = [chunk_id for chunk_id, _ in results]
        self.assertIn("c0", ids)
        self.assertIn("c4", ids)

    def test_invalid_query_and_missing_index_have_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            search_bm25(" ", index_path=self.bm25_path)
        with self.assertRaisesRegex(FileNotFoundError, "build-bm25"):
            search_bm25("seat", index_path=self.bm25_path)
        with self.assertRaisesRegex(FileNotFoundError, "build-dense"):
            search_dense(
                "seat",
                index_path=self.faiss_path,
                chunk_ids_path=self.ids_path,
                chunk_texts_path=self.texts_path,
                model=self.model,
            )


if __name__ == "__main__":
    unittest.main()
