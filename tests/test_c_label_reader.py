"""Synthetic reader-contract tests; no model, corpus text or gold answers."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.c_label_reader import (ASPECTS, COVERAGE_COLUMNS, METADATA_COLUMNS, SKIP_REASONS,
                                VERSION, load_b_labels, sha256)


def fixture():
    rows = []
    for index, aspect in enumerate(ASPECTS):
        objective = aspect == "other"
        rows.append({"chunk_id": f"s{index}", "review_id": f"r{index}", "is_eval": False,
                     "prediction_status": "predicted", "skip_reason": pd.NA,
                     "aspect": aspect, "subjectivity": "objective" if objective else "subjective",
                     "polarity": "NA" if objective else "positive", "model_version": VERSION,
                     "source": "hybrid_prediction", "aspect_source": "model",
                     "subjectivity_source": "model",
                     "polarity_source": "objective_gate" if objective else "model",
                     "parent_split": "unlabelled_parent"})
    for identifier, reason in [("gold", "eval_sentence"), ("sibling", "eval_parent"), ("fragment", "fragment")]:
        rows.append({"chunk_id": identifier, "review_id": "r_" + identifier,
                     "is_eval": reason == "eval_sentence", "prediction_status": "skipped",
                     "skip_reason": reason, **{column: pd.NA for column in METADATA_COLUMNS}})
    frame = pd.DataFrame(rows, columns=COVERAGE_COLUMNS)
    for column in COVERAGE_COLUMNS:
        if column != "is_eval":
            frame[column] = frame[column].astype("string")
    return frame


def write_package(directory, frame):
    directory = Path(directory)
    path = directory / "b_coverage.parquet"
    frame.to_parquet(path, index=False)
    manifest = {"schema_version": "c-b-sync-v1", "model_version": VERSION, "status": "candidate",
                "artifacts_sha256": {"b_coverage.parquet": sha256(path), "labels.parquet": "not-read",
                                     "coverage.parquet": "not-read"},
                "schemas": {"b_coverage.parquet": {column: str(frame[column].dtype) for column in frame}},
                "b_pool": {"rows": len(frame),
                           "predicted_rows": int(frame.prediction_status.eq("predicted").sum()),
                           "skipped_rows": int(frame.prediction_status.eq("skipped").sum()),
                           "excluded_by_first_reason": {reason: int(frame.skip_reason.eq(reason).sum())
                                                        for reason in SKIP_REASONS},
                           "ordered_chunk_ids_sha256": hashlib.sha256(
                               ("\n".join(frame.chunk_id.tolist()) + "\n").encode()).hexdigest()}}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


class CLabelReaderTests(unittest.TestCase):
    def test_roundtrip_reads_only_b_table_and_keeps_all_states(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture()
            write_package(directory, frame)
            original_reader = pd.read_parquet
            with patch("src.c_label_reader.pd.read_parquet", wraps=original_reader) as read:
                restored = load_b_labels(directory)
            self.assertEqual(read.call_count, 1)
            self.assertEqual(Path(read.call_args.args[0]).name, "b_coverage.parquet")
            pd.testing.assert_frame_equal(restored, frame)
            self.assertEqual(restored.loc[restored.is_eval, "skip_reason"].tolist(), ["eval_sentence"])
            self.assertTrue(restored.loc[restored.prediction_status.eq("skipped"), METADATA_COLUMNS].isna().all().all())

    def test_b_pool_reorders_by_id_with_extra_columns_and_custom_index(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture()
            write_package(directory, frame)
            pool = frame[["chunk_id", "review_id", "is_eval"]].iloc[::-1].copy()
            pool.index = range(100, 100 + len(pool))
            pool["unused_text"] = "ignored synthetic field"
            result = load_b_labels(directory, pool)
            pd.testing.assert_frame_equal(result, frame.iloc[::-1].reset_index(drop=True))

    def test_different_b_pool_missing_extra_duplicate_and_wrong_parent_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture()
            write_package(directory, frame)
            pool = frame[["chunk_id", "review_id"]].copy()
            extra = pool.copy()
            extra.loc[0, "chunk_id"] = "new_b_sentence"
            duplicate = pd.concat([pool, pool.iloc[:1]], ignore_index=True)
            wrong_parent = pool.copy()
            wrong_parent.loc[0, "review_id"] = "different_parent"
            for invalid in [pool.iloc[:-1], extra, duplicate, wrong_parent]:
                with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                    load_b_labels(directory, invalid)

    def test_optional_b_eval_flag_is_checked_by_id(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture()
            write_package(directory, frame)
            pool = frame[["chunk_id", "review_id", "is_eval"]].copy()
            pool.loc[0, "is_eval"] = True
            with self.assertRaisesRegex(ValueError, "is_eval"):
                load_b_labels(directory, pool)
            pool["is_eval"] = pool.is_eval.astype("string")
            with self.assertRaisesRegex(ValueError, "布尔"):
                load_b_labels(directory, pool)

    def test_bad_b_pool_schema_and_empty_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture()
            write_package(directory, frame)
            empty_id = frame[["chunk_id", "review_id"]].copy()
            empty_id.loc[0, "chunk_id"] = " "
            duplicate_columns = pd.concat([empty_id, empty_id.chunk_id], axis=1)
            for invalid in [None, [], frame[["chunk_id"]], empty_id, duplicate_columns]:
                if invalid is None:
                    continue
                with self.subTest(kind=type(invalid).__name__), self.assertRaises(ValueError):
                    load_b_labels(directory, invalid)

    def test_manifest_contract_hash_schema_counts_and_order_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture()
            for field, value in [("schema_version", "c-b-handoff-v1"), ("model_version", "v3"),
                                 ("status", "accepted"), ("artifacts_sha256", {}),
                                 ("schemas", {}), ("b_pool", {})]:
                manifest = write_package(directory, frame)
                manifest[field] = value
                Path(directory, "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_b_labels(directory)
            for field, value in [("rows", len(frame) - 1), ("predicted_rows", True),
                                 ("skipped_rows", 0), ("ordered_chunk_ids_sha256", "wrong")]:
                manifest = write_package(directory, frame)
                manifest["b_pool"][field] = value
                Path(directory, "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_b_labels(directory)
            manifest = write_package(directory, frame)
            manifest["b_pool"]["excluded_by_first_reason"]["eval_parent"] = 0
            Path(directory, "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "跳过计数"):
                load_b_labels(directory)

    def test_changed_artifact_is_rejected_before_parquet_read(self):
        with tempfile.TemporaryDirectory() as directory:
            write_package(directory, fixture())
            with Path(directory, "b_coverage.parquet").open("ab") as handle:
                handle.write(b"changed")
            with patch("src.c_label_reader.pd.read_parquet") as read:
                with self.assertRaisesRegex(ValueError, "哈希"):
                    load_b_labels(directory)
                read.assert_not_called()

    def test_mid_read_package_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture()
            manifest = write_package(directory, frame)
            original_reader = pd.read_parquet
            def changed_read(path):
                result = original_reader(path)
                manifest["changed_during_read"] = True
                Path(directory, "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                return result
            with patch("src.c_label_reader.pd.read_parquet", side_effect=changed_read):
                with self.assertRaisesRegex(ValueError, "检查期间"):
                    load_b_labels(directory)

    def test_coverage_columns_duplicate_ids_and_boolean_flags_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture()
            wrong_flag = frame.copy()
            wrong_flag["is_eval"] = wrong_flag.is_eval.astype("string")
            duplicate = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
            for invalid in [frame.drop(columns="aspect_source"), frame[COVERAGE_COLUMNS[::-1]],
                            wrong_flag, duplicate]:
                write_package(directory, invalid)
                with self.subTest(columns=list(invalid.columns)), self.assertRaises(ValueError):
                    load_b_labels(directory)

    def test_invalid_state_skip_reason_and_placeholder_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, column, value in [(0, "prediction_status", "unknown"), (0, "skip_reason", "fragment"),
                                         (8, "skip_reason", pd.NA), (8, "skip_reason", "unknown"),
                                         (8, "aspect", "other"), (8, "model_version", VERSION),
                                         (0, "aspect_source", " ")]:
                frame = fixture()
                frame.loc[index, column] = value
                write_package(directory, frame)
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    load_b_labels(directory)

    def test_eval_state_and_reason_must_agree_in_both_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, value in [(0, True), (8, False), (9, True)]:
                frame = fixture()
                frame.loc[index, "is_eval"] = value
                write_package(directory, frame)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    load_b_labels(directory)

    def test_model_version_source_and_label_enums_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            for column, value in [("model_version", "eight-aspect-subjectivity-polarity-v3"),
                                  ("source", "weak"), ("aspect", "missing"),
                                  ("subjectivity", "unclear"), ("parent_split", "test_parent")]:
                frame = fixture()
                frame.loc[0, column] = value
                write_package(directory, frame)
                with self.subTest(column=column), self.assertRaises(ValueError):
                    load_b_labels(directory)

    def test_objective_NA_and_subjective_binary_polarity_are_distinct_from_null(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, value in [(7, "negative"), (7, "na"), (7, pd.NA), (0, "NA")]:
                frame = fixture()
                frame.loc[index, "polarity"] = value
                write_package(directory, frame)
                with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                    load_b_labels(directory)


if __name__ == "__main__":
    unittest.main()
