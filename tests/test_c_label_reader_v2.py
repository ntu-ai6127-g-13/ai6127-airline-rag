"""Synthetic v2 reader tests; no project predictions, models or gold answers."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.c_label_reader_v2 import (ASPECTS, COVERAGE_COLUMNS, HELDOUT_PARENT_ROLE,
                                   INFERENCE_ORIGINS, METADATA_COLUMNS, SCHEMA_VERSION,
                                   SKIP_REASONS, VERSION, load_b_labels, sha256)


def fixture():
    rows = []
    for index, aspect in enumerate(ASPECTS):
        heldout = index in [2, 7]
        objective = index == 7
        rows.append({"chunk_id": f"s{index}", "review_id": f"r{index}", "is_eval": False,
                     "is_eval_parent": heldout, "prediction_status": "predicted", "skip_reason": pd.NA,
                     "aspect": aspect, "subjectivity": "objective" if objective else "subjective",
                     "polarity": "NA" if objective else "positive", "model_version": VERSION,
                     "source": "hybrid_prediction", "aspect_source": "model",
                     "subjectivity_source": "model", "polarity_source": "objective_gate" if objective else "model",
                     "parent_split": HELDOUT_PARENT_ROLE if heldout else "training_parent",
                     "prediction_origin": "supplemental_eval_parent" if heldout else "existing_v4_1"})
    for identifier, reason, parent in [("gold", "eval_sentence", True), ("fragment", "fragment", False),
                                        ("heldout_fragment", "fragment", True)]:
        rows.append({"chunk_id": identifier, "review_id": "r_" + identifier,
                     "is_eval": reason == "eval_sentence", "is_eval_parent": parent,
                     "prediction_status": "skipped", "skip_reason": reason,
                     **{column: pd.NA for column in METADATA_COLUMNS}})
    frame = pd.DataFrame(rows, columns=COVERAGE_COLUMNS)
    for column in COVERAGE_COLUMNS:
        if column not in ["is_eval", "is_eval_parent"]:
            frame[column] = frame[column].astype("string")
    return frame


def write_package(directory, frame):
    path = Path(directory) / "b_coverage.parquet"
    frame.to_parquet(path, index=False)
    manifest = {"schema_version": SCHEMA_VERSION, "model_version": VERSION, "status": "candidate",
                "artifacts_sha256": {"b_coverage.parquet": sha256(path), "labels.parquet": "not-read",
                                     "coverage.parquet": "not-read"},
                "schemas": {"b_coverage.parquet": {column: str(frame[column].dtype) for column in frame}},
                "b_pool": {"rows": len(frame),
                           "predicted_rows": int(frame.prediction_status.eq("predicted").sum()),
                           "skipped_rows": int(frame.prediction_status.eq("skipped").sum()),
                           "excluded_by_first_reason": {reason: int(frame.skip_reason.eq(reason).sum())
                                                        for reason in SKIP_REASONS},
                           "ordered_chunk_ids_sha256": hashlib.sha256(
                               ("\n".join(frame.chunk_id.tolist()) + "\n").encode()).hexdigest()},
                "inference_origin_counts": {origin: int(frame["prediction_origin"].eq(origin).sum())
                                            if "prediction_origin" in frame else 0
                                            for origin in INFERENCE_ORIGINS}}
    Path(directory, "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


class CLabelReaderV2Tests(unittest.TestCase):
    def test_roundtrip_reads_only_b_coverage_with_both_origins_and_empty_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture()
            write_package(directory, frame)
            original_reader = pd.read_parquet
            with patch("src.c_label_reader_v2.pd.read_parquet", wraps=original_reader) as read:
                restored = load_b_labels(directory)
            self.assertEqual(read.call_count, 1)
            self.assertEqual(Path(read.call_args.args[0]).name, "b_coverage.parquet")
            pd.testing.assert_frame_equal(restored, frame)
            self.assertEqual(restored.prediction_origin.value_counts().to_dict(),
                             {"existing_v4_1": 6, "supplemental_eval_parent": 2})
            self.assertTrue(restored.loc[restored.prediction_status.eq("skipped"), METADATA_COLUMNS].isna().all().all())

    def test_b_pool_order_optional_flags_and_extra_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture()
            write_package(directory, frame)
            pool = frame[["chunk_id", "review_id", "is_eval", "is_eval_parent"]].iloc[::-1].copy()
            pool.index = range(100, 100 + len(pool))
            pool["unused_text"] = "ignored synthetic field"
            result = load_b_labels(directory, pool)
            pd.testing.assert_frame_equal(result, frame.iloc[::-1].reset_index(drop=True))
            no_flags = load_b_labels(directory, pool[["chunk_id", "review_id"]])
            pd.testing.assert_frame_equal(no_flags, result)

    def test_missing_extra_duplicate_or_wrong_parent_b_pool_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture()
            write_package(directory, frame)
            pool = frame[["chunk_id", "review_id"]].copy()
            extra = pool.copy()
            extra.loc[0, "chunk_id"] = "new_b_sentence"
            wrong_parent = pool.copy()
            wrong_parent.loc[0, "review_id"] = "different_parent"
            duplicate = pd.concat([pool, pool.iloc[:1]], ignore_index=True)
            for invalid in [pool.iloc[:-1], extra, wrong_parent, duplicate]:
                with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                    load_b_labels(directory, invalid)

    def test_optional_pool_flags_are_matched_by_id_and_must_be_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture()
            write_package(directory, frame)
            for column in ["is_eval", "is_eval_parent"]:
                pool = frame[["chunk_id", "review_id", column]].copy()
                pool.loc[0, column] = not bool(pool.loc[0, column])
                with self.subTest(column=column), self.assertRaisesRegex(ValueError, column):
                    load_b_labels(directory, pool)
                pool[column] = pool[column].astype("string")
                with self.subTest(column=column), self.assertRaisesRegex(ValueError, "布尔"):
                    load_b_labels(directory, pool)

    def test_bad_b_pool_schema_empty_ids_and_repeated_columns_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture()
            write_package(directory, frame)
            empty = frame[["chunk_id", "review_id"]].copy()
            empty.loc[0, "chunk_id"] = " "
            duplicate_columns = pd.concat([empty, empty.chunk_id], axis=1)
            for invalid in [[], frame[["chunk_id"]], empty, duplicate_columns]:
                with self.subTest(kind=type(invalid).__name__), self.assertRaises(ValueError):
                    load_b_labels(directory, invalid)

    def test_old_schema_other_model_and_non_candidate_packages_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for field, value in [("schema_version", "c-b-sync-v1"), ("model_version", "v3"),
                                 ("status", "accepted"), ("artifacts_sha256", {}), ("schemas", {})]:
                manifest = write_package(directory, fixture())
                manifest[field] = value
                Path(directory, "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_b_labels(directory)

    def test_counts_origin_keys_types_and_order_hash_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture()
            for field, value in [("rows", len(frame) - 1), ("predicted_rows", True),
                                 ("skipped_rows", 0), ("ordered_chunk_ids_sha256", "wrong"),
                                 ("excluded_by_first_reason", {})]:
                manifest = write_package(directory, frame)
                manifest["b_pool"][field] = value
                Path(directory, "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_b_labels(directory)
            for origins in [{}, {"existing_v4_1": 6, "supplemental_eval_parent": 1},
                            {"existing_v4_1": 6, "supplemental_eval_parent": True},
                            {"existing_v4_1": 6, "supplemental_eval_parent": 2, "other": 0}]:
                manifest = write_package(directory, frame)
                manifest["inference_origin_counts"] = origins
                Path(directory, "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(origins=origins), self.assertRaises(ValueError):
                    load_b_labels(directory)

    def test_changed_artifact_is_rejected_before_parquet_read(self):
        with tempfile.TemporaryDirectory() as directory:
            write_package(directory, fixture())
            with Path(directory, "b_coverage.parquet").open("ab") as handle:
                handle.write(b"changed")
            with patch("src.c_label_reader_v2.pd.read_parquet") as read:
                with self.assertRaisesRegex(ValueError, "哈希"):
                    load_b_labels(directory)
                read.assert_not_called()

    def test_package_changes_during_read_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture()
            manifest = write_package(directory, frame)
            original_reader = pd.read_parquet
            def changed_read(path):
                result = original_reader(path)
                manifest["changed_during_read"] = True
                Path(directory, "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                return result
            with patch("src.c_label_reader_v2.pd.read_parquet", side_effect=changed_read):
                with self.assertRaisesRegex(ValueError, "检查期间"):
                    load_b_labels(directory)

    def test_coverage_order_duplicate_ids_and_both_boolean_columns_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = fixture()
            invalid_frames = [frame.drop(columns="prediction_origin"), frame[COVERAGE_COLUMNS[::-1]],
                              pd.concat([frame, frame.iloc[:1]], ignore_index=True)]
            for column in ["is_eval", "is_eval_parent"]:
                wrong = frame.copy()
                wrong[column] = wrong[column].astype("string")
                invalid_frames.append(wrong)
                missing = frame.copy()
                missing[column] = missing[column].astype("boolean")
                missing.loc[0, column] = pd.NA
                invalid_frames.append(missing)
            for invalid in invalid_frames:
                write_package(directory, invalid)
                with self.subTest(columns=list(invalid.columns)), self.assertRaises(ValueError):
                    load_b_labels(directory)

    def test_gold_is_never_predicted_and_eval_reason_is_bidirectional(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, column, value in [(2, "is_eval", True), (8, "is_eval", False),
                                         (9, "is_eval", True), (8, "is_eval_parent", False)]:
                frame = fixture()
                frame.loc[index, column] = value
                write_package(directory, frame)
                with self.subTest(index=index, column=column), self.assertRaises(ValueError):
                    load_b_labels(directory)

    def test_supplemental_parent_origin_and_heldout_role_are_inseparable(self):
        with tempfile.TemporaryDirectory() as directory:
            for column, value in [("is_eval_parent", False), ("prediction_origin", "existing_v4_1"),
                                  ("prediction_origin", "unknown"), ("parent_split", "training_parent"),
                                  ("parent_split", "development_parent"), ("parent_split", "unlabelled_parent")]:
                frame = fixture()
                frame.loc[2, column] = value
                write_package(directory, frame)
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    load_b_labels(directory)

    def test_existing_origin_must_keep_non_eval_parent_and_original_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            for column, value in [("is_eval_parent", True), ("prediction_origin", "supplemental_eval_parent"),
                                  ("parent_split", HELDOUT_PARENT_ROLE), ("parent_split", "test_parent")]:
                frame = fixture()
                frame.loc[0, column] = value
                write_package(directory, frame)
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    load_b_labels(directory)
            frame = fixture()
            frame.loc[1, "parent_split"] = "development_parent"
            frame.loc[3, "parent_split"] = "unlabelled_parent"
            write_package(directory, frame)
            pd.testing.assert_frame_equal(load_b_labels(directory), frame)

    def test_skipped_rows_have_no_origin_or_label_and_legal_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, column, value in [(0, "prediction_status", "unknown"), (0, "skip_reason", "fragment"),
                                         (8, "skip_reason", pd.NA), (8, "skip_reason", "unknown"),
                                         (8, "aspect", "other"), (8, "prediction_origin", "supplemental_eval_parent"),
                                         (8, "parent_split", HELDOUT_PARENT_ROLE), (0, "aspect_source", " ")]:
                frame = fixture()
                frame.loc[index, column] = value
                write_package(directory, frame)
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    load_b_labels(directory)

    def test_model_source_aspect_and_subjectivity_enums_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            for column, value in [("model_version", "eight-aspect-subjectivity-polarity-v3"),
                                  ("source", "weak"), ("aspect", "missing"), ("subjectivity", "unclear")]:
                frame = fixture()
                frame.loc[0, column] = value
                write_package(directory, frame)
                with self.subTest(column=column), self.assertRaises(ValueError):
                    load_b_labels(directory)

    def test_objective_literal_NA_and_subjective_polarity_are_not_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, value in [(7, "negative"), (7, "na"), (7, pd.NA), (0, "NA")]:
                frame = fixture()
                frame.loc[index, "polarity"] = value
                write_package(directory, frame)
                with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                    load_b_labels(directory)


if __name__ == "__main__":
    unittest.main()
