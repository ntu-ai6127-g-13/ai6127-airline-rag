"""仅用合成数据验证候选交接边界；不读取人工标注或现有模型。"""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import zipfile

import pandas as pd

from scripts.prepare_c_handoff import (ASPECTS, LABEL_COLUMNS, METADATA_COLUMNS, SKIP_REASONS,
                                      VERSION, make_tables, prepare, skip_reasons,
                                      validate_handoff, verify_packaging_sources, verify_provenance)
from src.weak_labels import normalize_text


def row(identifier, text, **overrides):
    result = dict(chunk_id=identifier, review_id="r_" + identifier, text=text, is_eval=False,
                  is_route_only=False, is_non_english=False, is_fragment=False)
    result.update(overrides)
    return result


def fixture():
    kb = pd.DataFrame([
        row("eval", "The seat was good.", is_eval=True, review_id="eval_parent", is_route_only=True),
        row("sibling", "The meal arrived.", review_id="eval_parent", is_route_only=True),
        row("duplicate", " THE SEAT WAS GOOD. ", is_route_only=True),
        row("empty", " \t ", is_route_only=True, is_non_english=True, is_fragment=True),
        row("route", "SIN to LHR.", is_route_only=True, is_non_english=True, is_fragment=True),
        row("nonenglish", "Synthetic non English text.", is_non_english=True, is_fragment=True),
        row("fragment", "Synthetic fragment.", is_fragment=True),
        row("keep_a", "The meal was terrible."),
        row("keep_b", "The aircraft was a Boeing 777."),
    ])
    selected = kb.loc[kb.chunk_id.isin(["keep_a", "keep_b"])].copy().reset_index(drop=True)
    selected["aspect"] = ["food", "other"]
    selected["subjectivity"] = ["subjective", "objective"]
    selected["polarity"] = ["negative", "NA"]
    selected["model_version"] = VERSION
    selected["source"] = "hybrid_prediction"
    selected["aspect_source"] = "model"
    selected["subjectivity_source"] = "model"
    selected["polarity_source"] = ["model", "objective_gate"]
    selected["parent_split"] = "unlabelled_parent"
    return kb, selected[[*LABEL_COLUMNS, "text", "is_eval"]], {"eval_parent"}, {normalize_text("The seat was good.")}


class HandoffTests(unittest.TestCase):
    def test_priority_reasons_and_complete_coverage(self):
        kb, predictions, reviews, texts = fixture()
        reason = skip_reasons(kb, reviews, texts)
        self.assertEqual(reason.iloc[:7].tolist(), SKIP_REASONS)
        self.assertTrue(reason.iloc[7:].isna().all())
        labels, coverage = make_tables(kb, predictions, reviews, texts)
        self.assertEqual(set(labels.chunk_id), {"keep_a", "keep_b"})
        self.assertEqual(coverage.chunk_id.tolist(), kb.chunk_id.tolist())
        self.assertTrue(coverage.chunk_id.is_unique)
        self.assertTrue(coverage.loc[coverage.prediction_status.eq("skipped"), METADATA_COLUMNS].isna().all().all())
        self.assertTrue(coverage.loc[coverage.prediction_status.eq("predicted"), "skip_reason"].isna().all())
        self.assertEqual(coverage.loc[coverage.is_eval, "prediction_status"].tolist(), ["skipped"])

    def test_left_join_keeps_every_kb_id_and_does_not_invent_labels(self):
        kb, predictions, reviews, texts = fixture()
        labels, coverage = make_tables(kb, predictions.sample(frac=1, random_state=2), reviews, texts)
        joined = kb[["chunk_id", "review_id"]].merge(
            labels.drop(columns="review_id"), on="chunk_id", how="left", validate="one_to_one").merge(
            coverage[["chunk_id", "prediction_status", "skip_reason"]], on="chunk_id", how="left", validate="one_to_one")
        self.assertEqual(len(joined), len(kb))
        self.assertEqual(joined.chunk_id.tolist(), kb.chunk_id.tolist())
        self.assertEqual(int(joined.aspect.isna().sum()), 7)
        self.assertEqual(int(joined.aspect.eq("other").sum()), 1)
        self.assertEqual(joined.loc[joined.chunk_id.eq("eval"), "skip_reason"].tolist(), ["eval_sentence"])

    def test_missing_prediction_id_is_rejected(self):
        kb, predictions, reviews, texts = fixture()
        with self.assertRaisesRegex(ValueError, "不精确一致"):
            make_tables(kb, predictions.iloc[:1], reviews, texts)

    def test_unknown_id_and_duplicate_prediction_id_are_rejected(self):
        kb, predictions, reviews, texts = fixture()
        unknown = predictions.copy()
        unknown.loc[0, "chunk_id"] = "not_in_kb"
        with self.assertRaisesRegex(ValueError, "当前 KB 不一致"):
            make_tables(kb, unknown, reviews, texts)
        with self.assertRaisesRegex(ValueError, "重复"):
            make_tables(kb, pd.concat([predictions, predictions.iloc[:1]], ignore_index=True), reviews, texts)
        with self.assertRaisesRegex(ValueError, "重复"):
            make_tables(pd.concat([kb, kb.iloc[:1]], ignore_index=True), predictions, reviews, texts)

    def test_prediction_text_and_parent_must_match_kb(self):
        kb, predictions, reviews, texts = fixture()
        for column in ["text", "review_id"]:
            changed = predictions.copy()
            changed.loc[0, column] = "different"
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "当前 KB 不一致"):
                make_tables(kb, changed, reviews, texts)

    def test_empty_or_illegal_prediction_labels_are_rejected(self):
        kb, predictions, reviews, texts = fixture()
        for column, value in [("aspect", pd.NA), ("aspect", ""), ("aspect", "unknown"),
                              ("subjectivity", None), ("polarity", "NA"), ("aspect_source", " ")]:
            changed = predictions.copy()
            changed.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                make_tables(kb, changed, reviews, texts)
        changed = predictions.copy()
        changed.loc[1, "polarity"] = "negative"
        with self.assertRaisesRegex(ValueError, "objective"):
            make_tables(kb, changed, reviews, texts)

    def test_skip_reason_status_or_placeholder_label_errors_are_rejected(self):
        kb, predictions, reviews, texts = fixture()
        labels, coverage = make_tables(kb, predictions, reviews, texts)
        for column, value, message in [("skip_reason", "fragment", "理由"),
                                       ("prediction_status", "predicted", "prediction_status"),
                                       ("aspect", "other", "空标签")]:
            changed = coverage.copy()
            changed.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, message):
                validate_handoff(kb, labels, changed, reviews, texts)
        with self.assertRaisesRegex(ValueError, "完整覆盖"):
            validate_handoff(kb, labels, coverage.iloc[:-1], reviews, texts)
        with self.assertRaisesRegex(ValueError, "重复"):
            validate_handoff(kb, labels, pd.concat([coverage, coverage.iloc[:1]], ignore_index=True), reviews, texts)

    def test_boolean_flags_and_prediction_eval_mask_are_strict(self):
        kb, predictions, reviews, texts = fixture()
        invalid_kb = kb.copy()
        invalid_kb["is_eval"] = invalid_kb.is_eval.astype("string")
        with self.assertRaisesRegex(ValueError, "布尔"):
            make_tables(invalid_kb, predictions, reviews, texts)
        invalid = predictions.copy()
        invalid.loc[0, "is_eval"] = True
        with self.assertRaisesRegex(ValueError, "评测"):
            make_tables(kb, invalid, reviews, texts)

    def test_parent_split_provenance_is_checked_without_using_models(self):
        kb, predictions, reviews, texts = fixture()
        bundle = {"train_review_ids": ["r_keep_a"], "dev_review_ids": [], "models": object()}
        with self.assertRaisesRegex(ValueError, "来源"):
            make_tables(kb, predictions, reviews, texts, bundle=bundle)
        predictions.loc[0, "parent_split"] = "training_parent"
        make_tables(kb, predictions, reviews, texts, bundle=bundle)

    def test_existing_output_is_rejected_before_reading_model_or_inputs(self):
        with tempfile.TemporaryDirectory() as directory, patch("scripts.prepare_c_handoff.joblib.load") as load:
            with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
                prepare(SimpleNamespace(output_dir=Path(directory)))
            load.assert_not_called()

    def test_report_and_model_hash_mismatches_are_rejected(self):
        hashes = {key: "synthetic_hash_" + key for key in ["model", "input", "kb", "eval_meta", "eval_text", "predictions"]}
        provenance = {key: {"sha256": hashes[key]} for key in ["model", "input", "kb", "eval_meta", "eval_text"]}
        checks = {key: True for key in ["unique_ids", "row_count_matches", "zero_eval_parent_text_overlap", "legal_aspects",
                                       "legal_subjectivity", "objective_is_NA", "subjective_has_binary_polarity", "inputs_unchanged", "roundtrip_identical"]}
        report = {"model_version": VERSION, "inputs": provenance, "output_sha256": hashes["predictions"], "checks": checks}
        bundle = {"version": VERSION, "tasks": {"aspect": ASPECTS, "subjectivity": ["objective", "subjective"],
                                                "polarity": ["negative", "positive"]},
                  "training_inputs": {key: provenance[key] for key in ["input", "kb", "eval_meta", "eval_text"]}}
        verify_provenance(report, bundle, hashes)
        for key in hashes:
            wrong = hashes.copy()
            wrong[key] = "changed"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "hash|training_inputs"):
                verify_provenance(report, bundle, wrong)
        wrong_bundle = copy.deepcopy(bundle)
        wrong_bundle["training_inputs"]["eval_meta"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "training_inputs"):
            verify_provenance(report, wrong_bundle, hashes)
        wrong_report = copy.deepcopy(report)
        wrong_report["checks"]["inputs_unchanged"] = False
        with self.assertRaisesRegex(ValueError, "未通过"):
            verify_provenance(wrong_report, bundle, hashes)


class HistoricalPackagingSourcesTests(unittest.TestCase):
    @staticmethod
    def digest(content):
        return hashlib.sha256(content).hexdigest()

    def test_current_sources_are_verified_without_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "src/classification_pipeline.py"
            path.parent.mkdir()
            path.write_bytes(b"fixed inference source")
            result = verify_packaging_sources(
                {"src/classification_pipeline.py": self.digest(path.read_bytes())}, root=root)
            self.assertEqual(result, {"current": ["src/classification_pipeline.py"], "archived": []})

    def test_explicit_archive_proves_old_packaging_bytes_without_executing_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper = "scripts/prepare_c_handoff.py"
            old_doc = "docs/C到B同步操作说明.md"
            inference = "src/classification_pipeline.py"
            (root / wrapper).parent.mkdir()
            (root / wrapper).write_bytes(b"new packaging implementation")
            (root / inference).parent.mkdir()
            (root / inference).write_bytes(b"fixed inference source")
            marker = root / "must_not_execute"
            archived_code = f"open({str(marker)!r}, 'w').write('executed')".encode()
            archive_path = root / "history.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(wrapper, archived_code)
                archive.writestr(old_doc, b"old README source")
            recorded = {wrapper: self.digest(archived_code), old_doc: self.digest(b"old README source"),
                        inference: self.digest(b"fixed inference source")}
            result = verify_packaging_sources(recorded, root=root, historical_sources=archive_path)
            self.assertEqual(result["current"], [inference])
            self.assertEqual(result["archived"], sorted([wrapper, old_doc]))
            self.assertFalse(marker.exists())
            self.assertFalse((root / old_doc).exists())

    def test_fallback_requires_explicit_archive_and_exact_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "scripts/prepare_c_b_sync.py"
            recorded = {name: self.digest(b"original bytes")}
            with self.assertRaisesRegex(ValueError, "hash"):
                verify_packaging_sources(recorded, root=root)
            archive_path = root / "history.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(name, b"changed bytes")
            with self.assertRaisesRegex(ValueError, "hash"):
                verify_packaging_sources(recorded, root=root, historical_sources=archive_path)

    def test_inference_rules_and_reader_never_use_historical_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "history.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("scripts/prepare_c_handoff.py", b"old wrapper")
            for name in ["src/classification_pipeline.py", "src/language_rules.py", "src/c_label_reader_v2.py"]:
                path = root / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"changed current implementation")
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "hash"):
                    verify_packaging_sources({name: self.digest(b"historical implementation")},
                                             root=root, historical_sources=archive_path)

    def test_archive_rejects_duplicate_unsafe_or_non_allowlisted_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = "scripts/prepare_c_handoff.py"
            archive_path = root / "history.zip"
            cases = [[allowed, allowed], ["../scripts/prepare_c_handoff.py"],
                     ["/scripts/prepare_c_handoff.py"], ["./scripts/prepare_c_handoff.py"],
                     ["scripts//prepare_c_handoff.py"], ["scripts\\prepare_c_handoff.py"],
                     ["src/c_label_reader_v2.py"], ["docs/"], ["C:/scripts/prepare_c_handoff.py"]]
            for names in cases:
                with self.subTest(names=names):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        with zipfile.ZipFile(archive_path, "w") as archive:
                            for name in names:
                                archive.writestr(name, b"old bytes")
                    with self.assertRaisesRegex(ValueError, "重复|路径|白名单"):
                        verify_packaging_sources({}, root=root, historical_sources=archive_path)

    def test_recorded_paths_and_digests_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["", ".", "../source.py", "/source.py", "./source.py",
                         "scripts//source.py", "scripts\\source.py", "C:/source.py"]:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "路径"):
                    verify_packaging_sources({name: self.digest(b"bytes")}, root=root)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_packaging_sources({"scripts/prepare_c_handoff.py": "not_a_digest"}, root=root)

    def test_source_symlink_outside_root_is_rejected_before_archive_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "repo"
            (root / "scripts").mkdir(parents=True)
            outside = parent / "outside.py"
            outside.write_bytes(b"original bytes")
            name = "scripts/prepare_c_handoff.py"
            (root / name).symlink_to(outside)
            archive_path = parent / "history.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(name, b"original bytes")
            with self.assertRaisesRegex(ValueError, "根目录"):
                verify_packaging_sources({name: self.digest(b"original bytes")}, root=root,
                                         historical_sources=archive_path)


if __name__ == "__main__":
    unittest.main()
