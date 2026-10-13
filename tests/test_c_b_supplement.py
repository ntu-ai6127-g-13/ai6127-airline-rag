"""合成材料验证独立补推理范围；不读取人工答案，不训练项目模型。"""
import json
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch

import pandas as pd

from scripts.prepare_c_handoff import make_tables
from scripts.supplement_c_b_predictions import (inference_reasons, make_extended_tables, run,
    select_supplement, supplemental_labels, validate_extended_tables, validate_jsonl,
    repackage_existing)
from scripts import check_c_b_extended as extended_checker
from src.c_label_reader_v2 import HELDOUT_PARENT_ROLE, LABEL_COLUMNS, VERSION
from src.training_guards import check_fit_boundary
from tests.test_c_handoff import fixture


def material():
    kb, predictions, reviews, texts = fixture()
    kb.loc[kb.chunk_id.isin(['eval', 'sibling']), 'is_route_only'] = False
    predictions['model_version'] = VERSION
    old, _ = make_tables(kb, predictions, reviews, texts, version=VERSION)
    rows = select_supplement(kb, old, reviews, texts, [], [])
    prediction = pd.DataFrame({'aspect': ['food'], 'subjectivity': ['objective'], 'polarity': ['NA'],
        'aspect_source': ['model'], 'subjectivity_source': ['rule:fact'], 'polarity_source': ['objective_gate']})
    added = supplemental_labels(rows, prediction)
    return kb, old, rows, added, reviews, texts


class SupplementTests(unittest.TestCase):
    def test_selects_only_missing_eligible_non_gold_siblings(self):
        kb, old, rows, added, reviews, texts = material()
        self.assertEqual(rows.chunk_id.tolist(), ['sibling'])
        self.assertEqual(added.parent_split.tolist(), [HELDOUT_PARENT_ROLE])
        self.assertTrue(added.is_eval_parent.all())
        self.assertFalse(rows.is_eval.any())
        self.assertEqual(inference_reasons(kb, texts).iloc[0], 'eval_sentence')
        self.assertEqual(inference_reasons(kb, texts).iloc[2], 'eval_duplicate_text')

    def test_missing_nonheldout_predictions_cannot_silently_expand_scope(self):
        kb, old, rows, added, reviews, texts = material()
        with self.assertRaisesRegex(ValueError, '兄弟句'):
            select_supplement(kb, old.iloc[:1], reviews, texts, [], [])

    def test_training_parent_overlap_rejected_before_prediction(self):
        kb, old, rows, added, reviews, texts = material()
        with self.assertRaisesRegex(ValueError, '隔离'):
            select_supplement(kb, old, reviews, texts, ['eval_parent'], [])

    def test_extended_coverage_preserves_original_values_and_null_gold(self):
        kb, old, rows, added, reviews, texts = material()
        labels, coverage, b, non_eval = make_extended_tables(kb, old, added, reviews, texts)
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(b), 5)
        self.assertEqual(len(non_eval), len(kb)-1)
        self.assertEqual(b.loc[b.prediction_status.eq('skipped'), 'skip_reason'].tolist(), ['eval_sentence', 'fragment'])
        self.assertTrue(coverage.loc[coverage.is_eval, 'aspect'].isna().all())
        self.assertEqual(labels.loc[labels.prediction_origin.eq('existing_v4_1'), 'polarity'].tolist(), old.polarity.tolist())

    def test_modified_existing_predictions_or_parent_flags_rejected(self):
        kb, old, rows, added, reviews, texts = material()
        labels, coverage, _, _ = make_extended_tables(kb, old, added, reviews, texts)
        wrong_labels, wrong_coverage = labels.copy(), coverage.copy()
        wrong_labels.loc[wrong_labels.chunk_id.eq('keep_a'), 'polarity'] = 'positive'
        wrong_coverage.loc[wrong_coverage.chunk_id.eq('keep_a'), 'polarity'] = 'positive'
        with self.assertRaises(AssertionError):
            validate_extended_tables(kb, old, wrong_labels, added, wrong_coverage, reviews, texts)
        wrong_coverage = coverage.copy()
        wrong_coverage.loc[wrong_coverage.chunk_id.eq('sibling'), 'is_eval_parent'] = False
        with self.assertRaises(ValueError):
            validate_extended_tables(kb, old, labels, added, wrong_coverage, reviews, texts)

    def test_jsonl_keeps_literal_NA_and_null_confidence(self):
        kb, old, rows, added, reviews, texts = material()
        labels, _, _, _ = make_extended_tables(kb, old, added, reviews, texts)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'predictions.jsonl'
            labels.assign(confidence=None).to_json(path, orient='records', lines=True)
            self.assertEqual(validate_jsonl(path, labels), len(labels))
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[-1]['polarity'], 'NA')
            self.assertIsNone(records[-1]['confidence'])
            records[-1]['confidence'] = 0.91
            path.write_text(''.join(json.dumps(record)+'\n' for record in records))
            with self.assertRaisesRegex(ValueError, '置信度'):
                validate_jsonl(path, labels)

    def test_supplemental_predictions_stay_rejected_as_training_targets(self):
        kb, old, rows, added, reviews, texts = material()
        train = added.assign(text=rows.text.to_numpy(), is_eval=False, split='train')
        dev = pd.DataFrame({'chunk_id': ['dev'], 'review_id': ['dev_parent'], 'text': ['Synthetic development sentence.'],
                            'is_eval': [False], 'source': ['weak'], 'split': ['dev']})
        with self.assertRaisesRegex(ValueError, '预测结果'):
            check_fit_boundary(train, dev, eval_reviews=reviews, eval_texts=texts)
        train['source'] = 'weak'
        with self.assertRaisesRegex(ValueError, '评测父评论'):
            check_fit_boundary(train, dev, eval_reviews=reviews, eval_texts=texts)

    def test_existing_output_stops_before_model_loading_or_inference(self):
        with tempfile.TemporaryDirectory() as folder, patch('scripts.supplement_c_b_predictions.predict_texts') as predict, \
                patch('scripts.supplement_c_b_predictions.audit_prior_package') as audit:
            with self.assertRaisesRegex(ValueError, '拒绝覆盖'):
                run(SimpleNamespace(output_dir=Path(folder)))
            predict.assert_not_called()
            audit.assert_not_called()


def documentation_fixture(root):
    original, package = root/'original', root/'package'
    original.mkdir()
    package.mkdir()
    digests = {}
    for name in extended_checker.ARTIFACTS:
        content = ('original bytes: '+name).encode()
        (original/name).write_bytes(content)
        (package/name).write_bytes(content)
        digests[name] = hashlib.sha256(content).hexdigest()
    old = {'schema_version': 'c-b-sync-v2', 'inference_lock': {'model_sha256': 'fixed'},
           'acceptance': {'human_gold_evaluated': False}, 'artifacts_sha256': digests}
    (original/'manifest.json').write_text(json.dumps(old))
    current = {}
    for name in extended_checker.PACKAGING_SOURCES:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('current source: '+name)
        current[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (package/'README.md').write_bytes((root/extended_checker.GUIDE_PATH).read_bytes())
    history = package/extended_checker.HISTORY_FILE
    with zipfile.ZipFile(history, 'w') as archive:
        archive.writestr('docs/C到B补预测交接说明.md', 'archived documentation')
    manifest = copy.deepcopy(old)
    manifest['artifacts_sha256']['README.md'] = current[extended_checker.GUIDE_PATH]
    manifest['artifacts_sha256'][extended_checker.HISTORY_FILE] = hashlib.sha256(history.read_bytes()).hexdigest()
    manifest['documentation_repack'] = {
        'version': 'single-guide-v1', 'created_at': 'synthetic',
        'original_package': {'path': str(original),
                             'manifest_sha256': hashlib.sha256((original/'manifest.json').read_bytes()).hexdigest()},
        'historical_sources_file': extended_checker.HISTORY_FILE,
        'historical_sources_sha256': manifest['artifacts_sha256'][extended_checker.HISTORY_FILE],
        'current_packaging_sources_sha256': current,
        'new_inference': False, 'model_training': False, 'data_files_unchanged': True,
    }
    return original, package, manifest


class DocumentationRepackTests(unittest.TestCase):
    def test_valid_documentation_copy_preserves_original_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, package, manifest = documentation_fixture(root)
            with patch.object(extended_checker, 'ROOT', root):
                result = extended_checker.validate_documentation_repack(package, manifest)
            self.assertEqual(result, (package/extended_checker.HISTORY_FILE).resolve())

    def test_changed_original_inference_or_acceptance_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, package, manifest = documentation_fixture(root)
            for key, value in [('inference_lock', {'model_sha256': 'different'}),
                               ('acceptance', {'human_gold_evaluated': True})]:
                bad = copy.deepcopy(manifest)
                bad[key] = value
                with self.subTest(key=key), patch.object(extended_checker, 'ROOT', root):
                    with self.assertRaisesRegex(ValueError, '改变了原推理'):
                        extended_checker.validate_documentation_repack(package, bad)

    def test_data_hash_or_original_artifact_changes_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original, package, manifest = documentation_fixture(root)
            bad = copy.deepcopy(manifest)
            bad['artifacts_sha256']['labels.parquet'] = 'a'*64
            with patch.object(extended_checker, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, '原标签或覆盖'):
                    extended_checker.validate_documentation_repack(package, bad)
                (original/'labels.parquet').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, '原补预测包产物'):
                    extended_checker.validate_documentation_repack(package, manifest)

    def test_incomplete_or_changed_current_packaging_sources_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, package, manifest = documentation_fixture(root)
            bad = copy.deepcopy(manifest)
            del bad['documentation_repack']['current_packaging_sources_sha256'][extended_checker.GUIDE_PATH]
            with patch.object(extended_checker, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, '来源不完整'):
                    extended_checker.validate_documentation_repack(package, bad)
                (root/extended_checker.GUIDE_PATH).write_text('changed')
                with self.assertRaisesRegex(ValueError, '来源 hash'):
                    extended_checker.validate_documentation_repack(package, manifest)

    def test_wrong_readme_or_history_digest_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, package, manifest = documentation_fixture(root)
            bad = copy.deepcopy(manifest)
            bad['documentation_repack']['historical_sources_sha256'] = 'b'*64
            with patch.object(extended_checker, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'ZIP 哈希'):
                    extended_checker.validate_documentation_repack(package, bad)
                (package/'README.md').write_text('different guide')
                with self.assertRaisesRegex(ValueError, 'README'):
                    extended_checker.validate_documentation_repack(package, manifest)

    def test_original_manifest_digest_or_self_reference_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, package, manifest = documentation_fixture(root)
            for change in [{'manifest_sha256': 'c'*64}, {'path': str(package)}]:
                bad = copy.deepcopy(manifest)
                bad['documentation_repack']['original_package'].update(change)
                with self.subTest(change=change), patch.object(extended_checker, 'ROOT', root):
                    with self.assertRaises(ValueError):
                        extended_checker.validate_documentation_repack(package, bad)

    def test_training_or_inference_claims_and_unknown_keys_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, package, manifest = documentation_fixture(root)
            for key in ['new_inference', 'model_training', 'unexpected']:
                bad = copy.deepcopy(manifest)
                bad['documentation_repack'][key] = True
                with self.subTest(key=key), patch.object(extended_checker, 'ROOT', root):
                    with self.assertRaisesRegex(ValueError, '仅整理说明'):
                        extended_checker.validate_documentation_repack(package, bad)

    def test_existing_repack_output_stops_before_model_or_audit(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch('scripts.supplement_c_b_predictions.predict_texts') as predict, \
                patch('scripts.supplement_c_b_predictions.joblib.load') as load, \
                patch.object(extended_checker, 'audit_package') as audit:
            with self.assertRaisesRegex(ValueError, '拒绝覆盖'):
                repackage_existing(Path(folder)/'missing', Path(folder), None)
            predict.assert_not_called()
            load.assert_not_called()
            audit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
