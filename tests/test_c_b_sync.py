"""用合成数据验证同步边界；不运行项目训练、推理或读取人工答案。"""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.prepare_c_b_sync import (BASE_VERSION, EXPECTED_TASKS, VERSION,
                                     b_source_at_commit, fitted_state_hash, make_b_coverage, prepare,
                                     verify_revision_provenance)
from scripts.prepare_c_handoff import make_tables
from tests.test_c_handoff import fixture


def provenance_fixture():
    names = ['model', 'input', 'kb', 'eval_meta', 'eval_text', 'predictions', 'base_model', 'training_report']
    hashes = {name: 'synthetic_' + name for name in names}
    inputs = {name: {'sha256': hashes[name]} for name in ['input', 'kb', 'eval_meta', 'eval_text']}
    base = {'version': BASE_VERSION, 'tasks': EXPECTED_TASKS, 'training_inputs': inputs,
            'vectorizer': 'frozen synthetic vocabulary', 'models': {'aspect': 'frozen synthetic weights'},
            'train_review_ids': ['r_train'], 'dev_review_ids': ['r_dev']}
    revised = copy.deepcopy(base)
    revised.update(version=VERSION, base_model_version=BASE_VERSION,
                   model_retrained_for_rule_revision=False,
                   base_model_sha256=hashes['base_model'],
                   base_training_report_sha256=hashes['training_report'])
    training = {'version': BASE_VERSION, 'inputs': inputs,
                'artifacts_sha256': {'model.joblib': hashes['base_model']}}
    report = {'model_version': VERSION, 'model_retrained_for_rule_revision': False,
              'inputs': {**inputs, 'model': {'sha256': hashes['model']}},
              'output_sha256': hashes['predictions'],
              'checks': {name: True for name in ['unique_ids', 'row_count_matches', 'zero_eval_parent_text_overlap',
                        'legal_aspects', 'legal_subjectivity', 'objective_is_NA', 'subjective_has_binary_polarity',
                        'inputs_unchanged', 'roundtrip_identical']}}
    revision = {'version': VERSION, 'training_base_version': BASE_VERSION, 'model_retrained': False,
                'artifacts_sha256': {'model.joblib': hashes['model']},
                'inputs': {**{name: {'sha256': hashes[name]} for name in ['kb', 'eval_meta', 'eval_text']},
                           'base_model': {'sha256': hashes['base_model']},
                           'base_training_report': {'sha256': hashes['training_report']}},
                'human_gold_answers_read': False, 'evaluation_exclusion': {'human_labels_read': False},
                'checks': {name: True for name in ['historical_v3_dev_exactly_reproduced',
                           'raw_model_predictions_unchanged', 'fitted_weights_and_vocabulary_unchanged',
                           'zero_dev_eval_id_parent_text_overlap', 'inputs_unchanged', 'model_reload_identical']}}
    return report, revision, training, revised, base, hashes


class SyncTests(unittest.TestCase):
    def test_runtime_cache_identity_is_distinct_from_vocabulary_state(self):
        original = TfidfVectorizer(stop_words='english')
        original.vocabulary_ = {'synthetic': 0}
        original._stop_words_id = 123
        revised = copy.deepcopy(original)
        revised._stop_words_id = 456
        base, new = {'vectorizer': original, 'models': {}}, {'vectorizer': revised, 'models': {}}
        self.assertEqual(fitted_state_hash(base), fitted_state_hash(new))
        revised.vocabulary_['different'] = 1
        self.assertNotEqual(fitted_state_hash(base), fitted_state_hash(new))

    def test_b_pool_preserves_unpredicted_gold_and_fragments(self):
        kb, predictions, reviews, texts = fixture()
        kb.loc[kb.chunk_id.isin(['eval', 'sibling']), 'is_route_only'] = False
        predictions['model_version'] = VERSION
        labels, coverage = make_tables(kb, predictions, reviews, texts, version=VERSION)
        result = make_b_coverage(kb, coverage)
        self.assertEqual(result.chunk_id.tolist(), ['eval', 'sibling', 'fragment', 'keep_a', 'keep_b'])
        self.assertEqual(set(result.loc[result.prediction_status.eq('predicted'), 'chunk_id']), set(labels.chunk_id))
        self.assertTrue(result.loc[result.chunk_id.eq('eval'), 'is_eval'].all())
        self.assertEqual(result.skip_reason.iloc[:3].tolist(), ['eval_sentence', 'eval_parent', 'fragment'])

    def test_b_pool_rejects_positional_misalignment(self):
        kb, predictions, reviews, texts = fixture()
        predictions['model_version'] = VERSION
        _, coverage = make_tables(kb, predictions, reviews, texts, version=VERSION)
        with self.assertRaisesRegex(ValueError, '编号'):
            make_b_coverage(kb, coverage.iloc[::-1])

    def test_revision_checks_do_not_invent_new_training(self):
        items = provenance_fixture()
        verify_revision_provenance(*items)
        mutations = [
            (0, lambda data: data['inputs']['eval_meta'].update(sha256='changed')),
            (0, lambda data: data.update(output_sha256='changed')),
            (1, lambda data: data.update(human_gold_answers_read=True)),
            (1, lambda data: data.update(model_retrained=True)),
            (2, lambda data: data.update(version=VERSION)),
            (3, lambda data: data['models'].update(aspect='changed weights')),
            (3, lambda data: data.update(train_review_ids=['different parent'])),
            (3, lambda data: data.update(base_model_sha256='changed')),
        ]
        for index, change in mutations:
            changed = copy.deepcopy(items)
            change(changed[index])
            with self.subTest(index=index), self.assertRaises(ValueError):
                verify_revision_provenance(*changed)

    def test_existing_package_is_rejected_before_loading_or_reading_inputs(self):
        with tempfile.TemporaryDirectory() as folder, patch('scripts.prepare_c_b_sync.joblib.load') as load:
            with self.assertRaisesRegex(ValueError, '拒绝覆盖'):
                prepare(SimpleNamespace(output_dir=Path(folder)))
            load.assert_not_called()

    def test_changed_b_pool_policy_rejected_without_executing_b_code(self):
        source = ("_REQUIRED_COLUMNS = ('chunk_id', 'text')\n"
                  "_NOISE_COLUMNS = ('is_route_only', 'is_non_english')\n"
                  "def _load_index_pool(path):\n    return path\n")
        with tempfile.TemporaryDirectory() as folder, patch('scripts.prepare_c_b_sync.subprocess.check_output',
                                                             return_value=source.encode()):
            path = Path(folder)/'retriever.py'
            path.write_text(source)
            _, info = b_source_at_commit('a'*40, path)
            self.assertTrue(info['pool_function_identical_to_verified_baseline'])
            path.write_text(source.replace("'is_non_english'", "'is_eval'"))
            with self.assertRaisesRegex(ValueError, '范围'):
                b_source_at_commit('a'*40, path)


if __name__ == '__main__':
    unittest.main()
