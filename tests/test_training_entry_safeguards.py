"""原始句子入口与独立 C3 训练保护；拟合阶段全部用 mock，绝不训练模型。"""
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from src import classification_pipeline as joint
from src import subjectivity, subjectivity_legacy
from src.training_guards import check_fit_boundary


C3_MODULES = (subjectivity, subjectivity_legacy)


def raw_chunks():
    common = dict(is_eval=False, is_route_only=False, is_non_english=False,
                  is_fragment=False, overall_rating=9,
                  **{'r_' + aspect: 4 for aspect in joint.TASKS['aspect'][:6]})
    return pd.DataFrame([
        dict(common, chunk_id='seat1', review_id='seat_parent',
             text='The seat was comfortable.'),
        dict(common, chunk_id='fare1', review_id='fare_parent',
             text='The ticket was 800 SGD.'),
    ])


def fit_parts():
    def part(role):
        return pd.DataFrame([
            dict(chunk_id=role + '1', review_id=role + '_parent1',
                 text=role + ' fact sentence.', split=role, source='weak_subjectivity',
                 is_eval=False, subjectivity='objective'),
            dict(chunk_id=role + '2', review_id=role + '_parent2',
                 text=role + ' evaluation sentence.', split=role, source='weak_subjectivity',
                 is_eval=False, subjectivity='subjective'),
        ])
    return part('train'), part('dev')


def joint_labels_and_reference():
    rows = []
    for parent in ('training_parent', 'development_parent'):
        for index, aspect in enumerate(joint.TASKS['aspect']):
            rows.append(dict(chunk_id=parent + '_' + aspect, review_id=parent,
                             text=parent + ' distinct ' + aspect, aspect=aspect,
                             subjectivity='subjective',
                             polarity='positive' if index % 2 else 'negative'))
        rows.append(dict(chunk_id=parent + '_fact', review_id=parent,
                         text=parent + ' distinct fact', aspect='value',
                         subjectivity='objective', polarity=None))
    labels = pd.DataFrame(rows)
    reference = labels[['chunk_id', 'review_id']].copy()
    reference['split'] = reference.review_id.map(
        {'training_parent': 'train', 'development_parent': 'dev'})
    return labels, reference


class TrainingEntrySafeguardTests(unittest.TestCase):
    def test_raw_chunks_still_generate_joint_and_both_c3_labels(self):
        raw = raw_chunks()
        before = raw.copy(deep=True)
        for module in (joint, *C3_MODULES):
            with self.subTest(module=module.__name__):
                labels, _ = module.build_labels(raw, set(), set())
                self.assertEqual(set(labels.chunk_id), set(raw.chunk_id))
                self.assertNotIn('split', labels)
        pd.testing.assert_frame_equal(raw, before)

    def test_label_generation_rejects_any_existing_split_before_losing_it(self):
        for module in (joint, *C3_MODULES):
            for role in ('train', 'dev', 'unassigned', 'blocked_dev_duplicate', None):
                raw = raw_chunks().assign(split=role)
                before = raw.copy(deep=True)
                with self.subTest(module=module.__name__, role=role):
                    with self.assertRaisesRegex(ValueError, 'split 角色'):
                        module.build_labels(raw, set(), set())
                    pd.testing.assert_frame_equal(raw, before)

    def test_resplit_helpers_reject_role_annotated_labels(self):
        train, dev = fit_parts()
        labels = pd.concat([train, dev], ignore_index=True)
        for module in (joint, *C3_MODULES):
            with self.subTest(module=module.__name__):
                with self.assertRaisesRegex(ValueError, 'split 角色'):
                    module.split_labels(labels)

    def test_joint_frozen_split_reference_remains_supported(self):
        labels, reference = joint_labels_and_reference()
        train, dev = joint.split_labels(labels, seed=123, reference=reference)
        self.assertEqual(set(train.review_id), {'training_parent'})
        self.assertEqual(set(dev.review_id), {'development_parent'})
        self.assertEqual(len(train) + len(dev), len(labels))
        pd.testing.assert_frame_equal(
            labels.sort_values('chunk_id').reset_index(drop=True),
            pd.concat([train, dev]).sort_values('chunk_id').reset_index(drop=True))

    def test_independent_c3_fit_stops_all_boundary_failures_before_vocabulary(self):
        train, dev = fit_parts()
        cases = [
            ('wrong_train_role', train.assign(split='dev'), dev),
            ('mixed_train_roles', pd.concat([train, dev]), dev),
            ('wrong_dev_role', train, dev.assign(split='unassigned')),
            ('prediction_source', train.assign(source='hybrid_prediction'), dev),
            ('unknown_source', train.assign(source='unknown'), dev),
            ('missing_source_value', train.assign(source=None), dev),
            ('eval_sentence', train.assign(is_eval=True), dev),
            ('nonboolean_eval', train.assign(is_eval='False'), dev),
            ('missing_eval_value', train.assign(is_eval=None), dev),
            ('same_ids', train, dev.assign(chunk_id=train.chunk_id.to_numpy())),
            ('same_parents', train, dev.assign(review_id=train.review_id.to_numpy())),
            ('same_texts', train, dev.assign(text=train.text.str.upper().to_numpy())),
            ('blank_text', train.assign(text=' '), dev),
            ('invalid_parent', train.assign(review_id=None), dev),
            ('empty_train', train.iloc[:0], dev),
        ]
        for module in C3_MODULES:
            for name, bad_train, bad_dev in cases:
                with self.subTest(module=module.__name__, case=name):
                    with patch(module.__name__ + '.TfidfVectorizer.fit_transform') as vocabulary:
                        with self.assertRaises(ValueError):
                            module.fit_model(bad_train, bad_dev)
                        vocabulary.assert_not_called()

    def test_weak_subjectivity_source_passes_shared_boundary(self):
        train, dev = fit_parts()
        checks = check_fit_boundary(train, dev, eval_reviews=set(), eval_texts=set())
        self.assertTrue(all(checks.values()))

    def test_valid_independent_c3_parts_reach_permitted_fit_without_training(self):
        train, dev = fit_parts()
        for module in C3_MODULES:
            with self.subTest(module=module.__name__):
                with patch(module.__name__ + '.TfidfVectorizer.fit_transform',
                           side_effect=RuntimeError('READY_TO_FIT')) as vocabulary:
                    with self.assertRaisesRegex(RuntimeError, 'READY_TO_FIT'):
                        module.fit_model(train, dev)
                    vocabulary.assert_called_once()

    def test_independent_c3_cli_rejects_role_input_before_fitting(self):
        frame = raw_chunks().assign(split='dev')
        kb = frame[['chunk_id', 'review_id', 'text', 'is_eval']]
        args = SimpleNamespace(input=Path('synthetic_raw.parquet'), seed=42,
                               output_dir=Path('/private/tmp/c3_guard_never_written'))
        for module in C3_MODULES:
            with self.subTest(module=module.__name__):
                with patch(module.__name__ + '.sha256', return_value='synthetic'), \
                        patch(module.__name__ + '.pd.read_parquet', side_effect=[kb, frame]), \
                        patch(module.__name__ + '.load_test_exclusions',
                              return_value=(set(), set(), {})), \
                        patch(module.__name__ + '.TfidfVectorizer.fit_transform') as vocabulary, \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, 'split 角色'):
                        module.train_command(args)
                    vocabulary.assert_not_called()
        self.assertFalse(args.output_dir.exists())

    def test_joint_cli_rejects_role_input_before_fitting_even_with_reference(self):
        args = SimpleNamespace(input=Path('synthetic_raw.parquet'), seed=42,
                               output_dir=Path('/private/tmp/joint_guard_never_written'),
                               max_features=500, max_iter=1000,
                               split_reference=Path('synthetic_reference.parquet'))
        with patch('src.classification_pipeline.sha256', return_value='synthetic'), \
                patch('src.classification_pipeline.checked_input',
                      return_value=(raw_chunks().assign(split='dev'), set(), set(), {})), \
                patch('src.classification_pipeline.TfidfVectorizer.fit_transform') as vocabulary, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'split 角色'):
                joint.train_command(args)
            vocabulary.assert_not_called()
        self.assertFalse(args.output_dir.exists())

    def test_independent_c3_cli_passes_actual_eval_lists_before_fitting(self):
        frame = raw_chunks()
        kb = frame[['chunk_id', 'review_id', 'text', 'is_eval']]
        train, dev = fit_parts()
        args = SimpleNamespace(input=Path('synthetic_raw.parquet'), seed=42,
                               output_dir=Path('/private/tmp/c3_guard_never_written'))
        for module in C3_MODULES:
            for reviews, texts in [({'train_parent1'}, set()),
                                   (set(), {'dev evaluation sentence.'})]:
                with self.subTest(module=module.__name__, reviews=reviews, texts=texts):
                    with patch(module.__name__ + '.sha256', return_value='synthetic'), \
                            patch(module.__name__ + '.pd.read_parquet', side_effect=[kb, frame]), \
                            patch(module.__name__ + '.load_test_exclusions',
                                  return_value=(reviews, texts, {})), \
                            patch(module.__name__ + '.build_labels', return_value=(train, {})), \
                            patch(module.__name__ + '.validate_labels', return_value={}), \
                            patch(module.__name__ + '.split_labels', return_value=(train, dev)), \
                            patch(module.__name__ + '.fit_model') as fit, \
                            patch(module.__name__ + '.TfidfVectorizer.fit_transform') as vocabulary, \
                            contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(ValueError, '评测'):
                            module.train_command(args)
                        fit.assert_not_called()
                        vocabulary.assert_not_called()
        self.assertFalse(args.output_dir.exists())


if __name__ == '__main__':
    unittest.main()
