"""训练防误用的失败路径测试；不读取真实数据、不拟合真实模型。"""
import unittest
from unittest.mock import patch

import pandas as pd

from src.classify import fit_baseline, validate_training_data
from src.classification_pipeline import fit_models
from scripts.compare_weak_label_strategies import fit_strategy
from src.training_guards import check_fit_boundary, select_training_rows, validate_roles
from src.weak_labels import normalize_text


def parts():
    train = pd.DataFrame([dict(chunk_id='train1', review_id='parent1',
                              text='The seat was comfortable.', split='train',
                              source='weak', is_eval=False, aspect='seat', polarity='positive')])
    dev = pd.DataFrame([dict(chunk_id='dev1', review_id='parent2',
                            text='The meal was terrible.', split='dev',
                            source='weak', is_eval=False, aspect='food', polarity='negative')])
    return train, dev


class TrainingGuardTests(unittest.TestCase):
    def test_selection_is_explicit_nonmutating_and_keeps_training_conflicts(self):
        train, dev = parts()
        opposite = train.assign(polarity='negative')
        unassigned = train.assign(chunk_id='unused', review_id='unused_parent', split='unassigned')
        blocked = train.assign(chunk_id='blocked', review_id='blocked_parent', split='blocked_dev_duplicate')
        labels = pd.concat([train, opposite, dev, unassigned, blocked], ignore_index=True)
        before = labels.copy(deep=True)
        selected = select_training_rows(labels)
        self.assertEqual(len(selected), 2)
        self.assertEqual(set(selected.polarity), {'positive', 'negative'})
        self.assertEqual(set(selected.chunk_id), {'train1'})
        pd.testing.assert_frame_equal(before, labels)
        self.assertTrue(all(check_fit_boundary(selected, dev, eval_reviews=set(), eval_texts=set()).values()))

    def test_missing_unknown_and_empty_training_roles_fail(self):
        train, _ = parts()
        for broken in [train.drop(columns='split'), train.assign(split=None),
                       train.assign(split='test'), train.assign(split='dev')]:
            with self.subTest(), self.assertRaises(ValueError):
                select_training_rows(broken)

    def test_parent_or_chunk_cannot_cross_roles(self):
        train, dev = parts()
        for changed in [dev.assign(review_id='parent1'), dev.assign(chunk_id='train1')]:
            with self.subTest(), self.assertRaises(ValueError):
                validate_roles(pd.concat([train, changed], ignore_index=True))

    def test_fit_rejects_mixed_roles_or_wrong_dev_role(self):
        train, dev = parts()
        with self.assertRaisesRegex(ValueError, '角色'):
            check_fit_boundary(pd.concat([train, dev]), dev)
        with self.assertRaisesRegex(ValueError, '角色'):
            check_fit_boundary(train, dev.assign(split='unassigned'))

    def test_train_dev_ids_parents_and_normalized_texts_are_disjoint(self):
        train, dev = parts()
        for changed in [dev.assign(chunk_id='train1'), dev.assign(review_id='parent1'),
                        dev.assign(text='  ＴＨＥ   SEAT WAS COMFORTABLE.  ')]:
            with self.subTest(), self.assertRaisesRegex(ValueError, '重叠'):
                check_fit_boundary(train, changed)

    def test_false_eval_flag_cannot_bypass_parent_or_text_exclusion(self):
        train, dev = parts()
        for reviews, texts in [({'parent1'}, set()), (set(), {normalize_text(train.text.iloc[0])}),
                               ({'parent2'}, set()), (set(), {normalize_text(dev.text.iloc[0])})]:
            with self.subTest(), self.assertRaisesRegex(ValueError, '评测'):
                check_fit_boundary(train, dev, eval_reviews=reviews, eval_texts=texts)

    def test_eval_boolean_schema_and_reference_pairs_are_required(self):
        train, dev = parts()
        for broken in [train.assign(is_eval=True), train.assign(is_eval='False'),
                       train.assign(is_eval=None), train.drop(columns='is_eval')]:
            with self.subTest(), self.assertRaises(ValueError):
                check_fit_boundary(broken, dev, eval_reviews=set(), eval_texts=set())
        with self.assertRaisesRegex(ValueError, '一起'):
            check_fit_boundary(train, dev, eval_reviews=set())

    def test_predictions_and_unknown_sources_are_rejected(self):
        train, dev = parts()
        for source in ['hybrid_prediction', 'model', None, 'weak_unknown']:
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, '来源'):
                check_fit_boundary(train.assign(source=source), dev)

    def test_invalid_ids_and_inconsistent_chunk_mapping_fail(self):
        train, dev = parts()
        for broken in [train.assign(chunk_id=None), train.assign(review_id=' '),
                       train.assign(text=''), pd.concat([train, train.assign(text='Other text.')])]:
            with self.subTest(), self.assertRaises(ValueError):
                check_fit_boundary(broken, dev)

    def test_all_public_fit_functions_stop_before_learning_vocabulary(self):
        train, dev = parts()
        contaminated = pd.concat([train, dev], ignore_index=True)
        for fit, module in [(fit_baseline, 'src.classify'), (fit_models, 'src.classification_pipeline'),
                            (fit_strategy, 'scripts.compare_weak_label_strategies')]:
            with self.subTest(module=module), patch(module + '.TfidfVectorizer.fit_transform') as learn:
                with self.assertRaisesRegex(ValueError, '角色'):
                    fit(contaminated, dev)
                learn.assert_not_called()

    def test_old_training_cli_cannot_resplit_role_annotated_labels(self):
        train, dev = parts()
        labeled = pd.concat([train, dev], ignore_index=True)
        with self.assertRaisesRegex(ValueError, '固定开发集'):
            validate_training_data(labeled, labeled, set(), set())


if __name__ == '__main__':
    unittest.main()
