"""宽松/严格对照的合成回归测试；不读取实际评测数据或写训练产物。"""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from scripts.compare_weak_label_strategies import (
    HYPERPARAMETERS, TASKS, assign_frozen_split, common_candidates,
    expand_broad, fit_strategy, select_strict,
)
from src.config import ASPECTS
from src.weak_labels import normalize_text


def row(chunk_id, text, **changes):
    result = dict(chunk_id=chunk_id, review_id='review_' + chunk_id, text=text,
                  is_eval=False, is_route_only=False, is_non_english=False,
                  is_fragment=False, split='train')
    result.update({'r_' + aspect: None for aspect in ASPECTS})
    result.update(changes)
    return result


class WeakStrategyComparisonTests(unittest.TestCase):
    def test_common_safety_filters_and_input_preservation(self):
        frame = pd.DataFrame([
            row('keep', 'The seat was comfortable.', r_seat=5),
            row('gold', 'The meal was tasty.', review_id='heldout', is_eval=True),
            row('sibling', 'The crew was kind.', review_id='heldout'),
            row('duplicate', '  ＴＨＥ   SEAT WAS AWFUL. '),
            row('empty', ' \t\n'),
            row('route', 'SIN to LHR.', is_route_only=True),
            row('language', 'Les repas sont excellents.', is_non_english=True),
            row('fragment', 'and the seat.', is_fragment=True),
        ])
        original = frame.copy(deep=True)
        out = common_candidates(frame, {'heldout'}, {normalize_text('The seat was awful.')})
        self.assertEqual(out.chunk_id.tolist(), ['keep'])
        self.assertEqual(out.columns.tolist(), frame.columns.tolist())
        pd.testing.assert_frame_equal(frame, original)

    def test_unsafe_schema_and_duplicate_ids_are_rejected(self):
        frame = pd.DataFrame([row('a', 'The seat was good.', r_seat=5)])
        with self.assertRaises(ValueError):
            common_candidates(pd.concat([frame, frame]), set(), set())
        for flag in ['is_eval', 'is_route_only', 'is_non_english', 'is_fragment']:
            with self.subTest(flag=flag):
                for value in ['False', None]:
                    broken = frame.copy()
                    broken[flag] = value
                    with self.assertRaises(ValueError):
                        common_candidates(broken, set(), set())
                with self.assertRaises(ValueError):
                    common_candidates(frame.drop(columns=flag), set(), set())

    def test_frozen_dev_duplicates_block_entire_training_parent(self):
        candidates = pd.DataFrame([
            row('train_duplicate', '  THE SEAT WAS GREAT. ', review_id='train_parent', r_seat=5),
            row('train_sibling', 'The meal was good.', review_id='train_parent', r_food=5),
            row('safe', 'The baggage arrived promptly.', review_id='safe_parent', r_ground=5),
            row('dev', 'The seat was great.', review_id='dev_parent', r_seat=5),
            row('unknown', 'The movie was fun.', review_id='new_parent', r_entertainment=5),
        ])
        reference = pd.DataFrame([
            {'chunk_id': 'old_train', 'review_id': 'train_parent', 'split': 'train'},
            {'chunk_id': 'old_safe', 'review_id': 'safe_parent', 'split': 'train'},
            {'chunk_id': 'dev', 'review_id': 'dev_parent', 'split': 'dev'},
        ])
        dev = candidates.loc[candidates.chunk_id.eq('dev')].copy()
        original = candidates.copy(deep=True)
        reference_original = reference.copy(deep=True)
        dev_original = dev.copy(deep=True)
        assigned, info = assign_frozen_split(candidates, reference, dev)
        roles = assigned.set_index('chunk_id').split.to_dict()
        self.assertEqual(roles, {'train_duplicate': 'blocked_dev_duplicate',
                                 'train_sibling': 'blocked_dev_duplicate', 'safe': 'train',
                                 'dev': 'dev', 'unknown': 'unassigned'})
        self.assertEqual(info['blocked_train_reviews'], 1)
        self.assertEqual(info['blocked_train_sentences'], 2)
        self.assertEqual(info['unassigned_reviews'], 1)
        broad = expand_broad(assigned)
        self.assertEqual(set(broad.loc[broad.split.eq('train'), 'chunk_id']), {'safe'})
        self.assertIn('unknown', set(broad.chunk_id))
        pd.testing.assert_frame_equal(candidates, original)
        pd.testing.assert_frame_equal(reference, reference_original)
        pd.testing.assert_frame_equal(dev, dev_original)

    def test_cross_parent_invalid_split_and_nondev_reference_are_rejected(self):
        candidates = pd.DataFrame([row('a', 'The seat was good.', review_id='parent')])
        dev = candidates.copy()
        cross = pd.DataFrame([
            {'chunk_id': 'a', 'review_id': 'parent', 'split': 'train'},
            {'chunk_id': 'b', 'review_id': 'parent', 'split': 'dev'},
        ])
        with self.assertRaises(ValueError):
            assign_frozen_split(candidates, cross, dev)
        for mark in [None, '', 'validation', 'train']:
            reference = cross.iloc[:1].copy()
            reference['split'] = mark
            with self.subTest(mark=mark), self.assertRaises(ValueError):
                assign_frozen_split(candidates, reference, dev)
        reference = cross.iloc[:1].assign(split='dev')
        with self.assertRaises(ValueError):
            assign_frozen_split(candidates, reference, dev.assign(review_id='unknown_parent'))

    def test_broad_preserves_multiaspect_opposite_labels_and_rating_boundaries(self):
        frame = pd.DataFrame([
            row('mixed', 'The seat was good but the food was terrible.',
                r_seat=5, r_food=1, r_staff=3, r_value='bad', r_ground=0,
                r_entertainment=2.5),
            row('valid', 'A factual flight description.', r_seat='2', r_staff=4,
                r_food=6, r_entertainment=-1, r_ground=np.nan),
        ])
        original = frame.copy(deep=True)
        broad = expand_broad(frame)
        actual = {(r.chunk_id, r.aspect): (int(r.rating), r.polarity)
                  for r in broad.itertuples()}
        self.assertEqual(actual, {('mixed', 'seat'): (5, 'positive'),
                                  ('mixed', 'food'): (1, 'negative'),
                                  ('valid', 'seat'): (2, 'negative'),
                                  ('valid', 'staff'): (4, 'positive')})
        self.assertTrue(broad.label_id.is_unique)
        self.assertFalse(broad.duplicated(['chunk_id', 'aspect']).any())
        for record in broad.itertuples():
            self.assertEqual(record.rating_column, 'r_' + record.aspect)
            self.assertEqual(record.label_id, record.chunk_id + '::' + record.aspect)
        self.assertTrue({'source', 'rule_version', 'is_eval', 'split'}.issubset(broad.columns))
        pd.testing.assert_frame_equal(frame, original)

    def test_strict_is_unchanged_single_aspect_subset_with_context_guards(self):
        frame = pd.DataFrame([
            row('seat', 'The seat had ample legroom.', r_seat=5, r_food=1),
            row('multi', 'The seat was good but the meal was terrible.', r_seat=5, r_food=1),
            row('none', 'I flew yesterday.', r_seat=5),
            row('unsupported', 'The seat was dirty.', r_seat=1),
            row('bare_staff', 'The staff were helpful.', r_staff=5),
            row('cabin_staff', 'The cabin staff were helpful.', r_staff=5),
            row('missing_target', 'The meal was excellent.', r_food=3, r_seat=5),
        ])
        broad = expand_broad(frame)
        frame_before, broad_before = frame.copy(deep=True), broad.copy(deep=True)
        strict = select_strict(broad, frame)
        self.assertEqual(set(strict.chunk_id), {'seat', 'cabin_staff'})
        self.assertLess(len(strict), len(broad))
        self.assertTrue(set(strict.label_id).issubset(set(broad.label_id)))
        preserved = ['chunk_id', 'review_id', 'text', 'aspect', 'rating', 'polarity',
                     'rating_column', 'is_eval', 'split']
        expected = broad.set_index('label_id').loc[strict.label_id, preserved].reset_index(drop=True)
        pd.testing.assert_frame_equal(strict[preserved].reset_index(drop=True), expected)
        pd.testing.assert_frame_equal(frame, frame_before)
        pd.testing.assert_frame_equal(broad, broad_before)

    def test_training_duplicate_conflicts_are_kept_and_sentiment_does_not_rewrite_ratings(self):
        frame = pd.DataFrame([
            row('a', 'The seat was awful.', r_seat=5),
            row('b', ' THE SEAT WAS AWFUL. ', r_seat=1),
            row('c', 'The seat was awful.', r_seat=5),
        ])
        broad = expand_broad(frame)
        strict = select_strict(broad, frame)
        self.assertEqual(len(broad), 3)
        self.assertEqual(len(strict), 3)
        self.assertEqual(strict.set_index('chunk_id').polarity.to_dict(),
                         {'a': 'positive', 'b': 'negative', 'c': 'positive'})
        self.assertEqual(strict.text.tolist(), frame.text.tolist())

    def test_weighted_compression_matches_explicit_repeated_training_and_excludes_dev_vocabulary(self):
        aspects = list(ASPECTS)
        raw = []
        for index, aspect in enumerate(aspects):
            for repetition, rating in enumerate([1, 4, 5]):
                sentiment = 'terrible' if rating == 1 else 'excellent'
                changes = {'r_' + aspect: rating,
                           'r_' + aspects[(index + 1) % len(aspects)]: rating,
                           'r_' + aspects[(index + 2) % len(aspects)]: 5 if rating == 1 else 1}
                raw.append(row(f'train_{index}_{repetition}',
                               f'The {aspect} was {sentiment} on flight number {index}{repetition}.',
                               **changes))
        train = expand_broad(pd.DataFrame(raw))
        dev = pd.DataFrame([
            {'chunk_id': f'dev_{index}_{polarity}', 'review_id': f'dev_parent_{index}_{polarity}',
             'text': f'The {aspect} was {sentiment} devexclusivevocabulary.',
             'aspect': aspect, 'polarity': polarity}
            for index, aspect in enumerate(aspects)
            for polarity, sentiment in [('negative', 'terrible'), ('positive', 'excellent')]
        ])
        train_original, dev_original = train.copy(deep=True), dev.copy(deep=True)
        with contextlib.redirect_stdout(io.StringIO()):
            bundle, metrics, predictions = fit_strategy(train, dev)
        vectorizer = bundle['vectorizer']
        self.assertNotIn('devexclusivevocabulary', vectorizer.vocabulary_)
        self.assertFalse(any('devexclusivevocabulary' in key for key in vectorizer.vocabulary_))
        # 只复用被测模型的同一 TF-IDF，隔离检验计数加权是否等价于逐条标签记录。
        x_train = vectorizer.transform(train.text)
        x_dev = vectorizer.transform(dev.text)
        hp = HYPERPARAMETERS
        for task in TASKS:
            explicit = LogisticRegression(C=hp['C'], solver=hp['solver'], max_iter=hp['max_iter'],
                                          random_state=hp['seed'], class_weight=hp['class_weight'])
            with threadpool_limits(limits=1):
                explicit.fit(x_train, train[task])
            weighted = bundle['models'][task]
            np.testing.assert_array_equal(weighted.classes_, explicit.classes_)
            np.testing.assert_allclose(weighted.coef_, explicit.coef_, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(weighted.intercept_, explicit.intercept_, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(weighted.predict_proba(x_dev), explicit.predict_proba(x_dev),
                                       atol=1e-5, rtol=1e-5)
            # 分数精确打平时，浮点微差可能改变 argmax；只比较有清晰间隔的预测。
            probabilities = explicit.predict_proba(x_dev)
            margin = np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2]
            clear = margin > 1e-3
            np.testing.assert_array_equal(predictions.loc[clear, 'predicted_' + task], explicit.predict(x_dev)[clear])
            self.assertEqual(metrics[task]['total_loss_weight'], len(train))
        self.assertLess(metrics['polarity']['weighted_training_rows'], len(train))
        pd.testing.assert_frame_equal(train, train_original)
        pd.testing.assert_frame_equal(dev, dev_original)


if __name__ == '__main__':
    unittest.main()
