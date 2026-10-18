"""C1 的行为与隔离测试；不访问真实人工标签，不写项目数据。"""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.weak_labels import (ASPECTS, build_weak_labels, load_test_exclusions,
                             main, normalize_text, validate_eval_flags)


def row(cid, text, **changes):
    item = dict(chunk_id=cid, review_id="review_" + cid, text=text,
                is_eval=False, is_route_only=False, is_non_english=False, is_fragment=False)
    item.update({"r_" + a: 5 for a in ASPECTS})
    item.update(changes)
    return item


class WeakLabelsTests(unittest.TestCase):
    def build(self, rows, reviews=None, texts=None):
        return build_weak_labels(pd.DataFrame(rows), reviews or set(), texts or set())

    def test_rating_boundaries_and_accounting(self):
        rows = [row(str(i), f"The seat was comfortable on flight {i}.", r_seat=score)
                for i, score in enumerate([1, 2, 3, 4, 5, None, 0, 6, 2.5, "bad"])]
        out, report = self.build(rows)
        self.assertEqual(dict(zip(out.chunk_id, out.polarity)),
                         {"0": "negative", "1": "negative", "3": "positive", "4": "positive"})
        self.assertEqual(report['excluded_by_first_reason']['rating_invalid'], 4)
        self.assertEqual(report['excluded_by_first_reason']['rating_missing'], 1)
        self.assertTrue(all(report['checks'].values()))
        self.assertNotIn('subjectivity', out)

    def test_corresponding_score_not_other_aspect_score(self):
        out, _ = self.build([row('a', 'The food was bad.', r_food=1, r_seat=5)])
        self.assertEqual(out.iloc[0].polarity, 'negative')
        self.assertEqual(out.iloc[0].rating_column, 'r_food')

    def test_excludes_entire_test_parent_and_normalized_test_text(self):
        rows = [row('a', 'The seat was awful.', review_id='heldout'),
                row('b', 'The food was nice.', review_id='heldout'),
                row('c', '  THE   SEAT WAS FINE. '), row('d', 'The meal was lovely.')]
        out, r = self.build(rows, {'heldout'}, {normalize_text('The seat was fine.')})
        self.assertEqual(out.chunk_id.tolist(), ['d'])
        self.assertEqual(r['excluded_by_first_reason']['test_review'], 2)
        self.assertEqual(r['excluded_by_first_reason']['test_text'], 1)

    def test_multiple_keywords_in_one_aspect_are_allowed(self):
        out, r = self.build([row('a', 'The SEAT had ample legroom.'),
                             row('b', 'The seat was good but the food was bad.'),
                             row('c', 'The moviegoer was seated yesterday.')])
        self.assertEqual(out.chunk_id.tolist(), ['a'])
        self.assertEqual(r['excluded_by_first_reason']['multiple_aspects'], 1)
        self.assertEqual(r['excluded_by_first_reason']['no_aspect'], 1)

    def test_quality_flags_and_empty_text(self):
        out, r = self.build([row('a', 'The seat was fine.'), row('b', ''),
                             row('c', 'The seat was good.', is_route_only=True),
                             row('d', 'The food was good.', is_non_english=True),
                             row('e', 'The meal was good.', is_fragment=True)])
        self.assertEqual(out.chunk_id.tolist(), ['a'])
        for name in ['empty_text', 'route_only', 'non_english', 'fragment']:
            self.assertEqual(r['excluded_by_first_reason'][name], 1)

    def test_old_schema_uses_real_quality_functions_without_mutating_input(self):
        frame = pd.DataFrame([row('a', 'The seat was comfortable.'),
                              row('b', ', the food was terrible.'),
                              row('c', 'Les bagages arrivent à temps et intacts !')])
        frame = frame.drop(columns=['is_fragment', 'is_non_english'])
        original = frame.copy(deep=True)
        out, r = build_weak_labels(frame, set(), set())
        self.assertEqual(out.chunk_id.tolist(), ['a'])
        self.assertEqual(set(r['quality_flags_derived_in_memory']), {'is_fragment', 'is_non_english'})
        self.assertEqual(r['excluded_by_first_reason']['non_english'], 1)
        self.assertEqual(r['excluded_by_first_reason']['fragment'], 1)
        pd.testing.assert_frame_equal(frame, original)

    def test_conflicting_duplicate_labels_removed_all_other_duplicates_once(self):
        rows = [row('a', 'The seat was fine.', r_seat=1),
                row('b', ' THE SEAT WAS FINE. ', r_seat=5),
                row('z', 'The food was tasty.'), row('c', 'The food was tasty.')]
        out, r = self.build(rows)
        self.assertEqual(out.chunk_id.tolist(), ['c'])
        self.assertEqual(r['excluded_by_first_reason']['duplicate_label_conflict'], 2)
        self.assertEqual(r['excluded_by_first_reason']['duplicate_text'], 1)

    def test_repeat_and_reordered_input_give_identical_results(self):
        frame = pd.DataFrame([row('b', 'The food was tasty.'), row('a', 'The seat was nice.')])
        a, _ = build_weak_labels(frame, set(), set())
        b, _ = build_weak_labels(frame.iloc[::-1], set(), set())
        pd.testing.assert_frame_equal(a, b)

    def test_duplicate_ids_missing_columns_invalid_flags_fail(self):
        with self.assertRaisesRegex(ValueError, '重复'):
            self.build([row('a', 'The food was good.'), row('a', 'The seat was good.')])
        with self.assertRaisesRegex(ValueError, '必需列'):
            build_weak_labels(pd.DataFrame([row('a', 'Food was good.')]).drop(columns=['r_food']), set(), set())
        with self.assertRaisesRegex(ValueError, '布尔'):
            self.build([row('a', 'Food was good.', is_route_only='False')])

    def test_no_valid_rows_fails(self):
        with self.assertRaisesRegex(ValueError, '没有符合'):
            self.build([row('a', 'The food was good.', r_food=3)])

    def test_gold_mapping_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            kb = pd.DataFrame([row(str(i), f'The seat was fine on flight {i}.', is_eval=True)
                               for i in range(1100)])
            meta = kb[['chunk_id', 'review_id']].copy()
            meta['id'] = ['E' + str(i) for i in range(1100)]
            gold = pd.DataFrame({'id': meta.id, 'text': kb.text})
            meta.to_csv(p/'meta.csv', index=False)
            gold.to_csv(p/'gold.csv', index=False)
            reviews, texts, info = load_test_exclusions(kb, p/'meta.csv', p/'gold.csv')
            self.assertEqual(len(reviews), 1100)
            self.assertEqual(len(texts), 1100)
            self.assertTrue(info['mapping_verified'])
            self.assertEqual(info['marked_sentences'], 1100)
            gold.loc[0, 'text'] = 'Wrong text'
            gold.to_csv(p/'gold.csv', index=False)
            with self.assertRaisesRegex(ValueError, '不一致'):
                load_test_exclusions(kb, p/'meta.csv', p/'gold.csv')

    def test_is_eval_excluded_before_parent_siblings(self):
        rows = [row('gold', 'The seat was fine.', review_id='heldout', is_eval=True),
                row('sibling', 'The food was fine.', review_id='heldout'),
                row('train', 'The meal was fine.')]
        out, report = self.build(rows, {'heldout'})
        self.assertEqual(out.chunk_id.tolist(), ['train'])
        self.assertFalse(out.is_eval.any())
        self.assertEqual(report['excluded_by_first_reason']['test_sentence'], 1)
        self.assertEqual(report['excluded_by_first_reason']['test_review'], 1)

    def test_same_count_wrong_eval_ids_rejected(self):
        kb = pd.DataFrame([row('a', 'A seat.', is_eval=False), row('b', 'Food.', is_eval=True)])
        with self.assertRaisesRegex(ValueError, '2 行不一致'):
            validate_eval_flags(kb, {'a'})

    def test_stale_1000_flags_with_1100_eval_ids_rejected(self):
        kb = pd.DataFrame([row(str(i), f'Seat {i}.', is_eval=i < 1000) for i in range(1100)])
        with self.assertRaisesRegex(ValueError, '100 行不一致'):
            validate_eval_flags(kb, set(kb.chunk_id))

    def test_eval_missing_null_string_and_unknown_ids_rejected(self):
        kb = pd.DataFrame([row('a', 'A seat.', is_eval=True)])
        with self.assertRaisesRegex(ValueError, '缺少 is_eval'):
            validate_eval_flags(kb.drop(columns=['is_eval']), {'a'})
        with self.assertRaisesRegex(ValueError, '未全部出现'):
            validate_eval_flags(kb, {'unknown'})
        for value in [None, 'False']:
            bad = kb.copy()
            bad['is_eval'] = value
            with self.assertRaisesRegex(ValueError, '布尔'):
                validate_eval_flags(bad, {'a'})

    def test_existing_output_and_input_alias_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'existing.parquet'
            p.write_bytes(b'original')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--output', str(p)]), 1)
                self.assertEqual(main(['--input', str(p), '--output', str(p), '--overwrite']), 1)
            self.assertEqual(p.read_bytes(), b'original')


if __name__ == '__main__':
    unittest.main()
