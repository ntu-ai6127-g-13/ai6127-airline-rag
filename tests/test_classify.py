"""C2 的关键行为：隔离、可复现切分、词表泄漏、保存后预测。"""
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.train_classifiers import (TASKS, fit_baseline, main, predict_texts,
                          split_by_review, validate_training_data)


def fixture():
    rows = []
    for group in range(20):
        for aspect in TASKS['aspect']:
            for polarity in TASKS['polarity']:
                rows.append(dict(chunk_id=f'{group:03d}_{aspect}_{polarity}',
                                 review_id=f'R{group:03d}',
                                 text=f'{aspect} was {polarity} token{group}',
                                 is_eval=False, aspect=aspect, polarity=polarity, source='weak'))
    return pd.DataFrame(rows)


class BaselineTests(unittest.TestCase):
    def test_group_split_is_disjoint_complete_and_order_independent(self):
        df = fixture()
        train, dev = split_by_review(df)
        other_train, other_dev = split_by_review(df.sample(frac=1, random_state=8))
        self.assertFalse(set(train.review_id) & set(dev.review_id))
        self.assertEqual(len(train) + len(dev), len(df))
        self.assertEqual(set(train.chunk_id), set(other_train.chunk_id))
        self.assertEqual(set(dev.chunk_id), set(other_dev.chunk_id))

    def test_eval_parent_or_duplicate_eval_text_is_rejected_even_with_false_flag(self):
        df = fixture()
        with self.assertRaisesRegex(ValueError, '所属评论'):
            validate_training_data(df, df, {'R000'}, set())
        with self.assertRaisesRegex(ValueError, '重复评测文本'):
            validate_training_data(df, df, set(), {df.text.iloc[0]})
        marked = df.copy()
        marked.loc[0, 'is_eval'] = True
        with self.assertRaisesRegex(ValueError, 'is_eval'):
            validate_training_data(marked, df, set(), set())

    def test_mismatched_kb_text_parent_or_flag_is_rejected(self):
        df = fixture()
        for col, value in [('text', 'changed original text'), ('review_id', 'different'), ('is_eval', True)]:
            with self.subTest(col=col):
                kb = df.copy()
                kb.loc[0, col] = value
                with self.assertRaisesRegex(ValueError, '知识库'):
                    validate_training_data(df, kb, set(), set())

    def test_duplicate_text_and_missing_classes_fail_closed(self):
        df = fixture()
        df.loc[1, 'text'] = '  ' + df.text.iloc[0].upper() + '  '
        with self.assertRaisesRegex(ValueError, '重复文本'):
            validate_training_data(df, df, set(), set())
        with self.assertRaisesRegex(ValueError, '类别不齐'):
            split_by_review(fixture().query("aspect == 'seat'"))

    def test_vocabulary_excludes_dev_only_words_and_model_roundtrip(self):
        train, dev = split_by_review(fixture())
        train = train.copy()
        train['rating'] = 'TRAIN_METADATA_NOT_A_FEATURE'
        dev = dev.copy()
        dev['text'] = dev.text + ' uniquedevonlymarker'
        bundle, metrics, predictions = fit_baseline(train, dev)
        vocabulary = bundle['vectorizer'].vocabulary_
        self.assertNotIn('uniquedevonlymarker', vocabulary)
        self.assertNotIn('train_metadata_not_a_feature', vocabulary)
        self.assertEqual(len(predictions), len(dev))
        self.assertEqual(sum(metrics['aspect']['confusion_matrix'][0]),
                         int(dev.aspect.eq(TASKS['aspect'][0]).sum()))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'model.joblib'
            joblib.dump(bundle, p)
            pd.testing.assert_frame_equal(predict_texts(bundle, dev.text.tolist()),
                                          predict_texts(joblib.load(p), dev.text.tolist()))

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / 'sentinel.txt'
            sentinel.write_text('keep')
            result = main(['train', '--input', 'nonexistent.parquet', '--output-dir', tmp])
            self.assertEqual(result, 1)
            self.assertEqual(sentinel.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
