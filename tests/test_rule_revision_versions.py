import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.classification_pipeline import TASKS, VERSION, V3_VERSION, build_labels, predict_texts
from scripts.build_c_rule_revision import build, make_revision_bundle


class Vectorizer:
    def transform(self, texts):
        return np.zeros((len(texts), 1))


class Model:
    def __init__(self, label):
        self.label = label

    def predict(self, matrix):
        return np.array([self.label] * len(matrix))


def bundle(version):
    return {'version': version, 'tasks': TASKS, 'vectorizer': Vectorizer(),
            'models': {'aspect': Model('overall'), 'subjectivity': Model('subjective'),
                       'polarity': Model('positive')}}


class RuleVersionTests(unittest.TestCase):
    def test_new_weak_label_source_names_actual_context_v4(self):
        frame = pd.DataFrame([dict(chunk_id='ground1', review_id='ground_parent',
            text='The ground-crew were rude.', is_eval=False, is_route_only=False,
            is_non_english=False, is_fragment=False, overall_rating=1,
            **{'r_'+a: 1 for a in TASKS['aspect'][:6]})])
        labels, _ = build_labels(frame, set(), set())
        self.assertEqual(labels.aspect.tolist(), ['ground'])
        self.assertEqual(labels.aspect_rule.tolist(), ['context_v4_single_aspect'])

    def test_v3_preserved_and_v4_fixes_idiom(self):
        text = "I can't recommend this airline enough."
        old = predict_texts(bundle(V3_VERSION), [text], include_raw=True).iloc[0]
        new = predict_texts(bundle(VERSION), [text], include_raw=True).iloc[0]
        self.assertEqual(old.polarity, 'negative')
        self.assertEqual(new.polarity, 'positive')
        self.assertEqual(old.raw_polarity, new.raw_polarity)
        self.assertEqual(new.polarity_source, 'rule:unambiguous_polarity_cues')

    def test_true_negative_unchanged_in_both_versions(self):
        for version in [V3_VERSION, VERSION]:
            self.assertEqual(predict_texts(bundle(version), ['I cannot recommend this airline.']).polarity.iloc[0], 'negative')

    def test_withdrawn_intermediate_v4_cannot_be_used(self):
        with self.assertRaisesRegex(ValueError, '不是当前八类联合模型'):
            predict_texts(bundle('eight-aspect-subjectivity-polarity-v4'), ['A test sentence.'])

    def test_ground_hyphen_fix_has_separate_version(self):
        text = 'The ground-crew were rude.'
        old = predict_texts(bundle(V3_VERSION), [text]).iloc[0]
        new = predict_texts(bundle(VERSION), [text]).iloc[0]
        self.assertEqual(old.focus_reason, 'unresolved_multi_aspect')
        self.assertEqual(new.focus_reason, 'whole_sentence')

    def test_v1_and_v2_remain_supported(self):
        text = "I can't recommend this airline enough."
        for version in ['eight-aspect-subjectivity-polarity-v1', 'eight-aspect-subjectivity-polarity-v2']:
            self.assertEqual(predict_texts(bundle(version), [text]).polarity.iloc[0], 'positive')

    def test_revision_inherits_weights_and_declares_no_training(self):
        base = bundle(V3_VERSION)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'base'; path.write_bytes(b'fixture')
            revised = make_revision_bundle(base, path, path)
        self.assertEqual(base['version'], V3_VERSION)
        self.assertEqual(revised['version'], VERSION)
        self.assertIs(revised['models'], base['models'])
        self.assertIs(revised['vectorizer'], base['vectorizer'])
        self.assertIs(revised['model_retrained_for_rule_revision'], False)
        self.assertEqual(revised['base_model_version'], V3_VERSION)

    def test_existing_revision_directory_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, '拒绝覆盖'):
                build(SimpleNamespace(output_dir=Path(folder)))


if __name__ == '__main__':
    unittest.main()
