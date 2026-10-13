"""审计修复的有限句式和方面回归；仅用合成文本，不训练或读取项目数据。"""
import unittest

import pandas as pd

from src.config import ASPECTS
from src.language_rules import aspect_hits, choose_focus, polarity_evidence
from src.weak_label_rules import match_series, rule_description
from src.weak_labels import build_weak_labels


def detected(text, ruleset='context_v4'):
    matches, blocked = match_series(pd.Series([text]), ruleset=ruleset)
    return {aspect for aspect in matches if matches.at[0, aspect]}, bool(blocked.iloc[0])


def row(chunk_id, text, **changes):
    result = {'chunk_id': chunk_id, 'review_id': 'review_' + chunk_id, 'text': text,
              'is_eval': False, 'is_route_only': False, 'is_non_english': False,
              'is_fragment': False}
    result.update({'r_' + aspect: None for aspect in ASPECTS})
    result.update(changes)
    return result


class RecommendationIdiomTests(unittest.TestCase):
    def test_bounded_strong_recommendations_are_positive(self):
        examples = [
            "I can't recommend this airline enough.",
            'I cannot recommend Singapore Airlines highly enough.',
            "I couldn't recommend them more highly!",
            'I could not recommend the cabin crew any higher.',
            "I can't recommend it strongly enough.",
            'I cannot recommend it too highly.',
            'CAN’T RECOMMEND THEM ENOUGH!',
            "Couldn't recommend the service higher.",
            'I cannot recommend this airline any more highly.',
            "I can't recommend enough.",
            "I can't recommend them enough for families.",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(polarity_evidence(text)[0], 'positive')

    def test_literal_inability_and_amounts_remain_negative(self):
        examples = [
            'I cannot recommend this airline.',
            "I can't recommend this airline to anyone.",
            "I couldn't recommend it after that delay.",
            "I can't recommend it highly.",
            "I can't recommend this airline any more.",
            "I can't recommend any higher fares.",
            "I can't recommend enough flights for a holiday.",
            "I can't recommend it because the seat wasn't comfortable enough.",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(polarity_evidence(text)[0], 'negative')

    def test_enough_in_another_sentence_or_clause_does_not_flip_recommend(self):
        examples = [
            'I cannot recommend this airline. Enough is enough.',
            'I cannot recommend this airline! Enough said.',
            'I cannot recommend this airline; enough said.',
            'I cannot recommend this airline: enough problems remain.',
            'I cannot recommend this airline.\nEnough said.',
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(polarity_evidence(text)[0], 'negative')
        for text in ["I can't recommend it after the delay enough.",
                     "I can't recommend it, however good enough the meal was."]:
            with self.subTest(text=text):
                self.assertNotEqual(polarity_evidence(text)[0], 'positive')

    def test_real_negative_evidence_is_not_swallowed_by_the_idiom(self):
        self.assertIsNone(polarity_evidence(
            "I can't recommend this airline enough, but the food was terrible.")[0])
        self.assertNotEqual(polarity_evidence(
            "I can't recommend this terrible airline enough.")[0], 'positive')

    def test_audited_avoidance_recommendation_is_negative(self):
        # 全库非评测句审查中的真实边界：建议远离，程度 enough 不代表赞扬。
        text = 'I cannot recommend enough to keep away from this airline.'
        self.assertEqual(polarity_evidence(text)[0], 'negative')
        for text in [
            "I can't recommend this airline enough to stay away from them.",
            "I couldn't recommend them highly enough to steer clear of this carrier.",
            'I could not recommend enough to avoid this airline.',
            'I cannot recommend enough to avoid flying with this airline.',
            "I can't recommend them enough to never fly with them.",
            "I couldn't recommend enough to not book this airline.",
        ]:
            with self.subTest(text=text):
                self.assertEqual(polarity_evidence(text)[0], 'negative')

    def test_recommend_avoiding_in_the_object_is_not_positive(self):
        for text in [
            'I cannot recommend avoiding this airline enough.',
            "I can't recommend avoiding them highly enough.",
            'I could not recommend steering clear of this airline more highly.',
            "I couldn't recommend staying away from this carrier enough.",
        ]:
            with self.subTest(text=text):
                self.assertEqual(polarity_evidence(text)[0], 'negative')

    def test_avoidance_with_an_unclear_target_abstains(self):
        for text in [
            "I can't recommend them enough to avoid delays.",
            'I cannot recommend avoiding delays enough.',
            'I cannot recommend enough to steer clear of trouble.',
            'I cannot recommend enough to keep away.',
        ]:
            with self.subTest(text=text):
                self.assertIsNone(polarity_evidence(text)[0])
        self.assertIsNone(polarity_evidence(
            "I can't recommend avoiding this airline enough, but the crew were excellent.")[0])

    def test_recommendation_to_people_and_for_families_stays_positive(self):
        for text in [
            'I cannot recommend this airline enough to others.',
            "I can't recommend it enough to anyone.",
            'I cannot recommend this airline enough for families.',
            "I couldn't recommend them higher for families.",
            'I cannot recommend this airline enough to keep our family entertained.',
        ]:
            with self.subTest(text=text):
                self.assertEqual(polarity_evidence(text)[0], 'positive')

    def test_existing_negation_and_special_cases_are_preserved(self):
        examples = {
            'The seat was not bad.': 'positive',
            'The seat was not particularly comfortable.': 'negative',
            "We couldn't have been happier.": 'positive',
            "It couldn't be worse.": 'negative',
            'No issues with my hand baggage.': 'positive',
            'No entertainment was available.': 'negative',
            'The flight was not delayed.': 'positive',
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(polarity_evidence(text)[0], expected)
        self.assertIsNone(polarity_evidence('The seat was not necessarily bad.')[0])
        self.assertIsNone(polarity_evidence('The food was good and bad.')[0])


class GroundCrewAuditTests(unittest.TestCase):
    def test_space_hyphen_and_unicode_variants_have_one_ground_aspect(self):
        examples = [
            'The ground crew were rude.',
            'The ground   crew were rude.',
            'The ground-crew were rude.',
            'The Ground‑Crew were rude.',
            'The ground–crew were rude.',
            'The ground—crew were rude.',
            'The ground − crew were rude.',
            'The ground - crew were rude.',
            'The ground\t-\tcrew were rude.',
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(detected(text), ({'ground'}, False))
                self.assertEqual(aspect_hits(text), {'ground'})

    def test_frozen_v3_still_exposes_its_original_hyphen_difference(self):
        text = 'The ground-crew were rude.'
        self.assertEqual(detected(text, 'context_v3')[0], {'staff', 'ground'})
        self.assertEqual(detected(text, 'context_v4')[0], {'ground'})
        self.assertEqual(detected('The ground crew were rude.', 'context_v3')[0], {'ground'})
        v3, v4 = rule_description('context_v3'), rule_description('context_v4')
        self.assertNotEqual(v3['patterns']['staff'], v4['patterns']['staff'])
        self.assertNotEqual(v3['policy'], v4['policy'])

    def test_other_crews_and_separate_cabin_mentions_still_match(self):
        examples = {
            'The cabin crew were helpful.': {'staff'},
            'The crew were friendly.': {'staff'},
            'The underground crew were helpful.': {'staff'},
            'Ground. Crew were helpful.': {'staff'},
            'The ground-crew and cabin crew were rude.': {'ground', 'staff'},
            'The airport staff and cabin crew were helpful.': {'ground', 'staff'},
            'The ground floor had a lounge.': set(),
            'The ground-crew and meals were poor.': {'ground', 'food'},
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(detected(text)[0], expected)
                self.assertEqual(aspect_hits(text), expected)

    def test_strict_filter_keeps_ground_rating_and_original_text(self):
        texts = ['The ground crew were rude.', 'The ground-crew were rude.',
                 'The Ground—Crew were rude.', 'The ground - crew were rude.']
        source = pd.DataFrame([row(str(i), text, r_ground=5, r_staff=1)
                               for i, text in enumerate(texts)])
        original = source.copy(deep=True)
        labels, report = build_weak_labels(source, set(), set(), ruleset='context_v4')
        self.assertEqual(labels.chunk_id.tolist(), ['0', '1', '2', '3'])
        self.assertEqual(labels.aspect.tolist(), ['ground'] * 4)
        self.assertEqual(labels.rating_column.tolist(), ['r_ground'] * 4)
        self.assertEqual(labels.rating.tolist(), [5] * 4)
        self.assertEqual(labels.polarity.tolist(), ['positive'] * 4)
        self.assertEqual(labels.text.tolist(), texts)
        self.assertEqual(report['excluded_by_first_reason']['multiple_aspects'], 0)
        pd.testing.assert_frame_equal(source, original)
        old, old_report = build_weak_labels(source.iloc[:2].copy(), set(), set(), ruleset='context_v3')
        self.assertEqual(old.chunk_id.tolist(), ['0'])
        self.assertEqual(old_report['excluded_by_first_reason']['multiple_aspects'], 1)

    def test_strict_filter_does_not_substitute_staff_rating_or_keep_multiaspect(self):
        source = pd.DataFrame([
            row('keep', 'The ground-crew were helpful.', r_ground=4, r_staff=1),
            row('missing', 'The ground-crew were rude.', r_staff=1),
            row('neutral', 'The ground-crew were polite.', r_ground=3, r_staff=5),
            row('multi', 'The ground-crew and meals were poor.', r_ground=1, r_food=1),
        ])
        labels, report = build_weak_labels(source, set(), set(), ruleset='context_v4')
        self.assertEqual(labels.chunk_id.tolist(), ['keep'])
        self.assertEqual(report['excluded_by_first_reason']['rating_missing'], 1)
        self.assertEqual(report['excluded_by_first_reason']['rating_three'], 1)
        self.assertEqual(report['excluded_by_first_reason']['multiple_aspects'], 1)

    def test_existing_context_and_contrast_guards_remain_available(self):
        self.assertEqual(detected('The website screen froze during payment.')[0], set())
        self.assertEqual(detected('The seatback screen offered films.')[0], {'entertainment'})
        self.assertEqual(detected('Customer service refused the refund.')[0], {'value'})
        self.assertTrue(detected('The seat was dirty.')[1])
        focus = choose_focus('The cabin crew were friendly but the ground-crew were rude.')
        self.assertEqual(focus['aspect'], 'ground')
        self.assertEqual(polarity_evidence(focus['text'])[0], 'negative')


if __name__ == '__main__':
    unittest.main()
