import unittest
import pandas as pd
from src.language_rules import choose_focus,polarity_evidence
from src.subjectivity import weak_decision
from src.subjectivity_legacy import weak_decision as old_decision
from src.classification_pipeline import build_labels,split_labels
from tests.test_classification_pipeline import row,fixture


class LanguageCorrectionsTests(unittest.TestCase):
    def test_journey_does_not_swallow_an_evaluation(self):
        for t in ['I flew from London to Paris and regret it.',
                  "We flew from CDG via DOH to JNB and couldn't have been happier!",
                  'I flew from London to Paris and missed my connection.']:
            self.assertEqual(weak_decision(t)[0],'subjective',t)
        # 连接词后未认识的谓语不应被城市模式吞入；需要模型/人工判断。
        self.assertIsNone(weak_decision('I flew from London to Paris and complained.')[0])
        self.assertEqual(weak_decision('I flew from London to Paris.')[0],'objective')
        self.assertEqual(old_decision('I flew from London to Paris and regret it.')[0],'objective')

    def test_fact_money_miles_and_compound_itinerary(self):
        for t in ['The fine was 50 dollars.','I earned 500 miles on this flight.',
                  'The crew cleaned the toilet after takeoff.',
                  'The flight departed from Singapore at nine in the morning and arrived in London later that day.']:
            self.assertEqual(weak_decision(t)[0],'objective',t)
        self.assertEqual(weak_decision('The seat was fine.')[0],'subjective')

    def test_negation_scope_and_fixed_idioms(self):
        cases={'No issues with my hand baggage.':'positive',
               'There were no problems with our checked baggage.':'positive',
               'The seats were not particularly comfortable.':'negative',
               'The seat was not bad.':'positive',
               'No entertainment was available.':'negative',
               'We did not lose our luggage.':'positive',
               "We couldn't have been happier.":'positive',
               "It couldn't be worse.":'negative',
               'The seat was not only comfortable but also spacious.':'positive',
               'I cannot recommend this airline.':'negative',
               'The flight was not delayed.':'positive'}
        for t,p in cases.items():self.assertEqual(polarity_evidence(t)[0],p,t)
        self.assertIsNone(polarity_evidence('The seat was not necessarily bad.')[0])
        self.assertIsNone(polarity_evidence('The food was good and bad.')[0])

    def test_contrast_focus_same_part_for_aspect_and_polarity(self):
        t='The seat was comfortable and the cabin crew were friendly, but the meal was cold, tasteless and almost impossible to eat.'
        f=choose_focus(t);self.assertEqual(f['aspect'],'food');self.assertEqual(polarity_evidence(f['text'])[0],'negative')
        f=choose_focus('The food was terrible but the seat was comfortable.')
        self.assertEqual(f['aspect'],'seat');self.assertEqual(polarity_evidence(f['text'])[0],'positive')
        f=choose_focus('Although the ticket seemed expensive, it was excellent value for money.')
        self.assertEqual(f['aspect'],'value');self.assertEqual(polarity_evidence(f['text'])[0],'positive')

    def test_no_contrast_r16_first_negative_and_three(self):
        self.assertEqual(choose_focus('The seat was good and the food was good.')['aspect'],'seat')
        self.assertEqual(choose_focus('The seat was good and the food was terrible.')['aspect'],'food')
        self.assertEqual(choose_focus('The seat was good, the food was good and the cabin crew were helpful.')['aspect'],'overall')
        self.assertEqual(choose_focus('The seat and food were good.')['reason'],'unresolved_multi_aspect')

    def test_rating_conflict_abstains_without_changing_source_or_aspect(self):
        source=pd.DataFrame([row('conflict','The seats were not particularly comfortable.',r_seat=5),
                             row('keep','The seat was comfortable.',r_seat=4)])
        before=source.copy(deep=True);out,r=build_labels(source,set(),set());out=out.set_index('chunk_id')
        self.assertTrue(pd.isna(out.loc['conflict','polarity']))
        self.assertEqual(out.loc['conflict','aspect'],'seat')
        self.assertEqual(out.loc['conflict','polarity_rule'],'rating_text_conflict_abstain')
        self.assertEqual(r['rating_text_conflicts_abstained'],1)
        pd.testing.assert_frame_equal(before,source)

    def test_reference_split_not_new_random_split(self):
        labels=fixture();train,dev=split_labels(labels)
        ref=pd.concat([train.assign(split='train'),dev.assign(split='dev')])[['chunk_id','review_id','split']]
        new_train,new_dev=split_labels(labels.sample(frac=1,random_state=7),seed=99,reference=ref)
        self.assertEqual(set(train.chunk_id),set(new_train.chunk_id));self.assertEqual(set(dev.chunk_id),set(new_dev.chunk_id))
        bad=ref.copy();bad.loc[bad.index[0],'split']='dev'
        with self.assertRaises(ValueError):split_labels(labels,reference=bad)


if __name__=='__main__':unittest.main()
