import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from src.classification_pipeline import (TASKS, VERSION, build_labels, split_labels,
                                          fit_models, predict_texts, train_command)
from src.subjectivity import weak_decision
from src.train_classifiers import predict_texts as public_predict


def row(id, text, **kw):
    r = dict(chunk_id=id, review_id='review_'+id, text=text, is_eval=False,
             is_route_only=False, is_fragment=False, is_non_english=False,
             overall_rating=9, **{'r_'+a: 4 for a in TASKS['aspect'][:6]})
    r.update(kw)
    return r


def fixture():
    rows = []
    for n in range(20):
        for a in TASKS['aspect']:
            for s in TASKS['subjectivity']:
                p = None if s == 'objective' else ('positive' if n % 2 else 'negative')
                rows.append(dict(chunk_id=f'{n:03}_{a}_{s}', review_id=f'r{n}',
                                 text=f'{a} is {s} {p} distinct{n}',
                                 aspect=a, subjectivity=s, polarity=p))
    return pd.DataFrame(rows)


class FixedModel:
    def __init__(self, labels):
        self.labels = labels
        self.calls = []
    def predict(self, x):
        self.calls.append(x.shape[0])
        return np.array(self.labels[:x.shape[0]])


class PipelineTests(unittest.TestCase):
    def test_fact_opinion_and_abstention_are_distinct(self):
        for text in ['The ticket was 800 SGD.', 'The aircraft was a Boeing 777.',
                     'The flight departed at 9am.', 'The seats were in a 3-3-3 configuration.']:
            self.assertEqual(weak_decision(text)[0], 'objective', text)
        for text in ['The seats were not comfortable.', 'The flight was delayed 3 hours.',
                     'No issues with my baggage.', 'The flight arrived on time.']:
            self.assertEqual(weak_decision(text)[0], 'subjective', text)
        self.assertIsNone(weak_decision('It was there when we arrived.')[0])

    def test_overall_and_other_do_not_become_catch_all(self):
        df = pd.DataFrame([row('overall','Overall a great experience.'),
                           row('other','The aircraft was a Boeing 777.'),
                           row('lav','The toilet was dirty.'),
                           row('clean','The cabin was clean.'),
                           row('unknown','It was there when we arrived.'),
                           row('offtopic','I went to a wedding.')])
        out,_ = build_labels(df,set(),set());out=out.set_index('chunk_id')
        self.assertEqual(out.loc['overall','aspect'],'overall')
        for id in ['other','lav','clean','offtopic']:
            self.assertEqual(out.loc[id,'aspect'],'other')
            self.assertTrue(pd.isna(out.loc[id,'polarity']))
        self.assertNotIn('unknown',out.index)
        self.assertEqual(out.loc['offtopic','subjectivity'],'objective')

    def test_overall_uses_ten_point_scale_and_abstains_in_middle(self):
        df=pd.DataFrame([row(str(v),f"Overall a {'bad' if v in [1,4] else 'great'} experience number {v}.",overall_rating=v)
                         for v in [0,1,4,5,6,7,10,11]])
        out,_=build_labels(df,set(),set());out=out.set_index('chunk_id')
        for v in [1,4]:self.assertEqual(out.loc[str(v),'polarity'],'negative')
        for v in [7,10]:self.assertEqual(out.loc[str(v),'polarity'],'positive')
        for v in [0,5,6,11]:self.assertTrue(pd.isna(out.loc[str(v),'polarity']))

    def test_objective_and_other_never_inherit_polarity_rating(self):
        df=pd.DataFrame([row('price','The ticket was 800 SGD.',r_value=1),
                         row('other','The toilet was awful.',overall_rating=10),
                         row('seat','The seat was comfortable.',r_seat=4)])
        out,_=build_labels(df,set(),set());out=out.set_index('chunk_id')
        self.assertTrue(pd.isna(out.loc['price','polarity']))
        self.assertTrue(pd.isna(out.loc['other','polarity']))
        self.assertEqual(out.loc['seat','polarity'],'positive')

    def test_mixed_topics_abstain_and_specific_beats_overall(self):
        df=pd.DataFrame([row('mixed','The meal was good but the toilet was dirty.'),
                         row('specific','Overall the seat was comfortable.'),
                         row('clean','The plane was dirty with food stuck to the seats.')])
        out,_=build_labels(df,set(),set());out=out.set_index('chunk_id')
        self.assertTrue(pd.isna(out.loc['mixed','aspect']))
        self.assertEqual(out.loc['specific','aspect'],'seat')
        self.assertEqual(out.loc['clean','aspect'],'other')

    def test_eval_parents_duplicate_eval_text_and_quality_excluded(self):
        df=pd.DataFrame([row('keep','Overall a great experience.'),
                         row('gold','The seat was good.',is_eval=True),
                         row('sibling','The meal was bad.',review_id='gold_parent'),
                         row('duplicate','THE CABIN WAS DIRTY.'),
                         row('noise','A great flight.',is_fragment=True)])
        out,_=build_labels(df,{'gold_parent'},{'the cabin was dirty.'})
        self.assertEqual(out.chunk_id.tolist(),['keep'])
        df['is_eval']=df.is_eval.astype(object)
        df.loc[0,'is_eval']='False'
        with self.assertRaises(ValueError):build_labels(df,set(),set())

    def test_conflicting_duplicate_scores_removed(self):
        df=pd.DataFrame([row('a','The seat was good.',r_seat=1),
                         row('b','The seat was good.',r_seat=5),
                         row('c','The meal was awful.')])
        out,r=build_labels(df,set(),set())
        self.assertEqual(out.chunk_id.tolist(),['c'])
        self.assertEqual(r['conflicting_duplicate_rows'],2)

    def test_all_tasks_share_disjoint_reproducible_split(self):
        x=fixture();a,b=split_labels(x);c,d=split_labels(x.sample(frac=1,random_state=3))
        self.assertFalse(set(a.review_id)&set(b.review_id))
        self.assertEqual(set(a.chunk_id),set(c.chunk_id))
        self.assertEqual(set(b.chunk_id),set(d.chunk_id))
        self.assertEqual(len(a)+len(b),len(x))
        with self.assertRaises(ValueError):split_labels(x.loc[x.aspect.ne('other')])
        x.loc[x.subjectivity.eq('objective'),'polarity']='positive'
        with self.assertRaises(ValueError):split_labels(x)

    def test_vocab_and_reload_and_public_prediction(self):
        train,dev=split_labels(fixture())
        dev['text']=dev.text+' onlydevmarker'
        bundle,metrics=fit_models(train,dev,max_features=500)
        self.assertNotIn('onlydevmarker',bundle['vectorizer'].vocabulary_)
        self.assertEqual(set(metrics),set(TASKS))
        pred=predict_texts(bundle,dev.text.tolist())
        self.assertTrue(pred.loc[pred.subjectivity.eq('objective'),'polarity'].eq('NA').all())
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'model.joblib';joblib.dump(bundle,p)
            pd.testing.assert_frame_equal(pred,public_predict(joblib.load(p),dev.text.tolist()))

    def test_objective_sentences_do_not_reach_polarity_model(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        v=TfidfVectorizer().fit(['first sentence','second sentence'])
        polar=FixedModel(['positive'])
        b={'version':VERSION,'tasks':TASKS,'vectorizer':v,
           'models':{'aspect':FixedModel(['overall','other']),
                     'subjectivity':FixedModel(['subjective','objective']),'polarity':polar}}
        pred=predict_texts(b,['first sentence','second sentence'])
        self.assertEqual(polar.calls,[1])
        self.assertEqual(pred.polarity.tolist(),['positive','NA'])
        b['models']['subjectivity']=FixedModel(['objective','objective']);polar.calls=[]
        predict_texts(b,['first sentence','second sentence'])
        self.assertEqual(polar.calls,[])

    def test_existing_output_protected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'已存在'):
                train_command(SimpleNamespace(output_dir=Path(td)))

    def test_nullable_weak_targets_parquet_roundtrip(self):
        df=pd.DataFrame([row('a','The ticket was 800 SGD.'),row('b','Overall a great flight.'),
                         row('c','The cabin was dirty.')])
        labels,_=build_labels(df,set(),set())
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'labels.parquet';labels.to_parquet(p,index=False)
            pd.testing.assert_frame_equal(labels,pd.read_parquet(p))

    def test_explicit_rules_guard_model_errors_with_auditable_source(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts=['The ticket was too expensive.','The flight departed at 9am.','I went to a wedding.']
        v=TfidfVectorizer().fit(texts)
        polar=FixedModel(['negative'])
        b={'version':VERSION,'tasks':TASKS,'vectorizer':v,
           'models':{'aspect':FixedModel(['value','ground','overall']),
                     'subjectivity':FixedModel(['objective','subjective','subjective']),'polarity':polar}}
        out=predict_texts(b,texts)
        self.assertEqual(out.subjectivity.tolist(),['subjective','objective','objective'])
        self.assertEqual(out.polarity.tolist(),['negative','NA','NA'])
        self.assertEqual(out.aspect.iloc[2],'other')
        self.assertTrue(out.subjectivity_source.str.startswith('rule:').all())
        self.assertEqual(polar.calls,[1])


if __name__=='__main__':unittest.main()
