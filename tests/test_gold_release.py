import unittest
import subprocess
import sys
import pandas as pd

from src.gold_release import align_cached_predictions, check_training_overlap


class GoldReleaseTests(unittest.TestCase):
    def setUp(self):
        self.gold = pd.DataFrame({"sentence_id":["b","a"],"chunk_id":["r2_s1","r1_s1"],
            "review_id":["r2","r1"],"text":["Good food.","Bad seat."]})
        self.cache = pd.DataFrame({"gold_sentence_id":["a","b"],"gold_text":["Bad seat.","Good food."],
            "pred_aspect":["seat","food"],"pred_subjectivity":["subjective","subjective"],
            "pred_polarity":["negative","positive"],"pred_label":["negative","positive"]})

    def test_alignment_uses_ids_not_row_order(self):
        pred = align_cached_predictions(self.gold, self.cache)
        self.assertEqual(pred.aspect.tolist(), ["food","seat"])

    def test_changed_text_missing_and_duplicate_ids_are_rejected(self):
        changed = self.cache.copy(); changed.loc[0,"gold_text"] = "Changed seat sentence."
        for cache in [changed, self.cache.iloc[:1], pd.concat([self.cache, self.cache.iloc[:1]])]:
            with self.subTest(rows=len(cache)), self.assertRaises(ValueError):
                align_cached_predictions(self.gold, cache)

    def test_parent_and_normalized_duplicate_leakage_are_visible(self):
        train = pd.DataFrame({"chunk_id":["new1","new2"],"review_id":["r2","new"],
                              "text":["Another sentence in the review."," BAD   SEAT. "]})
        self.assertEqual(check_training_overlap(self.gold, train),
            {"matching_chunk_ids":0,"matching_review_ids":1,"matching_normalized_texts":1})

    def test_training_cli_guard_blocks_final_answers_and_audit_reports(self):
        script = """
from src.training_isolation import install_answer_read_guard
denied=install_answer_read_guard()
for p in ['data/eval/gold_human_frozen_1000_20261019_v1.csv',
          'outputs/gold_release_audit_20261019/audit.json',
          'outputs/c_tuning_20261018/best.json']:
    try: open(p)
    except PermissionError: pass
    else: raise AssertionError(p)
assert len(denied)==4
with open('data/eval/to_label_meta.csv') as handle: assert handle.readline()
"""
        result=subprocess.run([sys.executable,"-c",script],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
