from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.quality import classification_metrics
from src.evaluate import evaluate_queries, evaluate_query_modes
from src.generate import evidence_answer
from src.query_service import rank_evidence, query_reviews, compare_airlines


class QueryQualityTests(unittest.TestCase):
    def test_missing_weak_labels_stay_unscored_and_false_objective_is_error(self):
        gold=pd.DataFrame({"aspect":["seat",None,"food"],"subjectivity":["subjective","subjective",None],
                           "polarity":["negative",None,None]})
        pred=pd.DataFrame({"aspect":["seat","seat","food"],"subjectivity":["objective","subjective","subjective"],
                           "polarity":["NA","positive","negative"]})
        result=classification_metrics(gold,pred)
        self.assertEqual(result["joint_scored_n"],1)
        self.assertEqual(result["subjective_joint_accuracy"],0.)
        self.assertEqual(result["subjective_recall"],.5)

    def test_objective_prediction_does_not_delete_review(self):
        rows=[{"chunk_id":"a","review_id":"r1","text":"The seat was uncomfortable."},
              {"chunk_id":"b","review_id":"r2","text":"A neutral itinerary."}]
        predictions=[{"aspect":"seat","subjectivity":"objective","polarity":"NA"},
                     {"aspect":"other","subjectivity":"objective","polarity":"NA"}]
        selected=rank_evidence(rows,predictions,aspect="seat",top_k=2)
        self.assertEqual({r["chunk_id"] for r in selected},{"a","b"})

    def test_duplicate_and_parent_limits_preserve_distinct_review_evidence(self):
        rows=[{"chunk_id":str(i),"review_id":"same","text":f"sentence {i}"} for i in range(5)]
        rows += [{"chunk_id":"x","review_id":"different","text":"sentence 0"},
                 {"chunk_id":"y","review_id":"different","text":"independent evidence"}]
        selected=rank_evidence(rows,top_k=5,max_per_review=2)
        self.assertEqual([r["chunk_id"] for r in selected],["0","1","y"])

    def test_one_review_with_many_sentences_is_insufficient(self):
        rows=[{"chunk_id":str(i),"review_id":"same","text":"evidence"} for i in range(5)]
        result=evidence_answer(rows)
        self.assertEqual(result["status"],"insufficient_evidence")
        self.assertEqual(result["citations"][0]["quote"],rows[0]["text"])

    def test_unjudged_queries_cannot_pass(self):
        report=evaluate_queries([{"query_id":"q1","expected_answerable":True,"returned_ids":["c1"]}])
        self.assertFalse(report["passed"])
        self.assertFalse(report["evaluation_complete"])
        self.assertIsNone(report["metrics"]["precision_at_5"])

    def test_short_retrieval_does_not_inflate_precision(self):
        report=evaluate_queries([{"query_id":"q1","expected_answerable":True,"returned_ids":["a"],
            "relevance":{"a":True},"answer_successful":True,"total_claims":1,"supported_claims":1}])
        self.assertEqual(report["metrics"]["precision_at_5"],.2)
        self.assertEqual(report["metrics"]["success_at_10"],1.)
        self.assertFalse(report["passed"])

    def test_full_judgments_can_pass_with_no_answer_cases(self):
        rows=[{"query_id":"q1","expected_answerable":True,"returned_ids":[str(i) for i in range(5)],
               "relevance":{str(i):True for i in range(5)},"answer_successful":True,"total_claims":2,"supported_claims":2},
              {"query_id":"q2","expected_answerable":False,"returned_ids":[],"correct_abstention":True}]
        report=evaluate_queries(rows)
        self.assertTrue(report["passed"])
        self.assertFalse(report["independent_acceptance_passed"])

    def test_paired_scores_are_separate_and_deltas_require_complete_judgments(self):
        rows = []
        for mode, relevant in [("baseline", 1), ("labels", 5)]:
            rows.extend([
                {"query_id":"q1_"+mode,"base_query_id":"q1","mode":mode,
                 "expected_answerable":True,"returned_ids":[str(i) for i in range(5)],
                 "relevance":{str(i):i<relevant for i in range(5)},"answer_successful":True,
                 "total_claims":1,"supported_claims":1},
                {"query_id":"q2_"+mode,"base_query_id":"q2","mode":mode,
                 "expected_answerable":False,"returned_ids":[],"correct_abstention":True}])
        report = evaluate_query_modes(rows)
        self.assertEqual(report["by_mode"]["baseline"]["metrics"]["precision_at_5"], .2)
        self.assertEqual(report["by_mode"]["labels"]["metrics"]["precision_at_5"], 1.)
        self.assertAlmostEqual(report["label_minus_baseline"]["precision_at_5"], .8)
        rows[-1]["correct_abstention"] = None
        self.assertIsNone(evaluate_query_modes(rows)["label_minus_baseline"])

    def test_empty_search_does_not_call_models_and_reports_insufficiency(self):
        with patch("src.query_service.search_with_filters",return_value=[]), patch("src.query_service.rerank") as rerank:
            result=query_reviews(aspect="seat",filters={"airline":"absent"})
        rerank.assert_not_called()
        self.assertEqual(result["answer"]["status"],"insufficient_evidence")

    def test_comparison_keeps_filters_and_detects_missing_airline_evidence(self):
        def fake(query,**kwargs):
            self.assertEqual(kwargs["filters"]["cabin"],"Economy Class")
            self.assertEqual(kwargs["aspect"],"seat")
            status="evidence_available" if kwargs["filters"]["airline"]=="A" else "insufficient_evidence"
            return {"answer":{"status":status}}
        with patch("src.query_service.query_reviews",side_effect=fake), \
             patch("src.query_service.summarize_ratings",return_value={"available":False}):
            result=compare_airlines(["A","B"],aspect="seat",filters={"airline":"ignored","cabin":"Economy Class"})
        self.assertEqual(result["status"],"insufficient_evidence")
        self.assertIn("B",result["text"])


if __name__=="__main__": unittest.main()
