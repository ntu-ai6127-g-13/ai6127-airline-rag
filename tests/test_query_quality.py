from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.quality import classification_metrics
from src.evaluate import evaluate_queries, evaluate_query_modes
from src.generate import evidence_answer
from src.query_service import (_direct_subtopic_match, compare_airlines,
                               compare_available_airlines, merge_subtopic_rankings,
                               query_aspects, query_reviews, rank_evidence,
                               scan_direct_subtopic_matches)


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

    def test_subtopic_merge_covers_each_facet_before_repeating(self):
        rankings = {
            "recline": [{"chunk_id": "r1", "text": "recline one"},
                        {"chunk_id": "r2", "text": "recline two"}],
            "comfort": [{"chunk_id": "c1", "text": "comfort one"},
                        {"chunk_id": "r1", "text": "also comfort"}],
            "legroom": [{"chunk_id": "l1", "text": "legroom one"}],
        }
        merged = merge_subtopic_rankings(rankings, limit=4)
        self.assertEqual([row["chunk_id"] for row in merged[:3]], ["r1", "c1", "l1"])
        self.assertEqual(merged[0]["matched_subtopics"], ["recline", "comfort"])

    def test_direct_subtopic_match_respects_word_boundaries(self):
        self.assertFalse(_direct_subtopic_match("The seat was comfortable.", "drinks"))
        self.assertTrue(_direct_subtopic_match("Tea and coffee were available.", "drinks"))
        self.assertTrue(_direct_subtopic_match("We were rebooked after cancellation.", "delay_handling"))
        self.assertTrue(_direct_subtopic_match("The dirty seat had crumbs in the pocket.", "seat_cleanliness"))
        self.assertFalse(_direct_subtopic_match("The seat was comfortable and reclined.", "seat_cleanliness"))
        self.assertFalse(_direct_subtopic_match("The tray table pressed into my stomach.", "seat_cleanliness"))
        self.assertTrue(_direct_subtopic_match("The tray table was dirty and sticky.", "seat_cleanliness"))
        self.assertFalse(_direct_subtopic_match("The toilet seat and basin were stained.", "seat_cleanliness"))

    def test_full_slice_subtopic_scan_finds_direct_matches_before_retrieval(self):
        frame = pd.DataFrame([
            {"chunk_id": "1", "review_id": "r1", "text": "The meal portion was tiny.",
             "airline": "A", "cabin": "Economy", "traveller_type": "Solo",
             "route_from": "SIN", "route_to": "LHR", "date": pd.Timestamp("2019-01-01")},
            {"chunk_id": "2", "review_id": "r2", "text": "The meal portion was generous.",
             "airline": "B", "cabin": "Economy", "traveller_type": "Solo",
             "route_from": "SIN", "route_to": "LHR", "date": pd.Timestamp("2019-01-01")},
            {"chunk_id": "3", "review_id": "r3", "text": "The food tasted good.",
             "airline": "A", "cabin": "Economy", "traveller_type": "Solo",
             "route_from": "SIN", "route_to": "LHR", "date": pd.Timestamp("2019-01-01")},
        ])
        with patch("src.query_service._subtopic_scan_frame", return_value=frame):
            result = scan_direct_subtopic_matches("portion", filters={"airline": "A"})
        self.assertEqual(result["scope_sentence_count"], 2)
        self.assertEqual(result["direct_sentence_count"], 1)
        self.assertEqual(result["direct_review_count"], 1)
        self.assertEqual(result["candidate_ids"], ["1"])

    def test_subtopic_queries_are_retrieved_and_reranked_separately(self):
        def search(phrase, **kwargs):
            topic = "portion" if "portion" in phrase else "variety"
            return [{"chunk_id": topic, "text": topic, "review_id": topic}]
        with patch("src.query_service.search_with_filters", side_effect=search) as search_mock, \
             patch("src.query_service.rerank", side_effect=lambda phrase, hits, **kwargs: hits), \
             patch("src.query_service.scan_direct_subtopic_matches", return_value={
                 "scope_sentence_count": 2, "direct_sentence_count": 0, "direct_review_count": 0,
                 "reranked_direct_count": 0, "rerank_cap": 240, "cap_applied": False,
                 "candidate_ids": []}), \
             patch("src.query_service.pd.read_parquet") as parquet_mock, \
             patch("src.query_service.summarize_ratings", return_value={"available": False}):
            parquet_mock.return_value.set_index.return_value.loc.__getitem__.return_value.items.return_value = []
            result = query_reviews(aspect="food", top_k=2, subtopics={
                "portion": "meal portion size", "variety": "meal variety choice"})
        self.assertEqual(search_mock.call_count, 2)
        self.assertEqual({row["chunk_id"] for row in result["evidence"]}, {"portion", "variety"})
        self.assertEqual(result["subtopic_evidence_counts"], {"portion": 1, "variety": 1})

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

    def test_multi_aspect_query_runs_each_aspect_with_shared_filters(self):
        def fake(query, **kwargs):
            aspect = kwargs["aspect"]
            self.assertEqual(kwargs["filters"], {"airline": "A Air"})
            return {"answer": {"status": "evidence_available"}, "evidence": [{"chunk_id": aspect}],
                    "rating_summary": {"available": True, "aspect": aspect}}
        with patch("src.query_service.query_reviews", side_effect=fake) as query_mock:
            result = query_aspects(["food", "value", "ground"], filters={"airline": "A Air"})
        self.assertEqual(query_mock.call_count, 3)
        self.assertEqual(list(result["results"]), ["food", "value", "ground"])
        self.assertEqual(set(result["rating_summaries"]), {"food", "value", "ground"})

    def test_open_comparison_uses_discovered_airlines_and_preserves_route_filters(self):
        selection = {"airlines": ["A Air", "B Air"], "candidates": [], "available": True}
        compared = {"type": "comparison", "results": {}, "rating_summary": {}}
        with patch("src.query_service.rank_airlines_by_rating", return_value=selection), \
             patch("src.query_service.compare_airlines", return_value=compared) as compare_mock:
            result = compare_available_airlines(
                aspect="seat", filters={"route_from": "Singapore", "cabin": "Economy Class"},
                subtopics={"legroom": "seat legroom"})
        self.assertEqual(result["candidate_selection"], selection)
        kwargs = compare_mock.call_args.kwargs
        self.assertEqual(kwargs["filters"]["route_from"], "Singapore")
        self.assertEqual(kwargs["subtopics"], {"legroom": "seat legroom"})


if __name__=="__main__": unittest.main()
