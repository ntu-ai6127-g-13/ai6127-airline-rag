import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import json

import pandas as pd

from experiments.downstream_evaluation.qa import replay_variants, review_record, citation_probes, proxy_judge
from experiments.downstream_evaluation.ranking import heldout, split_material, predict_heldout, review_predictions
from experiments.downstream_evaluation.review import summarize


class ProtocolBoundaryTests(unittest.TestCase):
    def test_heldout_parent_and_its_duplicate_text_cannot_enter_training(self):
        test_id = next(str(i) for i in range(100) if heldout(str(i), 6127, .2))
        train_id = next(str(i) for i in range(100) if not heldout(str(i), 6127, .2))
        rows = []
        for cid, rid, text in [("a", test_id, "The seat was good."),
                                ("b", train_id, " THE SEAT WAS GOOD. "),
                                ("c", train_id, "The meal was awful."),
                                ("d", "gold", "The crew were friendly.")]:
            rows.append(dict(chunk_id=cid, review_id=rid, text=text, airline="A", cabin="Economy Class",
                             is_route_only=False, is_non_english=False, is_fragment=False))
        pool, test, ids, _ = split_material(pd.DataFrame(rows), {"gold"}, set(), 6127, .2)
        self.assertEqual(set(pool.chunk_id), {"c"})
        self.assertEqual(set(test.chunk_id), {"a"})
        self.assertIn(test_id, ids)
        self.assertNotIn("gold", ids)

    def test_predictor_rejects_rating_columns_before_loading_a_model(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "unsafe.parquet"
            pd.DataFrame([dict(chunk_id="a", review_id="r", text="Seat.", airline="A",
                               cabin="Economy Class", seat_comfort=5)]).to_parquet(path)
            with self.assertRaisesRegex(ValueError, "whitelist"):
                predict_heldout("nonexistent-model", path, Path(folder)/"pred.parquet")

    def test_long_review_is_averaged_before_airline_aggregation(self):
        predicted = pd.DataFrame(dict(review_id=["a", "a", "b", "b"],
                                     aspect=["seat", "seat", "seat", "food"],
                                     subjectivity=["subjective"]*4,
                                     polarity=["positive", "negative", "positive", "negative"]))
        got = review_predictions(predicted).set_index("review_id")
        self.assertEqual(got.loc["a", "text_score"], .5)
        self.assertEqual(got.loc["b", "text_score"], 1.)

    def test_guard_ablation_keeps_actual_invalid_initial_answer(self):
        full = dict(text="Fixed [1].", mode="local_qwen_repaired", valid=True)
        calls = [{"response": {"text": "Raw [999].", "seconds": 2}},
                 {"response": {"text": "Fixed [1].", "seconds": 3}}]
        variants = replay_variants(full, calls, [{"valid": False, "errors": ["unknown_citations:999"]}], "Fallback [1].")
        self.assertEqual(variants["full"]["text"], "Fixed [1].")
        self.assertEqual(variants["no_repair"]["text"], "Fallback [1].")
        self.assertEqual(variants["no_guard"]["text"], "Raw [999].")
        self.assertIsNone(variants["no_guard"]["valid"])

    def test_deterministic_bypass_is_not_counted_as_model_ablation(self):
        variants = replay_variants(dict(text="No evidence.", mode="program_fallback"), [], [], "No evidence.")
        self.assertFalse(any(v["ablation_applicable"] for v in variants.values()))

    def test_empty_human_sheet_has_no_quality_scores(self):
        packet = {"records": [review_record("a", "Question?", "Answer [1].",
                                             [{"citation_id": "1", "text": "Evidence"}], {})]}
        report = summarize(packet)
        self.assertEqual(report["reviewed"], 0)
        self.assertFalse(report["evaluation_complete"])
        self.assertIsNone(report["reviewed_subset_metrics"]["faithfulness_micro"])

    def test_citation_probes_distinguish_valid_and_corrupt_ids(self):
        for probe in citation_probes():
            self.assertEqual(probe["expected_valid"], probe["actual_valid"])

    def test_judge_receives_only_citation_ids_and_original_sentences(self):
        response = {"text": json.dumps(dict(total_claims=2, supported_claims=1,
                                             answer_relevance_0_to_4=3, relevant_evidence_ids=["1"]))}
        with patch("src.grounded_qa._generate", return_value=response) as generate:
            result = proxy_judge("Question?", "Answer [1].", [dict(citation_id="1", review_id="R123",
                                 text="The seat was nice.", review_text="Very long unrelated parent", airline="A")], {})
            payload = json.loads(generate.call_args.args[0][1]["content"])
        self.assertEqual(payload["allowed_evidence_ids"], ["1"])
        self.assertNotIn("R123", json.dumps(payload))
        self.assertNotIn("Very long unrelated", json.dumps(payload))
        self.assertEqual(result["faithfulness"], .5)

    def test_judge_does_not_clamp_impossible_counts_to_a_passing_score(self):
        response = {"text": json.dumps(dict(total_claims=1, supported_claims=2,
                                             answer_relevance_0_to_4=4, relevant_evidence_ids=[]))}
        with patch("src.grounded_qa._generate", return_value=response):
            result = proxy_judge("Question?", "Answer.", [], {})
        self.assertTrue(result["judge_failed"])
        self.assertIsNone(result["faithfulness"])

    def test_judge_rejects_aggregate_field_names_as_evidence_ids(self):
        response = {"text": json.dumps(dict(total_claims=1, supported_claims=1,
                                             answer_relevance_0_to_4=4,
                                             relevant_evidence_ids=["rating_summary"]))}
        with patch("src.grounded_qa._generate", return_value=response):
            result = proxy_judge("Question?", "Answer [1].", [{"citation_id":"1","text":"Seat."}], {})
        self.assertTrue(result["judge_failed"])


if __name__ == "__main__":
    unittest.main()
