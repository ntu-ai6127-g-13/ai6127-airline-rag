from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

from src.grounded_qa import answer_question, collect_evidence, validate_answer
from src.question_parser import parse_question
from src.query_service import QueryBusyError, query_session


def evidence(chunk_id: str, airline: str, polarity: str) -> dict:
    return {"chunk_id": chunk_id, "review_id": "r" + chunk_id, "text": "Quoted review " + chunk_id,
            "airline": airline, "cabin": "Economy Class", "pred_subjectivity": "subjective",
            "pred_aspect": "seat", "pred_polarity": polarity}


class QuestionParserTests(unittest.TestCase):
    airlines = ["Singapore Airlines", "Qatar Airways", "Aer Lingus", "British Airways"]

    def test_chinese_question_extracts_corpus_safe_values(self):
        parsed = parse_question("比较新航和卡塔尔航空商务舱的餐饮", known_airlines=self.airlines)
        self.assertEqual(parsed.airlines, ["Singapore Airlines", "Qatar Airways"])
        self.assertEqual(parsed.aspect, "food")
        self.assertEqual(parsed.filters["cabin"], "Business Class")
        self.assertTrue(parsed.comparison)

    def test_explicit_question_overrides_sidebar_defaults(self):
        parsed = parse_question("Aer Lingus entertainment 好评", known_airlines=self.airlines,
                                default_aspect="seat", default_polarity="negative",
                                default_filters={"airline": "British Airways"})
        self.assertEqual(parsed.aspect, "entertainment")
        self.assertEqual(parsed.polarity, "positive")
        self.assertEqual(parsed.filters["airline"], "Aer Lingus")

    def test_general_question_maps_to_overall_experience(self):
        parsed = parse_question("新航总体怎么样？", known_airlines=self.airlines)
        self.assertEqual(parsed.aspect, "overall")


class GroundedAnswerTests(unittest.TestCase):
    def test_second_concurrent_query_fails_fast(self):
        def contender():
            with query_session(wait_seconds=0.01):
                return "unexpected"
        with query_session():
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(contender)
                with self.assertRaises(QueryBusyError):
                    future.result(timeout=1)

    def test_comparison_citation_ids_are_unique(self):
        result = {"type": "comparison", "results": {
            "A Air": {"evidence": [evidence("1", "A Air", "positive")]},
            "B Air": {"evidence": [evidence("2", "B Air", "negative")]},
        }}
        self.assertEqual([row["citation_id"] for row in collect_evidence(result)], ["A1", "B1"])

    def test_validation_rejects_unknown_citation_and_uncited_paragraph(self):
        self.assertFalse(validate_answer("结论。[9]", {"1"})[0])
        self.assertFalse(validate_answer("这家航空公司的客舱服务非常好。\n另一条有引用。[1]", {"1"})[0])
        self.assertFalse(validate_answer("本次有11条正面评价。[1]", {"1"})[0])
        self.assertTrue(validate_answer("结论在同一段落给出，随后说明原文依据。[1]", {"1"})[0])

    def test_invalid_model_answer_falls_back_to_program_answer(self):
        result = {"aspect": "seat", "evidence": [
            evidence("1", "A Air", "positive"), evidence("2", "A Air", "negative"),
            evidence("3", "A Air", "negative")], "filters": {}}
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"text": "无引用回答", "seconds": 1}).encode()
        with patch("urllib.request.urlopen", return_value=response):
            answer = answer_question("座位怎样", result)
        self.assertEqual(answer["mode"], "validation_fallback")
        self.assertIn("[", answer["text"])


if __name__ == "__main__":
    unittest.main()
