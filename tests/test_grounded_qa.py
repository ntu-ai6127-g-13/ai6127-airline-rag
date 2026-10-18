from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

from src.grounded_qa import _generate, answer_question, collect_evidence, validate_answer
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

    def test_food_subtopics_are_all_preserved(self):
        parsed = parse_question("新加坡航空的食物质量、分量和品种怎么样？",
                                known_airlines=self.airlines)
        self.assertEqual(parsed.aspect, "food")
        self.assertEqual(list(parsed.subtopics), ["quality", "portion", "variety"])

    def test_seat_subtopics_work_without_parent_aspect_word(self):
        parsed = parse_question("新航的靠背能否调整，是否舒服，腿部空间怎么样？",
                                known_airlines=self.airlines)
        self.assertEqual(parsed.aspect, "seat")
        self.assertEqual(set(parsed.subtopics), {"recline", "comfort", "legroom"})

    def test_ground_service_subtopics_are_distinct(self):
        parsed = parse_question("新航的登机口服务和延误处理怎么样？",
                                known_airlines=self.airlines)
        self.assertEqual(parsed.aspect, "ground")
        self.assertEqual(set(parsed.subtopics), {"gate_service", "delay_handling"})

    def test_seat_cleanliness_is_not_reduced_to_seat_comfort(self):
        parsed = parse_question("新航的座椅的干净程度怎么样？", known_airlines=self.airlines)
        self.assertEqual(parsed.aspect, "seat")
        self.assertEqual(set(parsed.subtopics), {"seat_cleanliness"})

    def test_multiple_top_level_aspects_are_preserved_in_question_order(self):
        parsed = parse_question("新加坡航空餐食、性价比和地勤怎么样？",
                                known_airlines=self.airlines)
        self.assertEqual(parsed.aspect, "food")
        self.assertEqual(parsed.aspects, ["food", "value", "ground"])

    def test_route_comparison_extracts_direction_and_does_not_filter_positive_only(self):
        parsed = parse_question(
            "新加坡飞伦敦经济舱，哪家航司腿部空间评价最好？",
            known_airlines=self.airlines,
            origins=["Singapore", "BA16 Singapore", "London"],
            destinations=["London", "London Heathrow", "London Gatwick", "Singapore"],
        )
        self.assertEqual(parsed.filters["route_from"], ["Singapore", "BA16 Singapore"])
        self.assertEqual(parsed.filters["route_to"], ["London", "London Heathrow", "London Gatwick"])
        self.assertEqual(parsed.filters["cabin"], "Economy Class")
        self.assertEqual(parsed.subtopics, {"legroom": "seat legroom leg room knee space pitch cramped"})
        self.assertIsNone(parsed.polarity)
        self.assertTrue(parsed.comparison)


class GroundedAnswerTests(unittest.TestCase):
    rating_summary = {"available": True, "scale": 5, "airlines": {
        "A Air": {"n": 100, "mean": 4.2, "median": 4.0, "std": .8,
                  "high_share": .8, "low_share": .1},
        "B Air": {"n": 120, "mean": 2.1, "median": 2.0, "std": 1.0,
                  "high_share": .2, "low_share": .7}},
        "comparison": {"status": "clear_difference", "winner": "A Air", "runner_up": "B Air",
                       "difference": 2.1}}

    def test_second_concurrent_query_fails_fast(self):
        def contender():
            with query_session(wait_seconds=0.01):
                return "unexpected"
        with query_session():
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(contender)
                with self.assertRaises(QueryBusyError):
                    future.result(timeout=1)

    def test_local_generation_stream_emits_cumulative_text(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def __iter__(self):
                return iter([
                    '{"type":"delta","text":"餐饮"}\n'.encode(),
                    '{"type":"delta","text":"评分"}\n'.encode(),
                    b'{"type":"done","seconds":1.25}\n',
                ])
        updates = []
        with patch("urllib.request.urlopen", return_value=Response()) as request:
            result = _generate([{"role": "user", "content": "test"}], endpoint="http://local",
                               timeout=2, on_stream=lambda text, event: updates.append((text, event)))
        self.assertTrue(request.call_args.args[0].full_url.endswith("/generate_stream"))
        self.assertEqual(updates, [("餐饮", "delta"), ("餐饮评分", "delta")])
        self.assertEqual(result, {"text": "餐饮评分", "seconds": 1.25})

    def test_comparison_citation_ids_are_unique(self):
        result = {"type": "comparison", "results": {
            "A Air": {"evidence": [evidence("1", "A Air", "positive")]},
            "B Air": {"evidence": [evidence("2", "B Air", "negative")]},
        }}
        self.assertEqual([row["citation_id"] for row in collect_evidence(result)], ["A1", "B1"])

    def test_multi_aspect_citation_ids_are_unique(self):
        result = {"type": "multi_aspect", "results": {
            "food": {"evidence": [evidence("1", "A Air", "positive")]},
            "value": {"evidence": [evidence("2", "A Air", "negative")]},
        }}
        rows = collect_evidence(result)
        self.assertEqual([row["citation_id"] for row in rows], ["A1", "B1"])
        self.assertEqual([row["query_aspect"] for row in rows], ["food", "value"])

    def test_validation_rejects_unknown_citation_and_uncited_paragraph(self):
        self.assertFalse(validate_answer("结论。[9]", {"1"})[0])
        self.assertFalse(validate_answer("这家航空公司的客舱服务非常好。\n另一条有引用。[1]", {"1"})[0])
        self.assertFalse(validate_answer("本次有11条正面评价。[1]", {"1"})[0])
        self.assertTrue(validate_answer("结论在同一段落给出，随后说明原文依据。[1]", {"1"})[0])

    def test_verified_rating_numbers_and_required_winner_are_enforced(self):
        valid = "A Air更好，原始评分为4.20/5，共100份评分。[A1]"
        self.assertTrue(validate_answer(valid, {"A1"}, rating_summary=self.rating_summary)[0])
        invalid = "B Air更好，原始评分为9.99/5。[B1]"
        ok, reasons = validate_answer(invalid, {"B1"}, rating_summary=self.rating_summary)
        self.assertFalse(ok)
        self.assertTrue(any("unverified_numeric_claim" in reason for reason in reasons))
        self.assertIn("missing_required_winner", reasons)

    def test_program_fallback_uses_rating_decision_instead_of_label_counts(self):
        result = {"type": "comparison", "aspect": "seat", "rating_summary": self.rating_summary,
                  "results": {"A Air": {"evidence": [evidence("1", "A Air", "positive")]},
                              "B Air": {"evidence": [evidence("2", "B Air", "negative")]}}}
        answer = answer_question("哪家更好", result, use_model=False)
        self.assertIn("A Air明显更好", answer["text"])
        self.assertIn("4.20/5", answer["text"])
        self.assertNotIn("正向", answer["text"])
        self.assertEqual(answer["mode"], "rating_decision")

    def test_broad_comparison_adds_one_sentence_full_review_distribution(self):
        distribution = lambda positive, negative: {
            "labeled_review_count": positive + negative,
            "positive_review_count": positive,
            "negative_review_count": negative,
            "positive_share": positive / (positive + negative),
        }
        result = {"type": "comparison", "aspect": "seat", "rating_summary": self.rating_summary,
                  "label_distributions": {
                      "A Air": distribution(75, 25), "B Air": distribution(40, 60)},
                  "results": {"A Air": {"evidence": [evidence("1", "A Air", "positive")]},
                              "B Air": {"evidence": [evidence("2", "B Air", "negative")]}}}
        text = answer_question("哪家座椅更好", result, use_model=False)["text"]
        self.assertIn("该方面全部相关评论的自动标签汇总：A Air共命中 100 条，其中可明确归类的 100 条里正向 "
                      "75 条（75.0%）、负向 25 条（25.0%）；B Air共命中 100 条，其中可明确归类的 100 条里"
                      "正向 40 条（40.0%）、负向 60 条（60.0%）。", text)

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

    def test_subtopic_comparison_does_not_use_broad_rating_as_winner(self):
        a = {**evidence("1", "A Air", "positive"), "matched_subtopics": ["legroom"],
             "direct_subtopic_matches": ["legroom"]}
        b = {**evidence("2", "B Air", "negative"), "matched_subtopics": ["legroom"],
             "direct_subtopic_matches": ["legroom"]}
        result = {"type": "comparison", "aspect": "seat", "subtopics": {"legroom": "seat legroom"},
                  "rating_summary": self.rating_summary,
                  "results": {"A Air": {"evidence": [a]}, "B Air": {"evidence": [b]}}}
        answer = answer_question("哪家腿部空间更好", result, use_model=False)
        self.assertEqual(answer["mode"], "program_fallback")
        self.assertIn("腿部空间", answer["text"])
        self.assertIn("原始座椅舒适度评分", answer["text"])
        self.assertIn("A Air 4.20/5（100 份）", answer["text"])
        self.assertIn("不足以可靠判定哪家最好", answer["text"])
        self.assertNotIn("A Air明显更好", answer["text"])

    def test_subtopic_comparison_uses_full_direct_aggregate_when_decisive(self):
        a = {**evidence("1", "Singapore Airlines", "positive"),
             "matched_subtopics": ["variety"], "direct_subtopic_matches": ["variety"]}
        b = {**evidence("2", "Emirates", "negative"),
             "matched_subtopics": ["variety"], "direct_subtopic_matches": ["variety"]}
        rating = {"available": True, "scale": 5, "airlines": {
            "Singapore Airlines": {"n": 704, "mean": 3.88, "median": 4.0, "std": 1.0,
                                   "high_share": .7, "low_share": .1},
            "Emirates": {"n": 1136, "mean": 3.18, "median": 3.0, "std": 1.0,
                         "high_share": .45, "low_share": .2}},
            "comparison": {"status": "clear_difference", "winner": "Singapore Airlines",
                           "runner_up": "Emirates", "difference": .70}}
        def analysis(positive, negative, low, high):
            total = positive + negative
            return {"variety": {"direct_review_count": total, "polarity_summary": {
                "labeled_review_count": total, "positive_review_count": positive,
                "negative_review_count": negative, "positive_share": positive / total,
                "positive_share_ci95_low": low, "positive_share_ci95_high": high}}}
        result = {"type": "comparison", "aspect": "food",
                  "subtopics": {"variety": "meal variety"}, "rating_summary": rating,
                  "results": {
                      "Singapore Airlines": {"evidence": [a], "subtopic_analysis": analysis(82, 41, .58, .74)},
                      "Emirates": {"evidence": [b], "subtopic_analysis": analysis(72, 95, .36, .51)},
                  }}
        text = answer_question("哪家餐食品种更好", result, use_model=False)["text"]
        self.assertIn("就整体餐饮而言，新加坡航空在这批同条件历史评分中明确更好", text)
        self.assertIn("完整直接原文的自动标签汇总明确更支持新加坡航空", text)
        self.assertIn("新加坡航空共命中 123 条，其中可明确归类的 123 条里正向 82 条（66.7%）、负向 41 条（33.3%）", text)
        self.assertIn("阿联酋航空共命中 167 条，其中可明确归类的 167 条里正向 72 条（43.1%）、负向 95 条（56.9%）", text)

    def test_single_subtopic_fallback_answers_directly(self):
        row = {**evidence("1", "Singapore Airlines", "negative"),
               "text": "The portion size was terribly small.",
               "pred_aspect": "food", "matched_subtopics": ["portion"],
               "direct_subtopic_matches": ["portion"]}
        result = {"aspect": "food", "subtopics": {"portion": "meal portion size"},
                  "rating_summary": {"available": False}, "evidence": [row]}
        answer = answer_question("新航餐食分量怎么样", result, use_model=False)
        self.assertIn("关于新加坡航空的餐食分量", answer["text"])
        self.assertIn("直接评论偏负面", answer["text"])
        self.assertNotIn("这些细分问题需要分别看原文", answer["text"])

    def test_style_validation_failure_is_repaired_once(self):
        row = {**evidence("1", "A Air", "negative"),
               "matched_subtopics": ["legroom"], "direct_subtopic_matches": ["legroom"]}
        result = {"aspect": "seat", "subtopics": {"legroom": "seat legroom"},
                  "rating_summary": {"available": False}, "evidence": [row]}
        first = unittest.mock.MagicMock()
        first.__enter__.return_value.read.return_value = json.dumps(
            {"text": "腿部空间方面，普遍认为很差。[1]", "seconds": 1}).encode()
        second = unittest.mock.MagicMock()
        second.__enter__.return_value.read.return_value = json.dumps(
            {"text": "腿部空间方面，有评论认为很差。[1]", "seconds": 1}).encode()
        with patch("urllib.request.urlopen", side_effect=[first, second]) as request:
            answer = answer_question("腿部空间怎么样", result)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(answer["mode"], "local_qwen_repaired")
        self.assertIn("有评论", answer["text"])

    def test_subtopic_validation_rejects_retrieval_sample_prevalence_claims(self):
        ok, reasons = validate_answer("座椅清洁度方面，更多人认为很脏。[1]", {"1"},
                                      required_subtopics=("seat_cleanliness",))
        self.assertFalse(ok)
        self.assertIn("unsupported_prevalence_claim", reasons)

    def test_multiple_subtopics_must_use_separate_paragraphs(self):
        same_paragraph = "腿部空间方面较窄。[1] 靠背调节与躺平方面可以后仰。[2]"
        ok, reasons = validate_answer(same_paragraph, {"1", "2"},
                                      required_subtopics=("legroom", "recline"))
        self.assertFalse(ok)
        self.assertIn("subtopics_not_separated", reasons)
        separated = "腿部空间方面较窄。[1]\n\n靠背调节与躺平方面可以后仰。[2]"
        self.assertTrue(validate_answer(separated, {"1", "2"},
                                        required_subtopics=("legroom", "recline"))[0])

    def test_multiple_subtopic_fallback_uses_separate_paragraphs(self):
        legroom = {**evidence("1", "Singapore Airlines", "negative"),
                   "matched_subtopics": ["legroom"], "direct_subtopic_matches": ["legroom"]}
        recline = {**evidence("2", "Singapore Airlines", "positive"),
                   "matched_subtopics": ["recline"], "direct_subtopic_matches": ["recline"]}
        result = {"aspect": "seat", "subtopics": {"legroom": "seat legroom", "recline": "seat recline"},
                  "rating_summary": {"available": False}, "evidence": [legroom, recline]}
        text = answer_question("腿部空间和靠背怎么样", result, use_model=False)["text"]
        self.assertIn("腿部空间", text)
        self.assertIn("靠背调节与躺平", text)
        self.assertGreaterEqual(text.count("\n\n"), 2)

    def test_multi_aspect_fallback_names_each_rating(self):
        def summary(aspect, field, mean, n):
            return {"available": True, "aspect": aspect, "rating_field": field, "scale": 5,
                    "airlines": {"Singapore Airlines": {"mean": mean, "n": n, "median": 4.0,
                        "std": 1.0, "high_share": .6, "low_share": .2}},
                    "comparison": {"status": "not_applicable"}}
        rows = {}
        for index, aspect in enumerate(("food", "value", "ground"), 1):
            rows[aspect] = {"evidence": [{**evidence(str(index), "Singapore Airlines", "positive"),
                                           "pred_aspect": aspect}]}
        ratings = {"food": summary("food", "food_bev", 3.8, 100),
                   "value": summary("value", "value_for_money", 3.6, 120),
                   "ground": summary("ground", "ground_service", 3.7, 80)}
        result = {"type": "multi_aspect", "aspects": ["food", "value", "ground"],
                  "results": rows, "rating_summaries": ratings}
        answer = answer_question("餐食、性价比和地勤怎么样", result, use_model=False)
        self.assertIn("原始餐饮评分为 3.80/5", answer["text"])
        self.assertIn("原始性价比评分为 3.60/5", answer["text"])
        self.assertIn("原始地面服务评分为 3.70/5", answer["text"])
        self.assertGreaterEqual(answer["text"].count("\n\n"), 3)

    def test_multi_aspect_validation_allows_intro_and_evidence_numbers(self):
        def summary(mean):
            return {"available": True, "scale": 5, "airlines": {
                "Singapore Airlines": {"mean": mean, "n": 100, "median": 4.0,
                                       "high_share": .6, "low_share": .2}},
                "comparison": {"status": "not_applicable"}}
        ratings = {"rating_summaries": {"food": summary(3.8), "ground": summary(4.0)}}
        text = ("餐饮和地面服务分别如下。\n\n"
                "餐饮：原始餐饮评分为 3.80/5，共 100 份；餐食不错。[A1]\n\n"
                "地面服务：原始地面服务评分为 4.00/5，共 100 份；有航班延误 15 分钟。[B1]")
        ok, reasons = validate_answer(text, {"A1", "B1"}, rating_summary=ratings,
                                      required_aspects=("food", "ground"), evidence_numbers={"15"})
        self.assertTrue(ok, reasons)

    def test_multi_aspect_validation_rejects_wrong_paragraph_order(self):
        def summary(mean):
            return {"available": True, "scale": 5, "airlines": {
                "Singapore Airlines": {"mean": mean, "n": 100, "median": 4.0,
                                       "high_share": .6, "low_share": .2}},
                "comparison": {"status": "not_applicable"}}
        ratings = {"rating_summaries": {"food": summary(3.8), "ground": summary(4.0)}}
        text = ("地面服务：原始地面服务评分为 4.00/5，共 100 份。[B1]\n\n"
                "餐饮：原始餐饮评分为 3.80/5，共 100 份。[A1]")
        ok, reasons = validate_answer(text, {"A1", "B1"}, rating_summary=ratings,
                                      required_aspects=("food", "ground"))
        self.assertFalse(ok)
        self.assertIn("aspects_out_of_order", reasons)


if __name__ == "__main__":
    unittest.main()
