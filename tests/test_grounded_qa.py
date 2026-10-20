from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

from src.grounded_qa import (_generate, answer_question, collect_evidence,
                             localize_evidence, validate_answer)
from src.question_parser import parse_question
from src.query_service import QueryBusyError, prioritize_polarity_coverage, query_session


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

    def test_mixed_language_airline_name_next_to_chinese_is_recognized(self):
        parsed = parse_question(
            "Spirit Airlines经济舱的行李服务怎么样？",
            known_airlines=[*self.airlines, "Spirit Airlines"],
        )
        self.assertEqual(parsed.airlines, ["Spirit Airlines"])
        self.assertEqual(parsed.filters["airline"], "Spirit Airlines")

    def test_long_airline_name_suppresses_nested_alias(self):
        parsed = parse_question(
            "国泰航空商务舱空乘服务态度怎么样？",
            known_airlines=[*self.airlines, "Cathay Pacific Airways", "Thai Airways"],
        )
        self.assertEqual(parsed.airlines, ["Cathay Pacific Airways"])

    def test_seat_cleanliness_remains_a_seat_subtopic(self):
        parsed = parse_question(
            "新加坡航空经济舱的座椅清洁度怎么样？",
            known_airlines=self.airlines,
        )
        self.assertEqual(parsed.aspects, ["seat"])
        self.assertEqual(list(parsed.subtopics), ["seat_cleanliness"])

    def test_delay_or_cancellation_is_delay_handling(self):
        parsed = parse_question(
            "Alaska Airlines经济舱遇到延误或取消时处理得怎么样？",
            known_airlines=[*self.airlines, "Alaska Airlines"],
        )
        self.assertEqual(parsed.aspect, "ground")
        self.assertEqual(list(parsed.subtopics), ["delay_handling"])

    def test_english_plural_dirty_seats_is_seat_cleanliness(self):
        parsed = parse_question(
            "How are legroom and dirty seats in Singapore Airlines economy class?",
            known_airlines=self.airlines,
        )
        self.assertEqual(parsed.aspects, ["seat"])
        self.assertEqual(list(parsed.subtopics), ["legroom", "seat_cleanliness"])

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

    def test_good_or_bad_question_keeps_both_sentiments(self):
        parsed = parse_question("新航经济舱腿部空间好不好？", known_airlines=self.airlines,
                                default_polarity="negative")
        self.assertEqual(parsed.aspect, "seat")
        self.assertEqual(parsed.subtopics, {"legroom": "seat legroom leg room knee space pitch cramped"})
        self.assertIsNone(parsed.polarity)

    def test_food_subtopics_are_all_preserved(self):
        parsed = parse_question("新加坡航空的食物质量、分量和品种怎么样？",
                                known_airlines=self.airlines)
        self.assertEqual(parsed.aspect, "food")
        self.assertEqual(list(parsed.subtopics), ["quality", "portion", "variety"])

    def test_what_food_question_is_concrete_menu_intent(self):
        parsed = parse_question("新航的餐食有什么吃？", known_airlines=self.airlines)
        self.assertEqual(parsed.aspect, "food")
        self.assertEqual(list(parsed.subtopics), ["menu_items"])
        self.assertEqual(parsed.answer_intent, "enumeration")

    def test_information_need_and_subtopic_routing_across_aspects(self):
        cases = {
            "新航有什么电影可以看？": ("entertainment", "content", "enumeration"),
            "新航有没有机上WiFi？": ("entertainment", "wifi", "availability"),
            "新航延误怎么处理？": ("ground", "delay_handling", "procedure"),
            "新航经济舱座位多宽？": ("seat", "width", "measurement"),
            "新航商务舱隐私怎么样？": ("seat", "privacy", "evaluation"),
            "新航转机衔接怎么样？": ("ground", "transfer", "evaluation"),
            "新航空乘的语言沟通怎么样？": ("staff", "communication", "evaluation"),
            "新航有没有毛毯和枕头？": ("other", "amenities", "availability"),
            "新航座位有储物空间吗？": ("seat", "storage", "availability"),
        }
        for question, (aspect, subtopic, intent) in cases.items():
            with self.subTest(question=question):
                parsed = parse_question(question, known_airlines=self.airlines)
                self.assertEqual(parsed.aspect, aspect)
                self.assertIn(subtopic, parsed.subtopics)
                self.assertEqual(parsed.answer_intent, intent)

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


class EvidenceCoverageTests(unittest.TestCase):
    def test_promotes_two_examples_from_each_available_direction(self):
        rows = [{"chunk_id": str(index), "text": str(index)} for index in range(1, 8)]
        predictions = {
            str(index): {"subjectivity": "subjective", "aspect": "ground",
                         "polarity": "negative" if index <= 5 else "positive"}
            for index in range(1, 8)
        }
        ordered = prioritize_polarity_coverage(
            rows, predictions, aspect="ground", preferred="negative", per_direction=2)
        self.assertEqual([row["chunk_id"] for row in ordered[:4]], ["1", "2", "6", "7"])


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

    def test_chinese_mode_attaches_translated_evidence_without_replacing_source(self):
        row = evidence("1", "A Air", "negative")
        result = {"language": "zh", "evidence": [row]}
        with patch("src.grounded_qa._translate_batch_cached", return_value=("翻译后的评论。",)):
            localize_evidence(result, language="zh")
        self.assertEqual(row["text"], "Quoted review 1")
        self.assertEqual(row["display_text"], "翻译后的评论。")
        self.assertEqual(result["evidence_translation"]["status"], "translated")

    def test_chinese_translation_keeps_successful_batches_when_one_batch_fails(self):
        rows = [evidence(str(index), "A Air", "negative") for index in range(1, 6)]
        result = {"language": "zh", "evidence": rows}
        with patch("src.grounded_qa._translate_batch_cached",
                   side_effect=[("译文一", "译文二", "译文三", "译文四"),
                                ValueError("bad batch json"), ValueError("bad item json")]):
            localize_evidence(result, language="zh")
        self.assertEqual(result["evidence_translation"]["status"], "partial_translation")
        self.assertEqual(result["evidence_translation"]["translated_count"], 4)
        self.assertEqual(result["evidence_translation"]["fallback_count"], 1)
        self.assertEqual(rows[0]["display_text"], "译文一")
        self.assertIn("翻译服务暂时未返回", rows[4]["display_text"])

    def test_english_program_fallback_contains_no_chinese(self):
        row = evidence("1", "A Air", "positive")
        result = {"language": "en", "aspect": "seat", "evidence": [row],
                  "rating_summary": {"available": True, "scale": 5, "airlines": {
                      "A Air": {"n": 100, "mean": 4.2, "median": 4.0, "std": .8,
                                "high_share": .8, "low_share": .1}}}}
        answer = answer_question("How are the seats?", result, use_model=False)
        self.assertIn("historical seat comfort rating", answer["text"])
        self.assertNotRegex(answer["text"], r"[\u3400-\u9fff]")

    def test_english_validation_rejects_chinese_output(self):
        ok, reasons = validate_answer("座椅不错。[1]", {"1"}, language="en")
        self.assertFalse(ok)
        self.assertIn("wrong_output_language", reasons)

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

    def test_narrow_route_without_direct_aspect_text_abstains(self):
        rows = [
            {**evidence("1", "Singapore Airlines", "positive"), "pred_aspect": "overall",
             "text": "This flight was slightly below standard but it was fine."},
            {**evidence("2", "Singapore Airlines", "positive"), "pred_aspect": "staff",
             "text": "The cabin crew were smiling and polite."},
        ]
        result = {
            "language": "zh", "aspect": "ground", "evidence": rows,
            "filters": {"airline": "Singapore Airlines", "route_from": ["Shanghai"],
                        "route_to": ["Singapore"]},
            "corpus_label_matches": {"sentence_count": 0, "review_count": 0,
                                     "labeled_review_count": 0},
            "rating_summary": {"available": True, "scale": 5, "minimum_reliable_n": 30,
                               "airlines": {"Singapore Airlines": {"mean": 4.5, "n": 2}},
                               "comparison": {"status": "not_applicable"}},
        }
        answer = answer_question("新航从上海飞新加坡的航线地勤怎么样", result, use_model=True)
        self.assertEqual(answer["mode"], "evidence_abstention")
        self.assertIn("只有 2 份原始地面服务评分", answer["text"])
        self.assertIn("没有评论原文直接讨论地面服务", answer["text"])
        self.assertIn("不能可靠判断这条精确航线的地面服务表现", answer["text"])
        self.assertNotIn("略低于平均水平", answer["text"])
        self.assertEqual(answer["citations"], [])

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
                  "rating_summary": {"available": False}, "evidence": [row],
                  "subtopic_analysis": {"portion": {"direct_review_count": 12, "polarity_summary": {
                      "labeled_review_count": 12, "positive_review_count": 0,
                      "negative_review_count": 12, "positive_share": 0.0,
                      "positive_share_ci95_low": 0.0, "positive_share_ci95_high": .24}}}}
        answer = answer_question("新航餐食分量怎么样", result, use_model=False)
        self.assertIn("关于新加坡航空的餐食分量", answer["text"])
        self.assertIn("完整直接匹配评论的自动标签总体偏负面", answer["text"])
        self.assertNotIn("这些细分问题需要分别看原文", answer["text"])

    def test_menu_question_lists_dishes_instead_of_rating_food(self):
        rows = [
            {**evidence("1", "Singapore Airlines", "positive"), "pred_aspect": "food",
             "text": "The beef rendang choice was served with rice.",
             "matched_subtopics": ["menu_items"], "direct_subtopic_matches": ["menu_items"]},
            {**evidence("2", "Singapore Airlines", "negative"), "pred_aspect": "food",
             "text": "Breakfast included sausage, mushrooms and omelette with fruit.",
             "matched_subtopics": ["menu_items"], "direct_subtopic_matches": ["menu_items"]},
        ]
        result = {"language": "zh", "aspect": "food",
                  "subtopics": {"menu_items": "specific dishes"}, "evidence": rows,
                  "rating_summary": {"available": True, "scale": 5, "airlines": {
                      "Singapore Airlines": {"mean": 3.87, "n": 1003}}}}
        with patch("src.grounded_qa.localize_evidence", return_value=result):
            answer = answer_question("新航的餐食有什么吃", result, use_model=True)
        self.assertEqual(answer["mode"], "menu_item_evidence")
        self.assertIn("牛肉仁当", answer["text"])
        self.assertIn("香肠", answer["text"])
        self.assertIn("煎蛋卷", answer["text"])
        self.assertNotIn("3.87", answer["text"])
        self.assertNotIn("正向", answer["text"])
        self.assertEqual(answer["citations"], ["1", "2"])

    def test_non_evaluative_fallback_does_not_substitute_parent_rating(self):
        row = {**evidence("1", "Singapore Airlines", "positive"),
               "text": "A power socket and USB charging port were available.",
               "matched_subtopics": ["power"], "direct_subtopic_matches": ["power"]}
        result = {"language": "en", "answer_intent": "availability", "aspect": "seat",
                  "subtopics": {"power": "seat charging"}, "evidence": [row],
                  "rating_summary": {"available": True, "scale": 5, "airlines": {
                      "Singapore Airlines": {"mean": 3.83, "n": 1015}}}}
        answer = answer_question("Does the seat have charging?", result, use_model=False)
        self.assertIn("power socket and USB charging", answer["text"])
        self.assertNotIn("3.83", answer["text"])
        self.assertNotIn("positive", answer["text"].lower())

    def test_validation_rejects_mixed_sentence_and_review_counts(self):
        text = ("自动标签汇总显示，749条评论中，正向评论884条，负向评论661条。"
                "一条评论提到餐食。[1]")
        ok, reasons = validate_answer(text, {"1"}, allow_aggregate_counts=True,
                                      evidence_numbers={"749", "884", "661"})
        self.assertFalse(ok)
        self.assertIn("incoherent_review_level_counts", reasons)

    def test_single_subtopic_direction_uses_full_aggregate_not_top_examples(self):
        negative_one = {**evidence("1", "Singapore Airlines", "negative"),
                        "text": "The seats were cramped and legroom was awful.",
                        "matched_subtopics": ["legroom"], "direct_subtopic_matches": ["legroom"]}
        negative_two = {**evidence("2", "Singapore Airlines", "negative"),
                        "text": "Leg and knee room were limited.",
                        "matched_subtopics": ["legroom"], "direct_subtopic_matches": ["legroom"]}
        positive = {**evidence("3", "Singapore Airlines", "positive"),
                    "text": "Legroom was generous and comfortable.",
                    "matched_subtopics": ["legroom"], "direct_subtopic_matches": ["legroom"]}
        result = {"aspect": "seat", "subtopics": {"legroom": "seat legroom"},
                  "rating_summary": {"available": False},
                  "evidence": [negative_one, negative_two, positive],
                  "subtopic_analysis": {"legroom": {"direct_review_count": 98, "polarity_summary": {
                      "labeled_review_count": 98, "positive_review_count": 78,
                      "negative_review_count": 20, "positive_share": .7959,
                      "positive_share_ci95_low": .706, "positive_share_ci95_high": .862}}}}
        text = answer_question("新航腿部空间怎么样", result, use_model=False)["text"]
        self.assertIn("完整直接匹配评论的自动标签总体偏正面", text)
        self.assertIn("正向 78 条（79.6%）、负向 20 条（20.4%）", text)
        self.assertIn("一条正向代表性评论", text)
        self.assertIn("一条负向反例", text)
        self.assertNotIn("偏负面", text)

    def test_representative_examples_avoid_obvious_label_text_conflicts(self):
        mislabeled = {**evidence("1", "AirAsia", "negative"),
                      "text": "He provided excellent professional customer service.",
                      "pred_aspect": "value", "matched_subtopics": ["support"]}
        consistent = {**evidence("2", "AirAsia", "negative"),
                      "text": "Customer service was rude and never helpful.",
                      "pred_aspect": "value", "matched_subtopics": ["support"]}
        positive = {**evidence("3", "AirAsia", "positive"),
                    "text": "Customer service was friendly and helpful.",
                    "pred_aspect": "value", "matched_subtopics": ["support"]}
        result = {"aspect": "value", "subtopics": {"support": "customer support"},
                  "rating_summary": {"available": False},
                  "evidence": [mislabeled, consistent, positive],
                  "subtopic_analysis": {"support": {"direct_review_count": 50, "polarity_summary": {
                      "labeled_review_count": 45, "positive_review_count": 5,
                      "negative_review_count": 40, "positive_share": .1111,
                      "positive_share_ci95_low": .05, "positive_share_ci95_high": .23}}}}
        text = answer_question("客服怎么样", result, use_model=False)["text"]
        self.assertIn("Customer service was rude and never helpful", text)
        self.assertNotIn("excellent professional customer service", text)

    def test_validation_rejects_direction_that_contradicts_full_aggregate(self):
        text = "关于座位腿部空间，完整评论偏负面，但自动标签中正向78条、负向20条。[1]"
        ok, reasons = validate_answer(
            text, {"1"}, required_subtopics=("legroom",),
            evidence_numbers={"78", "20"}, allow_aggregate_counts=True,
            expected_subtopic_directions={"legroom": "positive"})
        self.assertFalse(ok)
        self.assertIn("contradicts_full_subtopic_aggregate:legroom", reasons)

    def test_validation_allows_aggregate_scoped_overall_direction(self):
        text = ("这批历史评论中的座位腿部空间整体偏正面。\n\n"
                "自动标签为正向78条、负向20条；一条评论提到腿部空间充足。[1]")
        ok, reasons = validate_answer(
            text, {"1"}, required_subtopics=("legroom",), evidence_numbers={"78", "20"},
            allow_aggregate_counts=True, expected_subtopic_directions={"legroom": "positive"})
        self.assertTrue(ok, reasons)

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
