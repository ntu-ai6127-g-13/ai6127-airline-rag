"""Grounded answer generation with strict citation validation and fallback."""
from __future__ import annotations

import json
from functools import lru_cache
import re
import socket
import textwrap
import urllib.error
import urllib.request
from typing import Callable

from src.question_parser import SUBTOPIC_NAMES


DEFAULT_ENDPOINT = "http://127.0.0.1:8765"
_CITATION_RE = re.compile(r"\[([A-Z]?\d+)\]")
_PARAGRAPH_RE = re.compile(r"\n+")
_ASPECT_NAMES = {"seat": "座椅舒适度", "staff": "空乘服务", "food": "餐饮",
                 "entertainment": "机上娱乐", "ground": "地面服务", "value": "性价比",
                 "overall": "总体体验", "other": "其他服务"}
_AIRLINE_NAMES_ZH = {"Singapore Airlines": "新加坡航空", "United Airlines": "美国联合航空",
                     "Qatar Airways": "卡塔尔航空", "Cathay Pacific Airways": "国泰航空",
                     "British Airways": "英国航空", "Emirates": "阿联酋航空"}
_CABIN_NAMES_ZH = {"Economy Class": "经济舱", "Premium Economy": "超级经济舱",
                   "Business Class": "商务舱", "First Class": "头等舱"}
_SUBTOPIC_SUBJECTS = {
    "quality": "餐食质量与口味", "portion": "餐食分量", "variety": "餐食品种与选择",
    "drinks": "机上饮品", "special_meals": "特殊餐食", "temperature": "餐食温度",
    "content": "娱乐内容", "screen": "娱乐屏幕", "wifi": "机上 Wi-Fi",
    "audio": "耳机与音质", "usability": "娱乐系统操作", "legroom": "座位腿部空间",
    "width": "座位宽度", "recline": "座椅靠背调节与躺平", "comfort": "座椅舒适度",
    "condition": "座椅机械状况", "seat_cleanliness": "座椅清洁度",
    "power": "座位充电与电源", "attitude": "空乘服务态度",
    "efficiency": "空乘服务效率", "professionalism": "空乘专业程度",
    "availability": "空乘响应与在场服务", "punctuality": "航班准点表现",
    "delay_handling": "延误与取消处置", "baggage": "行李服务", "checkin": "值机服务",
    "gate_service": "登机口与登机服务", "lounge": "机场休息室", "price": "票价",
    "refund": "退款与改签", "support": "客服", "booking": "预订系统",
    "cleanliness": "客舱清洁", "toilet": "机上洗手间", "cabin_temperature": "客舱温度",
}
_ASPECT_NAMES_EN = {"seat": "seat comfort", "staff": "cabin crew service", "food": "food and beverages",
                    "entertainment": "inflight entertainment and Wi-Fi", "ground": "ground service",
                    "value": "value for money", "overall": "overall experience", "other": "other services"}
_SUBTOPIC_NAMES_EN = {
    "quality": "food quality and taste", "portion": "meal portion size", "variety": "meal variety and choice",
    "drinks": "onboard drinks", "special_meals": "special meals", "temperature": "meal temperature",
    "content": "entertainment content", "screen": "entertainment screen", "wifi": "onboard Wi-Fi",
    "audio": "headphones and audio quality", "usability": "entertainment system usability",
    "legroom": "seat legroom", "width": "seat width", "recline": "seat recline and lie-flat capability",
    "comfort": "seat comfort", "condition": "seat condition", "seat_cleanliness": "seat cleanliness",
    "power": "seat power and charging", "attitude": "crew attitude", "efficiency": "crew efficiency",
    "professionalism": "crew professionalism", "availability": "crew availability and responsiveness",
    "punctuality": "punctuality", "delay_handling": "delay and cancellation handling",
    "baggage": "baggage service", "checkin": "check-in service", "gate_service": "gate and boarding service",
    "lounge": "airport lounge", "price": "ticket price", "refund": "refunds and changes",
    "support": "customer support", "booking": "booking system", "cleanliness": "cabin cleanliness",
    "toilet": "onboard lavatory", "cabin_temperature": "cabin temperature",
}


def _citation_id(prefix: str, number: int) -> str:
    return f"{prefix}{number}"


def collect_evidence(result: dict) -> list[dict]:
    """Flatten a single/comparison query into uniquely cited evidence."""
    rows = []
    if result.get("type") == "multi_aspect":
        for position, (aspect, part) in enumerate(result["results"].items()):
            prefix = chr(ord("A") + position)
            for number, item in enumerate(part.get("evidence", []), 1):
                rows.append({**item, "citation_id": _citation_id(prefix, number),
                             "citation_prefix": prefix, "query_aspect": aspect})
    elif result.get("type") == "comparison":
        for position, (airline, part) in enumerate(result["results"].items()):
            prefix = chr(ord("A") + position)
            for number, item in enumerate(part.get("evidence", []), 1):
                rows.append({**item, "citation_id": _citation_id(prefix, number),
                             "citation_prefix": prefix, "airline": airline})
    else:
        for number, item in enumerate(result.get("evidence", []), 1):
            rows.append({**item, "citation_id": str(number), "citation_prefix": ""})
    return rows


def _rating_payload(summary: dict | None) -> dict | None:
    if not summary or not summary.get("available"):
        return None
    scale = summary["scale"]
    airlines = {}
    for airline, data in summary["airlines"].items():
        airlines[airline] = {"mean": f"{data['mean']:.2f}/{scale}" if data["mean"] is not None else None,
                             "rating_count": data["n"],
                             "median": f"{data['median']:.1f}/{scale}" if data["median"] is not None else None,
                             "high_rating_share": f"{data['high_share']*100:.1f}%" if data["high_share"] is not None else None,
                             "low_rating_share": f"{data['low_share']*100:.1f}%" if data["low_share"] is not None else None}
    comparison = dict(summary.get("comparison", {}))
    if comparison.get("difference") is not None:
        comparison["difference"] = f"{comparison['difference']:.2f}/{scale}"
    return {"source": "original review-level aspect ratings", "scale": f"1-{scale}",
            "airlines": airlines, "comparison": comparison}


def _allowed_numbers(summary: dict | None) -> set[str]:
    allowed = {"2002", "2019"}
    if summary and "rating_summaries" in summary:
        for child in summary["rating_summaries"].values():
            allowed.update(_allowed_numbers(child))
        return allowed
    if not summary or not summary.get("available"):
        return allowed
    scale = summary["scale"]
    allowed.update({"1", str(scale)})
    for data in summary["airlines"].values():
        allowed.add(str(data["n"]))
        for key, digits in (("mean", 2), ("median", 1)):
            if data.get(key) is not None: allowed.add(f"{data[key]:.{digits}f}")
        for key in ("high_share", "low_share"):
            if data.get(key) is not None: allowed.add(f"{data[key]*100:.1f}")
    comparison = summary.get("comparison", {})
    if comparison.get("difference") is not None:
        allowed.add(f"{comparison['difference']:.2f}")
    return allowed


def _prompt(question: str, result: dict, rows: list[dict]) -> list[dict]:
    language = result.get("language", "zh")
    aspect_names = _ASPECT_NAMES_EN if language == "en" else _ASPECT_NAMES
    subtopic_names = _SUBTOPIC_NAMES_EN if language == "en" else SUBTOPIC_NAMES
    aspect = result.get("aspect")
    aspect_name = aspect_names.get(aspect, "this aspect" if language == "en" else "该方面")
    evidence = [{
        "id": row["citation_id"], "airline": row.get("airline"), "cabin": row.get("cabin"),
        "date": row.get("date"), "route": row.get("route"),
        "text": (row.get("text") if language == "en" else row.get("text_zh") or row.get("display_text")),
        "matched_subtopics": [subtopic_names.get(topic, topic)
                              for topic in row.get("matched_subtopics", [])],
        "direct_subtopic_matches": [subtopic_names.get(topic, topic)
                                    for topic in row.get("direct_subtopic_matches", [])],
        "query_aspect": aspect_names.get(row.get("query_aspect"), row.get("query_aspect")),
        "predicted_aspect": row.get("pred_aspect"), "predicted_polarity": row.get("pred_polarity"),
    } for row in rows]
    subtopics = {topic: subtopic_names.get(topic, topic) for topic in result.get("subtopics", {})}
    requested_aspects = {aspect: aspect_names.get(aspect, aspect)
                         for aspect in result.get("aspects", [])}
    full_label_distributions = dict(result.get("label_distributions", {}))
    if not full_label_distributions and result.get("corpus_label_matches"):
        airline = result.get("filters", {}).get("airline")
        if isinstance(airline, (list, tuple, set)):
            airline = "所选航司"
        if not airline and rows:
            airline = rows[0].get("airline")
        full_label_distributions[str(airline or "所选航司")] = result["corpus_label_matches"]
    system = """你是航空评论分析助手。只能依据输入的原始评分汇总和评论证据作答，不能使用外部知识。
要求：
1. 第一行直接回答用户。若 comparison.status 是 clear_difference，必须明确说 winner 在这批历史评分中更好，不准用“褒贬不一”“各有优劣”回避结论。若是 no_clear_difference，说明评分差和为何不足以区分。
2. 评分均值、评分人数和差值只能逐字使用 rating_summary 中给出的数值；正负比例只能使用 full_label_distributions 或 comparison_subtopic_decisions 中的全量统计，不能统计页面展示的几条证据。
3. 评分汇总是主要定量依据，评论原文用来解释具体优缺点。引用格式严格为 [1]、[2] 或比较查询中的 [A1]、[B2]，只能使用存在的编号。
4. 自动分类标签仅供选证据参考，应按英文原文理解。不要把检索到的几句话当作总体好评率。
5. requested_subtopics 非空时，每个细分项必须单独成段，段与段之间空一行；单航司问题的每段以具体航司名和细分项开头，例如“关于新加坡航空的腿部空间”，不要写成“关于航司的腿部空间”。随后给出该项的发现和引用。禁止把多个细分项挤在同一段。优先引用 direct_subtopic_matches、其次引用 matched_subtopics 含对应名称的证据。原始方面评分只是上位方面的总体分，不能据此断言某个细分项好坏，也不能据此决定细分项比较的胜负；某项缺少直接证据就明确说明。回答中不要使用“多数、普遍、更多人、较多、整体、总体”等范围词，应写“检索到的原文同时包含……”并说明正反例。
6. 数据来自 2002–2019 年，只能描述这批历史评论，不能声称是当前服务。用自然、具体的中文，避免套话和“整体呈现某种倾向”。回答控制在 150–350 个中文字符。"""
    if subtopics:
        system = system.replace(
            "1. 第一行直接回答用户。若 comparison.status 是 clear_difference，必须明确说 winner 在这批历史评分中更好，不准用“褒贬不一”“各有优劣”回避结论。若是 no_clear_difference，说明评分差和为何不足以区分。",
            "1. 第一行直接回答用户，并按 requested_subtopics 给出具体发现。comparison 只是上位方面的评分比较，不能强制成为细分问题的结论。")
        if result.get("type") == "comparison":
            system += """
7. 这是细分属性的航司比较。必须逐家写出 parent_aspect_rating_summary 中的原始上位方面平均分、满分和评分份数，明确它不是细分项独立评分。上位 comparison 为 clear_difference 时，先明确说明 winner 在所问上位方面的历史评分更好。每个细分项严格服从 comparison_subtopic_decisions：clear_difference 才明确该项更支持 winner；no_clear_difference 应报告方向但说明差异不足以可靠区分；insufficient_evidence 才说证据不足。每个细分项把各航司的正向数量与百分比、负向数量与百分比合并在一句“自动标签汇总”中；这些汇总来自完整直接命中原文，不是页面展示的几条证据。随后用可见引用举例。没有直接原文的候选必须明确写出缺证据。"""
        else:
            system += """
7. subtopic_coverage 包含每个细分项完整直接命中原文的统计。若 polarity_summary 可用，在对应细分项段落里用一句话写出总评论数、正向数量与百分比、负向数量与百分比，并明确称为“自动标签汇总”。"""
    elif result.get("type") == "comparison":
        system += """
7. full_label_distributions 是各航司在当前结构化条件下、该上位方面的全量离线标签统计。用一句话分别写出每家航司的总评论数、正向数量与百分比、负向数量与百分比，并明确称为“自动标签汇总”。"""
    elif full_label_distributions:
        system += """
7. full_label_distributions 是当前结构化条件下、该方面的全量离线标签统计。用一句话写出总评论数、正向数量与百分比、负向数量与百分比，并明确称为“自动标签汇总”。"""
    if requested_aspects:
        system += """
7. requested_aspects 非空时，必须严格按照 requested_aspects 的顺序回答；每个评价方面单独成段，段与段之间空一行。每段必须明确写出方面名称、该方面的原始平均分、满分和评分份数，例如“原始餐饮评分为 3.80/5（900 份）”，再用属于该 query_aspect 的原文解释。不得用一个方面的评分回答另一个方面。回答控制在 220–550 个中文字符。"""
    if language == "en":
        system = """You are an airline review analysis assistant. Answer entirely in English and use only the supplied rating summaries, aggregate label statistics, and review evidence.
Rules:
1. Answer the question directly. If a parent rating comparison is marked clear_difference, name its winner. If it is no_clear_difference, report the gap and explain that it is not decisive.
2. Copy rating means, sample sizes, differences, and aggregate sentiment counts exactly from the payload. Never calculate sentiment from the few displayed quotations.
3. Use citations exactly as supplied, such as [1], [A1], or [B2]. Every substantive evidence claim needs a valid citation.
4. Automatic labels are supporting statistics rather than human gold labels. Describe the data as historical reviews from 2002–2019.
5. When requested_subtopics is nonempty, give each requested subtopic its own paragraph in the requested order. Parent-aspect ratings are context and cannot decide a subtopic winner. Follow comparison_subtopic_decisions: declare a subtopic winner only for clear_difference; describe direction without a winner for no_clear_difference; abstain only for insufficient_evidence.
6. When aggregate statistics are available, include each airline's total matched reviews and the positive and negative counts and percentages in one compact sentence. Then support the interpretation with representative evidence.
7. Translate nothing into Chinese and do not output Chinese characters. Write natural, specific English and avoid boilerplate. Keep the answer concise."""
    payload = {"question": question, "aspect": aspect, "filters": result.get("filters", {}),
               "requested_subtopics": subtopics,
               "rating_summary": None if subtopics else _rating_payload(result.get("rating_summary")),
               "parent_aspect_rating_summary": (_rating_payload(result.get("rating_summary"))
                                                if subtopics and result.get("type") == "comparison" else None),
               "comparison_subtopic_coverage": {
                   airline: part.get("subtopic_analysis", {})
                   for airline, part in result.get("results", {}).items()
               } if subtopics and result.get("type") == "comparison" else {},
               "subtopic_coverage": (result.get("subtopic_analysis", {})
                                     if subtopics and result.get("type") != "comparison" else {}),
               "comparison_subtopic_decisions": {
                   topic: _comparison_subtopic_decision(result, topic) for topic in subtopics
               } if subtopics and result.get("type") == "comparison" else {},
               "full_label_distributions": full_label_distributions,
               "requested_aspects": requested_aspects,
               "rating_summaries": {requested_aspects.get(key, key): _rating_payload(value)
                                    for key, value in result.get("rating_summaries", {}).items()},
               "evidence": evidence}
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def validate_answer(text: str, valid_ids: set[str], *, rating_summary: dict | None = None,
                    require_rating_winner: bool = True,
                    required_subtopics: tuple[str, ...] = (),
                    required_aspects: tuple[str, ...] = (),
                    evidence_numbers: set[str] | None = None,
                    require_all_rating_values: bool = False,
                    allow_aggregate_counts: bool = False,
                    language: str = "zh") -> tuple[bool, list[str]]:
    """Reject unknown/missing citations and obvious prompt-protocol leakage."""
    reasons = []
    clean = text.strip()
    cited = set(_CITATION_RE.findall(clean))
    if not clean: reasons.append("empty_answer")
    if not cited: reasons.append("missing_citations")
    if cited - valid_ids: reasons.append("unknown_citations:" + ",".join(sorted(cited - valid_ids)))
    if len(clean) > 1600: reasons.append("answer_too_long")
    if any(token in clean.casefold() for token in ("<|im_start|>", "program_computed_retrieved_sample", '"evidence"')):
        reasons.append("protocol_leak")
    if language == "en" and re.search(r"[\u3400-\u9fff]", clean):
        reasons.append("wrong_output_language")
    without_citations = _CITATION_RE.sub("", clean)
    numeric_tokens = [token.replace(",", "") for token in
                      re.findall(r"\d[\d,]*(?:\.\d+)?", without_citations)]
    unexpected = set(numeric_tokens) - (_allowed_numbers(rating_summary) | set(evidence_numbers or ()))
    if unexpected: reasons.append("unverified_numeric_claim:" + ",".join(sorted(unexpected)))
    if not allow_aggregate_counts and re.search(r"[零一二三四五六七八九十百]+\s*条", without_citations):
        reasons.append("retrieved_label_count_claim")
    comparison = (rating_summary or {}).get("comparison", {})
    if (require_rating_winner and comparison.get("status") == "clear_difference"
            and comparison.get("winner") not in clean):
        reasons.append("missing_required_winner")
    if require_all_rating_values and (rating_summary or {}).get("available"):
        for airline, data in rating_summary.get("airlines", {}).items():
            airline_names = {airline, _AIRLINE_NAMES_ZH.get(airline, airline)}
            if not any(name in clean for name in airline_names):
                reasons.append("missing_rating_airline:" + airline)
            expected = f"{data['mean']:.2f}" if data.get("mean") is not None else None
            if expected and expected not in clean:
                reasons.append("missing_rating_value:" + airline)
    paragraphs = [paragraph.strip() for paragraph in _PARAGRAPH_RE.split(clean) if paragraph.strip()]
    topic_paragraphs = {}
    for topic in required_subtopics:
        name = (_SUBTOPIC_NAMES_EN if language == "en" else SUBTOPIC_NAMES).get(topic, topic)
        containing = [index for index, paragraph in enumerate(paragraphs) if name in paragraph]
        topic_paragraphs[topic] = containing
        if not containing:
            reasons.append("missing_subtopic:" + topic)
    if len(required_subtopics) > 1:
        used = set()
        assigned = []
        names = [(_SUBTOPIC_NAMES_EN if language == "en" else SUBTOPIC_NAMES).get(topic, topic)
                 for topic in required_subtopics]
        for indices in topic_paragraphs.values():
            choices = [index for index in indices if index not in used]
            choices.sort(key=lambda index: (sum(name in paragraphs[index] for name in names), index))
            available = choices[0] if choices else None
            if available is not None:
                used.add(available)
                assigned.append(available)
        if len(used) != len([indices for indices in topic_paragraphs.values() if indices]):
            reasons.append("subtopics_not_separated")
        elif assigned != sorted(assigned):
            reasons.append("subtopics_out_of_order")
    aspect_paragraphs = {}
    summaries = (rating_summary or {}).get("rating_summaries", {})
    for aspect in required_aspects:
        name = (_ASPECT_NAMES_EN if language == "en" else _ASPECT_NAMES).get(aspect, aspect)
        containing = [index for index, paragraph in enumerate(paragraphs) if name in paragraph]
        aspect_paragraphs[aspect] = containing
        if not containing:
            reasons.append("missing_aspect:" + aspect)
            continue
        summary = summaries.get(aspect) or {}
        if summary.get("available"):
            data = next(iter(summary.get("airlines", {}).values()), {})
            expected = f"{data['mean']:.2f}" if data.get("mean") is not None else None
            has_rating = any(expected in paragraphs[index]
                             and f"/{summary['scale']}" in paragraphs[index].replace(" ", "")
                             for index in containing) if expected else True
            if not has_rating:
                reasons.append("missing_aspect_rating:" + aspect)
    if len(required_aspects) > 1:
        used = set()
        assigned = []
        names = [(_ASPECT_NAMES_EN if language == "en" else _ASPECT_NAMES).get(aspect, aspect)
                 for aspect in required_aspects]
        for indices in aspect_paragraphs.values():
            choices = [index for index in indices if index not in used]
            choices.sort(key=lambda index: (sum(name in paragraphs[index] for name in names), index))
            available = choices[0] if choices else None
            if available is not None:
                used.add(available)
                assigned.append(available)
        if len(used) != len([indices for indices in aspect_paragraphs.values() if indices]):
            reasons.append("aspects_not_separated")
        elif assigned != sorted(assigned):
            reasons.append("aspects_out_of_order")
    if required_subtopics and re.search(
            r"多数|普遍|大部分|更多(?:人|旅客|评论)|较多(?:人|旅客|评论)|整体|总体", clean):
        reasons.append("unsupported_prevalence_claim")
    # Each substantive paragraph must contain a citation.  A paragraph can open
    # with a short conclusion and immediately substantiate it in later cited
    # sentences, which is the normal answer form produced by the local model.
    # Generic caveats are allowed without citations because they describe system
    # limits, not the review evidence.
    exemptions = (("data", "sample", "retriev", "does not represent", "insufficient", "automatic-label", "rating")
                  if language == "en" else
                  ("数据", "样本", "检索", "不能代表", "证据不足", "自动标签", "截至", "评分"))
    for paragraph in paragraphs:
        aspect_mentions = sum(_ASPECT_NAMES.get(aspect, aspect) in paragraph for aspect in required_aspects)
        if aspect_mentions >= 2 and any(term in paragraph for term in ("如下", "分别")):
            continue
        searchable = paragraph.casefold() if language == "en" else paragraph
        if len(paragraph) < 8 or _CITATION_RE.search(paragraph) or any(term in searchable for term in exemptions):
            continue
        reasons.append("uncited_claim")
        break
    return not reasons, reasons


def _comparison_subtopic_decision(result: dict, topic: str) -> dict:
    summaries = {}
    for airline, part in result.get("results", {}).items():
        analysis = part.get("subtopic_analysis", {}).get(topic, {})
        polarity = analysis.get("polarity_summary") or {}
        summaries[airline] = {
            "direct_review_count": analysis.get("direct_review_count", 0),
            **polarity,
        }
    eligible = [(airline, data) for airline, data in summaries.items()
                if data.get("labeled_review_count", 0) >= 10
                and data.get("positive_share") is not None]
    if len(eligible) != len(summaries) or len(eligible) < 2:
        return {"status": "insufficient_evidence", "summaries": summaries,
                "minimum_labeled_reviews": 10}
    ranked = sorted(eligible, key=lambda item: (-item[1]["positive_share"], item[0]))
    leader, leader_data = ranked[0]
    runner, runner_data = ranked[1]
    difference = leader_data["positive_share"] - runner_data["positive_share"]
    decisive = (leader_data.get("positive_share_ci95_low") is not None
                and runner_data.get("positive_share_ci95_high") is not None
                and leader_data["positive_share_ci95_low"] > runner_data["positive_share_ci95_high"])
    return {"status": "clear_difference" if decisive else "no_clear_difference",
            "winner": leader if decisive else None, "leader": leader, "runner_up": runner,
            "positive_share_difference": round(difference, 4), "summaries": summaries,
            "minimum_labeled_reviews": 10}


def _polarity_distribution_sentence(summaries: dict[str, dict], *, direct: bool = False) -> str:
    """Format one compact sentence from full-corpus, review-level label totals."""
    parts = []
    for airline, data in summaries.items():
        labeled = int(data.get("labeled_review_count") or 0)
        positive = int(data.get("positive_review_count") or 0)
        negative = int(data.get("negative_review_count") or 0)
        if not labeled:
            continue
        positive_share = positive / labeled * 100
        negative_share = negative / labeled * 100
        label = _AIRLINE_NAMES_ZH.get(airline, airline)
        matched = int(data.get("direct_review_count") or data.get("review_count") or labeled)
        parts.append(
            f"{label}共命中 {matched:,} 条，其中可明确归类的 {labeled:,} 条里正向 "
            f"{positive:,} 条（{positive_share:.1f}%）、"
            f"负向 {negative:,} 条（{negative_share:.1f}%）"
        )
    if not parts:
        return ""
    scope = "全部直接相关评论" if direct else "该方面全部相关评论"
    return f"{scope}的自动标签汇总：" + "；".join(parts) + "。"


def _polarity_distribution_sentence_en(summaries: dict[str, dict], *, direct: bool = False) -> str:
    parts = []
    for airline, data in summaries.items():
        labeled = int(data.get("labeled_review_count") or 0)
        positive = int(data.get("positive_review_count") or 0)
        negative = int(data.get("negative_review_count") or 0)
        if not labeled:
            continue
        matched = int(data.get("direct_review_count") or data.get("review_count") or labeled)
        parts.append(
            f"{airline}: {matched:,} matched reviews; among the {labeled:,} with a clear direction, "
            f"{positive:,} were positive ({positive / labeled * 100:.1f}%) and "
            f"{negative:,} were negative ({negative / labeled * 100:.1f}%)"
        )
    if not parts:
        return ""
    scope = "Directly relevant review-level automatic-label summary" if direct else "Aspect-level automatic-label summary"
    return scope + ": " + "; ".join(parts) + "."


def _fallback_answer_en(result: dict, rows: list[dict]) -> str:
    if not rows:
        return ("The current filters do not provide enough review evidence to answer reliably. "
                "Try relaxing the airline, cabin, route, or date constraints.")
    aspect = result.get("aspect")
    aspect_name = _ASPECT_NAMES_EN.get(aspect, "this aspect")
    rating = result.get("rating_summary") or {}
    by_airline: dict[str, list[dict]] = {}
    for row in rows:
        by_airline.setdefault(str(row.get("airline") or "Selected airline"), []).append(row)

    def quote(row: dict) -> str:
        return textwrap.shorten(str(row.get("text", "")).strip(), width=155, placeholder="…")

    if result.get("type") == "multi_aspect":
        sections = []
        for key in result.get("aspects", []):
            name = _ASPECT_NAMES_EN.get(key, key)
            summary = result.get("rating_summaries", {}).get(key) or {}
            aspect_rows = [row for row in rows if row.get("query_aspect") == key]
            airline = str((aspect_rows[0].get("airline") if aspect_rows else "") or "Selected airline")
            data = next(iter(summary.get("airlines", {}).values()), {})
            rating_text = (f"the original {name} rating was {data['mean']:.2f}/{summary['scale']} "
                           f"from {data['n']:,} ratings" if summary.get("available") and data.get("mean") is not None
                           else f"the source table has no independent {name} rating")
            examples = "; ".join(f'“{quote(row)}”[{row["citation_id"]}]' for row in aspect_rows[:2])
            sections.append(f"For {airline}'s {name}, {rating_text}. Representative evidence: {examples}.")
        return "\n\n".join(sections) + "\n\nThe ratings and reviews cover 2002–2019 and do not describe current service."

    subtopics = result.get("subtopics") or {}
    if subtopics:
        sections = []
        comparison_query = result.get("type") == "comparison"
        if comparison_query and rating.get("available"):
            rating_parts = [f"{airline} {data['mean']:.2f}/{rating['scale']} ({data['n']:,} ratings)"
                            for airline, data in rating.get("airlines", {}).items() if data.get("mean") is not None]
            comparison = rating.get("comparison", {})
            if comparison.get("status") == "clear_difference":
                lead = (f"At the broader {aspect_name} level, {comparison['winner']} had the stronger historical rating, "
                        f"leading by {comparison['difference']:.2f} points.")
            elif comparison.get("status") == "no_clear_difference":
                lead = (f"At the broader {aspect_name} level, the {comparison['difference']:.2f}-point gap was not decisive.")
            else:
                lead = f"The broader {aspect_name} ratings were insufficient for a reliable ranking."
            sections.append(lead + " Original ratings: " + "; ".join(rating_parts)
                            + ". These are parent-aspect ratings, not independent subtopic scores.")
        for topic in subtopics:
            topic_name = _SUBTOPIC_NAMES_EN.get(topic, topic)
            matches = [row for row in rows if topic in row.get(
                "direct_subtopic_matches" if comparison_query else "matched_subtopics", [])]
            if comparison_query:
                decision = _comparison_subtopic_decision(result, topic)
                summaries = decision.get("summaries", {})
                chosen = []
                for airline in by_airline:
                    candidates = [row for row in matches if row.get("airline") == airline]
                    if candidates:
                        share = summaries.get(airline, {}).get("positive_share")
                        preferred = "positive" if share is not None and share >= .5 else "negative"
                        chosen.append(next((row for row in candidates if row.get("pred_polarity") == preferred), candidates[0]))
                aggregate = _polarity_distribution_sentence_en(summaries, direct=True)
                if decision.get("status") == "clear_difference":
                    finding = f"the full direct-evidence aggregate clearly supports {decision['winner']}"
                elif decision.get("status") == "no_clear_difference":
                    finding = f"the aggregate leans toward {decision['leader']}, but the gap is not decisive"
                else:
                    finding = "the direct evidence is insufficient to name a reliable winner"
            else:
                chosen = matches[:2]
                aggregate_data = result.get("subtopic_analysis", {}).get(topic, {}).get("polarity_summary")
                airline = next(iter(by_airline), "Selected airline")
                aggregate = _polarity_distribution_sentence_en({airline: aggregate_data}, direct=True) if aggregate_data else ""
                polarities = {row.get("pred_polarity") for row in chosen}
                finding = ("the directly relevant reviews lean positive" if polarities == {"positive"} else
                           "the directly relevant reviews lean negative" if polarities == {"negative"} else
                           "the directly relevant reviews include both positive and negative experiences")
            examples = "; ".join(
                f'A {row.get("airline", "selected-airline")} passenger wrote, “{quote(row)}”[{row["citation_id"]}]'
                for row in chosen)
            sections.append(f"For {topic_name}, {finding}. {aggregate} {examples}.".strip())
        return "\n\n".join(sections) + "\n\nThese are individual historical experiences and do not represent every flight."

    comparison = rating.get("comparison", {})
    distribution = _polarity_distribution_sentence_en(result.get("label_distributions", {}))
    if rating.get("available") and comparison.get("status") == "clear_difference":
        winner, runner = comparison["winner"], comparison["runner_up"]
        first, second = rating["airlines"][winner], rating["airlines"][runner]
        examples = "; ".join(f'{row.get("airline")} wrote, “{quote(row)}”[{row["citation_id"]}]' for row in rows[:2])
        return (f"{winner} was clearly stronger in these historical {aspect_name} ratings: "
                f"{first['mean']:.2f}/{rating['scale']} from {first['n']:,} ratings versus "
                f"{second['mean']:.2f}/{rating['scale']} from {second['n']:,} for {runner}, a "
                f"{comparison['difference']:.2f}-point gap. {distribution} {examples}. "
                "The data cover 2002–2019 and do not represent current service.")
    if rating.get("available") and comparison.get("status") == "no_clear_difference":
        leader, runner = comparison["leader"], comparison["runner_up"]
        first, second = rating["airlines"][leader], rating["airlines"][runner]
        return (f"These historical ratings do not reliably distinguish the airlines. {leader} scored "
                f"{first['mean']:.2f}/{rating['scale']} from {first['n']:,} ratings and {runner} scored "
                f"{second['mean']:.2f}/{rating['scale']} from {second['n']:,}, a "
                f"{comparison['difference']:.2f}-point gap. {distribution}")
    if rating.get("available") and len(rating.get("airlines", {})) == 1:
        airline, data = next(iter(rating["airlines"].items()))
        distribution = _polarity_distribution_sentence_en(
            {airline: result.get("corpus_label_matches")}) if result.get("corpus_label_matches") else ""
        examples = "; ".join(f'“{quote(row)}”[{row["citation_id"]}]' for row in rows[:2])
        return (f"{airline}'s historical {aspect_name} rating was {data['mean']:.2f}/{rating['scale']} from "
                f"{data['n']:,} ratings. {distribution} Representative evidence: {examples}. Data end in 2019.")
    examples = "; ".join(f'“{quote(row)}”[{row["citation_id"]}]' for row in rows[:2])
    return f"The source table does not provide enough comparable ratings for a firm conclusion. Available evidence: {examples}."


def fallback_answer(question: str, result: dict, rows: list[dict]) -> str:
    """Produce a useful answer without an LLM, grounded in visible evidence."""
    del question
    if result.get("language") == "en":
        return _fallback_answer_en(result, rows)
    if result.get("type") == "multi_aspect":
        sections = []
        for aspect in result.get("aspects", []):
            aspect_name = _ASPECT_NAMES.get(aspect, aspect)
            summary = result.get("rating_summaries", {}).get(aspect) or {}
            aspect_rows = [row for row in rows if row.get("query_aspect") == aspect]
            airline = str((aspect_rows[0].get("airline") if aspect_rows else "") or "")
            airline_label = _AIRLINE_NAMES_ZH.get(airline, airline or "所选航司")
            if summary.get("available"):
                data = next(iter(summary.get("airlines", {}).values()), {})
                if data.get("mean") is None:
                    rating_text = f"当前条件下没有可用的原始{aspect_name}评分"
                else:
                    rating_text = (f"原始{aspect_name}评分为 {data['mean']:.2f}/{summary['scale']}"
                                   f"（{data['n']:,} 份）")
            else:
                rating_text = f"原始表没有独立的{aspect_name}评分"
            positive = next((row for row in aspect_rows if row.get("pred_polarity") == "positive"), None)
            negative = next((row for row in aspect_rows if row.get("pred_polarity") == "negative"), None)
            selected = []
            for row in (positive, negative, *aspect_rows):
                if row is not None and row not in selected:
                    selected.append(row)
                if len(selected) == 2:
                    break
            details = []
            for index, row in enumerate(selected):
                prefix = "一条评论写道" if index == 0 else "另一条评论写道"
                quote = textwrap.shorten(str(row.get("text_zh") or row["text"]).strip(), width=125, placeholder="…")
                details.append(f"{prefix}：“{quote}”[{row['citation_id']}]")
            evidence_text = "；".join(details) if details else "当前筛选条件下没有检索到可核对的原文"
            distribution = (result.get("results", {}).get(aspect, {})
                            .get("corpus_label_matches"))
            distribution_text = _polarity_distribution_sentence(
                {airline_label: distribution}) if distribution else ""
            sections.append(f"关于{airline_label}的{aspect_name}，{rating_text}。"
                            f"{distribution_text}{evidence_text}。")
        return "\n\n".join(sections) + "\n\n评分和原文来自 2002–2019 年的历史评论，不能代表当前服务。"
    if not rows:
        return "当前筛选条件下没有检索到足够的评论证据，因此暂时无法回答。可以放宽航司、舱位、航线或日期条件后重试。"
    aspect = result.get("aspect")
    aspect_name = _ASPECT_NAMES.get(aspect, "该方面")
    rating = result.get("rating_summary") or {}
    by_airline: dict[str, list[dict]] = {}
    for row in rows:
        by_airline.setdefault(str(row.get("airline") or "所选航司"), []).append(row)
    def representative(airline: str, polarity: str) -> dict | None:
        candidates = [row for row in by_airline.get(airline, []) if row.get("pred_aspect") == aspect
                      and row.get("pred_polarity") == polarity]
        return candidates[0] if candidates else (by_airline.get(airline) or [None])[0]
    def quote(row: dict) -> str:
        return textwrap.shorten(str(row.get("text_zh") or row["text"]).strip(), width=145, placeholder="…")
    subtopics = result.get("subtopics") or {}
    if subtopics:
        sections = []
        comparison_query = result.get("type") == "comparison"
        if comparison_query and rating.get("available"):
            rating_parts = []
            for airline, data in rating.get("airlines", {}).items():
                label = _AIRLINE_NAMES_ZH.get(airline, airline)
                if data.get("mean") is None:
                    rating_parts.append(f"{label}无可用评分")
                else:
                    rating_parts.append(f"{label} {data['mean']:.2f}/{rating['scale']}（{data['n']:,} 份）")
            comparison = rating.get("comparison", {})
            if comparison.get("status") == "clear_difference":
                winner = comparison["winner"]
                winner_label = _AIRLINE_NAMES_ZH.get(winner, winner)
                lead = (f"就整体{aspect_name}而言，{winner_label}在这批同条件历史评分中明确更好，"
                        f"领先第二名 {comparison['difference']:.2f} 分。")
            elif comparison.get("status") == "no_clear_difference":
                lead = (f"就整体{aspect_name}而言，现有评分差 {comparison['difference']:.2f} 分，"
                        "不足以可靠区分胜负。")
            else:
                lead = f"就整体{aspect_name}而言，当前同条件评分样本不足以作可靠排名。"
            sections.append(lead + f"原始{aspect_name}评分为：" + "；".join(rating_parts)
                            + "。这些是上位方面评分，不是所问细分项的独立评分。")
        for topic in subtopics:
            matches = sorted(
                (row for row in rows if topic in (row.get("direct_subtopic_matches", [])
                                                  if comparison_query else row.get("matched_subtopics", []))),
                key=lambda row: topic not in row.get("direct_subtopic_matches", []))
            if comparison_query:
                chosen = []
                used_airlines = set()
                decision = _comparison_subtopic_decision(result, topic)
                summaries = decision.get("summaries", {})
                for airline in by_airline:
                    candidates = [row for row in matches if str(row.get("airline") or "") == airline]
                    if not candidates:
                        continue
                    share = summaries.get(airline, {}).get("positive_share")
                    preferred = "positive" if share is not None and share >= .5 else "negative"
                    selected = next((row for row in candidates
                                     if row.get("pred_polarity") == preferred), candidates[0])
                    chosen.append(selected)
                    used_airlines.add(airline)
                missing_airlines = [airline for airline in by_airline if airline not in used_airlines]
            else:
                chosen = matches[:2]
                missing_airlines = []
            name = _SUBTOPIC_SUBJECTS.get(topic, SUBTOPIC_NAMES.get(topic, topic))
            if not chosen:
                if comparison_query:
                    labels = "、".join(_AIRLINE_NAMES_ZH.get(airline, airline) for airline in by_airline)
                    sections.append(f"关于{name}，{labels}目前都没有检索到直接提及该细分项的原文，无法判断哪家最好。")
                else:
                    sections.append(f"关于{name}，目前没有找到足够直接的评论依据")
                continue
            polarities = [row.get("pred_polarity") for row in chosen
                          if row.get("pred_polarity") in {"positive", "negative"}]
            if polarities and all(value == "negative" for value in polarities):
                direction = "检索到的直接评论偏负面"
            elif polarities and all(value == "positive" for value in polarities):
                direction = "检索到的直接评论偏正面"
            elif {"positive", "negative"}.issubset(set(polarities)):
                direction = "检索到的直接评论看法不一"
            else:
                direction = "检索到的直接评论提供了一些具体信息"
            details = []
            for row_index, row in enumerate(chosen):
                airline = str(row.get("airline") or "")
                airline_label = _AIRLINE_NAMES_ZH.get(airline, airline)
                cabin = str(row.get("cabin") or "")
                cabin_label = _CABIN_NAMES_ZH.get(cabin, cabin)
                if comparison_query:
                    prefix = f"{airline_label}的旅客写道"
                elif cabin_label:
                    prefix = f"{'一位' if row_index == 0 else '另一位'}{cabin_label}旅客写道"
                else:
                    prefix = "一位旅客写道" if row_index == 0 else "另一位旅客写道"
                details.append(f"{prefix}：“{quote(row)}”[{row['citation_id']}]")
            if comparison_query:
                details.extend(f"{_AIRLINE_NAMES_ZH.get(airline, airline)}当前没有检索到直接提及该细分项的原文"
                               for airline in missing_airlines)
            airline = str(chosen[0].get("airline") or "")
            airline_label = _AIRLINE_NAMES_ZH.get(airline, airline)
            subject = f"{airline_label}的{name}" if airline_label and not comparison_query else name
            if comparison_query:
                aggregate = _polarity_distribution_sentence(
                    decision.get("summaries", {}), direct=True)
                if decision.get("status") == "clear_difference":
                    winner = decision["winner"]
                    direction = (f"完整直接原文的自动标签汇总明确更支持"
                                 f"{_AIRLINE_NAMES_ZH.get(winner, winner)}。{aggregate}")
                elif decision.get("status") == "no_clear_difference":
                    leader = decision["leader"]
                    direction = (f"自动标签汇总暂时偏向{_AIRLINE_NAMES_ZH.get(leader, leader)}，"
                                 f"但差异不足以可靠判定胜负。{aggregate}")
                else:
                    direction = f"现有直接原文仍不足以可靠判定哪家最好。{aggregate}"
            else:
                polarity_summary = (result.get("subtopic_analysis", {}).get(topic, {})
                                    .get("polarity_summary"))
                if polarity_summary:
                    direction += "。" + _polarity_distribution_sentence(
                        {airline_label or "所选航司": polarity_summary}, direct=True)
            sections.append(f"关于{subject}，{direction.rstrip('。')}。" + "；".join(details) + "。")
        return "\n\n".join(sections) + "\n\n这些评论记录的是具体旅客经历，不能代表所有航班。"
    if rating.get("available") and rating.get("comparison", {}).get("status") == "clear_difference":
        comparison = rating["comparison"]
        winner, runner = comparison["winner"], comparison["runner_up"]
        winner_label, runner_label = _AIRLINE_NAMES_ZH.get(winner, winner), _AIRLINE_NAMES_ZH.get(runner, runner)
        winner_data, runner_data = rating["airlines"][winner], rating["airlines"][runner]
        good, bad = representative(winner, "positive"), representative(runner, "negative")
        details = []
        if good: details.append(f"{winner_label}的旅客提到“{quote(good)}”[{good['citation_id']}]")
        if bad: details.append(f"{runner_label}的旅客则提到“{quote(bad)}”[{bad['citation_id']}]")
        distribution = _polarity_distribution_sentence(result.get("label_distributions", {}))
        return (f"{winner_label}明显更好，而且评分差距很大。原始{aspect_name}评分中，{winner_label}为"
                f"{winner_data['mean']:.2f}/{rating['scale']}（{winner_data['n']:,}份），{runner_label}为"
                f"{runner_data['mean']:.2f}/{rating['scale']}（{runner_data['n']:,}份），相差{comparison['difference']:.2f}分；"
                f"{winner_label}的高分占比为{winner_data['high_share']*100:.1f}%，{runner_label}仅为{runner_data['high_share']*100:.1f}%，"
                f"后者的低分占比达到{runner_data['low_share']*100:.1f}%。{distribution}"
                + "；".join(details) + "。评分和原文均来自2002–2019年的评论，不能直接代表当前服务。")
    if rating.get("available") and rating.get("comparison", {}).get("status") == "no_clear_difference":
        comparison = rating["comparison"]
        leader, runner = comparison["leader"], comparison["runner_up"]
        leader_label, runner_label = _AIRLINE_NAMES_ZH.get(leader, leader), _AIRLINE_NAMES_ZH.get(runner, runner)
        first, second = rating["airlines"][leader], rating["airlines"][runner]
        distribution = _polarity_distribution_sentence(result.get("label_distributions", {}))
        return (f"这批评分不足以可靠地区分谁更好。{leader_label}为{first['mean']:.2f}/{rating['scale']}（{first['n']:,}份），"
                f"{runner_label}为{second['mean']:.2f}/{rating['scale']}（{second['n']:,}份），只差{comparison['difference']:.2f}分，"
                f"没有达到预设的实际意义门槛；此时硬给出胜负会夸大数据差异。{distribution}")
    if rating.get("available") and len(rating.get("airlines", {})) == 1:
        airline, data = next(iter(rating["airlines"].items()))
        airline_label = _AIRLINE_NAMES_ZH.get(airline, airline)
        good, bad = representative(airline, "positive"), representative(airline, "negative")
        interpretation = "评价较好" if data["mean"] >= rating["scale"] * .7 else (
            "评价偏低" if data["mean"] < rating["scale"] * .55 else "评价处于中间水平")
        cited = [row for row in (good, bad) if row]
        detail = "；".join(f"“{quote(row)}”[{row['citation_id']}]" for row in cited)
        distribution = _polarity_distribution_sentence(
            {airline: result.get("corpus_label_matches")}) if result.get("corpus_label_matches") else ""
        return (f"{airline_label}的{aspect_name}历史评分为{data['mean']:.2f}/{rating['scale']}，共{data['n']:,}份评分，{interpretation}。"
                f"{distribution}代表性原文包括{detail}。数据截至2019年。")
    cited = rows[:2]
    details = "；".join(f"“{quote(row)}”[{row['citation_id']}]" for row in cited)
    return f"原始评论表没有足够的同口径评分来形成明确结论。目前能核对的原文包括{details}。"


def model_health(endpoint: str = DEFAULT_ENDPOINT, timeout: float = 0.5) -> dict:
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/health", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError, socket.timeout):
        return {"status": "unavailable"}


StreamCallback = Callable[[str, str], None]


def _generate(messages: list[dict], *, endpoint: str, timeout: float,
              on_stream: StreamCallback | None = None, max_tokens: int = 480) -> dict:
    body = json.dumps({"messages": messages, "max_tokens": max_tokens,
                       "temperature": 0.0}, ensure_ascii=False).encode("utf-8")
    path = "/generate_stream" if on_stream else "/generate"
    request = urllib.request.Request(endpoint.rstrip("/") + path, data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if on_stream is None:
            return json.loads(response.read().decode("utf-8"))
        chunks = []
        seconds = 0.0
        completed = False
        for raw_line in response:
            if not raw_line.strip():
                continue
            event = json.loads(raw_line.decode("utf-8"))
            if event.get("type") == "delta":
                chunks.append(str(event.get("text", "")))
                on_stream("".join(chunks), "delta")
            elif event.get("type") == "done":
                seconds = float(event.get("seconds", 0.0))
                completed = True
            elif event.get("type") == "error":
                raise RuntimeError(str(event.get("error", "stream_generation_failed")))
        if not completed:
            raise RuntimeError("stream_ended_without_completion")
        return {"text": "".join(chunks).strip(), "seconds": seconds}


def _evidence_items(result: dict) -> list[dict]:
    if result.get("results"):
        return [item for part in result["results"].values() for item in part.get("evidence", [])]
    return list(result.get("evidence", []))


@lru_cache(maxsize=512)
def _translate_batch_cached(texts: tuple[str, ...], endpoint: str) -> tuple[str, ...]:
    numbered = [{"id": index, "text": text} for index, text in enumerate(texts)]
    messages = [
        {"role": "system", "content":
         "Translate airline passenger review excerpts from English into concise, natural Simplified Chinese. "
         "Preserve concrete details, tone, negation, cabin products, numbers, and airline names. "
         "Return only valid JSON in the exact form {\"translations\":[{\"id\":0,\"text\":\"...\"}]}. "
         "Do not add explanations and do not omit any item."},
        {"role": "user", "content": json.dumps({"items": numbered}, ensure_ascii=False)},
    ]
    payload = _generate(messages, endpoint=endpoint, timeout=90.0, max_tokens=1200)
    raw = str(payload.get("text", "")).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("translation_json_missing")
    parsed = json.loads(raw[start:end + 1])
    translated = {int(item["id"]): str(item["text"]).strip()
                  for item in parsed.get("translations", [])}
    if set(translated) != set(range(len(texts))) or any(not value for value in translated.values()):
        raise ValueError("translation_items_incomplete")
    return tuple(translated[index] for index in range(len(texts)))


def localize_evidence(result: dict, *, language: str, endpoint: str = DEFAULT_ENDPOINT) -> dict:
    """Attach display-language evidence without replacing the source text."""
    result["language"] = language
    items = _evidence_items(result)
    if language == "en":
        for item in items:
            item["display_text"] = str(item.get("text", ""))
        result["evidence_translation"] = {"status": "not_needed", "language": "en"}
        return result
    unique = list(dict.fromkeys(str(item.get("text", "")).strip() for item in items if item.get("text")))
    translated_by_source: dict[str, str] = {}
    try:
        for offset in range(0, len(unique), 8):
            batch = tuple(unique[offset:offset + 8])
            translations = _translate_batch_cached(batch, endpoint)
            translated_by_source.update(zip(batch, translations))
        status = "translated"
    except (OSError, ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError, socket.timeout):
        status = "fallback_summary"
    for item in items:
        source = str(item.get("text", "")).strip()
        translated = translated_by_source.get(source)
        if not translated:
            aspect = _ASPECT_NAMES.get(item.get("pred_aspect"), "所问方面")
            polarity = {"positive": "正面", "negative": "负面"}.get(item.get("pred_polarity"), "相关")
            translated = f"这条评论被自动识别为关于{aspect}的{polarity}评价；翻译服务暂时未返回逐句译文。"
        item["text_zh"] = translated
        item["display_text"] = translated
    result["evidence_translation"] = {"status": status, "language": "zh", "translated_count": len(items)}
    return result


def answer_question(question: str, result: dict, *, endpoint: str = DEFAULT_ENDPOINT,
                    timeout: float = 45.0, use_model: bool = True,
                    on_stream: StreamCallback | None = None) -> dict:
    """Generate, validate, and fall back without ever losing the evidence."""
    language = result.get("language", "zh")
    # Legacy/library callers that do not declare a language keep the historical
    # behavior.  The UI always sets this field and therefore gets strict
    # single-language evidence and answers.
    if "language" in result:
        localize_evidence(result, language=language, endpoint=endpoint)
    rows = collect_evidence(result)
    valid_ids = {row["citation_id"] for row in rows}
    fallback = fallback_answer(question, result, rows)
    rating_comparison = (result.get("rating_summary") or {}).get("comparison", {})
    if (not result.get("subtopics") and result.get("type") == "comparison"
            and rating_comparison.get("status") in {"clear_difference", "no_clear_difference"}):
        return {"text": fallback, "mode": "rating_decision", "valid": True,
                "validation_errors": [], "citations": sorted(_CITATION_RE.findall(fallback)),
                "model_seconds": 0.0}
    if not use_model or not rows:
        return {"text": fallback, "mode": "program_fallback", "valid": True,
                "validation_errors": [], "citations": sorted(_CITATION_RE.findall(fallback)), "model_seconds": 0.0}
    messages = _prompt(question, result, rows)
    if result.get("type") == "multi_aspect":
        validation_rating = {"rating_summaries": result.get("rating_summaries", {})}
    elif result.get("subtopics") and result.get("type") == "comparison":
        validation_rating = result.get("rating_summary")
    else:
        validation_rating = None if result.get("subtopics") else result.get("rating_summary")
    required_aspects = tuple(result.get("aspects", ())) if result.get("type") == "multi_aspect" else ()
    require_all_rating_values = bool(result.get("subtopics") and result.get("type") == "comparison"
                                     and (result.get("rating_summary") or {}).get("available"))
    evidence_numbers = {token.replace(",", "") for row in rows for token in
                        re.findall(r"\d[\d,]*(?:\.\d+)?", str(row.get("text", "")))}
    allow_aggregate_counts = bool(
        result.get("corpus_label_matches") or result.get("label_distributions")
        or result.get("subtopics")
    )
    if allow_aggregate_counts:
        aggregate_parts = list(result.get("results", {}).values())
        aggregate_parts.append(result)
        for part in aggregate_parts:
            full = part.get("corpus_label_matches") or {}
            for value in full.values():
                if isinstance(value, int) and not isinstance(value, bool):
                    evidence_numbers.add(str(value))
                elif isinstance(value, float):
                    evidence_numbers.update({f"{value:.4f}", f"{value*100:.1f}"})
            full_total = int(full.get("labeled_review_count") or 0)
            if full_total:
                for key in ("positive_review_count", "negative_review_count"):
                    evidence_numbers.add(f"{int(full.get(key) or 0) / full_total * 100:.1f}")
            for analysis in part.get("subtopic_analysis", {}).values():
                for value in analysis.values():
                    if isinstance(value, int) and not isinstance(value, bool):
                        evidence_numbers.add(str(value))
                polarity_summary = analysis.get("polarity_summary") or {}
                for value in polarity_summary.values():
                    if isinstance(value, int) and not isinstance(value, bool):
                        evidence_numbers.add(str(value))
                    elif isinstance(value, float):
                        evidence_numbers.update({f"{value:.4f}", f"{value*100:.1f}"})
                polarity_total = int(polarity_summary.get("labeled_review_count") or 0)
                if polarity_total:
                    for key in ("positive_review_count", "negative_review_count"):
                        evidence_numbers.add(
                            f"{int(polarity_summary.get(key) or 0) / polarity_total * 100:.1f}")
    try:
        if on_stream:
            on_stream("", "start")
        payload = _generate(messages, endpoint=endpoint, timeout=timeout, on_stream=on_stream)
        text = str(payload.get("text", "")).strip()
        model_seconds = float(payload.get("seconds", 0.0))
        valid, errors = validate_answer(text, valid_ids, rating_summary=validation_rating,
                                        require_rating_winner=not bool(result.get("subtopics")),
                                        required_subtopics=tuple(result.get("subtopics", {})),
                                        required_aspects=required_aspects,
                                        evidence_numbers=evidence_numbers,
                                        require_all_rating_values=require_all_rating_values,
                                        allow_aggregate_counts=allow_aggregate_counts,
                                        language=language)
        if valid:
            if on_stream:
                on_stream(text, "complete")
            return {"text": text, "mode": "local_qwen", "valid": True,
                    "validation_errors": [], "citations": sorted(_CITATION_RE.findall(text)),
                    "model_seconds": model_seconds}
        repair_instruction = (
            "The previous answer failed validation (" + "; ".join(errors) + "). Rewrite it in English only. "
            "Keep valid citations, remove unsupported numbers and scope claims, and put every requested subtopic "
            "or aspect in its own paragraph. Output only the corrected answer."
            if language == "en" else
            "上一版未通过校验（" + "；".join(errors) + "）。请按原要求重写，只输出修正后的中文回答。"
            "保留有效引用，删除未经证实的数字和范围判断。每个 requested_subtopics 中文名称必须单独成段，"
            "段与段之间空一行，不要把多个细分项写在同一段。requested_aspects 也必须每方面单独成段，"
            "显式写出该方面的原始平均分、满分和评分份数。"
        )
        repair_messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": repair_instruction},
        ]
        if on_stream:
            on_stream("", "reset")
        repaired_payload = _generate(repair_messages, endpoint=endpoint, timeout=timeout,
                                     on_stream=on_stream)
        repaired = str(repaired_payload.get("text", "")).strip()
        model_seconds += float(repaired_payload.get("seconds", 0.0))
        repaired_valid, repaired_errors = validate_answer(
            repaired, valid_ids, rating_summary=validation_rating,
            require_rating_winner=not bool(result.get("subtopics")),
            required_subtopics=tuple(result.get("subtopics", {})), required_aspects=required_aspects,
            evidence_numbers=evidence_numbers,
            require_all_rating_values=require_all_rating_values,
            allow_aggregate_counts=allow_aggregate_counts,
            language=language)
        if repaired_valid:
            if on_stream:
                on_stream(repaired, "complete")
            return {"text": repaired, "mode": "local_qwen_repaired", "valid": True,
                    "validation_errors": errors, "citations": sorted(_CITATION_RE.findall(repaired)),
                    "model_seconds": model_seconds}
        if on_stream:
            on_stream(fallback, "replace")
        return {"text": fallback, "mode": "validation_fallback", "valid": True,
                "validation_errors": errors + repaired_errors,
                "citations": sorted(_CITATION_RE.findall(fallback)), "model_seconds": model_seconds}
    except (OSError, ValueError, KeyError, RuntimeError, urllib.error.URLError, socket.timeout) as error:
        if on_stream:
            on_stream(fallback, "replace")
        return {"text": fallback, "mode": "service_fallback", "valid": True,
                "validation_errors": [type(error).__name__], "citations": sorted(_CITATION_RE.findall(fallback)),
                "model_seconds": 0.0}
