"""Grounded answer generation with strict citation validation and fallback."""
from __future__ import annotations

import json
import re
import socket
import textwrap
import urllib.error
import urllib.request

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


def _citation_id(prefix: str, number: int) -> str:
    return f"{prefix}{number}"


def collect_evidence(result: dict) -> list[dict]:
    """Flatten a single/comparison query into uniquely cited evidence."""
    rows = []
    if result.get("type") == "comparison":
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
    aspect = result.get("aspect")
    aspect_name = _ASPECT_NAMES.get(aspect, "该方面")
    evidence = [{
        "id": row["citation_id"], "airline": row.get("airline"), "cabin": row.get("cabin"),
        "date": row.get("date"), "route": row.get("route"), "text": row.get("text"),
        "matched_subtopics": [SUBTOPIC_NAMES.get(topic, topic)
                              for topic in row.get("matched_subtopics", [])],
        "direct_subtopic_matches": [SUBTOPIC_NAMES.get(topic, topic)
                                    for topic in row.get("direct_subtopic_matches", [])],
        "predicted_aspect": row.get("pred_aspect"), "predicted_polarity": row.get("pred_polarity"),
    } for row in rows]
    subtopics = {topic: SUBTOPIC_NAMES.get(topic, topic) for topic in result.get("subtopics", {})}
    system = """你是航空评论分析助手。只能依据输入的原始评分汇总和评论证据作答，不能使用外部知识。
要求：
1. 第一行直接回答用户。若 comparison.status 是 clear_difference，必须明确说 winner 在这批历史评分中更好，不准用“褒贬不一”“各有优劣”回避结论。若是 no_clear_difference，说明评分差和为何不足以区分。
2. 评分均值、评分人数和差值只能逐字使用 rating_summary 中给出的数值；不要统计检索证据的正负条数。
3. 评分汇总是主要定量依据，评论原文用来解释具体优缺点。引用格式严格为 [1]、[2] 或比较查询中的 [A1]、[B2]，只能使用存在的编号。
4. 自动分类标签仅供选证据参考，应按英文原文理解。不要把检索到的几句话当作总体好评率。
5. requested_subtopics 非空时，每个细分项必须单独成段，段与段之间空一行；单航司问题的每段以具体航司名和细分项开头，例如“关于新加坡航空的腿部空间”，不要写成“关于航司的腿部空间”。随后给出该项的发现和引用。禁止把多个细分项挤在同一段。优先引用 direct_subtopic_matches、其次引用 matched_subtopics 含对应名称的证据。原始方面评分只是上位方面的总体分，不能据此断言某个细分项好坏，也不能据此决定细分项比较的胜负；某项缺少直接证据就明确说明。回答中不要使用“多数、普遍、更多人、较多、整体、总体”等范围词，应写“检索到的原文同时包含……”并说明正反例。
6. 数据来自 2002–2019 年，只能描述这批历史评论，不能声称是当前服务。用自然、具体的中文，避免套话和“整体呈现某种倾向”。回答控制在 150–350 个中文字符。"""
    if subtopics:
        system = system.replace(
            "1. 第一行直接回答用户。若 comparison.status 是 clear_difference，必须明确说 winner 在这批历史评分中更好，不准用“褒贬不一”“各有优劣”回避结论。若是 no_clear_difference，说明评分差和为何不足以区分。",
            "1. 第一行直接回答用户，并按 requested_subtopics 给出具体发现。comparison 只是上位方面的评分比较，不能强制成为细分问题的结论。")
    payload = {"question": question, "aspect": aspect, "filters": result.get("filters", {}),
               "requested_subtopics": subtopics,
               "rating_summary": None if subtopics else _rating_payload(result.get("rating_summary")),
               "evidence": evidence}
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def validate_answer(text: str, valid_ids: set[str], *, rating_summary: dict | None = None,
                    require_rating_winner: bool = True,
                    required_subtopics: tuple[str, ...] = ()) -> tuple[bool, list[str]]:
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
    without_citations = _CITATION_RE.sub("", clean)
    numeric_tokens = re.findall(r"\d+(?:\.\d+)?", without_citations)
    unexpected = set(numeric_tokens) - _allowed_numbers(rating_summary)
    if unexpected: reasons.append("unverified_numeric_claim:" + ",".join(sorted(unexpected)))
    if re.search(r"[零一二三四五六七八九十百]+\s*条", without_citations):
        reasons.append("retrieved_label_count_claim")
    comparison = (rating_summary or {}).get("comparison", {})
    if (require_rating_winner and comparison.get("status") == "clear_difference"
            and comparison.get("winner") not in clean):
        reasons.append("missing_required_winner")
    paragraphs = [paragraph.strip() for paragraph in _PARAGRAPH_RE.split(clean) if paragraph.strip()]
    topic_paragraphs = {}
    for topic in required_subtopics:
        name = SUBTOPIC_NAMES.get(topic, topic)
        containing = [index for index, paragraph in enumerate(paragraphs) if name in paragraph]
        topic_paragraphs[topic] = containing
        if not containing:
            reasons.append("missing_subtopic:" + topic)
    if len(required_subtopics) > 1:
        used = [indices[0] for indices in topic_paragraphs.values() if indices]
        if len(set(used)) != len(used):
            reasons.append("subtopics_not_separated")
    if required_subtopics and re.search(
            r"多数|普遍|大部分|更多(?:人|旅客|评论)|较多(?:人|旅客|评论)|整体|总体", clean):
        reasons.append("unsupported_prevalence_claim")
    # Each substantive paragraph must contain a citation.  A paragraph can open
    # with a short conclusion and immediately substantiate it in later cited
    # sentences, which is the normal answer form produced by the local model.
    # Generic caveats are allowed without citations because they describe system
    # limits, not the review evidence.
    exemptions = ("数据", "样本", "检索", "不能代表", "证据不足", "自动标签", "截至", "评分")
    for paragraph in paragraphs:
        if len(paragraph) < 8 or _CITATION_RE.search(paragraph) or any(term in paragraph for term in exemptions):
            continue
        reasons.append("uncited_claim")
        break
    return not reasons, reasons


def fallback_answer(question: str, result: dict, rows: list[dict]) -> str:
    """Produce a useful answer without an LLM, grounded in visible evidence."""
    del question
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
        return textwrap.shorten(str(row["text"]).strip(), width=145, placeholder="…")
    subtopics = result.get("subtopics") or {}
    if subtopics:
        sections = []
        comparison_query = result.get("type") == "comparison"
        for topic in subtopics:
            matches = sorted(
                (row for row in rows if topic in row.get("matched_subtopics", [])),
                key=lambda row: topic not in row.get("direct_subtopic_matches", []))
            if comparison_query:
                chosen = []
                used_airlines = set()
                for row in matches:
                    airline = str(row.get("airline") or "所选航司")
                    if airline not in used_airlines:
                        chosen.append(row)
                        used_airlines.add(airline)
                    if len(chosen) == len(by_airline):
                        break
            else:
                chosen = matches[:2]
            name = _SUBTOPIC_SUBJECTS.get(topic, SUBTOPIC_NAMES.get(topic, topic))
            if not chosen:
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
            airline = str(chosen[0].get("airline") or "")
            airline_label = _AIRLINE_NAMES_ZH.get(airline, airline)
            subject = f"{airline_label}的{name}" if airline_label and not comparison_query else name
            sections.append(f"关于{subject}，{direction}。" + "；".join(details))
        return "\n\n".join(sections) + "\n\n这些原文记录的是具体旅客经历，不能代表所有航班。"
    if rating.get("available") and rating.get("comparison", {}).get("status") == "clear_difference":
        comparison = rating["comparison"]
        winner, runner = comparison["winner"], comparison["runner_up"]
        winner_label, runner_label = _AIRLINE_NAMES_ZH.get(winner, winner), _AIRLINE_NAMES_ZH.get(runner, runner)
        winner_data, runner_data = rating["airlines"][winner], rating["airlines"][runner]
        good, bad = representative(winner, "positive"), representative(runner, "negative")
        details = []
        if good: details.append(f"{winner_label}的旅客提到“{quote(good)}”[{good['citation_id']}]")
        if bad: details.append(f"{runner_label}的旅客则提到“{quote(bad)}”[{bad['citation_id']}]")
        return (f"{winner_label}明显更好，而且评分差距很大。原始{aspect_name}评分中，{winner_label}为"
                f"{winner_data['mean']:.2f}/{rating['scale']}（{winner_data['n']:,}份），{runner_label}为"
                f"{runner_data['mean']:.2f}/{rating['scale']}（{runner_data['n']:,}份），相差{comparison['difference']:.2f}分；"
                f"{winner_label}的高分占比为{winner_data['high_share']*100:.1f}%，{runner_label}仅为{runner_data['high_share']*100:.1f}%，"
                f"后者的低分占比达到{runner_data['low_share']*100:.1f}%。"
                + "；".join(details) + "。评分和原文均来自2002–2019年的评论，不能直接代表当前服务。")
    if rating.get("available") and rating.get("comparison", {}).get("status") == "no_clear_difference":
        comparison = rating["comparison"]
        leader, runner = comparison["leader"], comparison["runner_up"]
        leader_label, runner_label = _AIRLINE_NAMES_ZH.get(leader, leader), _AIRLINE_NAMES_ZH.get(runner, runner)
        first, second = rating["airlines"][leader], rating["airlines"][runner]
        return (f"这批评分不足以可靠地区分谁更好。{leader_label}为{first['mean']:.2f}/{rating['scale']}（{first['n']:,}份），"
                f"{runner_label}为{second['mean']:.2f}/{rating['scale']}（{second['n']:,}份），只差{comparison['difference']:.2f}分，"
                "没有达到预设的实际意义门槛；此时硬给出胜负会夸大数据差异。")
    if rating.get("available") and len(rating.get("airlines", {})) == 1:
        airline, data = next(iter(rating["airlines"].items()))
        airline_label = _AIRLINE_NAMES_ZH.get(airline, airline)
        good, bad = representative(airline, "positive"), representative(airline, "negative")
        interpretation = "评价较好" if data["mean"] >= rating["scale"] * .7 else (
            "评价偏低" if data["mean"] < rating["scale"] * .55 else "评价处于中间水平")
        cited = [row for row in (good, bad) if row]
        detail = "；".join(f"“{quote(row)}”[{row['citation_id']}]" for row in cited)
        return (f"{airline_label}的{aspect_name}历史评分为{data['mean']:.2f}/{rating['scale']}，共{data['n']:,}份评分，{interpretation}。"
                f"代表性原文包括{detail}。数据截至2019年。")
    cited = rows[:2]
    details = "；".join(f"“{quote(row)}”[{row['citation_id']}]" for row in cited)
    return f"原始评论表没有足够的同口径评分来形成明确结论。目前能核对的原文包括{details}。"


def model_health(endpoint: str = DEFAULT_ENDPOINT, timeout: float = 0.5) -> dict:
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/health", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError, socket.timeout):
        return {"status": "unavailable"}


def _generate(messages: list[dict], *, endpoint: str, timeout: float) -> dict:
    body = json.dumps({"messages": messages, "max_tokens": 320,
                       "temperature": 0.0}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(endpoint.rstrip("/") + "/generate", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def answer_question(question: str, result: dict, *, endpoint: str = DEFAULT_ENDPOINT,
                    timeout: float = 45.0, use_model: bool = True) -> dict:
    """Generate, validate, and fall back without ever losing the evidence."""
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
    validation_rating = None if result.get("subtopics") else result.get("rating_summary")
    try:
        payload = _generate(messages, endpoint=endpoint, timeout=timeout)
        text = str(payload.get("text", "")).strip()
        model_seconds = float(payload.get("seconds", 0.0))
        valid, errors = validate_answer(text, valid_ids, rating_summary=validation_rating,
                                        require_rating_winner=not bool(result.get("subtopics")),
                                        required_subtopics=tuple(result.get("subtopics", {})))
        if valid:
            return {"text": text, "mode": "local_qwen", "valid": True,
                    "validation_errors": [], "citations": sorted(_CITATION_RE.findall(text)),
                    "model_seconds": model_seconds}
        repair_messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content":
             "上一版未通过校验（" + "；".join(errors) + "）。请按原要求重写，只输出修正后的中文回答。"
             "保留有效引用，删除未经证实的数字和范围判断。每个 requested_subtopics 中文名称必须单独成段，"
             "段与段之间空一行，不要把多个细分项写在同一段。"},
        ]
        repaired_payload = _generate(repair_messages, endpoint=endpoint, timeout=timeout)
        repaired = str(repaired_payload.get("text", "")).strip()
        model_seconds += float(repaired_payload.get("seconds", 0.0))
        repaired_valid, repaired_errors = validate_answer(
            repaired, valid_ids, rating_summary=validation_rating,
            require_rating_winner=not bool(result.get("subtopics")),
            required_subtopics=tuple(result.get("subtopics", {})))
        if repaired_valid:
            return {"text": repaired, "mode": "local_qwen_repaired", "valid": True,
                    "validation_errors": errors, "citations": sorted(_CITATION_RE.findall(repaired)),
                    "model_seconds": model_seconds}
        return {"text": fallback, "mode": "validation_fallback", "valid": True,
                "validation_errors": errors + repaired_errors,
                "citations": sorted(_CITATION_RE.findall(fallback)), "model_seconds": model_seconds}
    except (OSError, ValueError, KeyError, urllib.error.URLError, socket.timeout) as error:
        return {"text": fallback, "mode": "service_fallback", "valid": True,
                "validation_errors": [type(error).__name__], "citations": sorted(_CITATION_RE.findall(fallback)),
                "model_seconds": 0.0}
