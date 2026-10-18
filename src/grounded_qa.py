"""Grounded answer generation with strict citation validation and fallback."""
from __future__ import annotations

import json
import re
import socket
import textwrap
import urllib.error
import urllib.request


DEFAULT_ENDPOINT = "http://127.0.0.1:8765"
_CITATION_RE = re.compile(r"\[([A-Z]?\d+)\]")
_PARAGRAPH_RE = re.compile(r"\n+")
_ASPECT_NAMES = {"seat": "座椅舒适度", "staff": "空乘服务", "food": "餐饮",
                 "entertainment": "机上娱乐", "ground": "地面服务", "value": "性价比",
                 "overall": "总体体验", "other": "其他服务"}
_AIRLINE_NAMES_ZH = {"Singapore Airlines": "新加坡航空", "United Airlines": "美国联合航空",
                     "Qatar Airways": "卡塔尔航空", "Cathay Pacific Airways": "国泰航空",
                     "British Airways": "英国航空", "Emirates": "阿联酋航空"}


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
        "predicted_aspect": row.get("pred_aspect"), "predicted_polarity": row.get("pred_polarity"),
    } for row in rows]
    system = """你是航空评论分析助手。只能依据输入的原始评分汇总和评论证据作答，不能使用外部知识。
要求：
1. 第一行直接回答用户。若 comparison.status 是 clear_difference，必须明确说 winner 在这批历史评分中更好，不准用“褒贬不一”“各有优劣”回避结论。若是 no_clear_difference，说明评分差和为何不足以区分。
2. 评分均值、评分人数和差值只能逐字使用 rating_summary 中给出的数值；不要统计检索证据的正负条数。
3. 评分汇总是主要定量依据，评论原文用来解释具体优缺点。引用格式严格为 [1]、[2] 或比较查询中的 [A1]、[B2]，只能使用存在的编号。
4. 自动分类标签仅供选证据参考，应按英文原文理解。不要把检索到的几句话当作总体好评率。
5. 数据来自 2002–2019 年，只能描述这批历史评论，不能声称是当前服务。用自然、具体的中文，避免套话和“整体呈现某种倾向”。回答控制在 150–280 个中文字符。"""
    payload = {"question": question, "aspect": aspect, "filters": result.get("filters", {}),
               "rating_summary": _rating_payload(result.get("rating_summary")),
               "evidence": evidence}
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def validate_answer(text: str, valid_ids: set[str], *, rating_summary: dict | None = None) -> tuple[bool, list[str]]:
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
    if comparison.get("status") == "clear_difference" and comparison.get("winner") not in clean:
        reasons.append("missing_required_winner")
    # Each substantive paragraph must contain a citation.  A paragraph can open
    # with a short conclusion and immediately substantiate it in later cited
    # sentences, which is the normal answer form produced by the local model.
    # Generic caveats are allowed without citations because they describe system
    # limits, not the review evidence.
    exemptions = ("数据", "样本", "检索", "不能代表", "证据不足", "自动标签", "截至", "评分")
    for paragraph in _PARAGRAPH_RE.split(clean):
        paragraph = paragraph.strip()
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


def answer_question(question: str, result: dict, *, endpoint: str = DEFAULT_ENDPOINT,
                    timeout: float = 45.0, use_model: bool = True) -> dict:
    """Generate, validate, and fall back without ever losing the evidence."""
    rows = collect_evidence(result)
    valid_ids = {row["citation_id"] for row in rows}
    fallback = fallback_answer(question, result, rows)
    rating_comparison = (result.get("rating_summary") or {}).get("comparison", {})
    if result.get("type") == "comparison" and rating_comparison.get("status") in {
            "clear_difference", "no_clear_difference"}:
        return {"text": fallback, "mode": "rating_decision", "valid": True,
                "validation_errors": [], "citations": sorted(_CITATION_RE.findall(fallback)),
                "model_seconds": 0.0}
    if not use_model or not rows:
        return {"text": fallback, "mode": "program_fallback", "valid": True,
                "validation_errors": [], "citations": sorted(_CITATION_RE.findall(fallback)), "model_seconds": 0.0}
    body = json.dumps({"messages": _prompt(question, result, rows), "max_tokens": 320,
                       "temperature": 0.0}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(endpoint.rstrip("/") + "/generate", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = str(payload.get("text", "")).strip()
        valid, errors = validate_answer(text, valid_ids, rating_summary=result.get("rating_summary"))
        if valid:
            return {"text": text, "mode": "local_qwen", "valid": True,
                    "validation_errors": [], "citations": sorted(_CITATION_RE.findall(text)),
                    "model_seconds": float(payload.get("seconds", 0.0))}
        return {"text": fallback, "mode": "validation_fallback", "valid": True,
                "validation_errors": errors, "citations": sorted(_CITATION_RE.findall(fallback)),
                "model_seconds": float(payload.get("seconds", 0.0))}
    except (OSError, ValueError, KeyError, urllib.error.URLError, socket.timeout) as error:
        return {"text": fallback, "mode": "service_fallback", "valid": True,
                "validation_errors": [type(error).__name__], "citations": sorted(_CITATION_RE.findall(fallback)),
                "model_seconds": 0.0}
