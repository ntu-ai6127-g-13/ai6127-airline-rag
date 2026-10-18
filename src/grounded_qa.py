"""Grounded answer generation with strict citation validation and fallback."""
from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request


DEFAULT_ENDPOINT = "http://127.0.0.1:8765"
_CITATION_RE = re.compile(r"\[([A-Z]?\d+)\]")
_PARAGRAPH_RE = re.compile(r"\n+")


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


def _summary_counts(rows: list[dict], aspect: str | None) -> dict:
    matching = [row for row in rows if row.get("pred_subjectivity") == "subjective"
                and (aspect is None or row.get("pred_aspect") == aspect)]
    return {
        "evidence_count": len(rows),
        "review_count": len({str(row.get("review_id", row["chunk_id"])) for row in rows}),
        "matching_label_count": len(matching),
        "positive_count": sum(row.get("pred_polarity") == "positive" for row in matching),
        "negative_count": sum(row.get("pred_polarity") == "negative" for row in matching),
    }


def _qualitative_summary(rows: list[dict], aspect: str | None) -> dict[str, dict]:
    """Give the model a program-derived direction without exposing counts."""
    answer = {}
    for airline in dict.fromkeys(str(row.get("airline") or "所选航司") for row in rows):
        matching = [row for row in rows if str(row.get("airline") or "所选航司") == airline
                    and row.get("pred_subjectivity") == "subjective"
                    and (aspect is None or row.get("pred_aspect") == aspect)]
        positive = [row["citation_id"] for row in matching if row.get("pred_polarity") == "positive"]
        negative = [row["citation_id"] for row in matching if row.get("pred_polarity") == "negative"]
        tendency = "positive_lean" if len(positive) > len(negative) else (
            "negative_lean" if len(negative) > len(positive) else "mixed_or_tied")
        answer[airline] = {"sample_tendency": tendency,
                           "positive_evidence_ids": positive, "negative_evidence_ids": negative}
    return answer


def _prompt(question: str, result: dict, rows: list[dict]) -> list[dict]:
    aspect = result.get("aspect")
    evidence = [{
        "id": row["citation_id"], "airline": row.get("airline"), "cabin": row.get("cabin"),
        "date": row.get("date"), "route": row.get("route"), "text": row.get("text"),
        "predicted_aspect": row.get("pred_aspect"), "predicted_polarity": row.get("pred_polarity"),
    } for row in rows]
    system = """你是航空评论证据问答助手。只能依据用户提供的检索证据作答，不能使用外部知识。
要求：
1. 用简体中文直接回答问题，先给结论，再说明正反两面的证据；材料不足时明确说证据不足。
2. 每个关于航司服务的事实句末尾必须引用至少一个证据编号，格式严格为 [1]、[2] 或比较查询中的 [A1]、[B2]。
3. 只能使用输入中存在的编号。不要输出任何正负条数、百分比、评分、排名、日期或样本量；用“偏正面”“褒贬不一”等定性表达。
4. 自动分类标签可能有误，应优先按英文原文理解；评论来自 2002–2019 年，不能说成当前服务状况。
5. 不要把检索样本的正负数量说成全体旅客好评率。回答控制在 180–320 个中文字符。"""
    payload = {"question": question, "aspect": aspect, "filters": result.get("filters", {}),
               "program_summary_for_direction_only": _qualitative_summary(rows, aspect),
               "evidence": evidence}
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def validate_answer(text: str, valid_ids: set[str]) -> tuple[bool, list[str]]:
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
    without_citations = _CITATION_RE.sub("", clean).replace("2002–2019", "")
    if re.search(r"\d", without_citations) or re.search(r"[零一二三四五六七八九十百]+\s*条", without_citations):
        reasons.append("unverified_numeric_claim")
    # Each substantive paragraph must contain a citation.  A paragraph can open
    # with a short conclusion and immediately substantiate it in later cited
    # sentences, which is the normal answer form produced by the local model.
    # Generic caveats are allowed without citations because they describe system
    # limits, not the review evidence.
    exemptions = ("数据", "样本", "检索", "不能代表", "证据不足", "自动标签", "截至")
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
    by_airline: dict[str, list[dict]] = {}
    for row in rows:
        by_airline.setdefault(str(row.get("airline") or "所选航司"), []).append(row)
    parts = []
    for airline, items in by_airline.items():
        matching = [row for row in items if row.get("pred_subjectivity") == "subjective"
                    and (aspect is None or row.get("pred_aspect") == aspect)]
        positive = [row for row in matching if row.get("pred_polarity") == "positive"]
        negative = [row for row in matching if row.get("pred_polarity") == "negative"]
        cited = (negative[:1] + positive[:1]) or items[:2]
        citations = "".join(f"[{row['citation_id']}]" for row in cited)
        if matching:
            tendency = "负面证据较多" if len(negative) > len(positive) else (
                "正面证据较多" if len(positive) > len(negative) else "正负证据数量接近")
            parts.append(f"{airline}在本次检索样本中{tendency}：匹配该方面的自动标签共{len(matching)}条，"
                         f"其中正向{len(positive)}条、负向{len(negative)}条{citations}。")
        else:
            parts.append(f"{airline}已找到相关原文，但自动标签没有可靠地匹配到所问方面{citations}。")
    return "".join(parts) + "这些结论只描述当前检索到的历史评论样本；请结合下方原文核对，不能当作全航司好评率或当前排名。"


def model_health(endpoint: str = DEFAULT_ENDPOINT, timeout: float = 0.5) -> dict:
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/health", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError, socket.timeout):
        return {"status": "unavailable"}


def answer_question(question: str, result: dict, *, endpoint: str = DEFAULT_ENDPOINT,
                    timeout: float = 90.0, use_model: bool = True) -> dict:
    """Generate, validate, and fall back without ever losing the evidence."""
    rows = collect_evidence(result)
    valid_ids = {row["citation_id"] for row in rows}
    fallback = fallback_answer(question, result, rows)
    if not use_model or not rows:
        return {"text": fallback, "mode": "program_fallback", "valid": True,
                "validation_errors": [], "citations": sorted(_CITATION_RE.findall(fallback)), "model_seconds": 0.0}
    body = json.dumps({"messages": _prompt(question, result, rows), "max_tokens": 420,
                       "temperature": 0.0}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(endpoint.rstrip("/") + "/generate", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = str(payload.get("text", "")).strip()
        valid, errors = validate_answer(text, valid_ids)
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
