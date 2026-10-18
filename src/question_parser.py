"""Parse Chinese or English airline questions into safe retrieval constraints.

The parser is deliberately deterministic.  It extracts only values which exist
in the corpus (or a small, explicit alias table); the language model never gets
to invent filter values.  Sidebar values can be passed as defaults and explicit
phrases in the question take precedence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Iterable


ASPECT_ALIASES = {
    "seat": ("seat", "seats", "legroom", "leg room", "recline", "座位", "座椅", "腿部空间", "舒适度"),
    "staff": ("cabin crew", "flight attendant", "crew", "staff", "空乘", "乘务员", "机组服务"),
    "food": ("food", "meal", "meals", "drink", "catering", "餐饮", "餐食", "飞机餐", "饮料"),
    "entertainment": ("entertainment", "movie", "movies", "screen", "wi-fi", "wifi", "娱乐", "电影", "屏幕", "无线网"),
    "ground": ("check-in", "check in", "boarding", "baggage", "luggage", "delay", "punctual", "值机", "登机", "行李", "延误", "准点"),
    "value": ("value for money", "price", "fare", "refund", "customer service", "性价比", "价格", "票价", "退款", "客服"),
    "overall": ("overall", "experience", "recommended", "总体", "整体", "综合", "体验"),
    "other": ("cleanliness", "toilet", "cabin clean", "清洁", "卫生", "厕所", "洗手间"),
}

AIRLINE_ALIASES = {
    "新加坡航空": "Singapore Airlines", "新航": "Singapore Airlines",
    "卡塔尔航空": "Qatar Airways", "卡航": "Qatar Airways",
    "阿联酋航空": "Emirates", "阿联酋航": "Emirates",
    "国泰航空": "Cathay Pacific Airways", "国泰": "Cathay Pacific Airways",
    "英国航空": "British Airways", "英航": "British Airways",
    "澳洲航空": "Qantas Airways", "澳航": "Qantas Airways",
    "美国航空": "American Airlines", "美航": "American Airlines",
    "联合航空": "United Airlines", "美联航": "United Airlines",
    "达美航空": "Delta Air Lines", "达美": "Delta Air Lines",
    "汉莎航空": "Lufthansa", "汉莎": "Lufthansa",
    "法国航空": "Air France", "法航": "Air France",
    "荷兰皇家航空": "KLM Royal Dutch Airlines", "荷航": "KLM Royal Dutch Airlines",
    "土耳其航空": "Turkish Airlines", "土航": "Turkish Airlines",
    "泰国航空": "Thai Airways", "泰航": "Thai Airways",
    "全日空": "ANA All Nippon Airways", "日本航空": "Japan Airlines",
    "大韩航空": "Korean Air", "韩亚航空": "Asiana Airlines",
    "印度航空": "Air India", "加拿大航空": "Air Canada",
    "新西兰航空": "Air New Zealand", "维珍航空": "Virgin Atlantic Airways",
    "瑞士航空": "Swiss Intl Air Lines", "芬兰航空": "Finnair",
    "爱尔兰航空": "Aer Lingus", "冰岛航空": "Icelandair",
    "瑞安航空": "Ryanair", "易捷航空": "easyJet",
}

CABIN_ALIASES = {
    "First Class": ("first class", "头等舱"),
    "Business Class": ("business class", "商务舱", "公务舱"),
    "Premium Economy": ("premium economy", "超级经济舱", "高端经济舱"),
    "Economy Class": ("economy class", "economy", "经济舱"),
}

TRAVELLER_ALIASES = {
    "Business": ("business travel", "business trip", "商务出行", "出差"),
    "Couple Leisure": ("couple", "夫妻", "情侣"),
    "Family Leisure": ("family", "带孩子", "亲子", "家庭"),
    "Solo Leisure": ("solo", "alone", "独自", "单人"),
}

POLARITY_ALIASES = {
    "negative": ("negative", "bad", "worst", "complaint", "problem", "差评", "不好", "缺点", "问题", "投诉", "最差"),
    "positive": ("positive", "good", "best", "praise", "好评", "好的", "优点", "最好", "值得推荐"),
}

CITY_ALIASES = {
    "新加坡": "Singapore", "伦敦": "London", "香港": "Hong Kong", "东京": "Tokyo",
    "巴黎": "Paris", "悉尼": "Sydney", "墨尔本": "Melbourne", "纽约": "New York",
    "洛杉矶": "Los Angeles", "迪拜": "Dubai", "多哈": "Doha", "曼谷": "Bangkok",
    "北京": "Beijing", "上海": "Shanghai", "首尔": "Seoul", "法兰克福": "Frankfurt",
}


@dataclass
class ParsedQuestion:
    question: str
    aspect: str | None = None
    polarity: str | None = None
    airlines: list[str] = field(default_factory=list)
    filters: dict = field(default_factory=dict)
    retrieval_keywords: str = ""
    comparison: bool = False
    recognized: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _contains(text: str, phrase: str) -> bool:
    """Match CJK by substring and Latin aliases on token boundaries."""
    phrase_folded = phrase.casefold()
    if any("\u4e00" <= char <= "\u9fff" for char in phrase):
        return phrase_folded in text
    return re.search(r"(?<![\w])" + re.escape(phrase_folded) + r"(?![\w])", text) is not None


def _first_alias(text: str, mapping: dict[str, Iterable[str]]) -> str | None:
    matches = []
    for value, aliases in mapping.items():
        for alias in aliases:
            if _contains(text, alias):
                matches.append((text.find(alias.casefold()), -len(alias), value))
    return min(matches)[2] if matches else None


def _airlines(text: str, known_airlines: Iterable[str]) -> list[str]:
    found: list[tuple[int, int, str]] = []
    for alias, canonical in AIRLINE_ALIASES.items():
        if alias.casefold() in text:
            found.append((text.find(alias.casefold()), -len(alias), canonical))
    for canonical in known_airlines:
        if _contains(text, canonical):
            found.append((text.find(canonical.casefold()), -len(canonical), canonical))
    answer = []
    for _, _, canonical in sorted(found):
        if canonical not in answer and canonical in known_airlines:
            answer.append(canonical)
    return answer[:3]


def _route_filters(text: str, origins: Iterable[str], destinations: Iterable[str]) -> dict:
    known_from = {value.casefold(): value for value in origins}
    known_to = {value.casefold(): value for value in destinations}
    cities = {**{key.casefold(): value for key, value in CITY_ALIASES.items()}, **known_from, **known_to}
    names = sorted(cities, key=len, reverse=True)
    token = "(" + "|".join(re.escape(name) for name in names) + ")"
    patterns = [
        re.compile(r"(?:从|由)\s*" + token + r"\s*(?:到|飞往|去)\s*" + token),
        re.compile(r"(?:from)\s+" + token + r"\s+(?:to)\s+" + token),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            origin, destination = cities[match.group(1)], cities[match.group(2)]
            result = {}
            if origin.casefold() in known_from: result["route_from"] = known_from[origin.casefold()]
            if destination.casefold() in known_to: result["route_to"] = known_to[destination.casefold()]
            return result
    return {}


def parse_question(question: str, *, known_airlines: Iterable[str], origins: Iterable[str] = (),
                   destinations: Iterable[str] = (), default_aspect: str | None = None,
                   default_polarity: str | None = None, default_filters: dict | None = None) -> ParsedQuestion:
    """Return corpus-safe constraints; explicit question phrases override defaults."""
    raw = question.strip()
    folded = raw.casefold()
    aspect = _first_alias(folded, ASPECT_ALIASES) or default_aspect
    if aspect is None and any(term in folded for term in ("怎么样", "好不好", "哪家好", "哪家更好",
                                                           "评价如何", "how is", "how was", "which is better")):
        aspect = "overall"
    polarity = _first_alias(folded, POLARITY_ALIASES) or default_polarity
    airlines = _airlines(folded, list(known_airlines))
    filters = dict(default_filters or {})
    cabin = _first_alias(folded, CABIN_ALIASES)
    traveller = _first_alias(folded, TRAVELLER_ALIASES)
    if cabin: filters["cabin"] = cabin
    if traveller: filters["traveller_type"] = traveller
    filters.update(_route_filters(folded, origins, destinations))
    if len(airlines) == 1: filters["airline"] = airlines[0]

    # Retain only useful Latin keywords which are not already represented by a
    # structured airline/cabin value. Chinese questions still retrieve through
    # the English aspect expansion in query_service.ASPECT_QUERIES.
    words = re.findall(r"[a-z][a-z0-9'-]{2,}", folded)
    stop = {"which", "what", "how", "does", "have", "with", "from", "that", "this",
            "airline", "airlines", "airways", "class", "better", "reviews", "review"}
    airline_words = {word for name in airlines for word in re.findall(r"[a-z]+", name.casefold())}
    keywords = " ".join(dict.fromkeys(word for word in words if word not in stop | airline_words))
    recognized = []
    if aspect: recognized.append("aspect")
    if polarity: recognized.append("polarity")
    if airlines: recognized.append("airline")
    recognized.extend(key for key in ("cabin", "traveller_type", "route_from", "route_to") if key in filters)
    comparison_words = ("比较", "对比", "哪家", "哪个好", "更好", "better", "compare", "versus", " vs ")
    return ParsedQuestion(raw, aspect, polarity, airlines, filters, keywords,
                          len(airlines) >= 2 or any(word in folded for word in comparison_words), recognized)
