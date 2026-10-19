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
    "seat": ("seat", "seats", "legroom", "leg room", "recline", "backrest", "座位", "座椅", "靠背", "腿部空间", "舒适度"),
    "staff": ("cabin crew", "flight attendant", "crew", "staff", "空乘", "乘务员", "机组服务"),
    "food": ("food", "meal", "meals", "drink", "catering", "餐饮", "餐食", "食物", "飞机餐", "饮料"),
    "entertainment": ("entertainment", "movie", "movies", "screen", "wi-fi", "wifi", "娱乐", "电影", "屏幕", "无线网"),
    "ground": ("check-in", "check in", "boarding", "gate agent", "baggage", "luggage", "delay", "punctual", "值机", "登机", "登机口", "地勤", "地面服务", "行李", "延误", "准点"),
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

SUBTOPIC_RULES = {
    "food": {
        "menu_items": (("有什么吃", "有些什么吃", "吃什么", "有什么餐", "提供什么餐", "具体吃什么",
                        "具体餐食", "具体食物", "菜品", "菜式", "what food", "what meals",
                        "what do they serve", "what is served", "menu items", "which dishes"),
                       "specific onboard dishes meal items chicken beef fish rice noodles breakfast dessert"),
        "quality": (("质量", "味道", "口味", "好吃", "难吃", "新鲜", "taste", "tasty", "quality", "fresh"),
                    "food taste quality freshness delicious poor meal"),
        "portion": (("分量", "份量", "量够", "吃不饱", "portion", "quantity", "amount", "enough food"),
                    "meal portion size quantity amount enough small generous"),
        "variety": (("品种", "种类", "选择", "菜单", "variety", "selection", "choice", "menu", "options"),
                    "meal variety choice selection options menu limited"),
        "drinks": (("饮料", "酒水", "饮品", "咖啡", "茶", "drink", "beverage", "wine", "coffee", "tea"),
                   "onboard drinks beverages wine coffee tea selection quality"),
        "special_meals": (("特殊餐", "素食", "儿童餐", "清真", "过敏", "special meal", "vegetarian", "child meal", "halal", "allergy"),
                          "special meal vegetarian child meal dietary allergy halal availability"),
        "temperature": (("温度", "冷饭", "热饭", "冷的", "temperature", "cold meal", "hot meal"),
                        "meal temperature served hot cold reheated"),
        "meal_service": (("上餐", "送餐", "收餐", "用餐服务", "餐食服务", "预订餐", "预选餐", "名厨有约",
                          "meal service", "meal timing", "pre-order meal", "preorder meal", "book the cook"),
                         "meal service timing delivery collection preorder book the cook availability"),
    },
    "entertainment": {
        "content": (("片源", "电影", "影片", "有什么看", "能看什么", "看什么", "电视节目", "游戏", "音乐",
                     "电影选择", "内容", "节目", "movie selection", "content", "movies", "films", "games", "music"),
                    "entertainment movie selection content variety programs"),
        "screen": (("屏幕", "画质", "触屏", "screen", "display", "touchscreen"),
                   "entertainment screen display touch screen quality broken"),
        "wifi": (("无线网", "网络", "网速", "wi-fi", "wifi", "internet"),
                 "inflight Wi-Fi internet connection speed working"),
        "audio": (("耳机", "声音", "音质", "headphone", "headset", "audio", "sound"),
                  "inflight entertainment headphones headset audio sound quality working"),
        "usability": (("操作", "系统", "反应", "卡顿", "遥控器", "interface", "system", "remote", "responsive"),
                       "entertainment system interface remote control responsive frozen usability"),
    },
    "seat": {
        "legroom": (("腿部空间", "腿距", "legroom", "leg room", "knee room"),
                    "seat legroom leg room knee space pitch cramped"),
        "width": (("宽度", "座位宽", "多宽", "width", "wide", "narrow"), "seat width wide narrow cramped"),
        "recline": (("倾斜", "后仰", "躺平", "靠背", "椅背", "调整", "可调", "recline", "backrest", "seat back", "flat bed", "lie-flat"),
                    "seat backrest recline adjustable lie flat angle upright broken"),
        "comfort": (("舒服", "舒适", "软硬", "坐垫", "硌", "comfort", "comfortable", "uncomfortable", "hard seat", "padding", "cushion"),
                    "seat comfort comfortable uncomfortable padding cushion hard sore sleeping"),
        "condition": (("破损", "故障", "坏了", "broken seat", "damaged seat", "worn seat"),
                      "seat condition broken damaged worn functioning"),
        "seat_cleanliness": (("干净", "干净程度", "清洁", "卫生", "污渍", "脏座椅", "座椅脏", "seat cleanliness", "clean seat", "clean seats", "dirty seat", "dirty seats", "seat stain", "crumbs"),
                             "seat cleanliness clean dirty stains crumbs tray table hygiene"),
        "power": (("插座", "充电", "电源", "usb", "power socket", "charging", "outlet"),
                  "seat power socket USB charging outlet working"),
        "privacy": (("隐私", "隔板", "隔断", "包间", "套间门", "privacy", "divider", "suite door"),
                    "seat privacy divider partition suite door enclosed exposed"),
        "storage": (("储物", "收纳", "放东西", "储物格", "storage", "storage compartment"),
                    "seat storage compartment personal items space"),
    },
    "staff": {
        "attitude": (("态度", "礼貌", "友好", "粗鲁", "attitude", "friendly", "rude", "polite"),
                     "cabin crew attitude friendly polite rude helpful"),
        "efficiency": (("效率", "响应", "及时", "attention", "responsive", "efficient", "slow service"),
                       "cabin crew response efficiency attentive prompt slow"),
        "professionalism": (("专业", "熟练", "培训", "professional", "competent", "trained", "knowledgeable"),
                            "cabin crew professional competent trained knowledgeable service"),
        "availability": (("呼叫", "找不到", "巡视", "call button", "available", "presence", "attentive"),
                         "cabin crew available call button attentive present response"),
        "communication": (("沟通", "语言", "听不懂", "交流", "communication", "language skills", "understand"),
                          "cabin crew communication language understanding explanation announcements"),
    },
    "ground": {
        "punctuality": (("准点", "是否延误", "准时", "punctual", "on time", "delayed flight"),
                        "flight punctuality delay arrival departure on time"),
        "delay_handling": (("延误处理", "取消处理", "延误怎么处理", "延误如何处理", "取消怎么处理", "取消如何处理",
                            "延误或取消", "延误和取消", "延误后", "取消后", "改签安排", "补偿",
                            "delay handling", "cancellation handling", "disruption", "compensation", "rebooked"),
                           "airline delay cancellation disruption handling rebooking compensation hotel communication"),
        "baggage": (("行李", "丢失", "baggage", "luggage", "lost bag"),
                    "baggage luggage handling lost damaged delayed"),
        "checkin": (("值机", "值机柜台", "check-in", "check in", "checkin desk"),
                    "airport check-in desk queue staff process kiosk"),
        "gate_service": (("登机口服务", "登机服务", "登机口人员", "登机口", "登机顺序", "gate service", "gate agent", "boarding staff", "boarding process"),
                         "airport gate agents boarding service announcements queue priority process"),
        "lounge": (("休息室", "贵宾室", "lounge"),
                   "airport lounge seating food facilities staff crowded"),
        "transfer": (("转机", "中转", "衔接航班", "赶不上下一班", "transfer", "transit", "connection", "connecting flight"),
                     "airport transfer transit connecting flight missed connection assistance"),
        "assistance": (("轮椅", "特殊协助", "无障碍", "行动不便", "wheelchair", "special assistance", "mobility assistance"),
                       "airport wheelchair special assistance mobility disability support"),
    },
    "value": {
        "price": (("价格", "票价", "贵", "便宜", "price", "fare", "expensive", "cheap"),
                  "ticket price fare expensive cheap value for money"),
        "refund": (("退款", "退票", "改签", "refund", "rebooking", "cancellation fee"),
                   "refund rebooking cancellation fee repayment"),
        "support": (("客服", "电话", "投诉处理", "customer service", "hotline", "call centre"),
                    "customer service hotline call centre complaint handling"),
        "booking": (("预订", "订票", "网站", "app", "booking", "reservation", "website"),
                    "booking reservation website app ticket change usability problem"),
        "fees": (("附加费", "额外收费", "隐藏收费", "行李费", "选座费", "手续费", "surcharge", "extra fee", "hidden fee", "baggage fee", "seat fee"),
                 "airline surcharge extra fee hidden charge baggage fee seat selection fee"),
    },
    "other": {
        "cleanliness": (("清洁", "卫生", "干净", "脏", "clean", "cleanliness", "dirty"),
                        "aircraft cabin cleanliness clean dirty hygiene"),
        "toilet": (("厕所", "洗手间", "卫生间", "toilet", "lavatory", "washroom"),
                   "aircraft toilet lavatory cleanliness availability condition"),
        "cabin_temperature": (("客舱温度", "太冷", "太热", "空调", "cabin temperature", "too cold", "too hot", "air conditioning"),
                              "cabin temperature air conditioning too cold too hot"),
        "amenities": (("毛毯", "枕头", "洗漱包", "拖鞋", "睡衣", "眼罩", "耳塞", "amenity kit", "blanket", "pillow", "slippers", "pyjamas", "pajamas", "eye mask"),
                      "onboard amenities amenity kit blanket pillow slippers pajamas eye mask"),
        "noise": (("噪音", "吵", "发动机声", "noise", "noisy", "engine sound"),
                  "aircraft cabin noise noisy engine sound quiet"),
        "aircraft_condition": (("飞机新旧", "机舱老旧", "飞机老旧", "客舱状况", "机舱状况", "old aircraft", "dated cabin", "aircraft condition"),
                               "aircraft cabin condition old dated modern worn refurbished"),
    },
}

SUBTOPIC_NAMES = {"menu_items": "具体餐食与菜品", "quality": "质量与口味", "portion": "分量", "variety": "品种与选择",
                  "drinks": "饮品", "special_meals": "特殊餐食", "temperature": "餐食温度", "meal_service": "餐食服务流程",
                  "content": "娱乐内容", "screen": "屏幕", "wifi": "Wi-Fi", "audio": "耳机与音质",
                  "usability": "系统操作", "legroom": "腿部空间", "width": "座位宽度",
                  "recline": "靠背调节与躺平", "comfort": "座椅舒适度", "condition": "座椅机械状况",
                  "seat_cleanliness": "座椅清洁度", "power": "充电与电源", "privacy": "座位隐私", "storage": "座位储物",
                  "attitude": "服务态度", "efficiency": "服务效率",
                  "professionalism": "专业程度", "availability": "响应与在场服务", "communication": "沟通与语言",
                  "punctuality": "准点表现", "delay_handling": "延误与取消处置",
                  "baggage": "行李服务", "checkin": "值机服务", "gate_service": "登机口与登机服务",
                  "lounge": "机场休息室", "transfer": "转机衔接", "assistance": "特殊协助",
                  "price": "价格", "refund": "退款与改签", "support": "客服", "booking": "预订系统", "fees": "附加费用",
                  "cleanliness": "客舱清洁", "toilet": "洗手间", "cabin_temperature": "客舱温度",
                  "amenities": "机上用品", "noise": "客舱噪音", "aircraft_condition": "飞机与客舱状况"}


@dataclass
class ParsedQuestion:
    question: str
    aspect: str | None = None
    aspects: list[str] = field(default_factory=list)
    polarity: str | None = None
    airlines: list[str] = field(default_factory=list)
    filters: dict = field(default_factory=dict)
    retrieval_keywords: str = ""
    subtopics: dict[str, str] = field(default_factory=dict)
    subtopics_by_aspect: dict[str, dict[str, str]] = field(default_factory=dict)
    comparison: bool = False
    answer_intent: str = "evaluation"
    recognized: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _contains(text: str, phrase: str) -> bool:
    """Match CJK by substring and Latin aliases on token boundaries."""
    phrase_folded = phrase.casefold()
    if any("\u4e00" <= char <= "\u9fff" for char in phrase):
        return phrase_folded in text
    # ``\w`` also includes CJK characters, so it incorrectly rejects names in
    # mixed-language questions such as ``Spirit Airlines经济舱``.  Only Latin
    # letters and digits are meaningful token continuations for these aliases.
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase_folded)
                     + r"(?![a-z0-9])", text) is not None


def _first_alias(text: str, mapping: dict[str, Iterable[str]]) -> str | None:
    matches = []
    for value, aliases in mapping.items():
        for alias in aliases:
            if _contains(text, alias):
                matches.append((text.find(alias.casefold()), -len(alias), value))
    return min(matches)[2] if matches else None


def _all_aliases(text: str, mapping: dict[str, Iterable[str]]) -> list[str]:
    matches = []
    for value, aliases in mapping.items():
        for alias in aliases:
            if _contains(text, alias):
                matches.append((text.find(alias.casefold()), -len(alias), value))
    answer = []
    for _, _, value in sorted(matches):
        if value not in answer:
            answer.append(value)
    return answer


def _airlines(text: str, known_airlines: Iterable[str]) -> list[str]:
    found: list[tuple[int, int, str]] = []
    for alias, canonical in AIRLINE_ALIASES.items():
        if _contains(text, alias):
            start = text.find(alias.casefold())
            found.append((start, start + len(alias), canonical))
    for canonical in known_airlines:
        if _contains(text, canonical):
            start = text.find(canonical.casefold())
            found.append((start, start + len(canonical), canonical))
    # Prefer the longest name at a location.  This prevents ``国泰航空`` from
    # also matching the nested abbreviation ``泰航`` and prevents
    # ``Air Canada rouge`` from being shortened to ``Air Canada``.
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    answer = []
    accepted_spans: list[tuple[int, int]] = []
    for start, end, canonical in found:
        if any(existing_start <= start and end <= existing_end
               for existing_start, existing_end in accepted_spans):
            continue
        if canonical not in answer and canonical in known_airlines:
            answer.append(canonical)
            accepted_spans.append((start, end))
    return answer[:3]


def _route_filters(text: str, origins: Iterable[str], destinations: Iterable[str]) -> dict:
    origin_values, destination_values = list(origins), list(destinations)
    known_from = {value.casefold(): value for value in origin_values}
    known_to = {value.casefold(): value for value in destination_values}
    cities = {**{key.casefold(): value for key, value in CITY_ALIASES.items()}, **known_from, **known_to}
    names = sorted(cities, key=len, reverse=True)
    token = "(" + "|".join(re.escape(name) for name in names) + ")"
    patterns = [
        re.compile(r"(?:从|由)\s*" + token + r"\s*(?:到|飞往|去)\s*" + token),
        re.compile(token + r"\s*(?:飞往|飞到|飞|到|去往|去)\s*" + token),
        re.compile(r"(?:from)\s+" + token + r"\s+(?:to)\s+" + token),
    ]

    def variants(city: str, values: list[str]) -> str | list[str] | None:
        aliases = {
            "Singapore": ("singapore", "changi", "sin"),
            "London": ("london", "lhr", "lgw", "lcy", "ltn", "stn"),
            "New York": ("new york", "jfk", "ewr", "lga"),
            "Tokyo": ("tokyo", "narita", "haneda", "nrt", "hnd"),
            "Paris": ("paris", "cdg", "orly"),
        }.get(city, (city.casefold(),))
        matches = []
        for value in values:
            folded = value.casefold().strip()
            if re.search(r"\b(?:to|via)\b|[/,]", folded):
                continue
            if city == "London" and ("canada" in folded or "(on)" in folded):
                continue
            if any(re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", folded)
                   for alias in aliases):
                matches.append(value)
        matches = list(dict.fromkeys(matches))
        if not matches:
            return None
        return matches[0] if len(matches) == 1 else matches

    for pattern in patterns:
        match = pattern.search(text)
        if match:
            origin, destination = cities[match.group(1)], cities[match.group(2)]
            result = {}
            origin_values_for_city = variants(origin, origin_values)
            destination_values_for_city = variants(destination, destination_values)
            if origin_values_for_city is not None: result["route_from"] = origin_values_for_city
            if destination_values_for_city is not None: result["route_to"] = destination_values_for_city
            return result
    return {}


def parse_question(question: str, *, known_airlines: Iterable[str], origins: Iterable[str] = (),
                   destinations: Iterable[str] = (), default_aspect: str | None = None,
                   default_polarity: str | None = None, default_filters: dict | None = None) -> ParsedQuestion:
    """Return corpus-safe constraints; explicit question phrases override defaults."""
    raw = question.strip()
    folded = raw.casefold()
    if any(term in folded for term in ("有什么", "有哪些", "吃什么", "看什么", "能看什么", "what food",
                                       "what meals", "what movies", "which dishes", "what is served")):
        answer_intent = "enumeration"
    elif any(term in folded for term in ("怎么处理", "如何处理", "怎么办", "流程", "怎么安排",
                                         "how do", "how does", "process", "handled")):
        answer_intent = "procedure"
    elif (any(term in folded for term in ("有没有", "是否有", "是否提供", "能不能", "可不可以",
                                          "is there", "does it have", "do they provide", "available"))
          or re.search(r"(?:有|提供)(?!多好|多差).{1,16}吗[？?]?$", folded)):
        answer_intent = "availability"
    elif any(term in folded for term in ("多少", "多大", "多宽", "多长", "几种", "how much", "how many",
                                         "how wide", "how long")):
        answer_intent = "measurement"
    else:
        answer_intent = "evaluation"
    aspects = _all_aliases(folded, ASPECT_ALIASES)
    if not aspects:
        # A question may name only a concrete attribute (for example “靠背能否
        # 调整” or “延误后怎么处理”) without naming its parent aspect.
        subtopic_aspect_matches = []
        for candidate_aspect, rules in SUBTOPIC_RULES.items():
            for aliases, _ in rules.values():
                for alias in aliases:
                    if _contains(folded, alias):
                        subtopic_aspect_matches.append((folded.find(alias.casefold()), -len(alias), candidate_aspect))
        for _, _, candidate_aspect in sorted(subtopic_aspect_matches):
            if candidate_aspect not in aspects:
                aspects.append(candidate_aspect)
    if not aspects and default_aspect:
        aspects = [default_aspect]
    aspect = aspects[0] if aspects else None
    if aspect is None and any(term in folded for term in ("怎么样", "好不好", "哪家好", "哪家更好",
                                                           "评价如何", "how is", "how was", "which is better")):
        aspect = "overall"
        aspects = [aspect]
    polarity = _first_alias(folded, POLARITY_ALIASES) or default_polarity
    # Balanced evaluative questions ask for both sides.  In particular, the
    # substring “不好” inside “好不好” must not silently become a negative-only
    # evidence filter.
    balanced_sentiment_question = any(term in folded for term in (
        "好不好", "是好是坏", "正面还是负面", "good or bad", "positive or negative",
    ))
    if balanced_sentiment_question:
        polarity = None
    # “哪家最好 / which is best” asks for a comparison over both positive and
    # negative evidence; “最好” must not silently become a positive-only filter.
    if ("哪家" in folded or "which" in folded) and ("最好" in folded or "best" in folded):
        if not any(term in folded for term in ("好评", "positive", "praise", "优点", "值得推荐")):
            polarity = None
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
    subtopics_by_aspect = {}
    for requested_aspect in aspects:
        matches = []
        for key, (aliases, expansion) in SUBTOPIC_RULES.get(requested_aspect, {}).items():
            positions = [folded.find(alias.casefold()) for alias in aliases
                         if _contains(folded, alias)]
            if positions:
                matches.append((min(positions), key, expansion))
        requested = {key: expansion for _, key, expansion in sorted(matches)}
        if requested:
            subtopics_by_aspect[requested_aspect] = requested
    # “座椅清洁度” is a seat subtopic.  The generic word “清洁” also matches
    # the top-level cabin-cleanliness aspect, which used to turn this into an
    # unintended multi-aspect query.
    if ("seat" in aspects and "other" in aspects
            and "seat_cleanliness" in subtopics_by_aspect.get("seat", {})
            and not any(_contains(folded, term) for term in (
                "客舱清洁", "机舱清洁", "cabin cleanliness", "toilet", "厕所", "洗手间"))):
        aspects = [value for value in aspects if value != "other"]
    subtopics = subtopics_by_aspect.get(aspect, {})
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
    if subtopics: recognized.append("subtopic")
    return ParsedQuestion(question=raw, aspect=aspect, aspects=aspects, polarity=polarity,
                          airlines=airlines, filters=filters, retrieval_keywords=keywords,
                          subtopics=subtopics, subtopics_by_aspect=subtopics_by_aspect,
                          comparison=len(airlines) >= 2 or any(word in folded for word in comparison_words),
                          answer_intent=answer_intent,
                          recognized=recognized)
