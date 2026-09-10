"""文本清洗。抽样脚本和 build_kb.py 共用这里的函数。

发现的问题（实测 65,947 条评论）：
  1. 乱码       42.4% —— UTF-8 被按 cp1252 读过一遍，'✅'变成'âœ…'，'’'变成'â€™'
  2. 模板前缀   43.5% —— 网站自动加的 "✅ Trip Verified | 出发地 to 目的地."
  3. 模板尾巴   0.02% —— 少量评论末尾带 "⚑ British Airways Ranking"

为什么必须处理（报告 Q4 的答案）：
  · 乱码会破坏分词。"don't" 变成 "donâ€™t"，BM25 索引的是错的词，查询匹配不上
  · 模板前缀是噪声。它出现在两成以上的评论里，任何查询都会部分匹配到
    "Trip Verified"，稀释检索信号；而且前缀里的航线信息在 route 列已经有了

------------------------------------------------------------------------
2026-09-08 更新：乱码修复换成 ftfy，起因是残留率降不下去。

旧做法 `text.encode("cp1252").decode("utf-8")` 是**整条评论一起编码**，
只要评论里有一个 cp1252 装不下的字符（罗马尼亚语 ș、中文、emoji……），
整条的修复就被跳过 —— 一颗老鼠屎坏一锅汤。而且 cp1252 有 5 个未定义位
（0x81/8D/8F/90/9D），右双引号 '”' 的第三个字节正好是 0x9D，必然失败。

三种方案在 65,947 条真实评论上的实测（残留乱码率）：
    旧做法（整条 cp1252）   2.14%    0.4s
    只修「像乱码的那一段」   0.37%    0.5s
    ftfy                   0.03%   18.9s   ← 采用
ftfy 剩下的 21 条里有 12 条根本不是乱码，是真的法语/西语/葡语被误判，
真正的残留只有 9 条。误伤检查：只有 1 条评论被多改了，改的是删掉一个
不可见控制符 \x7f，是对的。

附带好处：ftfy 会把弯引号拉直（'’'→"'"、'“'→'"'）。这对关键词检索是
有利的 —— 用户搜 don't 能匹配到 don't，不拉直就匹配不上。
------------------------------------------------------------------------
"""
import re

import ftfy

# 网站的模板前缀共 7 种主要写法（实测）：
#   ✅ Trip Verified |     13,670      ✅ Verified Review |   12,126
#   Not Verified |          1,714      ❎ Unverified |           174
#   以及它们的乱码残留形式
# 统一处理：开头的任意符号 + Verified/Unverified 字样 + 竖线
_VERIFIED_PREFIX = re.compile(
    r'^[^A-Za-z]{0,6}'                                    # 开头的 ✅ / ❎ / 乱码残片
    r'(?:trip\s+verified|verified\s+review|not\s+verified|unverified|verified)'
    r'\s*\|\s*',
    flags=re.IGNORECASE)

# 少量评论末尾粘着网站的排行榜链接文字，例："⚑ British Airways Ranking"。
# 要求 Ranking 前面必须有 1-4 个非 ASCII 字符（原文里那个 ⚑，或者 ftfy 没修掉的
# 乱码残片 'âš‘'）—— 不加这个约束会误伤正文里正常提到 ranking 的句子，
# 加宽到 [^\w\s] 又会只切掉一半、把乱码残片留在末尾。
_RANKING_SUFFIX = re.compile(
    r"\s*[^\x00-\x7f\s]{1,4}[^\w\s]{0,2}\s*[A-Z][A-Za-z'\- ]{2,40}\s+Ranking\s*$")

# route 列的格式：'<出发> to <目的>' 或 '<出发> to <目的> via <中转>'
# 实测 24,432 / 24,549 条（99.5%）能匹配上
_ROUTE = re.compile(r'^(.*?)\s+to\s+(.*?)(?:\s+via\s+(.*))?$', flags=re.IGNORECASE)


# --- 分段兜底：ftfy 偶尔会保守地放过一些片段（例如 'Ã…' -> 'Å' 它就不改）---
# cp1252 里没有定义的 5 个位置，按 latin-1 直通补上
_CP1252_HOLES = {0x81, 0x8D, 0x8F, 0x90, 0x9D}


def _to_bytes(s: str):
    """把字符串按 cp1252 编回字节；有一个字符编不了就返回 None。"""
    out = bytearray()
    for ch in s:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) in _CP1252_HOLES:
                out.append(ord(ch))
            else:
                return None
    return bytes(out)


# 乱码长什么样：首字符落在 U+00C2..U+00F4（UTF-8 首字节被当 cp1252 解出来的范围），
# 后面跟 1-3 个能编回 cp1252 的字符
_LEAD = "".join(chr(i) for i in range(0xC2, 0xF5))
_TAIL = "".join(chr(i) for i in range(0x80, 0x2200) if _to_bytes(chr(i)) is not None)
_MOJI_SPAN = re.compile("[" + re.escape(_LEAD) + "][" + re.escape(_TAIL) + "]{1,3}")


def _repair_span(m: "re.Match") -> str:
    b = _to_bytes(m.group(0))
    if b is None:
        return m.group(0)
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        # 拼不成合法 UTF-8，说明本来就是真外语（compañía 里的 'ñí' 就会走到这），别动
        return m.group(0)


def fix_mojibake(text: str) -> str:
    """修复 UTF-8 被当成 cp1252 读取造成的乱码，并顺带做 Unicode 规范化。

    两步：先 ftfy（覆盖绝大多数），再对残留的片段做一次分段修复。
    第二步只在片段能还原成**合法 UTF-8** 时才替换，所以不会误伤真外语 ——
    全语料实测第二步只多改了 1 条评论（'Ã…lesund' -> 'Ålesund'），
    而 été / compañía / Târgu Mureș / Zürich 这些一个都没动。

    >>> fix_mojibake("The seat wasnâ€™t comfortable")
    "The seat wasn't comfortable"
    >>> fix_mojibake("Phuket to DÃ¼sseldorf")
    'Phuket to Düsseldorf'
    >>> fix_mojibake("Trondheim to Ã…lesund")
    'Trondheim to Ålesund'
    >>> fix_mojibake("compañía")
    'compañía'
    """
    if not isinstance(text, str):
        return text
    text = ftfy.fix_text(text)
    for _ in range(3):                      # 少量是多重编码，反复修到不变为止
        new = _MOJI_SPAN.sub(_repair_span, text)
        if new == text:
            break
        text = new
    return text


# 有 9 句的 "✅ Verified Review |" 不在评论开头，而是夹在中间
# （原始数据里有把两条评论拼在一起的）。带竖线的这几种写法足够明确，
# 不加 ^ 锚定也不会误伤正文，但这里只保留 4 种完整写法，不要光秃秃的 "verified"。
# [^A-Za-z.!?] 而不是 [^A-Za-z]：不能把前一句的句号吃掉，
# 否则 spaCy 会把前后两句粘成一句。
_VERIFIED_INLINE = re.compile(
    r'[^A-Za-z.!?]{0,6}'
    r'(?:trip\s+verified|verified\s+review|not\s+verified|unverified)'
    r'\s*\|\s*',
    flags=re.IGNORECASE)


def strip_boilerplate(text: str) -> str:
    """去掉网站自动加的 "Trip Verified |" 标记和 "… Ranking" 尾巴。"""
    text = _VERIFIED_PREFIX.sub("", text)
    text = _VERIFIED_INLINE.sub(" ", text)
    text = _RANKING_SUFFIX.sub("", text)
    return text.strip()


def clean_review(text: str) -> str:
    """完整的清洗流程：修乱码 -> 去模板 -> 压缩空白。

    注意**不做**的事：
      · 不转小写      —— 展示给用户看的要是原文
      · 不去标点感叹号 —— 那些是真实的情感信号
      · 不改拼写错误   —— 同上
    """
    if not isinstance(text, str):
        return text
    t = fix_mojibake(text)
    t = strip_boilerplate(t)
    return re.sub(r"\s+", " ", t).strip()


def clean_field(value):
    """清洗元数据短字段（airline / route / cabin）。

    只修乱码，不动模板 —— route 列里也有 'DÃ¼sseldorf' 这种，
    之前只洗了评论正文，把它漏了。
    """
    if not isinstance(value, str):
        return value
    return re.sub(r"\s+", " ", fix_mojibake(value)).strip()


def parse_route(route):
    """把 'Singapore to London via Dubai' 拆成 (出发, 目的, 中转)。

    作业范例问题就是「新加坡飞伦敦经济舱」这种带起降点的约束，
    拆开之后检索时可以直接按出发地/目的地过滤，不用去正文里模糊匹配。
    拆不开就返回 (None, None, None)，不猜。

    >>> parse_route("Singapore to London via Dubai")
    ('Singapore', 'London', 'Dubai')
    >>> parse_route("LHR to JFK")
    ('LHR', 'JFK', None)
    >>> parse_route("weird value")
    (None, None, None)
    """
    if not isinstance(route, str):
        return (None, None, None)
    m = _ROUTE.match(route.strip())
    if not m:
        return (None, None, None)
    a, b, via = m.group(1).strip(), m.group(2).strip(), m.group(3)
    if not a or not b:
        return (None, None, None)
    return (a, b, via.strip() if via else None)


# ---------------------------------------------------------------------
# 日期解析。原始 review_date 是 "2nd May 2019" 这种带序数后缀的字符串。
# 直接 pd.to_datetime() 会**静默失败 21.6%**：它认得 8th/29th（th），
# 却认不了 1st/2nd/3rd/21st/31st（st/nd/rd）。失败的变成 NaT，看起来
# 就像"这些评论没有日期"，其实原始数据 100% 有日期。
# 先把序数后缀去掉再按固定格式解析，100% 成功，而且比 format="mixed" 快 6 倍。
# ---------------------------------------------------------------------
_ORDINAL = re.compile(r"(\d+)(st|nd|rd|th)\b", flags=re.IGNORECASE)


def parse_review_date(series):
    """把 '2nd May 2019' 这类字符串解析成日期。传入传出都是 pandas Series。"""
    import pandas as pd
    cleaned = series.astype(str).str.replace(_ORDINAL, r"\1", regex=True)
    return pd.to_datetime(cleaned, errors="coerce", format="%d %B %Y")


def parse_date_flown(series):
    """date_flown 混着 datetime 和 'April 2019' 字符串，而且**只有月精度**。

    当初 build_kb 把这一列整个丢掉，就是因为混合类型 parquet 存不下。
    统一成 datetime 之后就能存了。注意它代表的是"哪个月飞的"，不是具体某天。
    """
    import pandas as pd
    def one(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return pd.NaT
        if isinstance(v, str):
            return pd.to_datetime(v, errors="coerce", format="%B %Y")
        return pd.Timestamp(v)
    return series.map(one)


# 评论正文开头常常就是航线："London to Izmir via Istanbul. First time I'd..."
# route 列有 38.7% 是空的，其中约 1,206 条能从正文开头把航线捞回来。
# 要求两边都是大写开头的地名、并以句号/逗号结束，避免把正常句子误当成航线。
_ROUTE_IN_TEXT = re.compile(
    r"^([A-Z][\w'\-.]*(?:\s+[A-Z][\w'\-.]*){0,3})"
    r"\s+to\s+"
    r"([A-Z][\w'\-.]*(?:\s+[A-Z][\w'\-.]*){0,3})"
    r"(?:\s+via\s+([A-Z][\w'\-.]*(?:\s+[A-Z][\w'\-.]*){0,3}))?"
    r"\s*[.,]")


def extract_route_from_text(text):
    """route 为空时，尝试从**清洗后**正文的开头把航线捞回来。

    捞不到就返回 None，不猜。

    >>> extract_route_from_text("London to Bangkok via Istanbul. Great flight.")
    'London to Bangkok via Istanbul'
    >>> extract_route_from_text("I went to the lounge. It was fine.")
    """
    if not isinstance(text, str):
        return None
    m = _ROUTE_IN_TEXT.match(text.strip())
    if not m:
        return None
    a, b, via = m.group(1), m.group(2), m.group(3)
    return f"{a} to {b}" + (f" via {via}" if via else "")


# 「整句就是航线」的句子，例如 "Guangzhou to New York."。
# 实测 16,558 句（3.03%）。它们和 route 列 100% 重复，没有任何情感极性：
# 进了检索索引只会占位置，进了弱标签只会加噪声。
# 这里只**打标记**不删除 —— 删不删是检索策略，属于 B 组的决定，
# 而且删掉不可逆；留一个布尔列，谁要用谁自己 filter。
_ROUTE_ONLY = re.compile(
    r"^[A-Z][\w'\-.]*(?:\s+[A-Z][\w'\-.]*){0,3}"
    r"\s+to\s+"
    r"[A-Z][\w'\-.]*(?:\s+[A-Z][\w'\-.]*){0,3}"
    r"(?:\s+via\s+[A-Z][\w'\-.]*(?:\s+[A-Z][\w'\-.]*){0,3})?"
    r"\s*\.?$")


def is_route_only(text) -> bool:
    """这句话是不是纯航线元数据（没有任何评价内容）。

    >>> is_route_only("Guangzhou to New York.")
    True
    >>> is_route_only("London to Izmir via Istanbul")
    True
    >>> is_route_only("I flew from London to Izmir and it was great.")
    False
    >>> is_route_only("Bangkok to Hong Kong was delayed.")
    False
    """
    if not isinstance(text, str):
        return False
    return bool(_ROUTE_ONLY.match(text.strip()))
