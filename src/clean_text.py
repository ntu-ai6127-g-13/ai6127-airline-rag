"""文本清洗。抽样脚本和 build_kb.py 共用这里的函数。

发现的问题（实测 65,947 条评论）：
  1. 乱码       42.4% —— UTF-8 被按 cp1252 读过一遍，'✅'变成'âœ…'，'’'变成'â€™'
  2. 模板前缀   43.5% —— 网站自动加的 "✅ Trip Verified | 出发地 to 目的地."

为什么必须处理（报告 Q4 的答案）：
  · 乱码会破坏分词。"don't" 变成 "donâ€™t"，BM25 索引的是错的词，查询匹配不上
  · 模板前缀是噪声。它出现在两成以上的评论里，任何查询都会部分匹配到
    "Trip Verified"，稀释检索信号；而且前缀里的航线信息在 route 列已经有了
"""
import re

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


def fix_mojibake(text: str) -> str:
    """修复 UTF-8 被当成 cp1252 读取造成的乱码。

    >>> fix_mojibake("The seat wasnâ€™t comfortable")
    'The seat wasn’t comfortable'
    """
    if not isinstance(text, str):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text          # 本来就是正常文本，原样返回


def strip_boilerplate(text: str) -> str:
    """去掉网站自动加的 "Trip Verified |" 前缀。"""
    return _VERIFIED_PREFIX.sub("", text).strip()


def clean_review(text: str) -> str:
    """完整的清洗流程：修乱码 -> 去模板前缀 -> 压缩空白。

    注意**不做**的事：
      · 不转小写      —— 展示给用户看的要是原文
      · 不去标点感叹号 —— 那些是真实的情感信号
      · 不改拼写错误   —— 同上
    """
    if not isinstance(text, str):
        return text
    t = fix_mojibake(text)
    t = fix_mojibake(t)          # 有少量是双重编码，再修一次
    t = strip_boilerplate(t)
    return re.sub(r"\s+", " ", t).strip()
