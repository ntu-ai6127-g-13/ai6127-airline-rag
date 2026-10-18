"""C1 本地规则版本；不改动 A/B 共用的 config.ASPECTS。

context_v3 保留历史行为；context_v4 修复 ground-crew 的方面归属。
宁可放弃不确定训练句，也不假装关键词是完整语义解析。
"""
import re
import unicodedata

import pandas as pd
from src.config import ASPECTS

VERSIONS = {'legacy': 'c1-single-aspect-v2-eval-guard',
            'context_v3': 'c1-single-aspect-v3-context',
            'context_v4': 'c1-single-aspect-v4-audit'}

# 用非捕获分组和完整单词边界，避免把 seated、moviegoer 等词误当关键词。
PATTERNS_V3 = {
    'seat': [r'seats?', r'leg[ -]?room', r'seat[ -]pitch',
             r'reclin(?:e|es|ed|ing|ers?)'],
    'staff': [r'(?<!ground )crew', r'(?:cabin|flight)[ -](?:staff|service|attendants?)',
              r'cabin[ -](?:manager|manger|director|supervisor)',
              r'stewards?', r'stewardess(?:es)?', r'air[ -]hostess(?:es)?', r'purser'],
    'food': [r'foods?', r'meals?', r'drinks?', r'breakfasts?', r'dinners?',
             r'snacks?', r'catering', r'beverages?', r'lunch(?:es)?'],
    'entertainment': [r'entertainment', r'movies?', r'ife', r'films?', r'headphones?',
                      r'audio', r'wi[ -]?fi'],
    'ground': [r'check[ -]?in', r'check(?:ed|ing)?(?: (?:me|us|them|him|her|you))? in',
               r'boarding', r'gates?', r'luggage', r'baggage', r'counters?', r'queues?',
               r'ground[ -](?:staff|crew|service)', r'airport[ -]staff',
               r'delay(?:s|ed)?', r'cancel(?:led|ed|lations?)', r'on[ -]time', r'punctual(?:ity)?'],
    'value': [r'value', r'prices?', r'expensive', r'cheap', r'worth', r'money',
              r'fares?', r'costs?(?![ -]savings?\b)', r'refund(?:s|ed)?',
              r'rebook(?:ed|ing)?', r'reschedul(?:e|ed|ing)',
              r'customer[ -]service', r'call[ -]cent(?:er|re)', r'hotline'],
}

# 历史模式不改写。新版在空格或连字符之后都把 ground crew 交给 ground，
# 仍允许同句的另一处 cabin crew 命中 staff。
PATTERNS = {aspect: list(parts) for aspect, parts in PATTERNS_V3.items()}
PATTERNS['staff'][0] = r'(?<!\bground[ -])crew'


def compile_words(parts):
    return re.compile(r'(?<![a-z])(?:' + '|'.join(parts) + r')(?![a-z])', re.I)


COMPILED_V3 = {a: compile_words(p) for a, p in PATTERNS_V3.items()}
COMPILED = {a: compile_words(p) for a, p in PATTERNS.items()}
CABIN_CONTEXT = compile_words([r'on[ -]?board', r'in[ -]?flight', r'cabin',
                               r'flight attendants?', r'stewardess(?:es)?', r'purser'])
NON_CABIN_CONTEXT = compile_words([r'airport', r'gates?', r'counters?', r'ground',
                                   r'check[ -]?in', r'check(?:ed|ing)?(?: (?:me|us|them))? in',
                                   r'customer service', r'refunds?', r'call cent(?:er|re)'])
STAFF_GENERIC = compile_words([r'staff', r'attendants?', r'service', r'friendly', r'rude', r'helpful'])
SCREEN_GENERIC = compile_words([r'screens?', r'tvs?', r'televisions?'])
SCREEN_CONTEXT = compile_words([r'in[ -]?flight', r'seat[ -]?back', r'entertainment', r'ife',
                                r'movies?', r'films?', r'channels?', r'arm[ -]?rests?'])
SCREEN_OTHER = compile_words([r'websites?', r'web sites?', r'booking', r'reservations?',
                              r'payments?', r'information desk', r'gates?', r'counters?',
                              r'luggage', r'baggage', r'claims?', r'checked in', r'check[ -]?in'])
SEAT_CONTEXT = compile_words([r'seats?', r'leg[ -]?room', r'reclin(?:e|ing|er)', r'seat[ -]pitch'])
SEAT_GENERIC = compile_words([r'comfortable', r'uncomfortable', r'space', r'pitch', r'cramped'])
# 按项目规范，明显的客舱/座椅脏污应交给 other，不能硬套 food/seat。
CLEANLINESS = re.compile(
    r'\b(?:cabin|aircraft|plane|seats?|toilets?|lavator(?:y|ies)|restrooms?)\b'
    r'[^.!?;]{0,45}\b(?:dirty|filthy|unclean|stained|sticky)\b|'
    r'\b(?:dirty|filthy|unclean|stained|sticky)\b[^.!?;]{0,30}'
    r'\b(?:cabin|aircraft|plane|seats?|toilets?|lavator(?:y|ies)|restrooms?)\b|'
    r'\bfood\b[^.!?;]{0,25}\bstuck\b[^.!?;]{0,25}\bseats?\b', re.I)


def matching_text_v3(text):
    value = unicodedata.normalize('NFKC', text).casefold()
    value = re.sub(r'[‐‑‒–—−]', '-', value)
    return ' '.join(value.split())


def matching_text(text):
    value = matching_text_v3(text)
    # 只收紧这一个复合词；不把其他标点或跨句片段改成关键词。
    return re.sub(r'\bground *- *crew\b', 'ground-crew', value)


def match_series(text: pd.Series, ruleset='context_v4'):
    """返回各方面真实命中词与需弃用的 unsupported-aspect 标记；不改原文。"""
    if ruleset not in VERSIONS:
        raise ValueError(f'未知规则版本：{ruleset}')
    if ruleset == 'legacy':
        normalized = text.str.lower()
        matchers = {a: compile_words([re.escape(w) for w in sorted(set(words), key=lambda w: (-len(w), w))])
                    for a, (_, words) in ASPECTS.items()}
        matches = pd.DataFrame({a: normalized.str.findall(p) for a, p in matchers.items()})
        return matches, pd.Series(False, index=text.index)
    normalizer = matching_text_v3 if ruleset == 'context_v3' else matching_text
    matchers = COMPILED_V3 if ruleset == 'context_v3' else COMPILED
    normalized = text.map(normalizer)
    matches = pd.DataFrame({a: normalized.str.findall(p) for a, p in matchers.items()})
    def add_conditional(aspect, matcher, allowed):
        extra = normalized.str.findall(matcher)
        extra = extra.where(allowed, other=None).map(lambda v: v if isinstance(v, list) else [])
        matches[aspect] = matches[aspect].combine(extra, lambda a, b: a + b)
    add_conditional('staff', STAFF_GENERIC,
                    normalized.str.contains(CABIN_CONTEXT) & ~normalized.str.contains(NON_CABIN_CONTEXT))
    add_conditional('entertainment', SCREEN_GENERIC,
                    normalized.str.contains(SCREEN_CONTEXT) & ~normalized.str.contains(SCREEN_OTHER))
    add_conditional('seat', SEAT_GENERIC, normalized.str.contains(SEAT_CONTEXT))
    return matches, normalized.str.contains(CLEANLINESS)


def rule_description(ruleset):
    if ruleset not in VERSIONS:
        raise ValueError(f'未知规则版本：{ruleset}')
    if ruleset == 'legacy':
        return {'ruleset': ruleset, 'keywords': {a: v[1] for a, v in ASPECTS.items()}}
    patterns = PATTERNS_V3 if ruleset == 'context_v3' else PATTERNS
    policy = ['plural and hyphen variants are explicit',
              'bare service/staff need cabin context; ground floor is not ground service',
              'bare screens/TV need entertainment context and no conflicting context',
              'comfort/space require a seat anchor; cost saving is not value for money',
              'clear cabin/seat cleanliness abstains as unsupported other',
              'multiple detected aspects abstain; no clause-level polarity inference']
    if ruleset == 'context_v4':
        policy.append('ground crew space/hyphen variants map to ground, not cabin staff')
    return {'ruleset': ruleset, 'patterns': patterns,
            'conditional_terms': {'staff': STAFF_GENERIC.pattern, 'entertainment': SCREEN_GENERIC.pattern,
                                  'seat': SEAT_GENERIC.pattern},
            'policy': policy}
