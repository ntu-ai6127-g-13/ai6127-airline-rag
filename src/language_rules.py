"""可审计的有限句式规则：否定范围与 R16 评价片段。不是完整语义解析器。"""
import re

from src.weak_label_rules import (COMPILED, CABIN_CONTEXT, NON_CABIN_CONTEXT, STAFF_GENERIC,
                                  SCREEN_CONTEXT, SCREEN_OTHER, SCREEN_GENERIC, matching_text)

RULE_VERSION = 'language-audit-v4.1'

# 强烈推荐习语只接受短宾语和明确的程度结尾，不跨标点或连接词搜寻 enough。
# 结尾不能接名词，避免把 “cannot recommend any higher fares” 当成赞扬。
RECOMMEND_OBJECT_WORD = (
    r"(?!(?:not|never|but|and|or|although|though|however|because|unless|if|when|while|"
    r"as|that|who|which|after|following|since|despite|due|to|for|with|from|"
    r"is|are|was|were|be|been|have|has|had|will|would|enough|higher|highly|"
    r"strongly|more|too)\b)[a-z0-9]+(?:['-][a-z0-9]+)*"
)
STRONG_RECOMMENDATION = re.compile(
    r"\b(?:cannot|can't|can not|could not|couldn't)\s+recommend\b"
    r'(?:\s+' + RECOMMEND_OBJECT_WORD + r'){0,8}\s+'
    r'(?:(?:highly|strongly)\s+enough|enough|(?:any\s+)?higher|'
    r'(?:any\s+)?more\s+highly|too\s+highly)\b'
    r'(?=\s*(?:$|[.!?;,]|(?:to|for|after|following|given|thanks|from|but|and|'
    r'although|however|because|with|as)\b))', re.I)

# recommend 的程度不能决定建议方向；劝避航司也是强烈建议。
AVOIDANCE_ACTION = re.compile(
    r'^(?:avoid(?:ing)?|(?:keep(?:ing)?|stay(?:ing)?)\s+(?:well\s+)?away(?:\s+from)?|'
    r'steer(?:ing)?\s+clear(?:\s+of)?|(?:not|never)\s+(?:fly|use|book|travel)(?:\s+with)?)\b', re.I)
AVOIDANCE_TARGET = re.compile(
    r'^(?:(?:flying|using|booking|travelling|traveling)\s+(?:with\s+)?)?'
    r'(?:it|them|(?:(?:this|that|the|these|those|any|our|their)\s+)?'
    r'(?:airlines?|carriers?|flights?|compan(?:y|ies)|services?|routes?|staff|crew))\b', re.I)


def recommendation_direction(text, match):
    """仅解释习语宾语或紧邻的 to 建议；目标不明时不把劝避误当赞扬。"""
    body = re.split(r'\brecommend\b', match.group(), maxsplit=1, flags=re.I)[1].strip()
    following = re.match(r'\s+to\s+', text[match.end():], re.I)
    candidates = [body]
    if following:
        candidates.append(text[match.end() + following.end():])
    for candidate in candidates:
        action = AVOIDANCE_ACTION.match(candidate)
        if action:
            target = candidate[action.end():].lstrip()
            return 'negative' if AVOIDANCE_TARGET.match(target) else None
    return 'positive'

POSITIVE = re.compile(r'\b(?:good|great|excellent|wonderful|amazing|pleasant|comfortable|'
                      r'friendly|helpful|attentive|delicious|tasty|spacious|acceptable|okay|'
                      r'enjoy(?:ed)?|happ(?:y|ier)|satisfied|impressed|recommend(?:ed)?)\b', re.I)
NEGATIVE = re.compile(r'\b(?:bad|poor|terrible|awful|horrible|rude|unfriendly|unhelpful|'
                      r'uncomfortable|cramped|dirty|filthy|tasteless|inedible|broken|'
                      r'expensive|overpriced|disappoint(?:ed|ing)|regret(?:ted)?|'
                      r'delay(?:ed|s)?|cancel(?:led|ed)|unacceptable|appalling)\b', re.I)
SPECIAL = [
    (STRONG_RECOMMENDATION, 'positive'),
    (re.compile(r"\b(?:could not|couldn't|cannot|can't) (?:have been|be) happier\b",re.I),'positive'),
    (re.compile(r"\b(?:could not|couldn't|cannot|can't) (?:have been|be) worse\b",re.I),'negative'),
    (re.compile(r'\b(?:no|without(?: any)?) (?:issues?|problems?|complaints?)\b',re.I),'positive'),
    (re.compile(r'\b(?:no|without) (?:food|meals?|entertainment|legroom|drinks?)\b',re.I),'negative'),
    (re.compile(r"\b(?:did not|didn't) lose\b[^.!?;,]{0,35}\b(?:luggage|baggage|bags?)\b",re.I),'positive'),
    (re.compile(r'\bnever (?:fly|use|travel)\b[^.!?;,]{0,40}\bagain\b',re.I),'negative'),
    (re.compile(r'\bmissed (?:my|our|the|a) (?:connection|flight)\b',re.I),'negative'),
    (re.compile(r'\bon[ -]time\b|\bpunctual\b',re.I),'positive'),
    (re.compile(r'\b(?:should|must|need to) improve\b',re.I),'negative'),
]


def polarity_evidence(text):
    """只有方向单一时返回正负；混合/歧义则弃权，不任意翻转整句。"""
    s=matching_text(text).replace('’', "'")
    occupied=[];signs=[]
    for pattern, sign in SPECIAL:
        for m in pattern.finditer(s):
            if pattern is STRONG_RECOMMENDATION:
                sign = recommendation_direction(s, m)
                if sign is None:
                    return None, 'uncertain_recommendation_direction'
            if pattern is STRONG_RECOMMENDATION and NEGATIVE.search(m.group()):
                # 不能借程度习语吞掉宾语里的明确负面评价。
                continue
            prefix=s[max(0,m.start()-20):m.start()]
            if re.search(r"\bnot\s+$|n't\s+$",prefix):
                sign_here='negative' if sign=='positive' else 'positive'
            else:sign_here=sign
            signs.append(sign_here);occupied.append(m.span())
    for pattern,sign in [(POSITIVE,'positive'),(NEGATIVE,'negative')]:
        for m in pattern.finditer(s):
            if any(a<=m.start()<b for a,b in occupied):continue
            prefix=re.split(r'[,;.!?]|\b(?:and|but|although|however)\b',s[:m.start()])[-1]
            # 不把 not only / not just 当成否定；不推断 not necessarily 的方向。
            if re.search(r'\bnot necessarily\b[^.!?;]{0,15}$',prefix):
                return None,'uncertain_negation'
            prefix=re.sub(r'\bnot (?:only|just|simply)\b','',prefix)
            negated=bool(re.search(r"(?:\b(?:not|cannot|never|hardly|no)|n't)(?:\s+[a-z]+){0,2}\s*$",prefix))
            signs.append(('negative' if sign=='positive' else 'positive') if negated else sign)
    values=set(signs)
    if len(values)==1:return values.pop(),'unambiguous_polarity_cues'
    return None,'mixed_polarity_cues' if values else 'no_polarity_cue'


def aspect_hits(text):
    s=matching_text(text)
    result={a for a,p in COMPILED.items() if p.search(s)}
    if CABIN_CONTEXT.search(s) and not NON_CABIN_CONTEXT.search(s) and STAFF_GENERIC.search(s):result.add('staff')
    if SCREEN_CONTEXT.search(s) and not SCREEN_OTHER.search(s) and SCREEN_GENERIC.search(s):result.add('entertainment')
    if re.search(r'\b(?:toilets?|lavator(?:y|ies)|restrooms?|washrooms?)\b',s):result.add('other')
    return result


def choose_focus(text):
    """返回原句内部片段；原文、句子 ID 和行数不变。无法可靠解析则保留整句。"""
    matches=list(re.finditer(r'\b(?:but|however|although|though)\b',text,re.I))
    for m in reversed(matches):
        if m.group().lower()=='but' and re.search(r'\bnot only\b',text[:m.start()],re.I):continue
        tail=text[m.end():].strip(' ,;:')
        if not tail:continue
        # 句首 Although X, Y 取主句 Y，避免把整个让步结构仍当作选中片段。
        if m.group().lower() in {'although','though'} and not text[:m.start()].strip():
            if ',' not in tail:continue
            tail=tail.split(',',1)[1].strip()
        if tail:
            hits=aspect_hits(tail)
            return {'text':tail,'reason':'contrast_focus','aspect':next(iter(hits)) if len(hits)==1 else None}
    hits=aspect_hits(text)
    if len(hits)<2:return {'text':text,'reason':'whole_sentence','aspect':None}
    clauses=[v.strip() for v in re.split(r'[,;]|\band\b',text,flags=re.I) if v.strip()]
    candidates=[]
    for clause in clauses:
        aspects=aspect_hits(clause);polarity,_=polarity_evidence(clause)
        if len(aspects)==1 and polarity:
            candidates.append((clause,next(iter(aspects)),polarity))
    if {c[1] for c in candidates} != hits:
        return {'text':text,'reason':'unresolved_multi_aspect','aspect':None}
    polarities={c[2] for c in candidates}
    if len(polarities)==1 and len(hits)>=3:
        return {'text':text,'reason':'r16_three_aspects','aspect':'overall'}
    if len(polarities)==1:
        selected=candidates[0];reason='r16_first_evaluated_aspect'
    else:
        selected=next(c for c in candidates if c[2]=='negative');reason='r16_negative_aspect'
    return {'text':selected[0],'reason':reason,'aspect':selected[1]}
