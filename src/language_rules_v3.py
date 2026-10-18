# 冻结的历史 v3 推理快照；只供旧模型复现，不作为新规则入口。
"""可审计的有限句式规则：否定范围与 R16 评价片段。不是完整语义解析器。"""
import re

from src.weak_label_rules_v3 import (COMPILED, CABIN_CONTEXT, NON_CABIN_CONTEXT, STAFF_GENERIC,
                                  SCREEN_CONTEXT, SCREEN_OTHER, SCREEN_GENERIC, matching_text)

POSITIVE = re.compile(r'\b(?:good|great|excellent|wonderful|amazing|pleasant|comfortable|'
                      r'friendly|helpful|attentive|delicious|tasty|spacious|acceptable|okay|'
                      r'enjoy(?:ed)?|happ(?:y|ier)|satisfied|impressed|recommend(?:ed)?)\b', re.I)
NEGATIVE = re.compile(r'\b(?:bad|poor|terrible|awful|horrible|rude|unfriendly|unhelpful|'
                      r'uncomfortable|cramped|dirty|filthy|tasteless|inedible|broken|'
                      r'expensive|overpriced|disappoint(?:ed|ing)|regret(?:ted)?|'
                      r'delay(?:ed|s)?|cancel(?:led|ed)|unacceptable|appalling)\b', re.I)
SPECIAL = [
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
