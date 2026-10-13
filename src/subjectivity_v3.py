"""冻结的历史 v3 主客观决策；不含训练入口，仅用于旧模型复现。"""
import re
from src.weak_labels import normalize_text

OPINION = re.compile(
    r'\b(?:good|great|excellent|amazing|wonderful|fantastic|superb|outstanding|'
    r'bad|poor|terrible|awful|horrible|appalling|worst|best|disappoint(?:ed|ing|ment)|'
    r'comfortabl[ey]|uncomfortable|cramped|spacious|tasty|delicious|inedible|'
    r'friendly|unfriendly|rude|helpful|unhelpful|attentive|inattentive|'
    r'efficient|inefficient|professional|unprofessional|pleasant|unpleasant|'
    r'expensive|overpriced|affordable|worth|worthless|dirty|filthy|clean|'
    r'broken|unacceptable|acceptable|enjoy(?:ed|able)?|love[ds]?|hate[ds]?|'
    r'recommend(?:ed)?|satisfied|dissatisfied|impressed|disgusting|appreciat(?:e|ed)|'
    r'frustrat(?:ed|ing)|slow|rubbish|disaster|okay)\b', re.I)

IMPLICIT_EVALUATION = re.compile(
    r'\b(?:delay(?:ed|s)?|cancel(?:led|ed|lation)|on[ -]time|punctual)\b|'
    r'\bno (?:issues?|problems?|complaints?)\b|'
    r'\b(?:no|without) (?:food|meals?|entertainment|legroom|drinks?)\b|'
    r'\b(?:should|must|need to) improve\b|\bnot (?:work|working|worth)\b', re.I)

CITY_TOKEN = r"(?!(?:and|but|although|however|with|who|which|regret|missed|could|couldn't)\b)[a-z][a-z'-]*"

CITY = rf"{CITY_TOKEN}(?: {CITY_TOKEN}){{0,4}}"

OBJECTIVE_PATTERNS = {
    'journey': re.compile(rf'(?:i|we) (?:flew|travelled|traveled) from {CITY} to {CITY}'
                          rf'(?: via {CITY})?(?: in (?:january|february|march|april|may|june|july|august|'
                          r'september|october|november|december)(?: \d{4})?)?[.!]?', re.I),
    'cabin': re.compile(r'(?:i|we) (?:flew|travelled|traveled|was seated|were seated) in '
                       r'(?:economy|premium economy|business|first) class[.!]?', re.I),
    'aircraft': re.compile(r'(?:the|our|this) (?:aircraft|plane|flight) (?:was|is) (?:an? )?'
                          r'(?:(?:airbus|boeing) ?)?[ab]?[2378]\d{2}(?:[ -]?\d{2,3})?(?:er|lr|neo)?[.!]?', re.I),
    'price': re.compile(r'(?:the|my|our) (?:return |one-way )?(?:ticket|fare) '
                       r'(?:was|is|cost|costs) (?:[£$€]\s?\d[\d,.]*|'
                       r'(?:sgd|usd|eur|gbp|aud) ?\d[\d,.]*|'
                       r'\d[\d,.]* (?:sgd|usd|eur|gbp|aud|dollars?|euros?|pounds?))[.!]?', re.I),
    'duration': re.compile(r'(?:the|our|this) (?:flight|journey) (?:took|lasted|was) '
                          r'(?:about |approximately )?\d+(?:\.\d+)? hours?'
                          r'(?: (?:and )?\d+ minutes?)?(?: long)?[.!]?', re.I),
    'clock_time': re.compile(r'(?:the|our) flight (?:departed|arrived|left|landed) at '
                            r'\d{1,2}(?::\d{2})? ?(?:am|pm|a\.m\.|p\.m\.)[.!]?', re.I),
    'seat_layout': re.compile(r'(?:the )?seats (?:were|are) (?:arranged )?in (?:a )?'
                             r'\d-\d(?:-\d)? (?:configuration|layout)[.!]?', re.I),
    'fine_amount': re.compile(r'the fine was (?:\d+(?:\.\d+)? (?:dollars|euros|pounds)|[$£€]\d+(?:\.\d+)?)[.!]?',re.I),
    'miles_earned': re.compile(r'(?:i|we) earned \d[\d,]* miles (?:on|for) (?:this|the|our) flight[.!]?',re.I),
    'cleaning_event': re.compile(r'the (?:cabin )?crew cleaned the (?:toilet|cabin) (?:after takeoff|before landing)[.!]?',re.I),
    'flight_itinerary': re.compile(rf'(?:our|the) flight departed from {CITY} at '
                                   r'(?:\d{1,2}(?::\d{2})? ?(?:am|pm)|(?:nine|ten|eight) in the morning)'
                                   rf',? (?:and )?arrived in {CITY} later that day[.!]?',re.I),
}

def weak_decision(text):
    from src.language_rules_v3 import polarity_evidence
    text = normalize_text(text).replace('’', "'")
    polarity, evidence=polarity_evidence(text)
    if polarity is not None or evidence in {'mixed_polarity_cues','uncertain_negation'}:
        return 'subjective', evidence
    if OPINION.search(text):
        return 'subjective', 'explicit_evaluation'
    if IMPLICIT_EVALUATION.search(text):
        return 'subjective', 'project_implicit_evaluation'
    if re.search(r'\b(?:was|were|is|are|seemed) (?:quite |perfectly )?fine\b',text):
        return 'subjective','fine_as_evaluation'
    for name, pattern in OBJECTIVE_PATTERNS.items():
        if pattern.fullmatch(text):
            return 'objective', 'fact_' + name
    return None, 'abstain'
