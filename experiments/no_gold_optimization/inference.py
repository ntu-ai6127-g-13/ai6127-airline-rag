"""Frozen embedding heads, local NLI zero-shot inference, and score fusion. No gold import."""
from functools import lru_cache
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from common import ASPECTS, MINILM, NLI_PATH
from src.language_rules import choose_focus

ASPECT_DESCRIPTIONS={
 'entertainment':'onboard entertainment, movies, screens, headphones or Wi-Fi',
 'food':'onboard meals, food, snacks, drinks or catering',
 'ground':'airport check-in, boarding, baggage handling, delays or cancellations',
 'other':'aircraft type, cabin cleanliness, toilets, loyalty miles, upgrades or a travel itinerary',
 'overall':'the overall quality of the airline or the whole travel experience',
 'seat':'passenger seats, seating comfort, legroom or seat recline',
 'staff':'the service and behaviour of cabin crew and flight attendants',
 'value':'ticket prices, value for money, refunds, rebooking or the customer service hotline',
}
HYPOTHESES={
 'descriptive':[f'The main topic of this airline review is {ASPECT_DESCRIPTIONS[a]}.' for a in ASPECTS]+[
    'The passenger states a factual detail without praise, criticism or a personal evaluation.',
    'The passenger expresses dissatisfaction, criticism or a negative evaluation.',
    'The passenger expresses satisfaction, praise or a positive evaluation.'],
 'sentiment':[f'This text is about {ASPECT_DESCRIPTIONS[a]}.' for a in ASPECTS]+[
    'This is an objective description, with no positive or negative opinion.',
    'The sentiment of this airline review is negative.',
    'The sentiment of this airline review is positive.'],
}


@lru_cache(maxsize=1)
def embedding_model(path):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(path),local_files_only=True)


class FrozenEmbedding:
    def __init__(self,path=MINILM):self.path=str(path)
    def transform(self,texts):
        return embedding_model(self.path).encode(list(texts),batch_size=128,normalize_embeddings=True,show_progress_bar=False).astype(np.float32)


@lru_cache(maxsize=1)
def nli_model(path):
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(str(path),local_files_only=True)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model=model.to(device).eval()
    labels={str(v).lower():int(k) for k,v in model.config.id2label.items()}
    assert 'entailment' in labels
    return tokenizer,model,device,labels['entailment']


def context(texts):
    return {'text':list(texts),'focus':[choose_focus(t)['text'] for t in texts]}


def model_scores(models,x,xf):
    si=list(models['subjectivity'].classes_).index('objective')
    pi=list(models['polarity'].classes_).index('positive')
    return {'objective':models['subjectivity'].predict_proba(x)[:,si],
            'aspect_raw':models['aspect'].predict_proba(x),'aspect_focus':models['aspect'].predict_proba(xf),
            'aspect_classes':np.asarray(models['aspect'].classes_,dtype=str),
            'positive_raw':models['polarity'].predict_proba(x)[:,pi],'positive_focus':models['polarity'].predict_proba(xf)[:,pi]}


def nli_scores(texts,variant,path=NLI_PATH,progress=None):
    import torch
    tokenizer,model,device,ent=nli_model(str(path))
    hyp=HYPOTHESES[variant]; nh=len(hyp)
    ctx=context(texts); unique=list(dict.fromkeys([*ctx['text'],*ctx['focus']]))
    loc={s:i for i,s in enumerate(unique)}
    # Bucket by premise length to reduce padding. Never truncate the hypothesis.
    order=np.argsort([len(t) for t in unique],kind='stable')
    logits=np.empty((len(unique),nh),dtype=np.float32)
    for start in range(0,len(order),8):
        rows=order[start:start+8]
        premises=[unique[i] for i in rows for h in hyp]
        hypotheses=hyp*len(rows)
        batch=tokenizer(premises,hypotheses,padding=True,truncation='only_first',max_length=192,return_tensors='pt').to(device)
        with torch.inference_mode():value=model(**batch).logits[:,ent].float().cpu().numpy().reshape(len(rows),nh)
        logits[rows]=value
        if progress:progress(min(start+len(rows),len(order)),len(order))
    raw=np.asarray([loc[t] for t in ctx['text']]);focused=np.asarray([loc[t] for t in ctx['focus']])
    label=softmax(logits[:,8:],axis=1)
    pos=label[:,2]/np.maximum(1e-10,label[:,1]+label[:,2])
    aspect=softmax(logits[:,:8],axis=1)
    return {'objective':label[raw,0],'aspect_raw':aspect[raw],'aspect_focus':aspect[focused],
            'aspect_classes':np.asarray(ASPECTS),'positive_raw':pos[raw],'positive_focus':pos[focused]}


def blend(left,right,weight):
    assert np.array_equal(left['aspect_classes'],right['aspect_classes'])
    return {k:(v if k=='aspect_classes' else (1-weight)*v+weight*right[k]) for k,v in left.items()}


def scores(bundle,texts,progress=None):
    if bundle['kind']=='nli':return nli_scores(texts,bundle['hypothesis_variant'],bundle['model_path'],progress)
    if bundle['kind']=='fusion':return blend(scores(bundle['weak'],texts,progress),scores(bundle['nli'],texts,progress),bundle['nli_weight'])
    ctx=context(texts);vec=bundle['vectorizer']
    return model_scores(bundle['models'],vec.transform(ctx['text']),vec.transform(ctx['focus']))


def decide(sc,texts,config):
    a=config.get('aspect_focus',.5);p=config.get('polarity_focus',.5)
    aspect=sc['aspect_classes'][((1-a)*sc['aspect_raw']+a*sc['aspect_focus']).argmax(1)]
    objective=sc['objective']>=config.get('objective_threshold',.5)
    positive=(1-p)*sc['positive_raw']+p*sc['positive_focus']>=config.get('positive_threshold',.5)
    sub=np.where(objective,'objective','subjective');pol=np.where(objective,'NA',np.where(positive,'positive','negative'))
    return pd.DataFrame({'text':list(texts),'subjectivity':sub,'polarity':pol,'aspect':aspect,'label':np.where(objective,'objective',pol)})


def predict(bundle,texts,progress=None):
    return decide(scores(bundle,texts,progress),texts,bundle['decision_config'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--text',action='append',required=True)
    args=parser.parse_args()
    print(predict(joblib.load(args.model),args.text).to_json(orient='records',force_ascii=False,indent=2))
