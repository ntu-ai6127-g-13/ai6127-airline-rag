"""Shared paths and partial-label metrics for experiments that exclude project gold."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/c_gold_smoke_20261018'
SOURCE = BASE / 'source'
sys.path.insert(0,str(SOURCE))
ASPECTS = ['entertainment','food','ground','other','overall','seat','staff','value']
TASKS = {'aspect':ASPECTS,'subjectivity':['objective','subjective'],'polarity':['negative','positive']}
GOALS = {'subjectivity_f1':.80,'aspect_f1':.80,'label_f1':.80,'polarity_f1':.88,
         'objective_precision':.70,'objective_recall':.70,'minimum_aspect_f1':.65,'joint_accuracy':.70}
MINILM = Path.home()/'.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
NLI_ID = 'MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33'
NLI_REVISION = '613e8c52c33e2bc0677ada4ad760f693e5e0f581'
NLI_PATH = Path.home()/'.cache/huggingface/hub/models--MoritzLaurer--deberta-v3-base-zeroshot-v1.1-all-33/snapshots'/NLI_REVISION


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path,value):
    p=Path(path); tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)


def f1(y,p,classes):
    values=[]
    for c in classes:
        tp=np.sum((y==c)&(p==c)); den=np.sum(y==c)+np.sum(p==c)
        values.append(float(2*tp/max(1,den)))
    return float(np.mean(values))


def reference_labels(frame):
    f=frame.copy()
    inferred=f.polarity.where(f.polarity.isin(['positive','negative']))
    inferred.loc[f.subjectivity.eq('objective')]='objective'
    f['label']=f['label'].where(f['label'].isin(['objective','negative','positive']),inferred) if 'label' in f else inferred
    return f


def metrics(frame,pred):
    g=reference_labels(frame).reset_index(drop=True);p=pred.reset_index(drop=True)
    out={'n':len(g),'task_counts':{}}
    for task,classes in {**TASKS,'label':['objective','negative','positive']}.items():
        mask=g[task].isin(classes).to_numpy()
        out[task+'_f1']=f1(g.loc[mask,task].to_numpy(),p.loc[mask,task].to_numpy(),classes)
        out['task_counts'][task]=int(mask.sum())
    mask=g.aspect.isin(ASPECTS)
    out['aspect_per_class_f1']={c:f1(g.loc[mask,'aspect'].to_numpy(),p.loc[mask,'aspect'].to_numpy(),[c]) for c in ASPECTS}
    out['minimum_aspect_f1']=min(out['aspect_per_class_f1'].values())
    mask=g.subjectivity.isin(TASKS['subjectivity']); actual=g.subjectivity.eq('objective')&mask
    chosen=p.subjectivity.eq('objective')&mask;tp=int((actual&chosen).sum())
    out.update(objective_count=int(actual.sum()),objective_correct=tp,
               objective_recall=tp/max(1,int(actual.sum())),objective_precision=tp/max(1,int(chosen.sum())))
    mask=g.aspect.isin(ASPECTS)&g.label.isin(['objective','negative','positive'])
    out['joint_accuracy']=float((g.loc[mask,'aspect'].eq(p.loc[mask,'aspect'])&g.loc[mask,'label'].eq(p.loc[mask,'label'])).mean()) if mask.any() else 0.
    return out


def rank(m):
    return (.3*m['aspect_f1']+.3*m['label_f1']+.15*m['subjectivity_f1']+.15*m['polarity_f1']+.1*m['minimum_aspect_f1']
            -.1*max(0,.7-m['objective_recall'])-.1*max(0,.7-m['objective_precision']))


def subset(sc,indices):
    return {k:(v if k=='aspect_classes' else np.asarray(v)[indices]) for k,v in sc.items()}


def install_gold_guard():
    """Fail closed if the fitting process attempts to open answer files or prior gold-tuned artifacts."""
    forbidden=[str(BASE/'inputs'),str(ROOT/'outputs/c_tuning_20261018'),str(ROOT/'outputs/c_tuning_quality_20261018'),
               str(ROOT/'outputs/ai_adjudicated_gold_20261018'),str(ROOT/'outputs/final_gold'),
               str(ROOT/'outputs/human_gold_review_1000_20261018'),str(ROOT/'data/eval'),
               str(ROOT/'outputs/agreement_selected_subset_20261018'),str(ROOT/'outputs/majority_merge_20261017')]
    denied=[]
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,Path)):
            value=args[0].decode() if isinstance(args[0],bytes) else args[0]
            p=str(Path(value).resolve())
            if any(p==f or p.startswith(f+'/') for f in forbidden):
                denied.append(p); raise PermissionError('Project gold / gold-tuned artifacts are forbidden in this process: '+p)
    sys.addaudithook(guard)
    try:open(BASE/'inputs/gold_1073.csv').close()
    except PermissionError:pass
    else:raise AssertionError('Gold guard self-check failed')
    return denied
