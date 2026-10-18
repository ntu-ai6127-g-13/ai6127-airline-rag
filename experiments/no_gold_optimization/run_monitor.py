"""Bounded, restart-safe, local optimization monitor without project gold in fitting or selection."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import threading
import time
import warnings
import faiss
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from common import ROOT,BASE,MINILM,NLI_ID,NLI_REVISION,NLI_PATH,ASPECTS,TASKS,GOALS,sha,write_json,metrics,rank,f1,subset,install_gold_guard
from inference import FrozenEmbedding,context,model_scores,scores,decide,blend,HYPOTHESES,predict
from challenge import generate
from src.training_guards import check_fit_boundary
from src.weak_labels import normalize_text

DEFAULT=ROOT/'outputs/c_no_gold_optimization_20261018'
SCHEDULE=[{'name':'r19_rule_balanced_C0.1','kind':'lr','C':.1},
          {'name':'r20_rule_balanced_C1','kind':'lr','C':1.},
          {'name':'r21_balanced_mlp64','kind':'mlp','hidden':64},
          {'name':'r22_balanced_mlp128','kind':'mlp','hidden':128},
          {'name':'r23_local_nli_descriptive','kind':'nli','variant':'descriptive'},
          {'name':'r24_local_nli_sentiment','kind':'nli','variant':'sentiment'}]


class Stopped(RuntimeError):pass


class Monitor:
    def __init__(self,out):
        self.out=out;self.path=out/'monitor.json';self.lock=threading.RLock();self.started=time.monotonic();self.finished=threading.Event()
        if self.path.exists():raise RuntimeError('Existing run is preserved; choose a new --output-dir')
        out.mkdir(exist_ok=True,parents=True)
        self.state={'status':'running','pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),
                    'phase':'initializing','rounds':[],'schedule':SCHEDULE,'quality_goals':GOALS,
                    'gold_used_for_fitting':False,'gold_used_for_selection':False,
                    'selection_policy':'Equal-weight weak-validation and synthetic-diagnostic score; never real-world quality evidence.',
                    'weak_validation_is_proxy':True,'synthetic_validation_is_proxy':True,
                    'max_wall_seconds':2700,'max_neural_epochs':20,'neural_patience':4,
                    'stop_policy':'All predeclared method families, then three fusion weights; wall bound or STOP file; gold evaluated after freezing only.'}
        self.flush()
        self.thread=threading.Thread(target=self.heartbeat,daemon=True);self.thread.start()
    def flush(self):
        with self.lock:
            self.state.update(updated_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-self.started)
            write_json(self.path,self.state)
    def heartbeat(self):
        while not self.finished.wait(10):self.flush()
    def check(self):
        if (self.out/'STOP').exists():raise Stopped('Local STOP file found')
        if time.monotonic()-self.started>self.state['max_wall_seconds']:raise Stopped('Wall-time bound reached')
    def phase(self,value):
        self.check()
        with self.lock:self.state['phase']=value;self.flush()
        print(value,flush=True)
    def close(self,status):
        self.finished.set();self.thread.join(timeout=2)
        self.state.update(status=status,phase='finished',finished_at=datetime.now(timezone.utc).isoformat());self.flush()


def combined_metric(frame,pred):
    weak=frame.source.eq('weak_validation_proxy').to_numpy();synthetic=~weak
    w=metrics(frame.loc[weak],pred.loc[weak]);s=metrics(frame.loc[synthetic],pred.loc[synthetic])
    return {'weak_proxy':w,'synthetic_diagnostic':s,'selection_score':.5*rank(w)+.5*rank(s)}


def tune(sc,frame):
    fit=np.flatnonzero(frame.role.eq('calibrate'));g=frame.iloc[fit].reset_index(drop=True);s=subset(sc,fit)
    config={'aspect_focus':0.,'polarity_focus':0.,'objective_threshold':.5,'positive_threshold':.5}
    best=None
    for mix in [0.,.5,1.]:
        c={**config,'aspect_focus':mix};pr=decide(s,g.text.tolist(),c);m=combined_metric(g,pr)
        score=sum(.3*d['aspect_f1']+.1*d['minimum_aspect_f1'] for d in [m['weak_proxy'],m['synthetic_diagnostic']])
        if best is None or score>best[0]+1e-12:best=(score,mix)
    config['aspect_focus']=best[1];best=None
    for focus in [0.,.5,1.]:
        for ot in [.0005,.002,.01,.03,.05,.1,.2,.35,.5,.65,.8]:
            for pt in [.4,.5,.6]:
                c={**config,'polarity_focus':focus,'objective_threshold':ot,'positive_threshold':pt}
                m=combined_metric(g,decide(s,g.text.tolist(),c));score=m['selection_score']
                if best is None or score>best[0]+1e-12:best=(score,c)
    return best[1]


def rule_weights(train,task):
    labels=train[task];rules=train[task+'_rule'].fillna('unknown')
    weights=np.zeros(len(train),dtype=np.float64)
    for c in TASKS[task]:
        use=labels.eq(c).fillna(False).to_numpy();counts=rules[use].value_counts()
        w=1/np.sqrt(rules[use].map(counts).to_numpy(dtype=float))
        weights[use]=w/w.sum()
    valid=weights>0;weights[valid]*=valid.sum()/len(TASKS[task])
    return weights


def main(out):
    monitor=Monitor(out)
    try:
        denied=install_gold_guard()
        monitor.state['gold_guard']={'enabled':True,'self_check_blocked_answer_read':True}
        monitor.phase('Validate fixed weak train, weak dev and original evaluation exclusions')
        wp=BASE/'reproduced_current_git_seed42/weak_labels.parquet';sp=BASE/'reproduced_current_git_seed42/split.parquet'
        original_path=BASE/'reproduced_current_git_seed42/model.joblib'
        weak=pd.read_parquet(wp).merge(pd.read_parquet(sp),on=['chunk_id','review_id'],validate='one_to_one')
        train=weak.loc[weak.split.eq('train')].reset_index(drop=True);dev=weak.loc[weak.split.eq('dev')].reset_index(drop=True)
        check_fit_boundary(train,dev)
        excluded_meta=pd.read_csv(BASE/'source/data/eval/to_label_meta.csv',usecols=['id','chunk_id','review_id'])
        excluded_text=pd.read_csv(BASE/'source/data/eval/to_label.csv',usecols=['id','text'])
        excluded_norm=set(excluded_text.text.map(normalize_text));excluded_parents=set(excluded_meta.review_id)
        for part in [train,dev]:
            assert not set(part.review_id)&excluded_parents
            assert not set(part.text.map(normalize_text))&excluded_norm
        monitor.state['training_counts']={t:train[t].value_counts().to_dict() for t in TASKS}
        monitor.state['input_hashes']={str(p):sha(p) for p in [wp,sp,original_path]}
        # Fixed sample from weak dev. No sampling choices depend on gold answers.
        rng=np.random.RandomState(20261018);indices=set(np.flatnonzero(dev.subjectivity.eq('objective').fillna(False)))
        for task,n in [('aspect',140),('polarity',140)]:
            for c in TASKS[task]:
                pool=np.flatnonzero(dev[task].eq(c).fillna(False));indices.update(rng.choice(pool,min(n,len(pool)),replace=False).tolist())
        proxy=dev.iloc[sorted(indices)].copy().reset_index(drop=True);proxy['source']='weak_validation_proxy'
        cal,held=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=20261018).split(proxy,groups=proxy.review_id))
        proxy['role']='calibrate';proxy.loc[held,'role']='monitor'
        assert not set(proxy.iloc[cal].review_id)&set(proxy.iloc[held].review_id)
        challenge=generate(excluded_norm|set(train.text.map(normalize_text))|set(dev.text.map(normalize_text)),normalize_text)
        validation=pd.concat([proxy,challenge],ignore_index=True)
        proxy.to_json(out/'weak_validation.jsonl',orient='records',lines=True,force_ascii=False)
        challenge.to_json(out/'synthetic_diagnostic.jsonl',orient='records',lines=True,force_ascii=False)
        validation.to_json(out/'selection_data.jsonl',orient='records',lines=True,force_ascii=False)
        monitor.state['selection_data_counts']=validation.groupby(['source','role']).size().to_dict()
        monitor.state['selection_data_counts']={str(k):int(v) for k,v in monitor.state['selection_data_counts'].items()}
        monitor.state['nli_model']={'id':NLI_ID,'revision':NLI_REVISION,'weight_sha256':sha(NLI_PATH/'model.safetensors')}
        write_json(out/'hypotheses.json',HYPOTHESES)
        monitor.phase('Recover frozen MiniLM vectors with exact chunk/text checks')
        index=faiss.read_index(str(ROOT/'index/faiss.index'))
        ids=pd.Index(np.load(ROOT/'index/chunk_ids.npy',allow_pickle=True));texts=np.load(ROOT/'index/chunk_texts.npy',allow_pickle=True)
        assert ids.is_unique
        def lookup(f):
            loc=ids.get_indexer(f.chunk_id);assert (loc>=0).all()
            assert np.array_equal(texts[loc],f.text.to_numpy())
            return index.reconstruct_batch(loc.astype(np.int64))
        x=lookup(train);xp=lookup(proxy);vec=FrozenEmbedding()
        test=rng.choice(len(train),16,replace=False)
        cosine=np.sum(vec.transform(train.iloc[test].text.tolist())*x[test],axis=1)
        assert cosine.min()>.9999
        monitor.state['cached_embedding_min_cosine']=float(cosine.min())
        xv=np.concatenate([xp,vec.transform(challenge.text.tolist())]);ctx=context(validation.text.tolist())
        xvf=xv.copy();changed=np.asarray(ctx['text'])!=np.asarray(ctx['focus'])
        xvf[changed]=vec.transform(np.asarray(ctx['focus'])[changed].tolist())
        del index,ids,texts,xp
        all_scores={};bundles={};results=[]
        def record(name,bundle,sc,details=None):
            monitor.phase(name+': select settings using non-gold calibration data')
            cfg=tune(sc,validation);pr=decide(sc,validation.text.tolist(),cfg)
            fit=validation.role.eq('calibrate');held=~fit
            result={'name':name,'details':details or {},'decision_config':cfg,
                    'calibrate':combined_metric(validation.loc[fit],pr.loc[fit]),
                    'monitor':combined_metric(validation.loc[held],pr.loc[held]),
                    'gold_seen':False,'proxy_metrics_are_not_quality_acceptance':True}
            bundle.update(decision_config=cfg,project_gold_used_for_fitting=False,project_gold_used_for_selection=False,
                          production_ready=False,inference_entry=str(Path(__file__).with_name('inference.py')))
            d=out/name;d.mkdir(exist_ok=False);joblib.dump(bundle,d/'model.joblib',compress=3)
            result['model_sha256']=sha(d/'model.joblib')
            write_json(d/'metrics.json',result);np.savez_compressed(d/'selection_scores.npz',**sc)
            validation.add_prefix('reference_').join(pr.add_prefix('pred_')).to_json(d/'selection_predictions.jsonl',orient='records',lines=True,force_ascii=False)
            all_scores[name]=sc;bundles[name]=bundle;results.append(result)
            monitor.state['rounds'].append(result);monitor.flush()
            print(json.dumps({'round':name,'non_gold_monitor_score':result['monitor']['selection_score'],
                              'weak_f1':{t:result['monitor']['weak_proxy'][t+'_f1'] for t in TASKS},
                              'synthetic_f1':{t:result['monitor']['synthetic_diagnostic'][t+'_f1'] for t in TASKS}}),flush=True)
        original=joblib.load(original_path)
        baseline={'kind':'ml','vectorizer':original['vectorizer'],'models':original['models']}
        monitor.phase('r0_word_weak: original weak-trained baseline, discard all historical gold-tuned decisions')
        record('r0_word_weak',baseline,scores(baseline,validation.text.tolist()))
        for spec in SCHEDULE:
            name=spec['name'];monitor.phase(name+': preparing')
            if spec['kind']=='nli':
                bundle={'kind':'nli','model_path':str(NLI_PATH),'model_id':NLI_ID,'model_revision':NLI_REVISION,'hypothesis_variant':spec['variant']}
                def progress(done,total):
                    monitor.check()
                    if done%64==0 or done==total:monitor.phase(f'{name}: local NLI {done}/{total} unique texts')
                sc=scores(bundle,validation.text.tolist(),progress)
                record(name,bundle,sc,spec);continue
            models={};epoch_history={}
            for task,classes in TASKS.items():
                monitor.phase(name+': fitting '+task+' from weak labels only')
                weights=rule_weights(train,task);use=weights>0
                if spec['kind']=='lr':
                    model=LogisticRegression(C=spec['C'],max_iter=1000,random_state=42)
                    with threadpool_limits(limits=1),warnings.catch_warnings():
                        warnings.simplefilter('error',ConvergenceWarning)
                        model.fit(x[use],train.loc[use,task],sample_weight=weights[use])
                else:
                    model=MLPClassifier(hidden_layer_sizes=(spec['hidden'],),alpha=.01,batch_size=256,
                                        learning_rate_init=.001,max_iter=1,random_state=42,early_stopping=False)
                    selected=[]
                    for c in classes:
                        pool=np.flatnonzero(train[task].eq(c).fillna(False));p=weights[pool]/weights[pool].sum()
                        selected.extend(rng.choice(pool,4000,replace=True,p=p).tolist())
                    selected=np.asarray(selected);xt=x[selected];yt=train.iloc[selected][task].to_numpy(dtype=str)
                    best=None;patience=0;history=[]
                    for epoch in range(1,21):
                        monitor.phase(f'{name}: {task} epoch {epoch}/20')
                        with threadpool_limits(limits=1):model.partial_fit(xt,yt,classes=np.asarray(classes))
                        pred=model.predict(xv);scores_by_source=[]
                        for source in ['weak_validation_proxy','synthetic_taxonomy_diagnostic_not_gold']:
                            mask=validation.role.eq('calibrate')&validation.source.eq(source)&validation[task].isin(classes)
                            scores_by_source.append(f1(validation.loc[mask,task].to_numpy(),pred[mask],classes))
                        criterion=float(np.mean(scores_by_source));history.append({'epoch':epoch,'non_gold_calibration_f1':criterion})
                        if best is None or criterion>best[0]+.002:
                            best=(criterion,deepcopy(model),epoch);patience=0
                        else:patience+=1
                        if patience>=4:break
                    model=best[1];epoch_history[task]={'selected_epoch':best[2],'history':history}
                models[task]=model
            bundle={'kind':'ml','vectorizer':vec,'models':models,'training_spec':spec,'fitting_data':'original weak train only; class/rule balancing'}
            record(name,bundle,model_scores(models,xv,xvf),{**spec,'neural_epoch_selection':epoch_history})
        # Choose source families on non-gold calibration only, then compare fixed fusion weights on monitor.
        weak_options=[r for r in results if bundles[r['name']]['kind']=='ml']
        nli_options=[r for r in results if bundles[r['name']]['kind']=='nli']
        weak_best=max(weak_options,key=lambda r:r['calibrate']['selection_score'])['name']
        nli_best=max(nli_options,key=lambda r:r['calibrate']['selection_score'])['name']
        for weight in [.25,.5,.75]:
            name=f'r25_fusion_nli{weight:g}';bundle={'kind':'fusion','weak':bundles[weak_best],'nli':bundles[nli_best],'nli_weight':weight}
            record(name,bundle,blend(all_scores[weak_best],all_scores[nli_best],weight),{'weak_source':weak_best,'nli_source':nli_best,'nli_weight':weight})
        champion=max(results,key=lambda r:r['monitor']['selection_score'])
        # Freeze every candidate and selected pointer BEFORE any gold evaluation process can start.
        frozen={'frozen_at':datetime.now(timezone.utc).isoformat(),'selected_without_gold':champion['name'],
                'model':str(out/champion['name']/'model.joblib'),'selection_metric':champion['monitor']['selection_score'],
                'candidates':{r['name']:r['model_sha256'] for r in results},'gold_used_for_selection':False,
                'scripts':{str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
                'input_hashes':monitor.state['input_hashes'],'quality_goals':GOALS}
        write_json(out/'frozen_candidates.json',frozen)
        write_json(out/'selected_without_gold.json',frozen)
        monitor.state['selected_without_gold']=champion['name'];monitor.state['freeze_sha256']=sha(out/'frozen_candidates.json')
        monitor.state['gold_guard']['denied_reads']=denied
        monitor.state['gold_guard']['only_attempt_was_self_check']=len(denied)==1
        assert len(denied)==1
        assert all(sha(p)==digest for p,digest in monitor.state['input_hashes'].items())
        monitor.close('completed_frozen_pending_gold_audit')
    except Stopped as exc:
        monitor.state['stop_reason']=str(exc);monitor.close('stopped_before_complete');raise
    except BaseException as exc:
        monitor.state['error']=repr(exc);monitor.close('failed');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=DEFAULT)
    args=parser.parse_args();main(args.output_dir.resolve())
