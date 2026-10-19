"""Optional pre-gold extension: consensus pseudo-labels only for abstained weak-training rows."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import warnings
import faiss
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from common import ROOT,BASE,TASKS,sha,write_json,install_gold_guard
from inference import FrozenEmbedding,nli_scores,model_scores,context,decide
from run_monitor import Monitor,Stopped,tune,combined_metric,rule_weights,check_fit_boundary


def main(out):
    if (out/'gold_audit.json').exists() or (out/'audit_monitor.json').exists():
        raise RuntimeError('Gold audit has begun; do not modify frozen candidates in response')
    stage1=json.loads((out/'frozen_candidates.json').read_text())
    mainstate=json.loads((out/'monitor.json').read_text())
    if mainstate['status']!='completed_frozen_pending_gold_audit':raise RuntimeError('Need completed first stage')
    for name,digest in stage1['candidates'].items():assert sha(out/name/'model.joblib')==digest
    mon=Monitor(out/'teacher_control')
    try:
        denied=install_gold_guard()
        mon.state.update(schedule=[{'name':'r26_consensus_C0.1','C':.1},{'name':'r27_consensus_C1','C':1.}],
                         gold_guard={'enabled':True,'self_check_blocked_answer_read':True},
                         pseudo_label_policy={'pool_size':2048,'objective_min_probability_each_view':.5,
                                              'subjective_max_objective_probability_each_view':.1,
                                              'polarity_min_conditional_probability_each_view':.85,
                                              'required_view_agreement':True,'only_previously_abstained_subjectivity':True,
                                              'aspect_labels_changed':False})
        mainstate.update(status='running_teacher_extension',phase='See teacher_control/monitor.json for progress')
        write_json(out/'monitor.json',mainstate)
        mon.phase('Load original weak-training rows; preserve existing non-gold validation')
        weak=pd.read_parquet(BASE/'reproduced_current_git_seed42/weak_labels.parquet').merge(
            pd.read_parquet(BASE/'reproduced_current_git_seed42/split.parquet'),on=['chunk_id','review_id'],validate='one_to_one')
        train=weak.loc[weak.split.eq('train')].reset_index(drop=True);dev=weak.loc[weak.split.eq('dev')]
        check_fit_boundary(train,dev)
        validation=pd.read_json(out/'selection_data.jsonl',lines=True)
        assert not set(train.review_id)&set(validation.review_id)
        unknown=np.flatnonzero(train.subjectivity.isna())
        rng=np.random.RandomState(20261018);pool=np.sort(rng.choice(unknown,min(2048,len(unknown)),replace=False))
        target=train.iloc[pool].copy();views=[]
        for variant in ['descriptive','sentiment']:
            def progress(done,total):
                mon.check()
                if done%128==0 or done==total:mon.phase(f'Teacher {variant}: {done}/{total} unique training texts')
            views.append(nli_scores(target.text.tolist(),variant,progress=progress))
        a,b=views
        objective=(a['objective']>=.5)&(b['objective']>=.5)
        positive_a=a['positive_raw']>=.5;positive_b=b['positive_raw']>=.5
        subjective=(a['objective']<=.1)&(b['objective']<=.1)&(positive_a==positive_b)
        subjective&=(np.maximum(a['positive_raw'],1-a['positive_raw'])>=.85)&(np.maximum(b['positive_raw'],1-b['positive_raw'])>=.85)
        assert not (objective&subjective).any()
        augmented=train.copy();obj_idx=pool[objective];sub_idx=pool[subjective]
        augmented.loc[obj_idx,'subjectivity']='objective';augmented.loc[sub_idx,'subjectivity']='subjective'
        augmented.loc[np.r_[obj_idx,sub_idx],'subjectivity_rule']='teacher_nli_two_view_consensus'
        augmented.loc[sub_idx,'polarity']=np.where(positive_a[subjective],'positive','negative')
        augmented.loc[sub_idx,'polarity_rule']='teacher_nli_two_view_consensus'
        assert augmented.loc[train.subjectivity.notna(),'subjectivity'].equals(train.loc[train.subjectivity.notna(),'subjectivity'])
        assert augmented.aspect.equals(train.aspect)
        target['teacher_subjectivity']=augmented.iloc[pool].subjectivity.to_numpy()
        target['teacher_polarity']=augmented.iloc[pool].polarity.to_numpy()
        for i,view in enumerate(views):
            target[f'view_{i}_objective_score']=view['objective'];target[f'view_{i}_positive_score']=view['positive_raw']
        target['accepted']=objective|subjective
        target.to_json(out/'teacher_pseudo_labels.jsonl',orient='records',lines=True,force_ascii=False)
        mon.state['pseudo_label_counts']={'pool':len(pool),'new_objective':len(obj_idx),'new_subjective':len(sub_idx),
                                         'still_abstained':int((~(objective|subjective)).sum()),
                                         'new_positive':int(positive_a[subjective].sum()),'new_negative':int((~positive_a[subjective]).sum())}
        mon.phase('Consensus counts: '+json.dumps(mon.state['pseudo_label_counts']))
        mon.phase('Recover frozen embeddings; fit heads with original weak labels plus accepted teacher labels')
        ids=pd.Index(np.load(ROOT/'index/chunk_ids.npy',allow_pickle=True));texts=np.load(ROOT/'index/chunk_texts.npy',allow_pickle=True)
        ix=ids.get_indexer(train.chunk_id);assert (ix>=0).all() and np.array_equal(texts[ix],train.text.to_numpy())
        index=faiss.read_index(str(ROOT/'index/faiss.index'));x=index.reconstruct_batch(ix.astype(np.int64))
        del ids,texts,index
        vec=FrozenEmbedding();ctx=context(validation.text.tolist());xv=vec.transform(ctx['text']);xvf=vec.transform(ctx['focus'])
        # The aspect head stays from a previously frozen semantic candidate chosen on non-gold calibration.
        base_options=[r for r in mainstate['rounds'] if r['name'] in ['r19_rule_balanced_C0.1','r20_rule_balanced_C1','r21_balanced_mlp64','r22_balanced_mlp128']]
        base_name=max(base_options,key=lambda r:r['calibrate']['selection_score'])['name']
        original=joblib.load(out/base_name/'model.joblib')
        added=[]
        for spec in mon.state['schedule']:
            name=spec['name'];models=deepcopy(original['models'])
            for task in ['subjectivity','polarity']:
                mon.phase(name+': fit '+task)
                weights=rule_weights(augmented,task);use=weights>0
                model=LogisticRegression(C=spec['C'],max_iter=1000,random_state=42)
                with threadpool_limits(limits=1),warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    model.fit(x[use],augmented.loc[use,task],sample_weight=weights[use])
                models[task]=model
            sc=model_scores(models,xv,xvf);cfg=tune(sc,validation);pred=decide(sc,validation.text.tolist(),cfg)
            fit=validation.role.eq('calibrate');held=~fit
            result={'name':name,'details':{**spec,'aspect_source':base_name,'pseudo_label_counts':mon.state['pseudo_label_counts']},
                    'decision_config':cfg,'calibrate':combined_metric(validation.loc[fit],pred.loc[fit]),
                    'monitor':combined_metric(validation.loc[held],pred.loc[held]),'gold_seen':False,
                    'proxy_metrics_are_not_quality_acceptance':True}
            bundle={'kind':'ml','vectorizer':vec,'models':models,'decision_config':cfg,
                    'project_gold_used_for_fitting':False,'project_gold_used_for_selection':False,'production_ready':False,
                    'fitting_data':'Original weak train, plus two-view teacher labels on 2048 abstained training rows',
                    'inference_entry':str(Path(__file__).with_name('inference.py'))}
            d=out/name;d.mkdir();joblib.dump(bundle,d/'model.joblib',compress=3);result['model_sha256']=sha(d/'model.joblib')
            write_json(d/'metrics.json',result);np.savez_compressed(d/'selection_scores.npz',**sc)
            validation.add_prefix('reference_').join(pred.add_prefix('pred_')).to_json(d/'selection_predictions.jsonl',orient='records',lines=True,force_ascii=False)
            added.append(result);mon.state['rounds'].append(result);mon.flush()
        combined=mainstate['rounds']+added;selected=max(combined,key=lambda r:r['monitor']['selection_score'])
        # Keep the first freeze intact as an audit trail. No gold has been read in either stage.
        write_json(out/'stage1_frozen_candidates.json',stage1)
        final={**stage1,'frozen_at':datetime.now(timezone.utc).isoformat(),'selected_without_gold':selected['name'],
               'model':str(out/selected['name']/'model.joblib'),'selection_metric':selected['monitor']['selection_score'],
               'candidates':{r['name']:r['model_sha256'] for r in combined},
               'scripts':{str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
               'teacher_extension':mon.state['pseudo_label_counts'],'stage1_freeze_file':'stage1_frozen_candidates.json'}
        write_json(out/'frozen_candidates.json',final);write_json(out/'selected_without_gold.json',final)
        mainstate.update(status='completed_frozen_pending_gold_audit',phase='finished',rounds=combined,
                         selected_without_gold=selected['name'],freeze_sha256=sha(out/'frozen_candidates.json'),
                         teacher_extension=mon.state['pseudo_label_counts'],finished_at=datetime.now(timezone.utc).isoformat())
        write_json(out/'monitor.json',mainstate)
        mon.state['gold_guard']['denied_reads']=denied;assert len(denied)==1
        mon.close('completed_frozen_pending_gold_audit')
    except BaseException as exc:
        mon.state['error']=repr(exc);mon.close('failed')
        mainstate.update(status='teacher_extension_failed',error=repr(exc));write_json(out/'monitor.json',mainstate)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,default=ROOT/'outputs/c_no_gold_optimization_20261018')
    main(p.parse_args().run_dir.resolve())
