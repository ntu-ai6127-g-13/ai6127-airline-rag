"""Five predeclared inference candidates, selected only on non-gold development data."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.training_isolation import install_answer_read_guard
from src.weak_labels import normalize_text
from src.quality import classification_metrics, CLASSIFIER_TARGETS
from experiments.aspect_quality.prompts import majority_aspects,ASPECTS

OLD=ROOT/'outputs/query_quality_20261018'
MODEL=Path.home()/'.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b'
MLX=ROOT/'outputs/query_quality_runtime/venv311/bin/python'
VARIANTS=['baseline','hierarchy','fewshot','rotated','ensemble']


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)


def consensus(qwen,nli):
    result=qwen.copy();recover=result.subjectivity.eq('objective') & nli.subjectivity.eq('subjective')
    for col in ['label','subjectivity','polarity']:result.loc[recover,col]=nli.loc[recover,col]
    return result


def main(out):
    if (out/'monitor.json').exists():raise ValueError('Use a fresh experiment; prior results are immutable')
    denied=install_answer_read_guard();started=time.monotonic()
    state={'status':'running','phase':'preparing','pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),
           'schedule':VARIANTS,'rounds':[],'max_seconds':5400,'gold_used_for_training':False,
           'gold_used_for_selection':False,'weight_fine_tuning':False,'classifier_targets':CLASSIFIER_TARGETS}
    def flush():
        state.update(elapsed_seconds=round(time.monotonic()-started,2),heartbeat_at=datetime.now(timezone.utc).isoformat());write(out/'monitor.json',state)
    def run(command,name):
        state['phase']=name;flush()
        with (out/(name+'.log')).open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['worker_pid']=process.pid
            try:
                while process.poll() is None:
                    if (out/'STOP').exists() or time.monotonic()-started>state['max_seconds']:
                        raise RuntimeError('STOP requested or deadline reached')
                    lines=(out/(name+'.log')).read_text().splitlines()
                    state['worker_progress']=next((l for l in reversed(lines) if l.startswith('{')),None)
                    flush();time.sleep(2)
                if process.returncode:raise RuntimeError(f'{name} failed; see {name}.log')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    try:
        frozen=json.loads((OLD/'frozen_candidates.json').read_text())
        for section in ['source_hashes','input_hashes','model_weight_hashes']:
            for path,digest in frozen[section].items():
                if sha(path)!=digest:raise ValueError('Prior frozen artifact changed: '+path)
        oldsilver=pd.read_json(ROOT/'experiments/query_quality/development_silver.jsonl',lines=True)
        fresh=pd.read_json(HERE/'development_silver.jsonl',lines=True)
        blind=pd.read_json(out/'blind_development_texts.jsonl',lines=True)
        if not fresh[['id','chunk_id','review_id','text']].equals(blind[['id','chunk_id','review_id','text']]):raise ValueError('Silver does not match blind inputs')
        weak=pd.read_parquet(ROOT/'outputs/weak_rebuild_20261019/weak_labels.parquet')
        split=pd.read_parquet(ROOT/'outputs/weak_rebuild_20261019/split.parquet')
        train=weak.merge(split,on=['chunk_id','review_id']).query("split=='train'")
        if set(fresh.review_id)&set(train.review_id) or set(fresh.text.map(normalize_text))&set(train.text.map(normalize_text)):
            raise ValueError('Fresh development overlaps weak training parents/texts')
        proxy=pd.read_json(OLD/'selection_data.jsonl',lines=True).query("role=='monitor'").reset_index(drop=True)
        previous=oldsilver.query("role=='monitor'").copy()
        calibration=oldsilver.query("role=='calibrate'").copy()
        examples=[]
        for aspect in ASPECTS:
            g=calibration.loc[calibration.aspect.eq(aspect)].copy();g['length']=g.text.str.len()
            g=g.sort_values(['length','id'])
            if g.empty:raise ValueError('No calibration example for '+aspect)
            examples.append(g.iloc[0][['id','text','aspect','review_id']].to_dict())
            if len(g)>1:examples.append(g.iloc[-1][['id','text','aspect','review_id']].to_dict())
        parts=[]
        for name,frame in [('fresh',fresh),('previous',previous),('proxy',proxy)]:
            frame=frame.copy();frame['group']=name;frame['id']=[name+'_'+str(i) for i in range(len(frame))];parts.append(frame)
        reference=pd.concat(parts,ignore_index=True)
        if set(reference.review_id)&{r['review_id'] for r in examples}:raise ValueError('Examples overlap monitor parents')
        if set(reference.text.map(normalize_text))&{normalize_text(r['text']) for r in examples}:raise ValueError('Examples overlap monitor texts')
        reference.to_json(out/'selection_data.jsonl',orient='records',lines=True,force_ascii=False)
        write(out/'examples.json',examples)
        inputs=reference[['id','text']];inputs.to_json(out/'texts.jsonl',orient='records',lines=True,force_ascii=False)
        source_paths=[*HERE.glob('*.py'),ROOT/'src/training_isolation.py',ROOT/'src/quality.py']
        source_hashes={str(p):sha(p) for p in source_paths}
        source_hashes.update(frozen['source_hashes'])
        input_paths=[out/'selection_data.jsonl',out/'texts.jsonl',out/'examples.json',HERE/'development_silver.jsonl',out/'development_manifest.json',
                     ROOT/'docs/选题与标注方案.md',ROOT/'outputs/weak_rebuild_20261019/weak_labels.parquet',ROOT/'outputs/weak_rebuild_20261019/split.parquet']
        input_hashes={str(p):sha(p) for p in input_paths}
        policy={'group_weights':{'fresh':.7,'previous':.2,'proxy':.1},'within_group_weights':{'aspect_f1':.65,'joint_accuracy':.25,'subjective_recall':.1},
                'minimum_improvement_over_baseline':.005,'maximum_pooled_silver_polarity_and_recall_regression':.02,
                'tie_break':'earlier predeclared candidate','selection_is_silver_proxy_not_independent_acceptance':True}
        write(out/'plan.json',{'policy':policy,'schedule':VARIANTS,'source_hashes':source_hashes,'input_hashes':input_hashes,
            'weight_hashes':frozen['model_weight_hashes'],'n':len(reference),'examples':len(examples),'frozen_before_predictions':True})
        manifest={**frozen,'selected_spec':{'mode':'direct','source':'r25_fusion_nli0.5'}}
        write(out/'nli_manifest.json',manifest)
        run([sys.executable,str(HERE/'component_worker.py'),'--manifest',str(out/'nli_manifest.json'),'--input',str(out/'texts.jsonl'),'--output',str(out/'nli.jsonl')],'nli')
        nli=pd.read_json(out/'nli.jsonl',lines=True)
        if nli.id.tolist()!=reference.id.tolist():raise ValueError('NLI output order differs')
        predictions={};reports=[]
        for variant in VARIANTS:
            command=[str(MLX),str(HERE/'worker.py'),'--input',str(out/'texts.jsonl'),'--output',str(out/(variant+'.jsonl')),
                     '--model',str(MODEL),'--examples',str(out/'examples.json'),'--variant',variant]
            if variant=='ensemble':
                aspects=majority_aspects([predictions[v].aspect.tolist() for v in VARIANTS[:4]])
                pd.DataFrame({'id':reference.id,'aspect':aspects}).to_json(out/'ensemble_aspects.jsonl',orient='records',lines=True)
                command+=['--aspects',str(out/'ensemble_aspects.jsonl')]
            run(command,variant)
            raw=pd.read_json(out/(variant+'.jsonl'),lines=True)
            if raw.id.tolist()!=reference.id.tolist():raise ValueError('Qwen output IDs differ')
            pred=consensus(raw,nli);predictions[variant]=pred
            pred.to_json(out/(variant+'_consensus.jsonl'),orient='records',lines=True)
            metrics={group:classification_metrics(reference.loc[reference.group.eq(group)],pred.loc[reference.group.eq(group)]) for group in policy['group_weights']}
            score=sum(weight*(.65*metrics[g]['subjective_aspect_f1']+.25*metrics[g]['subjective_joint_accuracy']+.1*metrics[g]['subjective_recall']) for g,weight in policy['group_weights'].items())
            silver=reference.group.ne('proxy');pooled=classification_metrics(reference.loc[silver],pred.loc[silver])
            report={'variant':variant,'score':score,'groups':metrics,'pooled_silver':pooled};reports.append(report)
            state['rounds'].append(report);flush();write(out/(variant+'_metrics.json'),report)
            print(json.dumps({'variant':variant,'proxy_score':score,'fresh_aspect_f1':metrics['fresh']['subjective_aspect_f1'],'fresh_joint':metrics['fresh']['subjective_joint_accuracy']}),flush=True)
        baseline=reports[0]
        for report in reports:
            report['non_gold_regression_gate_passed']=all(report['pooled_silver'][k]>=baseline['pooled_silver'][k]-.02 for k in ['subjective_polarity_f1','subjective_recall'])
        best=max([r for r in reports if r['non_gold_regression_gate_passed']],key=lambda r:r['score'])
        if best['score']<baseline['score']+.005:best=baseline
        for path,digest in {**source_hashes,**input_hashes}.items():
            if sha(path)!=digest:raise ValueError('Source/input changed during experiment: '+path)
        assert len(denied)==1
        result={'selected_without_gold':best['variant'],'candidates':VARIANTS,'reports':reports,'policy':policy,
                'source_hashes':source_hashes,'input_hashes':input_hashes,'model_weight_hashes':frozen['model_weight_hashes'],
                'component_model_hashes':frozen['input_hashes'],'model_path':str(MODEL),'examples_path':str(out/'examples.json'),
                'gold_used_for_training':False,'gold_used_for_selection':False,'weight_fine_tuning':False,
                'candidate_prediction_hashes':{v:sha(out/(v+'_consensus.jsonl')) for v in VARIANTS},
                'frozen_at':datetime.now(timezone.utc).isoformat(),'production_ready':False}
        write(out/'frozen_candidates.json',result)
        state.update(status='completed_frozen_pending_audit',phase='finished',selected_without_gold=best['variant'],gold_guard_denials=denied)
    except BaseException as error:
        state.update(status='failed_or_stopped',error=repr(error));raise
    finally:flush()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/aspect_quality_20261019')
    main(p.parse_args().output_dir.resolve())
