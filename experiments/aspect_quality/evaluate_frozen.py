"""One final-gold audit after development-only selection; never change the selection."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.gold_release import load_frozen_gold,align_cached_predictions
from src.quality import classification_metrics,CLASSIFIER_TARGETS
from experiments.aspect_quality.run_monitor import sha,write,consensus,MODEL,MLX,HERE,OLD
from experiments.aspect_quality.prompts import majority_aspects


def main(out):
    freeze=out/'frozen_candidates.json';frozen=json.loads(freeze.read_text())
    if (out/'audit.json').exists() or (out/'audit_monitor.json').exists():raise ValueError('Audit already exists; preserve it')
    for section in ['source_hashes','input_hashes','model_weight_hashes','component_model_hashes']:
        for path,digest in frozen[section].items():
            if sha(path)!=digest:raise ValueError('Frozen dependency changed: '+path)
    # Gold is first read in this separate process, after selection was frozen.
    gold,release=load_frozen_gold()
    previous=json.loads((ROOT/'outputs/gold_release_audit_20261019/audit.json').read_text())
    if previous['gold_sha256']!=release['gold_sha256'] or sha(OLD/'frozen_candidates.json')!=previous['frozen_sha256']:
        raise ValueError('Prior evaluation release changed')
    def cached(name):
        entry=previous['models'][name];path=ROOT/entry['prediction_cache']
        if sha(path)!=entry['prediction_cache_sha256']:raise ValueError('Prior prediction cache changed')
        return align_cached_predictions(gold,pd.read_json(path,lines=True))
    raw_baseline=cached('r34_qwen_closed_choice')
    nli=cached('r35_choice_aspect_nli_sentiment')
    baseline=cached('r36_choice_subjective_consensus')
    pd.testing.assert_frame_equal(consensus(raw_baseline,nli),baseline)
    directory=out/'final_evaluation';directory.mkdir(exist_ok=False)
    rows=gold[['sentence_id','text']].rename(columns={'sentence_id':'id'})
    rows.to_json(directory/'texts.jsonl',orient='records',lines=True,force_ascii=False)
    started=time.monotonic();state={'status':'running','selected_without_gold':frozen['selected_without_gold'],
        'frozen_sha256':sha(freeze),'gold_sha256':release['gold_sha256'],'max_seconds':3600,'gold_used_for_training':False}
    def flush():
        state.update(elapsed_seconds=round(time.monotonic()-started,2),heartbeat_at=datetime.now(timezone.utc).isoformat());write(out/'audit_monitor.json',state)
    def predict(variant,aspects=None):
        output=directory/(variant+'.jsonl')
        command=[str(MLX),str(HERE/'worker.py'),'--input',str(directory/'texts.jsonl'),'--output',str(output),
                 '--model',str(MODEL),'--examples',frozen['examples_path'],'--variant',variant]
        if aspects is not None:
            pd.DataFrame({'id':rows.id,'aspect':aspects}).to_json(directory/'ensemble_aspects.jsonl',orient='records',lines=True)
            command+=['--aspects',str(directory/'ensemble_aspects.jsonl')]
        state['phase']=variant;flush()
        with (directory/(variant+'.log')).open('x') as log:
            p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['worker_pid']=p.pid
            try:
                while p.poll() is None:
                    if (out/'STOP_AUDIT').exists() or time.monotonic()-started>state['max_seconds']:raise RuntimeError('Audit stopped or timed out')
                    lines=(directory/(variant+'.log')).read_text().splitlines()
                    state['worker_progress']=next((l for l in reversed(lines) if l.startswith('{')),None)
                    flush();time.sleep(2)
                if p.returncode:raise RuntimeError('Prediction failed: '+variant)
            finally:
                if p.poll() is None:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        frame=pd.read_json(output,lines=True)
        if frame.id.tolist()!=rows.id.tolist():raise ValueError('Evaluation prediction IDs differ')
        return frame.drop(columns='id')
    flush()
    try:
        selected=frozen['selected_without_gold'];computed={}
        if selected=='baseline':raw=raw_baseline
        elif selected=='ensemble':
            computed={v:predict(v) for v in ['hierarchy','fewshot','rotated']}
            aspects=majority_aspects([raw_baseline.aspect.tolist()]+[computed[v].aspect.tolist() for v in ['hierarchy','fewshot','rotated']])
            raw=predict('ensemble',aspects)
        else:raw=predict(selected)
        pred=consensus(raw,nli)
        metrics=classification_metrics(gold,pred);old=classification_metrics(gold,baseline)
        subjective=gold.subjectivity.eq('subjective').to_numpy()
        groups,ids=pd.factorize(gold.review_id);counts=np.bincount(groups,weights=subjective)
        good=subjective & gold.aspect.eq(pred.aspect).to_numpy() & gold.polarity.eq(pred.polarity).to_numpy()
        old_good=subjective & gold.aspect.eq(baseline.aspect).to_numpy() & gold.polarity.eq(baseline.polarity).to_numpy()
        hits=np.bincount(groups,weights=good);old_hits=np.bincount(groups,weights=old_good)
        samples=np.random.RandomState(20261020).randint(0,len(ids),size=(2000,len(ids)))
        deltas=(hits[samples].sum(axis=1)-old_hits[samples].sum(axis=1))/counts[samples].sum(axis=1)
        report={'status':'completed','selected_without_gold':selected,'frozen_sha256':sha(freeze),'gold_sha256':release['gold_sha256'],
            'gold_n':len(gold),'baseline_r36':old,'selected_metrics':metrics,'targets':CLASSIFIER_TARGETS,
            'delta_vs_r36':{k:metrics[k]-old[k] for k in CLASSIFIER_TARGETS},
            'joint_delta_parent_bootstrap_95':np.quantile(deltas,[.025,.975]).tolist(),
            'gold_used_for_training':False,'gold_used_for_selection':False,'independent_test':False,
            'model_promoted':False,'weight_fine_tuning':False,
            'limitations':['Final labels have historical development exposure and post-hoc subset selection.',
                          'Development labels are AI-assisted silver; proxy gains can fail on human gold.',
                          'Only the preselected candidate is assessed; selection is never changed after this audit.']}
        gold.add_prefix('gold_').join(pred.add_prefix('pred_')).to_json(out/'audit_predictions.jsonl',orient='records',lines=True,force_ascii=False)
        write(out/'audit.json',report)
        write(out/'release_gate.json',{'classifier_point_targets_passed':metrics['point_targets_passed'],
              'independent_query_acceptance_passed':False,'production_ready':False,'previous_model_retained':True})
        lines=['# 本轮方面分类优化结果','',f'非 gold 开发阶段预选：`{selected}`。5 个预先规定候选，固定本地 Qwen 权重；未做权重微调。',
               '新开发 silver 96 句，加旧开发 monitor 40 句与弱标签/合成代理 294 句。新 silver 在查看预测前由助手标注，不是人工 gold。','',
               '| 指标 | 目标 | 原 r36 | 本轮预选 | 变化（百分点） |','|---|---:|---:|---:|---:|']
        for k,target in CLASSIFIER_TARGETS.items():lines.append(f'| {k} | {target:.1%} | {old[k]:.1%} | {metrics[k]:.1%} | {(metrics[k]-old[k])*100:+.2f} |')
        lines+=['',f'全部分类点估计过线：{metrics["point_targets_passed"]}。原模型未替换。',
                '这是定稿 gold 上的开发审计；历史曝光不会因标签定稿而消失。没有把 silver 或小批查询体验通过当成独立定量验收。']
        (out/'README.md').write_text('\n'.join(lines)+'\n')
        state.update(status='completed',phase='finished',point_targets_passed=metrics['point_targets_passed'])
        print(json.dumps(report,ensure_ascii=False),flush=True)
    except BaseException as error:
        state.update(status='failed_or_stopped',error=repr(error));raise
    finally:flush()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=ROOT/'outputs/aspect_quality_20261019')
    main(p.parse_args().run_dir.resolve())
