"""Pick aspect and sentiment specialists on calibration silver, then freeze before gold."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

import pandas as pd

from runtime import ROOT,HERE,PRIOR,DEFAULT_OUT,sha,write_json,guard_gold
from run_monitor import Monitor,combined,combine_predictions
from predict_cli import predict_frozen
from src.quality import classification_metrics,proxy_rank


def main(out):
    if (out/'audit_monitor.json').exists() or (out/'audit.json').exists():raise RuntimeError('Cannot extend after gold')
    frozen=json.loads((out/'frozen_candidates.json').read_text())
    if not frozen.get('silver_validation_used'):raise RuntimeError('Complete real-development silver validation first')
    if (out/'before_specialists_frozen_candidates.json').exists():raise RuntimeError('Specialist extension already started')
    write_json(out/'before_specialists_frozen_candidates.json',frozen)
    monitor=Monitor(out/'specialist_control')
    monitor.state['schedule']=['aspect selected on silver calibration','three NLI sentiment components on silver calibration','two fixed specialist combinations']
    try:
        denied=guard_gold();silver=pd.read_json(out/'development_silver.jsonl',lines=True)
        proxy=pd.read_json(out/'selection_data.jsonl',lines=True);components={};proxy_components={}
        for name in ['taxonomy','evidence_first','closed_choice',frozen['component_sources']['weak']]:
            components[name]=pd.read_json(out/('silver_component_'+name)/'predictions.jsonl',lines=True)
            if name=='closed_choice':proxy_components[name]=pd.read_json(out/'closed_choice_inference/predictions.jsonl',lines=True)
            elif name in ['taxonomy','evidence_first']:
                proxy_components[name]=pd.read_json(out/('r29_qwen_'+name)/'selection_predictions.jsonl',lines=True)
            else:
                raw=pd.read_json(PRIOR/name/'selection_predictions.jsonl',lines=True).set_index('reference_chunk_id')
                columns=['aspect','subjectivity','polarity','label']
                pred=raw.loc[proxy.chunk_id,['pred_'+c for c in columns]].reset_index(drop=True);pred.columns=columns
                proxy_components[name]=pred
        calibration=silver.role.eq('calibrate')
        aspect_source=max(components,key=lambda name: classification_metrics(silver.loc[calibration],components[name].loc[calibration])['subjective_aspect_f1'])
        sources={str(p):sha(p) for p in [*HERE.glob('*.py'),ROOT/'src/quality.py']}
        rows=[{'id':row.id,'text':row.text} for row in silver.itertuples()]
        sentiment_options=['r25_fusion_nli0.25','r25_fusion_nli0.5','r25_fusion_nli0.75']
        for name in sentiment_options:
            existing=out/('silver_component_'+name)/'predictions.jsonl'
            if existing.exists():components[name]=pd.read_json(existing,lines=True)
            else:
                directory=out/('silver_specialist_'+name);directory.mkdir(exist_ok=False)
                manifest={**frozen,'selected_spec':{'mode':'direct','source':name},'source_hashes':sources}
                write_json(directory/'manifest.json',manifest);monitor.phase(name+': real development sentiment scoring')
                components[name]=predict_frozen(directory/'manifest.json',rows)
                components[name].to_json(directory/'predictions.jsonl',orient='records',lines=True,force_ascii=False)
            raw=pd.read_json(PRIOR/name/'selection_predictions.jsonl',lines=True).set_index('reference_chunk_id')
            columns=['aspect','subjectivity','polarity','label']
            pred=raw.loc[proxy.chunk_id,['pred_'+c for c in columns]].reset_index(drop=True);pred.columns=columns
            proxy_components[name]=pred
        def sentiment_score(name):
            m=classification_metrics(silver.loc[calibration],components[name].loc[calibration])
            return m['subjective_polarity_f1']-2*max(0,.97-m['subjective_recall'])
        sentiment_source=max(sentiment_options,key=sentiment_score)
        specs={
            'r37_calibrated_task_specialists':{'mode':'combine','aspect_source':aspect_source,'sentiment_source':sentiment_source},
            'r38_specialists_subjective_consensus':{'mode':'subjective_consensus','aspect_source':aspect_source,'sentiment_source':sentiment_source,'other_source':'closed_choice'},
        }
        reports=[json.loads((out/name/'silver_metrics.json').read_text()) for name in frozen['candidates']]
        for name,spec in specs.items():
            pred=combine_predictions(spec,components,silver.text)
            proxy_pred=combine_predictions(spec,proxy_components,proxy.text)
            report={'name':name,'spec':spec};old_report={'name':name,'spec':spec,'proxy_only':True}
            for role in ['calibrate','monitor']:
                mask=silver.role.eq(role);report[role]=classification_metrics(silver.loc[mask],pred.loc[mask])
                mask=proxy.role.eq(role);old_report[role]=combined(proxy.loc[mask],proxy_pred.loc[mask])
            report['selection_score']=.8*proxy_rank(report['monitor'])+.2*old_report['monitor']['proxy_selection_score']
            directory=out/name;directory.mkdir(exist_ok=False)
            write_json(directory/'candidate.json',spec);write_json(directory/'metrics.json',old_report);write_json(directory/'silver_metrics.json',report)
            proxy_pred.to_json(directory/'selection_predictions.jsonl',orient='records',lines=True)
            silver.add_prefix('reference_').join(pred.add_prefix('pred_')).to_json(directory/'silver_predictions.jsonl',orient='records',lines=True)
            reports.append(report);monitor.state['rounds'].append(report);monitor.flush()
        frozen['candidates'].update(specs)
        frozen['component_sources']['specialist_sentiment']=sentiment_source
        best=max(reports,key=lambda r:r['selection_score'])
        frozen.update(selected_without_gold=best['name'],selected_spec=best['spec'],source_hashes=sources,
            frozen_at=datetime.now(timezone.utc).isoformat(),specialist_selection_before_gold=True,
            specialist_components={'aspect':aspect_source,'sentiment':sentiment_source})
        write_json(out/'frozen_candidates.json',frozen)
        state=json.loads((out/'monitor.json').read_text());state.update(selected_without_gold=best['name'],
            frozen_sha256=sha(out/'frozen_candidates.json'),silver_selection_results=reports,
            status='completed_frozen_pending_audit',candidate_count=len(reports))
        write_json(out/'monitor.json',state)
        assert len(denied)==1
        monitor.state.update(selected_without_gold=best['name'],gold_guard_only_self_test=True)
        monitor.close('completed_frozen_pending_audit')
        print(json.dumps({'selected':best['name'],'specialist_sources':frozen['specialist_components'],'selection_score':best['selection_score']},ensure_ascii=False),flush=True)
    except BaseException as error:
        monitor.state['error']=repr(error);monitor.close('failed_or_stopped');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=DEFAULT_OUT)
    main(parser.parse_args().run_dir.resolve())
