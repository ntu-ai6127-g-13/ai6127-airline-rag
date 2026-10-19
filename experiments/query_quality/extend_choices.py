"""Additional fixed-choice experiments; refuse to run once the gold audit starts."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

import pandas as pd

from runtime import ROOT,HERE,PRIOR,DEFAULT_OUT,sha,write_json,guard_gold,llm_predict
from run_monitor import Monitor,combined,combine_predictions


def main(out):
    if (out/'audit.json').exists() or (out/'audit_monitor.json').exists():
        raise RuntimeError('No extensions after gold audit has begun')
    frozen=json.loads((out/'frozen_candidates.json').read_text())
    if (out/'stage1_frozen_candidates.json').exists():raise RuntimeError('Extension already started')
    write_json(out/'stage1_frozen_candidates.json',frozen)
    monitor=Monitor(out/'choice_control')
    monitor.state['schedule']=['closed_choice','closed_choice_aspect_nli_sentiment','closed_choice_subjective_consensus']
    try:
        denied=guard_gold()
        frame=pd.read_json(out/'selection_data.jsonl',lines=True)
        components={}
        for variant in ['taxonomy','evidence_first']:
            components[variant]=pd.read_json(out/('r29_qwen_'+variant)/'selection_predictions.jsonl',lines=True)
        nli=frozen['component_sources']['nli']
        old=pd.read_json(PRIOR/nli/'selection_predictions.jsonl',lines=True).set_index('reference_chunk_id')
        fields=['aspect','subjectivity','polarity','label']
        p=old.loc[frame.chunk_id,['pred_'+field for field in fields]].reset_index(drop=True);p.columns=fields
        components[nli]=p
        rows=[{'id':str(i),'text':text} for i,text in enumerate(frame.text)]
        components['closed_choice']=pd.DataFrame(llm_predict(rows,'closed_choice',out/'closed_choice_inference',
            lambda done,total:monitor.phase(f'closed choice: {done}/{total} equivalent sentences')))
        specs={
            'r34_qwen_closed_choice':{'mode':'direct','source':'closed_choice','llm_variant':'closed_choice'},
            'r35_choice_aspect_nli_sentiment':{'mode':'combine','aspect_source':'closed_choice','sentiment_source':nli},
            'r36_choice_subjective_consensus':{'mode':'subjective_consensus','aspect_source':'closed_choice','sentiment_source':'closed_choice','other_source':nli},
        }
        results=[]
        for name in frozen['candidates']:
            results.append(json.loads((out/name/'metrics.json').read_text()))
        for name,spec in specs.items():
            pred=combine_predictions(spec,components,frame.text)
            report={'name':name,'spec':spec,'proxy_only':True}
            for role in ['calibrate','monitor']:
                mask=frame.role.eq(role);report[role]=combined(frame.loc[mask],pred.loc[mask])
            directory=out/name;directory.mkdir(exist_ok=False)
            write_json(directory/'candidate.json',spec);write_json(directory/'metrics.json',report)
            pred.to_json(directory/'selection_predictions.jsonl',orient='records',lines=True,force_ascii=False)
            results.append(report);monitor.state['rounds'].append(report);monitor.flush()
            print(json.dumps({'candidate':name,'proxy_score':report['monitor']['proxy_selection_score']}),flush=True)
        best=max(results,key=lambda r:r['monitor']['proxy_selection_score'])
        frozen['candidates'].update(specs)
        frozen.update(selected_without_gold=best['name'],selected_spec=best['spec'],
            frozen_at=datetime.now(timezone.utc).isoformat(),
            source_hashes={str(p):sha(p) for p in [*HERE.glob('*.py'),ROOT/'src/quality.py']},
            closed_choice_extension_before_gold=True)
        assert len(denied)==1
        write_json(out/'frozen_candidates.json',frozen)
        mainstate=json.loads((out/'monitor.json').read_text())
        mainstate.update(selected_without_gold=best['name'],rounds=results,
                         frozen_sha256=sha(out/'frozen_candidates.json'),status='completed_frozen_pending_audit')
        write_json(out/'monitor.json',mainstate)
        monitor.state.update(selected_without_gold=best['name'],gold_guard_only_self_test=True)
        monitor.close('completed_frozen_pending_audit')
    except BaseException as error:
        monitor.state['error']=repr(error);monitor.close('failed_or_stopped');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=DEFAULT_OUT)
    main(parser.parse_args().run_dir.resolve())
