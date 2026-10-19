"""Selection check on new real dev sentences with explicitly AI-generated silver labels."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

import pandas as pd

from runtime import ROOT,HERE,DEFAULT_OUT,sha,write_json,guard_gold
from run_monitor import Monitor,combine_predictions
from predict_cli import predict_frozen
from src.quality import classification_metrics,proxy_rank


def main(out):
    if (out/'audit.json').exists() or (out/'audit_monitor.json').exists():raise RuntimeError('Silver selection must precede gold audit')
    if (out/'before_silver_frozen_candidates.json').exists():raise RuntimeError('Preserve the previous extension')
    frozen=json.loads((out/'frozen_candidates.json').read_text())
    if not frozen.get('closed_choice_extension_before_gold'):raise RuntimeError('Complete the fixed-choice extension first')
    silver_path=out/'development_silver.jsonl'
    if not silver_path.exists():silver_path.write_text((HERE/'development_silver.jsonl').read_text())
    if not (out/'blind_development_texts.jsonl').exists():
        pd.read_json(silver_path,lines=True)[['id','chunk_id','review_id','text','role']].to_json(
            out/'blind_development_texts.jsonl',orient='records',lines=True,force_ascii=False)
    write_json(out/'before_silver_frozen_candidates.json',frozen)
    monitor=Monitor(out/'silver_control')
    monitor.state.update(schedule=['score all frozen components on new real dev silver','select once before gold'],
        label_provenance='AI-assisted development labels; NOT human gold or independent acceptance',
        selection_policy='0.8 real-development silver monitor + 0.2 original weak/synthetic proxy monitor; calibration reported separately')
    try:
        denied=guard_gold()
        frame=pd.read_json(out/'development_silver.jsonl',lines=True)
        base=ROOT/'outputs/c_gold_smoke_20261018'
        split=pd.read_parquet(base/'reproduced_current_git_seed42/split.parquet')
        train_parents=set(split[split.split.eq('train')].review_id)
        dev_parents=set(split[split.split.eq('dev')].review_id)
        evaluation=pd.read_csv(base/'source/data/eval/to_label_meta.csv',usecols=['review_id'])
        assert set(frame.review_id)<=dev_parents and not set(frame.review_id)&train_parents
        assert not set(frame.review_id)&set(evaluation.review_id)
        assert not set(frame[frame.role.eq('calibrate')].review_id)&set(frame[frame.role.eq('monitor')].review_id)
        previous=pd.read_json(ROOT/'outputs/c_no_gold_optimization_20261018/selection_data.jsonl',lines=True)
        assert not set(frame.review_id)&set(previous.review_id)
        required={value for spec in frozen['candidates'].values() for key,value in spec.items()
                  if key in {'source','aspect_source','sentiment_source','other_source'}}
        components={};rows=[{'id':row.id,'text':row.text} for row in frame.itertuples()]
        sources={str(p):sha(p) for p in [*HERE.glob('*.py'),ROOT/'src/quality.py']}
        for name in sorted(required):
            directory=out/('silver_component_'+name);directory.mkdir(exist_ok=False)
            component_manifest={**frozen,'selected_spec':{'mode':'direct','source':name},'source_hashes':sources}
            write_json(directory/'manifest.json',component_manifest)
            monitor.phase(name+': real development silver inference')
            pred=predict_frozen(directory/'manifest.json',rows,
                lambda done,total:monitor.phase(f'{name}: {done}/{total} development rows'))
            components[name]=pred
            pred.to_json(directory/'predictions.jsonl',orient='records',lines=True,force_ascii=False)
        # Imported lazily by legacy-component inference above.
        rule_func=sys.modules['src.language_rules'].choose_focus
        reports=[]
        for name,spec in frozen['candidates'].items():
            pred=combine_predictions(spec,components,frame.text,rule_func)
            report={'name':name,'spec':spec}
            for role in ['calibrate','monitor']:
                mask=frame.role.eq(role);report[role]=classification_metrics(frame.loc[mask],pred.loc[mask])
            original=json.loads((out/name/'metrics.json').read_text())
            report['selection_score']=.8*proxy_rank(report['monitor'])+.2*original['monitor']['proxy_selection_score']
            write_json(out/name/'silver_metrics.json',report)
            frame.add_prefix('reference_').join(pred.add_prefix('pred_')).to_json(out/name/'silver_predictions.jsonl',orient='records',lines=True,force_ascii=False)
            reports.append(report);monitor.state['rounds'].append(report);monitor.flush()
        best=max(reports,key=lambda r:r['selection_score'])
        frozen.update(selected_without_gold=best['name'],selected_spec=best['spec'],
                      frozen_at=datetime.now(timezone.utc).isoformat(),source_hashes=sources,
                      silver_validation_used=True,silver_is_human_gold=False,
                      selection_policy=monitor.state['selection_policy'])
        frozen['input_hashes'][str(out/'development_silver.jsonl')]=sha(out/'development_silver.jsonl')
        frozen['input_hashes'][str(HERE/'development_silver.jsonl')]=sha(HERE/'development_silver.jsonl')
        frozen['input_hashes'][str(out/'blind_development_texts.jsonl')]=sha(out/'blind_development_texts.jsonl')
        assert len(denied)==1
        write_json(out/'frozen_candidates.json',frozen)
        state=json.loads((out/'monitor.json').read_text())
        state.update(selected_without_gold=best['name'],selection_policy=monitor.state['selection_policy'],
                     silver_selection_results=reports,frozen_sha256=sha(out/'frozen_candidates.json'),status='completed_frozen_pending_audit')
        write_json(out/'monitor.json',state)
        monitor.state.update(selected_without_gold=best['name'],gold_guard_only_self_test=True)
        monitor.close('completed_frozen_pending_audit')
        print(json.dumps({'selected_without_gold':best['name'],'silver_monitor':best['monitor'],'score':best['selection_score']},ensure_ascii=False),flush=True)
    except BaseException as error:
        monitor.state['error']=repr(error);monitor.close('failed_or_stopped');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=DEFAULT_OUT)
    main(parser.parse_args().run_dir.resolve())
