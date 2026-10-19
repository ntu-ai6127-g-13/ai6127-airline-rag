"""Infer new sentences with the frozen, development-selected experimental candidate."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.training_isolation import install_answer_read_guard
from experiments.aspect_quality.run_monitor import sha,consensus,MLX,HERE,VARIANTS
from experiments.aspect_quality.prompts import majority_aspects


def main():
    parser=argparse.ArgumentParser()
    for key in ['manifest','input','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();denied=install_answer_read_guard()
    if args.output.exists():raise ValueError('Output already exists')
    frozen=json.loads(args.manifest.read_text());variant=frozen['selected_without_gold']
    if variant not in VARIANTS:raise ValueError('Unknown selected variant')
    for section in ['source_hashes','model_weight_hashes']:
        for path,digest in frozen[section].items():
            if sha(path)!=digest:raise ValueError('Frozen source or weight changed: '+path)
    examples=Path(frozen['examples_path'])
    if sha(examples)!=frozen['input_hashes'][str(examples)]:raise ValueError('Prompt examples changed')
    rows=[json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    if not rows or any(set(r)!={'id','text'} or not isinstance(r['id'],str) or not isinstance(r['text'],str) or not r['text'].strip() for r in rows):
        raise ValueError('Each input needs only a string ID and nonempty text')
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate IDs')
    def run(command):
        p=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        try:
            log,_=p.communicate(timeout=3600)
            if p.returncode:raise RuntimeError(log[-4000:])
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    with tempfile.TemporaryDirectory(prefix='aspect_candidate_') as temporary:
        temp=Path(temporary);input_path=temp/'texts.jsonl'
        input_path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        nli_manifest={'source_hashes':frozen['source_hashes'],'model_weight_hashes':frozen['model_weight_hashes'],
                      'input_hashes':frozen['component_model_hashes'],'selected_spec':{'mode':'direct','source':'r25_fusion_nli0.5'}}
        (temp/'nli_manifest.json').write_text(json.dumps(nli_manifest))
        run([sys.executable,str(HERE/'component_worker.py'),'--manifest',str(temp/'nli_manifest.json'),'--input',str(input_path),'--output',str(temp/'nli.jsonl')])
        nli=pd.read_json(temp/'nli.jsonl',lines=True)
        if nli.id.tolist()!=ids:raise ValueError('NLI output IDs differ')
        def predict(name,aspects=None):
            command=[str(MLX),str(HERE/'worker.py'),'--input',str(input_path),'--output',str(temp/(name+'.jsonl')),
                     '--model',frozen['model_path'],'--examples',str(examples),'--variant',name]
            if aspects is not None:
                pd.DataFrame({'id':ids,'aspect':aspects}).to_json(temp/'aspects.jsonl',orient='records',lines=True)
                command+=['--aspects',str(temp/'aspects.jsonl')]
            run(command);pred=pd.read_json(temp/(name+'.jsonl'),lines=True)
            if pred.id.tolist()!=ids:raise ValueError('Qwen output IDs differ')
            return pred
        if variant=='ensemble':
            components=[predict(v) for v in VARIANTS[:4]]
            raw=predict('ensemble',majority_aspects([p.aspect.tolist() for p in components]))
        else:raw=predict(variant)
        result=consensus(raw,nli)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:f.write(result.to_json(orient='records',lines=True,force_ascii=False))
    assert len(denied)==1
    print(json.dumps({'n':len(rows),'variant':variant,'experimental':True,'gold_guard_self_test_only':True}))


if __name__=='__main__':main()
