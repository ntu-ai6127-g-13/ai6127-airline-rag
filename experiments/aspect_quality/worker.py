"""Isolated local inference; all variants use identical sentiment policy and weights."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.training_isolation import install_answer_read_guard
from experiments.aspect_quality.prompts import system_prompt
sys.path.insert(0,str(ROOT/'experiments/query_quality'))
from mlx_choice_worker import ASPECT_SYSTEM,ASPECT_CHOICES,SENTIMENT_SYSTEM


def main():
    parser=argparse.ArgumentParser()
    for arg in ['input','output','model','examples']:parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--variant',choices=['baseline','hierarchy','fewshot','rotated','ensemble'],required=True)
    parser.add_argument('--aspects',type=Path)
    args=parser.parse_args();denied=install_answer_read_guard()
    if args.output.exists():raise ValueError('Preserve prior predictions')
    rows=[json.loads(l) for l in args.input.read_text().splitlines()]
    if not rows or any(set(r)!={'id','text'} for r in rows):raise ValueError('Only IDs and texts are accepted')
    examples=json.loads(args.examples.read_text())
    os.environ['HF_HUB_OFFLINE']='1'
    import mlx.core as mx
    from mlx_lm import load,batch_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.cache import make_prompt_cache
    model,tokenizer=load(str(args.model));started=time.monotonic()
    def choose(system,users,letters,task):
        prompts=[tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':u}],add_generation_prompt=True) for u in users]
        prefix=min(map(len,prompts))-1
        for prompt in prompts[1:]:
            for j in range(prefix):
                if prompt[j]!=prompts[0][j]:prefix=j;break
        cache=make_prompt_cache(model)
        model(mx.array([prompts[0][:prefix]]),cache=cache);mx.eval([c.state for c in cache])
        option_ids=[tokenizer.encode(l,add_special_tokens=False) for l in letters]
        if not all(len(x)==1 for x in option_ids):raise ValueError('Options must be one token')
        allowed=mx.array([x[0] for x in option_ids])
        def restrict(tokens,logits):
            return mx.where(mx.any(mx.arange(logits.shape[-1])[:,None]==allowed[None,:],axis=1),logits,-float('inf'))
        answers=[]
        for offset in range(0,len(prompts),16):
            batch=prompts[offset:offset+16]
            result=batch_generate(model,tokenizer,[p[prefix:] for p in batch],
                prompt_caches=[copy.deepcopy(cache) for _ in batch],max_tokens=1,
                sampler=make_sampler(temp=0.),logits_processors=[restrict],prefill_batch_size=8,completion_batch_size=16,verbose=False)
            values=[t.strip() for t in result.texts]
            if not all(t in letters for t in values):raise ValueError('Invalid constrained answer')
            answers.extend(values);mx.clear_cache()
            print(json.dumps({'task':task,'completed':len(answers),'total':len(rows),'seconds':round(time.monotonic()-started,1)}),flush=True)
        return answers
    if args.variant=='ensemble':
        given=[json.loads(l) for l in args.aspects.read_text().splitlines()]
        if [r['id'] for r in given]!=[r['id'] for r in rows]:raise ValueError('Ensemble IDs differ')
        aspects=[r['aspect'] for r in given]
        if any(a not in ASPECT_CHOICES for a in aspects):raise ValueError('Unknown ensemble aspect')
    else:
        system,order=(ASPECT_SYSTEM,ASPECT_CHOICES) if args.variant=='baseline' else system_prompt(args.variant,examples)
        answers=choose(system,['Review: '+json.dumps(r['text'])+'\nMain evaluated aspect option:' for r in rows],list('ABCDEFGH'),'aspect')
        aspects=[order[ord(a)-65] for a in answers]
    letters=choose(SENTIMENT_SYSTEM,['Review: '+json.dumps(r['text'])+'\nSpecified aspect: '+a+'\nSentiment option:' for r,a in zip(rows,aspects)],list('ABC'),'sentiment')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        for row,aspect,letter in zip(rows,aspects,letters):
            label={'A':'positive','B':'negative','C':'objective'}[letter]
            f.write(json.dumps({'id':row['id'],'aspect':aspect,'label':label,'subjectivity':'objective' if label=='objective' else 'subjective','polarity':'NA' if label=='objective' else label})+'\n')
    assert len(denied)==1
    print(json.dumps({'status':'completed','guard_self_test_only':True}),flush=True)


if __name__=='__main__':main()
