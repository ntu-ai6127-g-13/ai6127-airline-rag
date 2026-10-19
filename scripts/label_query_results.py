"""Paired label-ranking audit on the exact top-10 pool of a completed query benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.classify import classify_texts
from src.query_service import rank_evidence
from src.generate import evidence_answer
from src.evaluate import evaluate_query_modes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-dir',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    progress=json.loads((args.baseline_dir/'progress.json').read_text())
    if progress['status']!='completed_pending_judgments':raise RuntimeError('Baseline queries must finish first')
    documents=[json.loads(path.read_text()) for path in sorted(args.baseline_dir.glob('Q*.json'))]
    unique={row['chunk_id']:row['text'] for doc in documents for row in doc['result']['baseline_evidence']}
    stop=threading.Event();started=time.monotonic()
    state={'status':'running','phase':'batch_classification','queries':len(documents),'unique_sentences':len(unique),
           'candidate_pool':'same baseline top 10; measures reranking, not candidate recall',
           'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    def heartbeat():
        while not stop.is_set():
            current={**state,'elapsed_seconds':time.monotonic()-started}
            tmp=out/'progress.tmp';tmp.write_text(json.dumps(current,indent=2));tmp.replace(out/'progress.json')
            stop.wait(10)
    worker=threading.Thread(target=heartbeat,daemon=True);worker.start()
    try:
        labels=classify_texts(list(unique.values()),args.manifest,timeout_seconds=3600)
        mapping=dict(zip(unique,labels));judgments=[];changed=0
        (out/'predictions.jsonl').write_text(''.join(json.dumps({'chunk_id':cid,**label})+'\n' for cid,label in mapping.items()))
        for doc in documents:
            case=doc['case'];result=doc['result'];baseline=result['baseline_evidence']
            selected=rank_evidence(baseline,[mapping[row['chunk_id']] for row in baseline],
                                  aspect=case['aspect'],polarity=case['polarity'],top_k=10)
            changed+=([r['chunk_id'] for r in baseline[:5]]!=[r['chunk_id'] for r in selected[:5]])
            result.update(evidence=selected,answer=evidence_answer(selected,aspect=case['aspect']),
                          baseline_answer=evidence_answer(baseline,aspect=case['aspect']),classifier_used=True,
                          classifier_is_experimental=True,label_candidate_pool_k=10,
                          interactive_label_latency_measured=False)
            (out/(case['query_id']+'.json')).write_text(json.dumps(doc,ensure_ascii=False,indent=2))
            for mode,items in [('baseline',baseline),('labels',selected)]:
                judgments.append({'query_id':case['query_id']+'_'+mode,'base_query_id':case['query_id'],'mode':mode,
                    'expected_answerable':None,'returned_ids':[r['chunk_id'] for r in items],
                    'relevance':{r['chunk_id']:None for r in items},'answer_successful':None,
                    'total_claims':None,'supported_claims':None,'correct_abstention':None,
                    'judgment_provenance':'pending_human_review','human_reviewed':False,'independent_test':False})
        (out/'judgments_to_review.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in judgments))
        (out/'evaluation_pending.json').write_text(json.dumps(evaluate_query_modes(judgments),ensure_ascii=False,indent=2))
        state.update(status='completed_pending_judgments',phase='finished',queries_with_changed_top5=changed,
                     human_judgments_available=False,passed=False)
        print(json.dumps(state),flush=True)
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:
        stop.set();worker.join(timeout=2)
        (out/'progress.json').write_text(json.dumps({**state,'elapsed_seconds':time.monotonic()-started},indent=2))


if __name__=='__main__':main()
