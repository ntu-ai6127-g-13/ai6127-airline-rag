"""Build 120 fixed query cases; export paired rankings and blank judgments for review."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.query_service import query_reviews, rank_evidence
from src.classify import classify_texts
from src.generate import evidence_answer
from src.evaluate import evaluate_query_modes


def make_cases():
    airlines=["Singapore Airlines","Emirates","Qatar Airways","British Airways","United Airlines"]
    aspects=["seat","staff","food","entertainment","ground","value"]
    cases=[]
    def add(kind,aspect,filters,polarity=None,query="",comparison_group=None):
        cases.append({"query_id":f"Q{len(cases)+1:03d}","type":kind,"query":query,"aspect":aspect,
                      "polarity":polarity,"filters":filters,"comparison_group":comparison_group,
                      "expected_answerable":None,"split":"query_evaluation_not_classifier_training"})
    for airline in airlines:
        for aspect in aspects:
            add("single_aspect",aspect,{"airline":airline})
            for polarity in ["positive","negative"]: add("polar_opinion",aspect,{"airline":airline},polarity)
    for airline in airlines:
        for aspect in ["seat","food"]: add("cabin_filter",aspect,{"airline":airline,"cabin":"Economy Class"})
    for i,aspect in enumerate(["seat","staff","food","ground","value"]):
        for airline in airlines[:2]: add("comparison_component",aspect,{"airline":airline},comparison_group=f"compare_{i}")
    # Temporal and nonexistent-metadata queries exercise explicit evidence insufficiency.
    for aspect in aspects: add("outside_corpus_date",aspect,{"date":["2025-01-01","2026-12-31"]})
    for aspect in ["seat","staff","food","ground"]: add("absent_airline",aspect,{"airline":"Nonexistent Evaluation Airline"})
    assert len(cases)==120
    return cases


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--limit",type=int,default=12)
    parser.add_argument("--classifier-manifest",type=Path)
    args=parser.parse_args();out=args.output_dir
    out.mkdir(parents=True,exist_ok=False)
    cases=make_cases()
    case_file=out/"queries.jsonl"
    case_file.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in cases))
    digest=hashlib.sha256(case_file.read_bytes()).hexdigest()
    manifest={"query_count":120,"queries_sha256":digest,"frozen_before_retrieval":True,
              "created_at":datetime.now(timezone.utc).isoformat(),"human_judgments_available":False,
              "query_relevance_labels_used_for_tuning":False}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    if not args.run: return
    if not 1<=args.limit<=120: raise ValueError("limit must be between 1 and 120")
    # Spread the pilot across aspect, polarity, cabin, comparisons and no-answer cases.
    import numpy as np
    indices=np.linspace(0,len(cases)-1,args.limit,dtype=int)
    judgments=[]; completed=[]
    for index in indices:
        case=cases[index];started=time.monotonic()
        result=query_reviews(case["query"],aspect=case["aspect"],polarity=case["polarity"],filters=case["filters"],
                             include_candidates=bool(args.classifier_manifest))
        (out/(case["query_id"]+".json")).write_text(json.dumps({"case":case,"result":result},ensure_ascii=False,indent=2))
        completed.append((case,result))
        (out/"progress.json").write_text(json.dumps({"status":"running","phase":"retrieval","completed":len(completed),
            "total":args.limit,"last_query":case["query_id"],"last_seconds":time.monotonic()-started},indent=2))
        print(case["query_id"],round(time.monotonic()-started,1),result["answer"]["status"],flush=True)
    if args.classifier_manifest:
        # Label shared candidates once across queries. Benchmark batch time is not interactive latency.
        unique={item["chunk_id"]:item["text"] for _,result in completed for item in result["candidate_evidence"]}
        print('Batch labeling',len(unique),'unique candidate sentences',flush=True)
        (out/"progress.json").write_text(json.dumps({"status":"running","phase":"batch_classification",
            "unique_sentences":len(unique),"retrieved_queries":len(completed)},indent=2))
        started=time.monotonic()
        labels=classify_texts(list(unique.values()),args.classifier_manifest,timeout_seconds=3600)
        mapped=dict(zip(unique,labels));elapsed=time.monotonic()-started
        (out/"batch_predictions.jsonl").write_text("".join(json.dumps({"chunk_id":cid,**prediction})+"\n" for cid,prediction in mapped.items()))
        for case,result in completed:
            items=result["candidate_evidence"]
            selected=rank_evidence(items,[mapped[item["chunk_id"]] for item in items],
                                  aspect=case["aspect"],polarity=case["polarity"],top_k=10)
            result.update(evidence=selected,answer=evidence_answer(selected,aspect=case["aspect"]),
                          classifier_used=True,classifier_is_experimental=True,
                          label_batch_seconds=elapsed,interactive_label_latency_measured=False)
            (out/(case["query_id"]+".json")).write_text(json.dumps({"case":case,"result":result},ensure_ascii=False,indent=2))
    for case,result in completed:
        for mode,key in [("baseline","baseline_evidence"),("labels","evidence")]:
            if mode=="labels" and not args.classifier_manifest: continue
            items=result[key]
            judgments.append({"query_id":case["query_id"]+"_"+mode,"base_query_id":case["query_id"],"mode":mode,
                "expected_answerable":None,"returned_ids":[r["chunk_id"] for r in items],
                "relevance":{r["chunk_id"]:None for r in items},"answer_successful":None,
                "total_claims":None,"supported_claims":None,"correct_abstention":None,
                "judgment_provenance":"pending_human_review"})
        (out/"judgments_to_review.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in judgments))
    report=evaluate_query_modes(judgments)
    (out/"evaluation_pending.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/"progress.json").write_text(json.dumps({"status":"completed_pending_judgments","executed_queries":args.limit,
        "paired_modes":bool(args.classifier_manifest),"passed":False},indent=2))


if __name__=="__main__":main()
