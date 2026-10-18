"""Prepare a new run from the versioned AI-silver texts, checking evaluation isolation."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.training_isolation import install_answer_read_guard
from src.weak_labels import load_test_exclusions,normalize_text,sha256


def main(out):
    denied=install_answer_read_guard()
    if out.exists():raise ValueError('Use a new run directory')
    silver_path=Path(__file__).with_name('development_silver.jsonl')
    silver=pd.read_json(silver_path,lines=True)
    kb=pd.read_parquet(ROOT/'data/processed/chunks.parquet')
    excluded,texts,_=load_test_exclusions(kb,ROOT/'data/eval/to_label_meta.csv',ROOT/'data/eval/to_label.csv')
    split=pd.read_parquet(ROOT/'outputs/weak_rebuild_20261019/split.parquet')
    train_parents=set(split.loc[split.split.eq('train'),'review_id']);dev_parents=set(split.loc[split.split.eq('dev'),'review_id'])
    old=pd.read_json(ROOT/'outputs/c_no_gold_optimization_20261018/selection_data.jsonl',lines=True)
    previous=pd.read_json(ROOT/'experiments/query_quality/development_silver.jsonl',lines=True)
    forbidden=excluded|train_parents|set(old.review_id)|set(previous.review_id)
    if set(silver.review_id)&forbidden or not set(silver.review_id)<=dev_parents:raise ValueError('Development parent isolation failed')
    if set(silver.text.map(normalize_text))&texts:raise ValueError('Evaluation text overlap')
    if not silver.id.is_unique or not silver.chunk_id.is_unique:raise ValueError('Duplicate IDs')
    original=kb.set_index('chunk_id').loc[silver.chunk_id]
    if original.text.tolist()!=silver.text.tolist() or original.review_id.tolist()!=silver.review_id.tolist():raise ValueError('Texts or parent mapping changed')
    out.mkdir(parents=True)
    path=out/'blind_development_texts.jsonl'
    silver[['id','chunk_id','review_id','text','sampling_stratum','role']].to_json(path,orient='records',lines=True,force_ascii=False)
    manifest={'n':len(silver),'text_sha256':sha256(path),'silver_sha256':sha256(silver_path),
              'train_parent_overlap':0,'original_eval_parent_overlap':0,'old_proxy_and_silver_parent_overlap':0,
              'original_eval_normalized_text_overlap':0,'human_labels_read':False,'gold_guard_denials':denied,
              'selection':'Replay of fixed AI-silver development cohort, not fresh independent evaluation'}
    (out/'development_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    main(p.parse_args().output_dir.resolve())
