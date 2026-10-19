"""Freeze new development texts before inspecting any candidate predictions."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.training_isolation import install_answer_read_guard
from src.weak_labels import normalize_text, sha256, load_test_exclusions


def main():
    denied = install_answer_read_guard()
    out = ROOT / 'outputs/aspect_quality_20261019'
    out.mkdir(exist_ok=False)
    chunks = pd.read_parquet(ROOT/'data/processed/chunks.parquet')
    excluded, texts, _ = load_test_exclusions(chunks, ROOT/'data/eval/to_label_meta.csv', ROOT/'data/eval/to_label.csv')
    split = pd.read_parquet(ROOT/'outputs/weak_rebuild_20261019/split.parquet')
    train = set(split.loc[split.split.eq('train'),'review_id'])
    dev = set(split.loc[split.split.eq('dev'),'review_id'])
    old = pd.read_json(ROOT/'outputs/c_no_gold_optimization_20261018/selection_data.jsonl',lines=True)
    silver = pd.read_json(ROOT/'experiments/query_quality/development_silver.jsonl',lines=True)
    forbidden = train | excluded | set(old.review_id) | set(silver.review_id)
    weak = pd.read_parquet(ROOT/'outputs/weak_rebuild_20261019/weak_labels.parquet').set_index('chunk_id')
    eligible = chunks.loc[chunks.review_id.isin(dev) & ~chunks.review_id.isin(forbidden) & ~chunks.is_eval].copy()
    eligible = eligible.loc[~eligible[['is_route_only','is_non_english','is_fragment']].any(axis=1)]
    eligible = eligible.loc[~eligible.text.map(normalize_text).isin(texts)]
    eligible = eligible.loc[eligible.text.str.split().str.len().between(5,100)].sort_values('chunk_id')
    eligible['sampling_stratum'] = eligible.chunk_id.map(weak.aspect).fillna('uncovered')
    chosen=[]; parents=set(); norms=set()
    for stratum in sorted(eligible.sampling_stratum.unique()):
        need=32 if stratum=='uncovered' else 8
        for row in eligible.loc[eligible.sampling_stratum.eq(stratum)].sample(frac=1,random_state=20261019).itertuples():
            norm=normalize_text(row.text)
            if row.review_id in parents or norm in norms: continue
            chosen.append({'id':f'N{len(chosen):03d}','chunk_id':row.chunk_id,'review_id':row.review_id,
                           'text':row.text,'sampling_stratum':stratum,'role':'monitor'})
            parents.add(row.review_id);norms.add(norm);need-=1
            if not need:break
        if need:raise ValueError('Insufficient development pool: '+stratum)
    assert len(chosen)==96 and not parents & forbidden
    path=out/'blind_development_texts.jsonl'
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in chosen))
    manifest={'n':96,'seed':20261019,'text_sha256':sha256(path),'train_parent_overlap':0,
              'original_eval_parent_overlap':0,'old_proxy_and_silver_parent_overlap':0,
              'original_eval_normalized_text_overlap':0,'human_labels_read':False,
              'selection':'8 per weak aspect plus 32 uncovered; unique development parents; not random population sampling',
              'gold_guard_denials':denied}
    (out/'development_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    for row in chosen: print(json.dumps({'id':row['id'],'text':row['text']},ensure_ascii=False))


if __name__=='__main__':main()
