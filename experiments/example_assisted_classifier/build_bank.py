"""Rebuild the frozen 63-example silver bank from local corpus, not human gold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import (HERE, ROOT, checked_corpus, sha256, text_sha256,
                   validate_bank, write_json, write_jsonl)
from .guard import install_guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kb', type=Path, default=ROOT / 'data/processed/chunks.parquet')
    parser.add_argument('--eval-meta', type=Path, default=ROOT / 'data/eval/to_label_meta.csv')
    parser.add_argument('--eval-text', type=Path, default=ROOT / 'data/eval/to_label.csv')
    parser.add_argument('--out', type=Path, required=True, help='New directory under outputs/')
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / 'outputs') or args.out.exists():
        raise ValueError('Use a NEW directory under this checkout outputs/; never overwrite')
    denied = install_guard(ROOT, [args.eval_meta, args.eval_text])
    spec_path = HERE / 'bank_spec.json'
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    if spec['human_reviewed'] or len(spec['annotations']) != 63:
        raise ValueError('Unexpected bank specification')
    kb, parents, texts, exclusion = checked_corpus(args.kb, args.eval_meta, args.eval_text)
    corpus = kb.set_index('chunk_id')
    bank = []
    for annotation in spec['annotations']:
        source = corpus.loc[annotation['chunk_id']]
        if source.review_id != annotation['review_id'] or text_sha256(source.text) != annotation['text_sha256']:
            raise ValueError('Corpus version differs from frozen bank annotations')
        bank.append({key: annotation[key] for key in ['id', 'chunk_id', 'review_id', 'aspect', 'label']}
                    | {'text': source.text})
    validate_bank(bank, kb, parents, texts)
    if len(denied) != 1:
        raise RuntimeError('Unexpected protected answer read')
    args.out.mkdir(parents=True)
    bank_path = args.out / 'bank.jsonl'
    write_jsonl(bank_path, bank)
    manifest = {'schema_version': 1, 'rows': len(bank), 'human_reviewed': False,
        'production_ready': False, 'bank_sha256': sha256(bank_path),
        'spec_sha256': sha256(spec_path), 'kb_sha256': sha256(args.kb),
        'eval_meta_sha256': sha256(args.eval_meta), 'eval_text_sha256': sha256(args.eval_text),
        'gold_exclusion': exclusion, 'gold_answers_read': False,
        'source': spec['source'], 'limitations': spec['limitations']}
    write_json(args.out / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
