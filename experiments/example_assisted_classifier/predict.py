"""Opt-in CLI; leaves the default classifier, app, indexes and weights unchanged."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .core import (ROOT, HERE, MODEL_REVISION, ENCODER_REVISION, checked_corpus,
                   merge_fallback, read_jsonl, retrieve_examples, sha256,
                   source_hashes, validate_bank, validate_inputs, write_json, write_jsonl)
from .guard import install_guard


def run_worker(args, mode, inputs, model_files, sources):
    lock_path = args.out / (mode + '_lock.json')
    write_json(lock_path, {'source_hashes': sources, 'input_sha256': sha256(inputs),
                           'model_files': model_files})
    directory = args.out / mode
    command = [str(args.mlx_python.resolve()), '-m',
        'experiments.example_assisted_classifier.worker', '--mode', mode,
        '--input', str(inputs.resolve()), '--lock', str(lock_path.resolve()),
        '--model', str(args.model.resolve()), '--out', str(directory.resolve())]
    with (args.out / (mode + '.log')).open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            status = process.wait(timeout=args.timeout)
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            raise
    if status:
        raise RuntimeError('MLX worker failed; see ' + str(args.out / (mode + '.log')))
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    output = directory / 'predictions.jsonl'
    if (manifest['status'] != 'completed' or manifest['input_sha256'] != sha256(inputs)
            or manifest['lock_sha256'] != sha256(lock_path)
            or manifest['output_sha256'] != sha256(output)):
        raise ValueError('Worker manifest or predictions changed')
    return read_jsonl(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='JSONL: string id/text only')
    parser.add_argument('--bank', type=Path, required=True, help='Rebuilt bank.jsonl')
    parser.add_argument('--out', type=Path, required=True, help='NEW directory under outputs/')
    parser.add_argument('--kb', type=Path, default=ROOT / 'data/processed/chunks.parquet')
    parser.add_argument('--eval-meta', type=Path, default=ROOT / 'data/eval/to_label_meta.csv')
    parser.add_argument('--eval-text', type=Path, default=ROOT / 'data/eval/to_label.csv')
    parser.add_argument('--mlx-python', type=Path,
                        default=ROOT / 'outputs/query_quality_runtime/venv311/bin/python')
    cache = Path.home() / '.cache/huggingface/hub'
    parser.add_argument('--model', type=Path, default=cache /
        ('models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/' + MODEL_REVISION))
    parser.add_argument('--encoder', type=Path, default=cache /
        ('models--sentence-transformers--all-MiniLM-L6-v2/snapshots/' + ENCODER_REVISION))
    parser.add_argument('--timeout', type=int, default=600, help='Seconds per worker')
    parser.add_argument('--experimental', action='store_true')
    args = parser.parse_args()
    if not args.experimental:
        parser.error('Requires --experimental: independent human acceptance is incomplete')
    if args.timeout <= 0:
        raise ValueError('Timeout must be positive')
    if not args.out.resolve().is_relative_to(ROOT / 'outputs') or args.out.exists():
        raise ValueError('Use a NEW directory under this checkout outputs/; never overwrite')
    denied = install_guard(ROOT, [args.eval_meta, args.eval_text])
    rows = read_jsonl(args.input)
    validate_inputs(rows)
    bank = read_jsonl(args.bank)
    bank_manifest_path = args.bank.parent / 'manifest.json'
    bank_manifest = json.loads(bank_manifest_path.read_text(encoding='utf-8'))
    if (bank_manifest['bank_sha256'] != sha256(args.bank)
            or bank_manifest['spec_sha256'] != sha256(HERE / 'bank_spec.json')
            or bank_manifest['rows'] != len(bank) or bank_manifest['human_reviewed']):
        raise ValueError('Bank manifest changed or unexpected provenance')
    kb, parents, eval_texts, exclusion = checked_corpus(args.kb, args.eval_meta, args.eval_text)
    if (bank_manifest['kb_sha256'] != sha256(args.kb)
            or bank_manifest['eval_meta_sha256'] != sha256(args.eval_meta)
            or bank_manifest['eval_text_sha256'] != sha256(args.eval_text)):
        raise ValueError('Bank was built for a different corpus/eval metadata version')
    validate_bank(bank, kb, parents, eval_texts)
    # Bind annotations to the checked-in spec as well as the generated file hash.
    spec = json.loads((HERE / 'bank_spec.json').read_text(encoding='utf-8'))
    if [{key: row[key] for key in ['id', 'chunk_id', 'review_id', 'aspect', 'label']}
        for row in bank] != [{key: row[key] for key in ['id', 'chunk_id', 'review_id', 'aspect', 'label']}
                             for row in spec['annotations']]:
        raise ValueError('Bank labels/order differ from frozen silver specification')
    from .core import text_sha256
    if any(text_sha256(row['text']) != entry['text_sha256']
           for row, entry in zip(bank, spec['annotations'])):
        raise ValueError('Bank texts differ from frozen silver specification')
    fingerprints = json.loads((HERE / 'model_fingerprints.json').read_text(encoding='utf-8'))
    for directory, files in [(args.model, fingerprints['qwen_files']),
                             (args.encoder, fingerprints['encoder_files'])]:
        for name, digest in files.items():
            if sha256(directory / name) != digest:
                raise ValueError('Wrong fixed model/tokenizer file: ' + str(directory / name))
    if not args.mlx_python.is_file():
        raise FileNotFoundError('Prepare the separate MLX runtime; see experiment README')
    sources = source_hashes()
    started = time.monotonic()
    prepared = retrieve_examples(rows, bank, args.encoder)
    args.out.mkdir(parents=True)
    inputs = args.out / 'example_inputs.jsonl'
    write_jsonl(inputs, prepared)
    # Leave an auditable status record even if a subprocess fails.
    write_json(args.out / 'run_started.json', {'started_at': datetime.now(timezone.utc).isoformat(),
        'source_hashes': sources, 'input_sha256': sha256(args.input),
        'bank_manifest_sha256': sha256(bank_manifest_path), 'human_reviewed': False,
        'production_ready': False})
    raw = run_worker(args, 'examples', inputs, fingerprints['qwen_files'], sources)
    if ([row['id'] for row in raw] != [row['id'] for row in rows]
            or [row['text'] for row in raw] != [row['text'] for row in rows]):
        raise ValueError('Inference IDs/text do not match inputs')
    invalid_inputs = [{key: row[key] for key in ['id', 'text']} for row in raw if not row['valid']]
    fallback = []
    if invalid_inputs:
        fallback_inputs = args.out / 'fallback_inputs.jsonl'
        write_jsonl(fallback_inputs, invalid_inputs)
        fallback = run_worker(args, 'words', fallback_inputs, fingerprints['qwen_files'], sources)
    output = merge_fallback(raw, fallback)
    target = args.out / 'predictions.jsonl'
    write_jsonl(target, output)
    if len(denied) != 1:
        raise RuntimeError('Unexpected protected answer read')
    from src.weak_labels import normalize_text
    bank_texts = {normalize_text(row['text']) for row in bank}
    report = {'status': 'completed', 'rows': len(rows), 'fallback_n': len(fallback),
        'seconds': round(time.monotonic() - started, 2), 'human_reviewed': False,
        'production_ready': False, 'experimental': True, 'gold_answers_read': False,
        'model_weights_updated': False, 'source_hashes': sources,
        'model_fingerprints_sha256': sha256(HERE / 'model_fingerprints.json'),
        'input_sha256': sha256(args.input), 'predictions_sha256': sha256(target),
        'bank_sha256': sha256(args.bank), 'gold_exclusion': exclusion,
        'input_bank_normalized_overlap_n': sum(normalize_text(row['text']) in bank_texts for row in rows),
        'limitations': ['AI silver is not human gold or independent human acceptance.',
            'Literal quotes verify source text, not semantic correctness.',
            'Fallback labels have no verified extractive support.',
            'Input overlap with the bank invalidates independent accuracy claims.',
            'No application integration or full-corpus throughput claim.']}
    write_json(args.out / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
