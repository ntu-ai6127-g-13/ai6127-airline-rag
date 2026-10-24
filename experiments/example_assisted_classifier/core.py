"""Portable data validation, frozen example retrieval and prediction assembly."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .evidence import ASPECTS, LABELS

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL_REVISION = '50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b'
ENCODER_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
WEIGHT_SHA256 = '2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f'


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def validate_inputs(rows):
    if not rows or any(not isinstance(row, dict) or set(row) != {'id', 'text'}
            or not isinstance(row['id'], str) or not row['id'].strip()
            or not isinstance(row['text'], str) or not row['text'].strip() for row in rows):
        raise ValueError('Expected nonempty JSONL with nonempty string id/text only')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Input IDs must be unique')


def label_prediction(aspect, label):
    if aspect not in ASPECTS or label not in LABELS:
        raise ValueError('Illegal predicted label')
    return {'aspect': aspect, 'label': label,
            'subjectivity': 'objective' if label == 'objective' else 'subjective',
            'polarity': 'NA' if label == 'objective' else label}


def merge_fallback(rows, fallback_rows):
    expected = [row['id'] for row in rows if not row['valid']]
    if [row['id'] for row in fallback_rows] != expected:
        raise ValueError('Fallback IDs must exactly match invalid rows in order')
    fallback = iter(fallback_rows)
    result = []
    for row in rows:
        pred = dict(row)
        pred['inference_route'] = 'retrieved_silver_examples'
        if not pred['valid']:
            item = next(fallback)
            pred.update(label_prediction(item['aspect'], item['label']))
            pred['inference_route'] = 'word_fallback'
            # Preserve valid=False and invalid_reason: fallback has no verified quote.
        result.append(pred)
    return result


def select_examples(bank_rows, scores):
    """Keep the frozen 0.7/0.3 rank order; at most two examples per aspect."""
    import numpy as np

    selected = []
    counts = {}
    for position in np.argsort(-np.asarray(scores), kind='stable'):
        row = bank_rows[int(position)]
        if counts.get(row['aspect'], 0) >= 2:
            continue
        selected.append({key: row[key] for key in ['id', 'text', 'aspect', 'label']}
                        | {'retrieval_score': float(scores[position])})
        counts[row['aspect']] = counts.get(row['aspect'], 0) + 1
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise ValueError('Bank must support four examples with at most two per aspect')
    return selected


def retrieve_examples(rows, bank, encoder_path):
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    encoder = SentenceTransformer(str(encoder_path), local_files_only=True, device='cpu')
    bank_texts = [row['text'] for row in bank]
    texts = [row['text'] for row in rows]
    bank_vectors = encoder.encode(bank_texts, normalize_embeddings=True, show_progress_bar=False)
    query_vectors = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(bank_texts)
    lexical = (vectorizer.transform(texts) @ vectorizer.transform(bank_texts).T).toarray()
    scores = .7 * (query_vectors @ bank_vectors.T) + .3 * lexical
    return [{**row, 'examples': select_examples(bank, scores[i])} for i, row in enumerate(rows)]


def checked_corpus(kb_path, eval_meta, eval_text):
    """Read only eval IDs/text for exclusion; never the human answers."""
    import pandas as pd
    from src.weak_labels import load_test_exclusions

    kb = pd.read_parquet(kb_path, columns=['chunk_id', 'review_id', 'text', 'is_eval'])
    parents, texts, info = load_test_exclusions(kb, Path(eval_meta), Path(eval_text))
    if info['marked_sentences'] != 1100:
        raise ValueError('Expected updated corpus with exactly 1,100 is_eval sentences')
    return kb, parents, texts, info


def validate_bank(bank, kb, parents, eval_texts):
    from src.weak_labels import normalize_text

    if not bank:
        raise ValueError('Empty example bank')
    expected = {'id', 'chunk_id', 'review_id', 'text', 'aspect', 'label'}
    if any(not isinstance(row, dict) or set(row) != expected
            or any(not isinstance(value, str) or not value.strip() for value in row.values())
            for row in bank):
        raise ValueError('Wrong bank schema')
    for row in bank:
        label_prediction(row['aspect'], row['label'])
    for key in ['id', 'chunk_id', 'review_id']:
        if len({row[key] for row in bank}) != len(bank):
            raise ValueError('Duplicate bank ' + key)
    if len({normalize_text(row['text']) for row in bank}) != len(bank):
        raise ValueError('Duplicate bank normalized text')
    corpus = kb.set_index('chunk_id')
    for row in bank:
        source = corpus.loc[row['chunk_id']]
        if source['text'] != row['text'] or source['review_id'] != row['review_id']:
            raise ValueError('Bank ID/text/parent differs from corpus')
        if source['is_eval'] or row['review_id'] in parents or normalize_text(row['text']) in eval_texts:
            raise ValueError('Bank overlaps eval sentence, parent review or normalized text')


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(HERE.glob('*.py'))}


def check_source_hashes(expected):
    if source_hashes() != expected:
        raise ValueError('Experiment source changed after input preparation')
