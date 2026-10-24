"""Portable packaging and safety regression checks; no weights or real corpus needed."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from experiments.example_assisted_classifier.core import (
    HERE, ROOT, merge_fallback, select_examples, validate_bank, validate_inputs)


@pytest.mark.parametrize('rows', [[], [{'id': 1, 'text': 'hello'}],
    [{'id': 'x', 'text': ' '}], [{'id': ' ', 'text': 'hello'}],
    [{'id': 'x', 'text': 'hello', 'label': 'positive'}],
    [{'id': 'x', 'text': 'hello'}, {'id': 'x', 'text': 'other'}], [None]])
def test_bad_inputs_rejected(rows):
    with pytest.raises(ValueError):
        validate_inputs(rows)


def test_valid_input_and_deterministic_diverse_retrieval():
    validate_inputs([{'id': '001', 'text': 'The seats were good.'}])
    bank = [{'id': str(i), 'text': str(i), 'aspect': aspect, 'label': 'positive'}
            for i, aspect in enumerate(['seat', 'seat', 'seat', 'food', 'staff'])]
    selected = select_examples(bank, np.array([.9, .9, .8, .7, .6]))
    assert [row['id'] for row in selected] == ['0', '1', '3', '4']
    with pytest.raises(ValueError):
        select_examples(bank[:3], np.ones(3))


def test_fallback_keeps_invalid_evidence_and_ids():
    raw = [{'id': 'a', 'valid': True, 'aspect': 'seat', 'label': 'positive'},
           {'id': 'b', 'valid': False, 'aspect': 'invalid', 'label': 'invalid',
            'invalid_reason': 'quote_not_in_sentence'}]
    result = merge_fallback(raw, [{'id': 'b', 'aspect': 'value', 'label': 'objective'}])
    assert result[0]['inference_route'] == 'retrieved_silver_examples'
    assert result[1]['inference_route'] == 'word_fallback'
    assert result[1]['polarity'] == 'NA'
    assert result[1]['valid'] is False
    assert result[1]['invalid_reason'] == 'quote_not_in_sentence'
    assert raw[1]['aspect'] == 'invalid'
    with pytest.raises(ValueError):
        merge_fallback(raw, [{'id': 'wrong', 'aspect': 'value', 'label': 'objective'}])


def bank_fixture():
    bank = [{'id': 'B0', 'chunk_id': 'r0_s0', 'review_id': 'r0',
             'text': 'Seats were good.', 'aspect': 'seat', 'label': 'positive'}]
    corpus = pd.DataFrame([{'chunk_id': 'r0_s0', 'review_id': 'r0',
                            'text': 'Seats were good.', 'is_eval': False}])
    return bank, corpus


def test_bank_must_match_corpus():
    bank, corpus = bank_fixture()
    validate_bank(bank, corpus, set(), set())
    bad = [{**bank[0], 'text': 'A different sentence.'}]
    with pytest.raises(ValueError, match='differs from corpus'):
        validate_bank(bad, corpus, set(), set())


@pytest.mark.parametrize('condition', ['flag', 'parent', 'duplicate_text'])
def test_eval_excluded_at_sentence_parent_and_normalized_text(condition):
    bank, corpus = bank_fixture()
    parents, texts = set(), set()
    if condition == 'flag':
        corpus.loc[0, 'is_eval'] = True
    elif condition == 'parent':
        parents.add('r0')
    else:
        texts.add('seats were good.')
    with pytest.raises(ValueError, match='overlaps eval'):
        validate_bank(bank, corpus, parents, texts)


def test_bank_rejects_duplicate_ids_and_labels():
    bank, corpus = bank_fixture()
    with pytest.raises(ValueError, match='Duplicate'):
        validate_bank(bank + bank, corpus, set(), set())
    with pytest.raises(ValueError, match='Illegal'):
        validate_bank([{**bank[0], 'label': 'unknown'}], corpus, set(), set())


def test_packaged_spec_has_no_text_or_human_gold():
    spec = json.loads((HERE / 'bank_spec.json').read_text())
    assert spec['human_reviewed'] is False
    assert spec['gold_used'] is False
    assert len(spec['annotations']) == 63
    assert all(set(row) == {'id', 'chunk_id', 'review_id', 'aspect', 'label', 'text_sha256'}
               for row in spec['annotations'])
    assert len({row['chunk_id'] for row in spec['annotations']}) == 63
    assert all(len(row['text_sha256']) == 64 for row in spec['annotations'])


def test_guard_denies_answers_but_allows_original_exclusion_text(tmp_path):
    root = tmp_path / 'project'
    data = root / 'data/eval'
    data.mkdir(parents=True)
    allowed = data / 'to_label.csv'
    allowed.write_text('id,text\n1,hello\n')
    answer = data / 'gold.csv'
    answer.write_text('must not be read')
    code = '''from pathlib import Path
from experiments.example_assisted_classifier.guard import install_guard
import sys
root, allowed, answer = map(Path, sys.argv[1:])
denied = install_guard(root, [allowed])
assert allowed.read_text().startswith('id,text')
try:
    answer.read_text()
except PermissionError:
    pass
else:
    raise AssertionError('gold read allowed')
assert len(denied) == 2
'''
    subprocess.run([sys.executable, '-c', code, str(root), str(allowed), str(answer)],
                   cwd=ROOT, check=True)


def test_cli_requires_explicit_opt_in_without_creating_output(tmp_path):
    out = tmp_path / 'not_created'
    process = subprocess.run([sys.executable, '-m',
        'experiments.example_assisted_classifier.predict', '--input', 'missing',
        '--bank', 'missing', '--out', str(out)], cwd=ROOT, capture_output=True, text=True)
    assert process.returncode != 0
    assert '--experimental' in process.stderr
    assert not out.exists()


def test_bank_builder_refuses_output_outside_ignored_directory(tmp_path):
    out = tmp_path / 'not_created'
    process = subprocess.run([sys.executable, '-m',
        'experiments.example_assisted_classifier.build_bank', '--out', str(out)],
        cwd=ROOT, capture_output=True, text=True)
    assert process.returncode != 0
    assert 'NEW directory' in process.stderr
    assert not out.exists()
