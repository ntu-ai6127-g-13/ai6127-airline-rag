"""Check evidence provenance and R16 reduction without a model or real data."""
import json

import pytest

from experiments.example_assisted_classifier.evidence import parse_evidence


def spans(text, items, focus=''):
    return parse_evidence(text, json.dumps({'focus': focus, 'items': [
        {'quote': quote, 'aspect': aspect, 'label': label} for quote, aspect, label in items]}), variant='spans')


def test_contrast_moves_aspect_and_polarity_together():
    text = 'Seats were great but food was awful.'
    result = spans(text, [('Seats were great', 'seat', 'positive'),
        ('food was awful', 'food', 'negative')], focus='food was awful.')
    assert (result['aspect'], result['polarity']) == ('food', 'negative')


def test_no_contrast_first_same_polarity_and_negative_mixed():
    text = 'Seats were good, food was good.'
    assert spans(text, [('Seats were good', 'seat', 'positive'),
        ('food was good', 'food', 'positive')])['aspect'] == 'seat'
    text = 'Seats were good, food was bad.'
    assert spans(text, [('Seats were good', 'seat', 'positive'),
        ('food was bad', 'food', 'negative')])['aspect'] == 'food'


def test_three_distinct_evaluated_aspects_and_incidental_facts():
    text = 'A Boeing 777, comfy seats, lovely meals, rude crew.'
    result = spans(text, [('A Boeing 777', 'other', 'objective'),
        ('comfy seats', 'seat', 'positive'), ('lovely meals', 'food', 'positive'),
        ('rude crew', 'staff', 'negative')])
    assert (result['aspect'], result['polarity']) == ('overall', 'negative')
    text = 'A Boeing 777 with comfy seats and lovely meals.'
    result = spans(text, [('A Boeing 777', 'other', 'objective'),
        ('comfy seats', 'seat', 'positive'), ('lovely meals', 'food', 'positive')])
    assert result['aspect'] == 'seat'


def test_repeated_category_is_counted_once():
    text = 'Rude crew, unfriendly crew, bad staff.'
    result = spans(text, [('Rude crew', 'staff', 'negative'),
        ('unfriendly crew', 'staff', 'negative'), ('bad staff', 'staff', 'negative')])
    assert result['aspect'] == 'staff'


@pytest.mark.parametrize('raw', [
    'not JSON', '{"quote":"very comfortable","aspect":"seat","label":"positive"}',
    '{"quote":"Seats not comfortable","aspect":"seat","label":"positive","label":"negative"}',
    '{"quote":"Seats not comfortable","aspect":"new_label","label":"negative"}',
    '{"quote":"","aspect":"seat","label":"negative"}',
])
def test_malformed_or_fabricated_direct_evidence_abstains(raw):
    text = 'Seats not comfortable'
    result = parse_evidence(text, raw, variant='direct')
    assert not result['valid']


def test_neutral_tail_cannot_erase_a_complaint():
    result = spans('Food was bad but aircraft was a 777.', [
        ('Food was bad', 'food', 'negative'), ('aircraft was a 777', 'other', 'objective')],
        focus='aircraft was a 777.')
    assert not result['valid']


def test_empty_evidence_abstains_and_plain_fact_keeps_topic():
    assert not spans('I took a flight.', [])['valid']
    result = spans('The fare was 500 dollars.', [('The fare was 500 dollars.', 'value', 'objective')])
    assert (result['aspect'], result['subjectivity'], result['polarity']) == ('value', 'objective', 'NA')
