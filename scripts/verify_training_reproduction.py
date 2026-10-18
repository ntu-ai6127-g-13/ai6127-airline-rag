"""Compare rebuilt weak labels, partitions, predictions and fitted parameters."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--rebuilt-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the previous verification; choose a new output path')
    report = {'reference_run': str(args.reference_run), 'rebuilt_run': str(args.rebuilt_run),
              'data': {}, 'parameters': {}, 'quality_improvement_claimed': False}
    for filename in ['weak_labels.parquet', 'split.parquet', 'dev_predictions.parquet']:
        paths = [directory / filename for directory in [args.reference_run, args.rebuilt_run]]
        left, right = [pd.read_parquet(path).sort_values('chunk_id').reset_index(drop=True) for path in paths]
        pd.testing.assert_frame_equal(left, right)
        report['data'][filename] = {'rows': len(right), 'columns': len(right.columns),
            'dataframe_exact_match': True, 'sha256': [sha256(path) for path in paths]}
    old, new = [joblib.load(directory / 'model.joblib') for directory in [args.reference_run, args.rebuilt_run]]
    for key in ['version', 'tasks', 'input_features', 'train_review_ids', 'dev_review_ids']:
        if old[key] != new[key]: raise ValueError('Model metadata changed: ' + key)
    if old['vectorizer'].vocabulary_ != new['vectorizer'].vocabulary_:
        raise ValueError('Training vocabulary changed')
    np.testing.assert_array_equal(old['vectorizer'].idf_, new['vectorizer'].idf_)
    if set(old['models']) != set(new['models']): raise ValueError('Model tasks changed')
    for task in new['models']:
        for field in ['classes_', 'coef_', 'intercept_']:
            np.testing.assert_array_equal(getattr(old['models'][task], field), getattr(new['models'][task], field))
        report['parameters'][task] = 'classes, coefficients and intercepts exactly equal'
    report.update(status='passed', vocabulary_and_idf_exactly_equal=True,
                  model_sha256=[sha256(directory / 'model.joblib') for directory in [args.reference_run, args.rebuilt_run]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__': main()
