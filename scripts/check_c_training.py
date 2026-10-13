"""只读核验已有 C 组训练材料；不加载模型、不拟合、不读取人工答案。

默认只打印结果；--output 可将本次检查另存为一个新 JSON，拒绝覆盖旧报告。
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from src.config import ROOT, CHUNKS, DATA_EVAL
from src.training_guards import check_fit_boundary, select_training_rows, validate_roles
from src.weak_labels import load_test_exclusions, sha256, validate_ids

CORE = ['chunk_id', 'review_id', 'text', 'is_eval', 'source']


def frozen_parts(labels, split):
    validate_ids(labels, '固定切分弱标签')
    validate_ids(split, '固定切分清单')
    validate_roles(split)
    if not split.split.isin(['train', 'dev']).all() or set(labels.chunk_id) != set(split.chunk_id):
        raise ValueError('弱标签与冻结切分的编号范围不完全一致')
    merged = labels.merge(split[['chunk_id', 'review_id', 'split']],
                          on=['chunk_id', 'review_id'], how='left', validate='one_to_one')
    if merged.split.isna().any():
        raise ValueError('冻结清单与标签的父评论对应关系不一致')
    return select_training_rows(merged), merged.loc[merged.split.eq('dev')].copy()


def check_mapping(frame, kb):
    """宽松策略可有每句多个标签，但每句的原文/父评论必须唯一且与库一致。"""
    unique = frame[['chunk_id', 'review_id', 'text', 'is_eval']].drop_duplicates()
    if not unique.chunk_id.is_unique:
        raise ValueError('标签中的同一句子编号对应不同原文、父评论或评测标记')
    joined = unique.merge(kb, on='chunk_id', how='left', suffixes=('_label', '_kb'),
                          validate='one_to_one', indicator=True)
    if not joined._merge.eq('both').all() or any(
            not joined[c + '_label'].eq(joined[c + '_kb']).all()
            for c in ['review_id', 'text', 'is_eval']):
        raise ValueError('训练材料与当前知识库的编号/原文/父评论/评测标记不一致')
    return True


def audit_existing():
    paths = {'kb': CHUNKS, 'eval_meta': DATA_EVAL/'to_label_meta.csv',
             'eval_text': DATA_EVAL/'to_label.csv',
             'old_labels': ROOT/'data/processed/weak_labels.parquet',
             'old_split': ROOT/'outputs/c2_baseline_full/split.parquet',
             'frozen_comparison_dev': ROOT/'outputs/weak_strategy_comparison_v1/frozen_dev.parquet'}
    for version in ['v2', 'v3']:
        for kind in ['labels', 'split']:
            name = 'weak_labels' if kind == 'labels' else kind
            paths[f'{version}_{kind}'] = ROOT/f'outputs/classification_eight_full_{version}/{name}.parquet'
    for strategy in ['broad', 'strict']:
        paths[strategy] = ROOT/f'outputs/weak_strategy_comparison_v1/{strategy}/weak_labels.parquet'
    inputs = {name: {'path': str(path), 'sha256': sha256(path)} for name, path in paths.items()}
    kb = pd.read_parquet(paths['kb'], columns=['chunk_id', 'review_id', 'text', 'is_eval'])
    validate_ids(kb, '知识库')
    reviews, texts, evaluation = load_test_exclusions(kb, paths['eval_meta'], paths['eval_text'])
    result = {'created_at': datetime.now(timezone.utc).isoformat(),
              'kind': 'read_only_existing_training_preflight',
              'model_fitting_performed': False, 'models_loaded': False,
              'human_gold_answers_read': False,
              'evaluation_exclusion': evaluation,
              'knowledge_base': {'rows': len(kb), 'eval_sentences': int(kb.is_eval.sum())},
              'inputs': inputs, 'datasets': {}}

    def record(name, train, dev, source_rows):
        checks = check_fit_boundary(train, dev, eval_reviews=reviews, eval_texts=texts)
        checks['train_rows_match_current_kb'] = check_mapping(train, kb)
        checks['dev_rows_match_current_kb'] = check_mapping(dev, kb)
        result['datasets'][name] = {'source_records': source_rows,
            'train_records': len(train), 'dev_records': len(dev),
            'train_unique_sentences': int(train.chunk_id.nunique()),
            'train_parent_reviews': int(train.review_id.nunique()),
            'dev_parent_reviews': int(dev.review_id.nunique()), 'checks': checks}

    for name, label_key, split_key in [('old_six_baseline', 'old_labels', 'old_split'),
                                     ('eight_v2', 'v2_labels', 'v2_split'),
                                     ('eight_v3_candidate', 'v3_labels', 'v3_split')]:
        print(f'核验 {name}（不训练）...', flush=True)
        labels = pd.read_parquet(paths[label_key], columns=CORE)
        train, dev = frozen_parts(labels, pd.read_parquet(paths[split_key]))
        record(name, train, dev, len(labels))
        del labels, train, dev
    dev = pd.read_parquet(paths['frozen_comparison_dev'], columns=CORE)
    dev = dev.assign(split='dev')
    for strategy in ['broad', 'strict']:
        print(f'核验 {strategy}：显式选择 split=train（不训练）...', flush=True)
        labels = pd.read_parquet(paths[strategy], columns=CORE + ['split'])
        train = select_training_rows(labels)
        record(strategy, train, dev, len(labels))
        result['datasets'][strategy]['excluded_nontraining_records'] = len(labels)-len(train)
        result['datasets'][strategy]['source_records_by_role'] = labels.split.value_counts().to_dict()
        del labels, train
    result['checks'] = {'all_five_dataset_checks_passed': all(
        all(info['checks'].values()) for info in result['datasets'].values()),
        'input_files_unchanged': all(sha256(path) == inputs[name]['sha256'] for name, path in paths.items())}
    if not all(result['checks'].values()):
        raise ValueError('核验期间输入变化或隔离检查未通过')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise ValueError('检查报告已存在，拒绝覆盖；省略 --output 即只读再次核验')
    report = audit_existing()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'checks': report['checks'], 'datasets': {
        key: {column: item[column] for column in ['train_records', 'dev_records', 'train_unique_sentences']}
        for key, item in report['datasets'].items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f'核验停止：{error}', file=sys.stderr)
        raise SystemExit(1)
