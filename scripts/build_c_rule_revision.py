"""沿用 v3 已训练权重，创建明确标记为 v4.1 的规则修正候选；不训练、不读人工答案。

旧 v3 模型及结果保持原样，输出目录存在时拒绝覆盖。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classification_pipeline import (TASKS, VERSION, V3_VERSION, predict_texts, score)
from src.training_guards import check_eval_exclusion
from src.weak_labels import load_test_exclusions, sha256, validate_ids


def make_revision_bundle(base, base_path, base_report_path):
    if base.get('version') != V3_VERSION or base.get('tasks') != TASKS:
        raise ValueError('只接受原有八方面 v3 模型')
    revised = dict(base)
    revised.update(version=VERSION, base_model_version=V3_VERSION,
                   base_model_sha256=sha256(base_path),
                   base_training_report_sha256=sha256(base_report_path),
                   model_retrained_for_rule_revision=False,
                   inference_rule_revision='c-audit-fixes-v4.1',
                   training_policy='inherit v3 fitted vocabulary, estimators and parent split; only inference rules revised')
    return revised


def weak_metrics(frame, predictions):
    result = {}
    for task, classes in TASKS.items():
        use = frame['weak_'+task].notna().to_numpy()
        result[task] = score(frame.loc[use, 'weak_'+task], predictions.loc[use, task], classes)
        if task == 'polarity':
            result[task]['gate_NA_errors'] = int(predictions.loc[use, task].eq('NA').sum())
    return result


def build(args):
    started = time.perf_counter()
    destination = args.output_dir.resolve()
    if destination.exists():
        raise ValueError('修正版输出目录已存在，拒绝覆盖；请换新目录')
    paths = {'base_model': args.model.resolve(), 'base_training_report': args.training_report.resolve(),
             'base_dev': args.dev.resolve(), 'kb': ROOT/'data/processed/chunks.parquet',
             'eval_meta': ROOT/'data/eval/to_label_meta.csv', 'eval_text': ROOT/'data/eval/to_label.csv'}
    inputs = {name: {'path': str(path), 'sha256': sha256(path)} for name, path in paths.items()}
    base = joblib.load(paths['base_model'])
    training = json.loads(paths['base_training_report'].read_text())
    if training.get('version') != V3_VERSION or training.get('inputs') != base.get('training_inputs'):
        raise ValueError('v3 模型与训练报告来源不一致')
    for name in ['kb', 'eval_meta', 'eval_text']:
        if base['training_inputs'].get(name, {}).get('sha256') != inputs[name]['sha256']:
            raise ValueError(f'v3 模型与当前 {name} 版本不一致')
    for name, key in [('base_model', 'model.joblib'), ('base_dev', 'dev_predictions.parquet')]:
        if training.get('artifacts_sha256', {}).get(key) != inputs[name]['sha256']:
            raise ValueError(f'原报告与 {name} 文件校验值不一致')
    kb = pd.read_parquet(paths['kb'], columns=['chunk_id', 'review_id', 'text', 'is_eval'])
    reviews, texts, eval_info = load_test_exclusions(kb, paths['eval_meta'], paths['eval_text'])
    dev = pd.read_parquet(paths['base_dev']).reset_index(drop=True)
    validate_ids(dev, 'v3 弱标签开发题')
    mapped = dev[['chunk_id', 'review_id', 'text']].merge(kb, on='chunk_id', how='left',
                         suffixes=('_dev', '_kb'), validate='one_to_one', indicator=True)
    if (not mapped._merge.eq('both').all() or not mapped.review_id_dev.eq(mapped.review_id_kb).all()
            or not mapped.text_dev.eq(mapped.text_kb).all()):
        raise ValueError('v3 开发题编号、父评论或原文与 KB 不一致')
    check_eval_exclusion(pd.DataFrame({'is_eval': mapped.is_eval, 'review_id': mapped.review_id_dev,
                                      'text': mapped.text_dev}), reviews, texts)
    if (not dev.review_id.isin(base['dev_review_ids']).all()
            or dev.review_id.isin(base['train_review_ids']).any()):
        raise ValueError('开发题不在原冻结开发父评论范围')
    print(f'[1/3] 核验并复现原 v3 的 {len(dev):,} 条弱标签开发预测...', flush=True)
    historical = predict_texts(base, dev.text.tolist(), include_raw=True)
    audit_cols = ['subjectivity_source', 'aspect_source', 'polarity_source', 'focus_text', 'focus_reason']
    for task in TASKS:
        for old_column, new_column in [('predicted_'+task, task), ('raw_'+task, 'raw_'+task)]:
            if not dev[old_column].eq(historical[new_column]).all():
                raise ValueError(f'旧 v3 的 {old_column} 未精确复现，拒绝创建新候选')
    for column in audit_cols:
        if not dev[column].eq(historical[column]).all():
            raise ValueError(f'旧 v3 的 {column} 未精确复现')
    revised = make_revision_bundle(base, paths['base_model'], paths['base_training_report'])
    weights_before = joblib.hash({'vectorizer': base['vectorizer'], 'models': base['models']})
    print('[2/3] 相同权重上应用修正规则，比较同一份弱标签开发题...', flush=True)
    predicted = predict_texts(revised, dev.text.tolist(), include_raw=True)
    output = dev[['chunk_id', 'review_id', 'text', *['weak_'+t for t in TASKS]]].copy()
    changed = pd.Series(False, index=dev.index)
    counts = {}
    for task in TASKS:
        if not predicted['raw_'+task].eq(historical['raw_'+task]).all():
            raise ValueError('原始模型预测改变；本修正版只允许调整后置规则')
        output['v3_'+task], output['revised_'+task] = historical[task], predicted[task]
        mask = historical[task].ne(predicted[task])
        counts[task] = int(mask.sum())
        changed |= mask
    for column in audit_cols:
        output['v3_'+column], output['revised_'+column] = historical[column], predicted[column]
    counts['any_final_label'] = int(changed.sum())
    examples = ["I can't recommend this airline enough.", "I cannot recommend this airline.",
                'The ground-crew were rude.', 'The ground crew were rude.',
                'The aircraft was a Boeing 777.', 'The seat was comfortable.',
                'I cannot recommend enough to keep away from this airline.']
    cases = predict_texts(revised, examples, include_raw=True)
    if cases.loc[0, 'polarity'] != 'positive' or cases.loc[1, 'polarity'] != 'negative':
        raise ValueError('推荐习语回归验证失败')
    if cases.loc[2, 'focus_reason'] == 'unresolved_multi_aspect':
        raise ValueError('ground-crew 仍被当成多方面句')
    if cases.loc[6, 'polarity'] != 'negative':
        raise ValueError('劝人远离航司的建议仍被误判为正面')
    source_paths = [ROOT/'src'/name for name in ['classification_pipeline.py', 'language_rules.py',
                    'language_rules_v3.py', 'weak_label_rules.py', 'weak_label_rules_v3.py',
                    'subjectivity.py', 'subjectivity_v3.py', 'subjectivity_legacy.py', 'training_guards.py', 'weak_labels.py']]
    report = {'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(), 'inputs': inputs,
              'purpose': 'rule-only revision candidate using inherited v3 estimators; not a newly trained v4.1 model',
              'training_base_version': V3_VERSION, 'model_retrained': False, 'human_gold_answers_read': False,
              'evaluation_kind': 'same historical v3 weak-label development questions, not human gold',
              'evaluation_exclusion': eval_info, 'dev_rows': len(dev), 'changed_predictions': counts,
              'v3_final_weak_metrics': weak_metrics(dev, historical),
              'revised_final_weak_metrics': weak_metrics(dev, predicted),
              'implementation_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__).resolve(), *source_paths]},
              'acceptance': {'human_accepted': False, 'human_gold_evaluated': False,
                             'final_version_selected': False, 'b_integration_verified': False},
              'limits': ['客观训练样本少等 v3 训练局限仍保留。',
                         '修正后的 C1 context_v4 标签尚未用于重新训练；本候选继承 v3 权重。',
                         '弱标签开发分数不能当作人工准确率，规则修正不等于整体效果提升。',
                         '历史 v3 交接包保留原样，不自动替换 B 的候选输入。']}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.c_rule_revision_', dir=destination.parent))
    try:
        joblib.dump(revised, temporary/'model.joblib', compress=3)
        restored = joblib.load(temporary/'model.joblib')
        weights_after = joblib.hash({'vectorizer': restored['vectorizer'], 'models': restored['models']})
        if weights_before != weights_after:
            raise ValueError('模型权重或词表发生变化')
        pd.testing.assert_frame_equal(cases, predict_texts(restored, examples, include_raw=True))
        output.to_parquet(temporary/'dev_rule_comparison.parquet', index=False)
        pd.testing.assert_frame_equal(output, pd.read_parquet(temporary/'dev_rule_comparison.parquet'))
        output.loc[changed].to_csv(temporary/'dev_rule_changes.csv', index=False, encoding='utf-8-sig')
        cases.to_csv(temporary/'rule_cases.csv', index=False, encoding='utf-8-sig')
        report['checks'] = {'historical_v3_dev_exactly_reproduced': True, 'raw_model_predictions_unchanged': True,
                            'fitted_weights_and_vocabulary_unchanged': True, 'synthetic_cases_passed': True,
                            'zero_dev_eval_id_parent_text_overlap': True, 'model_reload_identical': True,
                            'inputs_unchanged': all(sha256(p) == inputs[k]['sha256'] for k, p in paths.items())}
        if not all(report['checks'].values()):
            raise ValueError('修正候选核验失败')
        report['elapsed_seconds'] = round(time.perf_counter()-started, 2)
        report['artifacts_sha256'] = {p.name: sha256(p) for p in temporary.iterdir()}
        (temporary/'revision.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        if destination.exists():
            raise ValueError('运行中目标目录出现，拒绝覆盖')
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps({'output': str(destination), 'changed_predictions': counts,
                      'model_retrained': False, 'checks': report['checks']}, ensure_ascii=False, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT/'outputs/classification_eight_full_v3/model.joblib')
    parser.add_argument('--training-report', type=Path, default=ROOT/'outputs/classification_eight_full_v3/metrics.json')
    parser.add_argument('--dev', type=Path, default=ROOT/'outputs/classification_eight_full_v3/dev_predictions.parquet')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/classification_eight_rules_v4_1')
    build(parser.parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
