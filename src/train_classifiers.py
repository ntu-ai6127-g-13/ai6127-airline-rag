"""C2：TF-IDF + 逻辑回归基线；只在留出的弱标签上自评。

训练：python -m src.train_classifiers train --output-dir outputs/c2_baseline_full
预测：python -m src.train_classifiers predict --model outputs/c2_baseline_full/baseline.joblib --text 'The seat was comfortable.'
train 保留旧六类基线；train-all 训练八方面+主客观+情感联合基线。
predict 自动识别模型版本；联合模型的客观句输出 polarity=NA。
不使用人工金标准答案，不改写输入数据。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from src.config import ASPECTS, CHUNKS, DATA_EVAL, OUT_DIR, ROOT, WEAK_LABELS
from src.weak_labels import load_test_exclusions, normalize_text, sha256, validate_ids
from src.training_guards import check_fit_boundary

VERSION = 'c2-tfidf-lr-v1'
TASKS = {'aspect': list(ASPECTS), 'polarity': ['negative', 'positive']}
REQUIRED = ['chunk_id', 'review_id', 'text', 'is_eval', 'aspect', 'polarity', 'source']


def validate_training_data(frame, kb, test_reviews, test_texts):
    """拒绝泄漏或不一致的数据，而不是悄悄删行后继续。"""
    missing = set(REQUIRED) - set(frame)
    if missing:
        raise ValueError(f'弱标签缺少列：{sorted(missing)}；请先刷新 C1')
    if frame.empty:
        raise ValueError('弱标签为空')
    if 'split' in frame:
        raise ValueError('带 split 的标签必须使用固定开发集的对照入口；禁止整文件重新切分训练')
    validate_ids(frame, '弱标签')
    if not frame.text.map(lambda t: isinstance(t, str) and bool(t.strip())).all():
        raise ValueError('弱标签存在空白或非文本句子')
    if (not pd.api.types.is_bool_dtype(frame.is_eval) or frame.is_eval.isna().any()
            or frame.is_eval.any()):
        raise ValueError('is_eval 必须是无空值且全部 False 的布尔列')
    if not frame.source.eq('weak').all():
        raise ValueError('本基线只接受 source=weak 的训练材料')
    for task, labels in TASKS.items():
        if not frame[task].isin(labels).all():
            raise ValueError(f'{task} 存在非法或缺失标签')
    if frame.review_id.isin(test_reviews).any():
        raise ValueError('弱标签混入最终评测句所属评论，拒绝训练，请刷新 C1')
    normalized = frame.text.map(normalize_text)
    if normalized.isin(test_texts).any():
        raise ValueError('弱标签包含重复评测文本，拒绝训练')
    if not normalized.is_unique:
        raise ValueError('弱标签有规范化重复文本，请在 C1 排重后再训练')
    validate_ids(kb, '参考知识库')
    mapped = frame[['chunk_id', 'review_id', 'text']].merge(
        kb[['chunk_id', 'review_id', 'text', 'is_eval']], on='chunk_id', how='left',
        suffixes=('_weak', '_kb'), validate='one_to_one', indicator=True)
    if not (mapped._merge.eq('both').all() and
            mapped.review_id_weak.eq(mapped.review_id_kb).all() and
            mapped.text_weak.eq(mapped.text_kb).all() and mapped.is_eval.eq(False).all()):
        raise ValueError('弱标签与当前知识库的编号、原文、父评论或评测标记不一致')
    return {'zero_eval_sentence_overlap': True, 'zero_eval_parent_overlap': True,
            'zero_eval_normalized_text_overlap': True, 'unique_normalized_texts': True,
            'weak_rows_match_current_kb': True}


def split_by_review(frame, dev_fraction=0.2, seed=42):
    """约 20% 的父评论留作开发验证；排序使结果不受输入行顺序影响。"""
    if not 0 < dev_fraction < 1:
        raise ValueError('dev_fraction 必须介于 0 和 1 之间')
    ordered = frame.sort_values('chunk_id').reset_index(drop=True)
    if ordered.review_id.nunique() < 2:
        raise ValueError('至少需要两条不同的父评论进行分组切分')
    train_index, dev_index = next(GroupShuffleSplit(
        n_splits=1, test_size=dev_fraction, random_state=seed
    ).split(ordered, groups=ordered.review_id))
    train, dev = ordered.iloc[train_index].copy(), ordered.iloc[dev_index].copy()
    if set(train.review_id) & set(dev.review_id):
        raise ValueError('训练和验证出现相同父评论')
    if set(train.text.map(normalize_text)) & set(dev.text.map(normalize_text)):
        raise ValueError('训练和验证出现重复文本')
    for name, part in [('train', train), ('dev', dev)]:
        for task, labels in TASKS.items():
            if set(part[task]) != set(labels):
                raise ValueError(f'{name} 的 {task} 类别不齐，请扩大输入样本；不要查看金标准调切分')
    return train, dev


def score_task(actual, predicted, labels):
    report = classification_report(actual, predicted, labels=labels,
                                   output_dict=True, zero_division=0)
    return {'accuracy': float(accuracy_score(actual, predicted)),
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'per_class': {label: report[label] for label in labels},
            'confusion_matrix': confusion_matrix(actual, predicted, labels=labels).tolist(),
            'confusion_labels': labels,
            'confusion_axes': 'rows=weak reference; columns=model prediction'}


def fit_baseline(train, dev, *, max_features=20000, max_iter=1000, seed=42):
    check_fit_boundary(train, dev)
    # 保留否定词；词表和 IDF 只在 train 上拟合，dev 只能 transform。
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                                 stop_words=None)
    x_train = vectorizer.fit_transform(train.text)
    x_dev = vectorizer.transform(dev.text)
    models, metrics = {}, {}
    predictions = dev[['chunk_id', 'review_id', 'text', 'aspect', 'polarity']].rename(
        columns={'aspect': 'weak_aspect', 'polarity': 'weak_polarity'}).reset_index(drop=True)
    for task, labels in TASKS.items():
        print(f'      训练 {task}：{len(train):,} 句，{x_train.shape[1]:,} 个文本特征 ...', flush=True)
        model = LogisticRegression(max_iter=max_iter, solver='lbfgs', C=1.0,
                                   random_state=seed)
        # 不让未收敛模型被当作成功基线保存，也限制本机 CPU 线程占用。
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(x_train, train[task])
        predicted = model.predict(x_dev)
        metrics[task] = score_task(dev[task], predicted, labels)
        dummy = DummyClassifier(strategy='most_frequent').fit(x_train, train[task])
        metrics[task]['majority_reference'] = score_task(dev[task], dummy.predict(x_dev), labels)
        metrics[task]['iterations'] = model.n_iter_.tolist()
        predictions['predicted_' + task] = predicted
        predictions[task + '_matches_weak'] = predicted == dev[task].to_numpy()
        models[task] = model
    return {'version': VERSION, 'vectorizer': vectorizer, 'models': models,
            'input_features': ['text'], 'tasks': TASKS}, metrics, predictions


def predict_texts(bundle, texts):
    from src.classification_pipeline import SUPPORTED_VERSIONS, predict_texts as predict_all
    if bundle.get('version') in SUPPORTED_VERSIONS:
        return predict_all(bundle, texts)
    if bundle.get('version') != VERSION:
        raise ValueError('模型文件不是当前 C2 基线版本')
    if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        raise ValueError('请输入至少一个非空句子')
    features = bundle['vectorizer'].transform(texts)
    result = pd.DataFrame({'text': texts})
    for task in TASKS:
        result[task] = bundle['models'][task].predict(features)
    return result


def report_markdown(report):
    split = report['split']
    lines = ['# C2 基线实际运行结果', '',
             '**本页指标来自留出的弱标签，不是人工金标准成绩。**', '',
             f"运行时间：{report['created_at']}；版本：{VERSION}。", '',
             f"输入 {report['input_rows']:,} 句。训练 {split['train_rows']:,} 句 / "
             f"{split['train_reviews']:,} 条评论；验证 {split['dev_rows']:,} 句 / "
             f"{split['dev_reviews']:,} 条评论。按父评论划分，seed={split['seed']}。", '',
             '| 任务 | Accuracy | 宏平均 P | 宏平均 R | 宏平均 F1 | 多数类参照宏 F1 |',
             '|---|---:|---:|---:|---:|---:|']
    for task, m in report['metrics'].items():
        lines.append(f"| {task} | {m['accuracy']:.4f} | {m['macro_precision']:.4f} | "
                     f"{m['macro_recall']:.4f} | {m['macro_f1']:.4f} | "
                     f"{m['majority_reference']['macro_f1']:.4f} |")
    for task, m in report['metrics'].items():
        lines += ['', f'## {task} 各类结果（参照为弱标签）', '',
                  '| 类别 | Precision | Recall | F1 | 验证句数 |', '|---|---:|---:|---:|---:|']
        for label, scores in m['per_class'].items():
            lines.append(f"| {label} | {scores['precision']:.4f} | {scores['recall']:.4f} | "
                         f"{scores['f1-score']:.4f} | {int(scores['support']):,} |")
    lines += ['', '## 怎样理解', '',
              '- TF-IDF 把词和相邻两词组合变成数字；逻辑回归学习这些特征与标签的关系。',
              '- 两个模型共用只从训练集学习的词表。评分、命中词、人工答案和父评论全文不作为特征。',
              '- 多数类参照始终猜训练中最多的类别，帮助判断模型是否比简单猜测更有用。',
              '- 方面标签本身由关键词规则产生，因此方面高分主要说明模型学会了规则，不能证明真实方面识别准确。',
              '- 极性参照来自评论评分，可能与单句意思不同。预测与弱标签不同，不一定是模型错。',
              '- errors.csv 是与弱标签不一致的样例，尚未人工裁定。',
              '- 金标准只读取编号和原文用于防泄漏核验，未读取人工标签，未进行预测或评分。',
              '- 未实现主观/客观分类、overall/other、多方面输出、全库打标或神经网络训练。',
              '- 没有完成这些后续任务，不代表本次 C2 基线没有跑通。', '']
    return '\n'.join(lines)


def train_command(args):
    start = time.perf_counter()
    destination = args.output_dir.resolve()
    if destination.exists():
        raise ValueError(f'输出目录已存在：{destination}；请用新的 --output-dir 保留已有实验')
    if args.max_features < 1 or args.max_iter < 1:
        raise ValueError('max-features 和 max-iter 必须大于 0')
    paths = {'weak_labels': args.input.resolve(), 'knowledge_base': CHUNKS.resolve(),
             'eval_meta': (DATA_EVAL / 'to_label_meta.csv').resolve(),
             'eval_text': (DATA_EVAL / 'to_label.csv').resolve()}
    source_hashes = {name: {'path': str(p), 'sha256': sha256(p)} for name, p in paths.items()}
    print('[1/5] 核对弱标签与评测集隔离（不读取人工答案）...', flush=True)
    frame = pd.read_parquet(paths['weak_labels'])
    kb = pd.read_parquet(CHUNKS, columns=['chunk_id', 'review_id', 'text', 'is_eval'])
    test_reviews, test_texts, eval_info = load_test_exclusions(kb, paths['eval_meta'], paths['eval_text'])
    checks = validate_training_data(frame, kb, test_reviews, test_texts)
    print('[2/5] 按 review_id 划分训练/验证...', flush=True)
    train, dev = split_by_review(frame, args.dev_fraction, args.seed)
    checks.update(check_fit_boundary(train, dev, eval_reviews=test_reviews, eval_texts=test_texts))
    checks.update({'zero_train_dev_review_overlap': True, 'zero_train_dev_text_overlap': True,
                   'all_rows_in_one_split': len(train) + len(dev) == len(frame),
                   'tfidf_fitted_on_train_only': True, 'model_input_text_only': True})
    print(f'      训练 {len(train):,} 句；验证 {len(dev):,} 句；金标准 {eval_info["records"]:,} 句均隔离', flush=True)
    print('[3/5] 训练两个基线，并在弱标签验证集上预测...', flush=True)
    bundle, metrics, predictions = fit_baseline(train, dev, max_features=args.max_features,
                                               max_iter=args.max_iter, seed=args.seed)
    report = {'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
              'evaluation_kind': 'held_out_weak_labels_not_human_gold', 'input_rows': len(frame),
              'input_reviews': int(frame.review_id.nunique()), 'evaluation_exclusion': eval_info,
              'split': {'method': 'GroupShuffleSplit by review_id, sorted chunk_id',
                        'seed': args.seed, 'dev_review_fraction_requested': args.dev_fraction,
                        'train_rows': len(train), 'dev_rows': len(dev),
                        'train_reviews': int(train.review_id.nunique()),
                        'dev_reviews': int(dev.review_id.nunique())},
              'hyperparameters': {'max_features': args.max_features, 'ngram_range': [1, 2],
                                  'stop_words': None, 'C': 1.0, 'solver': 'lbfgs',
                                  'max_iter': args.max_iter, 'tuned_on_dev': False},
              'vocabulary_size': len(bundle['vectorizer'].vocabulary_), 'metrics': metrics,
              'checks': checks, 'inputs': source_hashes,
              'runtime': {'python': sys.version.split()[0], 'scikit_learn': sklearn.__version__,
                          'pandas': pd.__version__, 'joblib': joblib.__version__},
              'implementation_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                                        [Path(__file__).resolve(), ROOT / 'src/weak_labels.py', ROOT / 'src/config.py',
                                         ROOT / 'src/training_guards.py']}}
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    report['base_commit'] = git.stdout.strip() if git.returncode == 0 else None
    bundle['training_inputs'] = source_hashes
    bundle['split'] = report['split']
    print('[4/5] 保存模型、切分清单、预测、指标和不一致样例...', flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.c2_baseline_', dir=destination.parent))
    try:
        model_path = temporary / 'baseline.joblib'
        joblib.dump(bundle, model_path, compress=3)
        restored = joblib.load(model_path)
        # 全验证集回读预测，确保保存的模型能独立加载且结果不变。
        recovered = predict_texts(restored, dev.text.tolist())
        for task in TASKS:
            if not np.array_equal(recovered[task], predictions['predicted_' + task]):
                raise ValueError('保存后模型预测不一致')
        checks['model_reload_predictions_identical'] = True
        split = pd.concat([train.assign(split='train'), dev.assign(split='dev')])
        split[['chunk_id', 'review_id', 'split', 'aspect', 'polarity']].sort_values('chunk_id').to_parquet(
            temporary / 'split.parquet', index=False)
        predictions.to_parquet(temporary / 'dev_predictions.parquet', index=False)
        errors = []
        for task in TASKS:
            bad = predictions.loc[~predictions[task + '_matches_weak']].copy()
            bad = bad.groupby('weak_' + task, sort=True).head(10)
            bad['task'] = task
            errors.append(bad)
        pd.concat(errors, ignore_index=True).to_csv(temporary / 'errors.csv', index=False, encoding='utf-8-sig')
        checks['input_files_unchanged'] = all(sha256(p) == source_hashes[n]['sha256'] for n, p in paths.items())
        if not all(checks.values()):
            raise ValueError(f'最终检查未通过：{checks}')
        report['elapsed_seconds'] = round(time.perf_counter() - start, 2)
        report['artifacts_sha256'] = {p.name: sha256(p) for p in temporary.iterdir() if p.is_file()}
        (temporary / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
        (temporary / '结果说明.md').write_text(report_markdown(report))
        if destination.exists():
            raise ValueError('输出目录在运行期间已出现，拒绝覆盖')
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print('[5/5] 基线完成（以下不是金标准成绩）', flush=True)
    for task, m in metrics.items():
        print(f"      {task}: Accuracy={m['accuracy']:.4f}, Macro-F1={m['macro_f1']:.4f}")
    print(f'      保存位置：{destination}\n      耗时：{report["elapsed_seconds"]} 秒')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train', help='从 C1 弱标签训练基线，输出新实验目录')
    train.add_argument('--input', type=Path, default=WEAK_LABELS)
    train.add_argument('--output-dir', type=Path, default=OUT_DIR / 'c2_baseline_full')
    train.add_argument('--dev-fraction', type=float, default=0.2)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--max-features', type=int, default=20000)
    train.add_argument('--max-iter', type=int, default=1000)
    all_train = sub.add_parser('train-all', help='八方面、主客观、情感共用父评论切分的联合基线')
    all_train.add_argument('--input', type=Path, default=CHUNKS)
    all_train.add_argument('--output-dir', type=Path, default=OUT_DIR / 'classification_eight_full')
    all_train.add_argument('--seed', type=int, default=42)
    all_train.add_argument('--max-features', type=int, default=20000)
    all_train.add_argument('--max-iter', type=int, default=1000)
    all_train.add_argument('--split-reference', type=Path, help='沿用已有父评论训练/验证名单；未知父评论排除')
    export = sub.add_parser('export', help='联合模型批量预测合格非评测句，输出独立本地文件')
    export.add_argument('--model', type=Path, required=True)
    export.add_argument('--input', type=Path, default=CHUNKS)
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--batch-size', type=int, default=8192)
    pred = sub.add_parser('predict', help='读取本地训练好的模型预测新句子')
    pred.add_argument('--model', type=Path, required=True)
    pred.add_argument('--text', action='append', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'train':
            return train_command(args)
        if args.command == 'train-all':
            from src.classification_pipeline import train_command as train_all_command
            return train_all_command(args)
        if args.command == 'export':
            from src.classification_pipeline import export_command
            return export_command(args)
        result = predict_texts(joblib.load(args.model), args.text)
        print(result.to_json(orient='records', force_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, ConvergenceWarning) as exc:
        print(f'错误：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in {'train', 'train-all'}:
        from src.training_isolation import install_answer_read_guard
        blocked_reads = install_answer_read_guard()
        print('Gold answer-read guard enabled; probe blocked.', flush=True)
    raise SystemExit(main())
