"""评分全量下放与高精度子集的本地对照；只评估冻结的弱标签开发集。

python -B scripts/compare_weak_label_strategies.py
不读取人工答案，不覆盖旧实验，不自动选择或替换交付模型。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from src.classify import TASKS, VERSION as MODEL_VERSION, predict_texts, score_task, validate_training_data
from src.config import ASPECTS, CHUNKS, DATA_EVAL, ROOT
from src.weak_label_rules import match_series
from src.weak_labels import load_test_exclusions, normalize_text, read_input, sha256, validate_ids
from src.training_guards import check_fit_boundary, select_training_rows

VERSION = 'broad-versus-single-aspect-v1'
QUALITY = ['is_route_only', 'is_non_english', 'is_fragment']
HYPERPARAMETERS = {'max_features': 20000, 'ngram_range': [1, 2], 'stop_words': None,
                   'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'seed': 42,
                   'class_weight': None, 'tuned_on_dev': False}


def common_candidates(frame, eval_reviews, eval_texts):
    """双方共同的安全/质量筛选，不按关键词、情感或评分选择父评论。"""
    validate_ids(frame, '共同输入')
    for col in ['is_eval', *QUALITY]:
        if col not in frame or not pd.api.types.is_bool_dtype(frame[col]) or frame[col].isna().any():
            raise ValueError(f'{col} 必须是无空值布尔列')
    if not frame.text.map(lambda t: isinstance(t, str)).all():
        raise ValueError('原文必须为文本')
    norm = frame.text.map(normalize_text)
    keep = ~frame.is_eval & ~frame.review_id.isin(eval_reviews) & ~norm.isin(eval_texts)
    keep &= frame.text.str.strip().ne('') & ~frame[QUALITY].any(axis=1)
    return frame.loc[keep].sort_values('chunk_id').reset_index(drop=True).copy()


def assign_frozen_split(candidates, reference_split, frozen_dev):
    """沿用旧父评论名单，先共同屏蔽与旧开发文本重复的整条训练父评论。"""
    validate_ids(reference_split, '参考切分')
    if not reference_split.split.isin(['train', 'dev']).all():
        raise ValueError('参考切分有非法标记')
    if reference_split.groupby('review_id').split.nunique().gt(1).any():
        raise ValueError('同一父评论跨集合')
    mapping = reference_split.drop_duplicates('review_id').set_index('review_id').split
    if not frozen_dev.review_id.map(mapping).eq('dev').all():
        raise ValueError('冻结开发父评论不属于参考开发集合')
    result = candidates.copy()
    result['split'] = result.review_id.map(mapping).fillna('unassigned')
    duplicate = result.split.eq('train') & result.text.map(normalize_text).isin(
        set(frozen_dev.text.map(normalize_text)))
    blocked = set(result.loc[duplicate, 'review_id'])
    result.loc[result.review_id.isin(blocked), 'split'] = 'blocked_dev_duplicate'
    train = result.loc[result.split.eq('train')]
    if set(train.review_id) & set(frozen_dev.review_id) or set(train.text.map(normalize_text)) & set(frozen_dev.text.map(normalize_text)):
        raise ValueError('开发父评论/重复文本泄漏')
    return result, {'blocked_train_reviews': len(blocked),
                    'blocked_train_sentences': int(result.split.eq('blocked_dev_duplicate').sum()),
                    'unassigned_reviews': int(result.loc[result.split.eq('unassigned'), 'review_id'].nunique()),
                    'candidate_sentences_by_split': result.split.value_counts().to_dict(),
                    'candidate_reviews_by_split': result.groupby('split').review_id.nunique().to_dict()}


def expand_broad(frame):
    """每个句子×每个可用方面评分一条。矛盾/多方面记录刻意保留。"""
    validate_ids(frame, '宽松输入')
    parts = []
    for aspect in ASPECTS:
        values = pd.to_numeric(frame['r_' + aspect], errors='coerce')
        take = values.isin([1, 2, 4, 5])
        part = frame.loc[take, ['chunk_id', 'review_id', 'text', 'is_eval', 'split']].copy()
        part['aspect'] = aspect
        part['rating'] = values[take].astype('int8')
        part['polarity'] = np.where(part.rating.ge(4), 'positive', 'negative')
        part['rating_column'] = 'r_' + aspect
        part['source'] = 'weak_rating_broadcast'
        part['rule_version'] = VERSION + ':broad'
        part['label_id'] = part.chunk_id + '::' + aspect
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    if out.empty or not out.label_id.is_unique or out.duplicated(['chunk_id', 'aspect']).any():
        raise ValueError('宽松标签为空或重复句子/方面记录')
    return out


def select_strict(broad, candidates):
    """严格版是宽松版的真子集；只新增单方面关键词约束，不反写评分。"""
    matches, unsupported = match_series(candidates.text, 'context_v3')
    hits = matches.map(bool)
    select = hits.sum(axis=1).eq(1) & ~unsupported
    chosen = hits.idxmax(axis=1)
    ids = set(candidates.loc[select, 'chunk_id'] + '::' + chosen[select])
    result = broad.loc[broad.label_id.isin(ids)].copy().reset_index(drop=True)
    result['source'] = 'weak_single_aspect'
    result['rule_version'] = VERSION + ':strict-context-v3'
    if result.empty or not result.chunk_id.is_unique:
        raise ValueError('严格标签为空或句子重复')
    return result


def describe_labels(labels):
    norm = labels.text.map(normalize_text)
    temporary = pd.DataFrame({'text': norm, 'aspect': labels.aspect, 'polarity': labels.polarity})
    by_text = temporary.groupby('text', sort=False)
    return {'label_records': len(labels), 'unique_sentences': int(labels.chunk_id.nunique()),
            'parent_reviews': int(labels.review_id.nunique()), 'unique_normalized_texts': int(norm.nunique()),
            'texts_with_multiple_aspect_labels': int(by_text.aspect.nunique().gt(1).sum()),
            'texts_with_both_polarities': int(by_text.polarity.nunique().gt(1).sum()),
            'aspect_polarity_records': {a: {p: int(n) for p, n in g.polarity.value_counts().items()}
                                       for a, g in labels.groupby('aspect')},
            'records_by_split': labels.split.value_counts().to_dict()}


def fit_strategy(train, dev, params=None):
    """同一算法/参数；各策略 TF-IDF 仅用自己的训练句，句子 ID 只计一次。

    分类损失按原始标签记录计权。情感任务相同 chunk+label 压缩为计数权重，
    等价于保留重复记录；同一句相反标签不合并、不投票。
    """
    hp = HYPERPARAMETERS if params is None else params
    check_fit_boundary(train, dev)
    if train.empty or dev.empty:
        raise ValueError('空训练/开发集合')
    for task, classes in TASKS.items():
        if set(train[task]) != set(classes) or set(dev[task]) != set(classes):
            raise ValueError(f'{task} 类别不齐')
    sentences = train.drop_duplicates('chunk_id').sort_values('chunk_id')
    position = pd.Series(np.arange(len(sentences)), index=sentences.chunk_id)
    vectorizer = TfidfVectorizer(max_features=hp['max_features'], ngram_range=tuple(hp['ngram_range']),
                                 stop_words=hp['stop_words'])
    x = vectorizer.fit_transform(sentences.text)
    x_dev = vectorizer.transform(dev.text)
    predictions = dev[['chunk_id', 'review_id', 'text', 'aspect', 'polarity']].rename(
        columns={'aspect': 'weak_aspect', 'polarity': 'weak_polarity'}).reset_index(drop=True)
    models, metrics = {}, {}
    for task, classes in TASKS.items():
        weighted = train.groupby(['chunk_id', task], sort=True).size().rename('weight').reset_index()
        if int(weighted.weight.sum()) != len(train):
            raise ValueError('压缩训练权重没有保留全部原始标签记录')
        print(f'  {task}: {len(train):,} 标签记录，{len(weighted):,} 加权训练行', flush=True)
        features = x[position.loc[weighted.chunk_id].to_numpy()]
        model = LogisticRegression(C=hp['C'], solver=hp['solver'], max_iter=hp['max_iter'],
                                   random_state=hp['seed'], class_weight=hp['class_weight'])
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(features, weighted[task], sample_weight=weighted.weight.to_numpy())
        guess = model.predict(x_dev)
        metrics[task] = score_task(dev[task], guess, classes)
        majority = train[task].value_counts().idxmax()
        metrics[task]['majority_reference'] = score_task(dev[task], [majority] * len(dev), classes)
        metrics[task]['iterations'] = model.n_iter_.tolist()
        metrics[task]['weighted_training_rows'] = len(weighted)
        metrics[task]['total_loss_weight'] = int(weighted.weight.sum())
        predictions['predicted_' + task] = guess
        predictions[task + '_matches_weak'] = guess == dev[task].to_numpy()
        models[task] = model
        del features
    bundle = {'version': MODEL_VERSION, 'experiment_version': VERSION, 'vectorizer': vectorizer,
              'models': models, 'input_features': ['text'], 'tasks': TASKS,
              'tfidf_document_unit': 'unique training chunk_id', 'hyperparameters': dict(hp)}
    return bundle, metrics, predictions


def audit_sample(labels, seed=42):
    pieces = [g.sample(min(10, len(g)), random_state=seed)
              for _, g in labels.groupby(['aspect', 'polarity'], sort=True)]
    audit = pd.concat(pieces, ignore_index=True)
    for col in ['human_aspect_correct', 'human_polarity_correct', 'note']:
        audit[col] = ''
    return audit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/weak_strategy_comparison_v1')
    args = ap.parse_args(argv)
    dest = args.output_dir.resolve()
    if dest.exists():
        raise ValueError('目标已存在，拒绝覆盖；请使用新实验目录')
    start = time.perf_counter()
    paths = {'kb': CHUNKS, 'eval_meta': DATA_EVAL / 'to_label_meta.csv',
             'eval_text': DATA_EVAL / 'to_label.csv',
             'old_labels': ROOT / 'data/processed/weak_labels.parquet',
             'reference_split': ROOT / 'outputs/c2_baseline_full/split.parquet'}
    inputs = {k: {'path': str(p.resolve()), 'sha256': sha256(p)} for k, p in paths.items()}
    print('[1/5] 核验语料和冻结的旧弱标签开发集；不读取人工答案', flush=True)
    kb = read_input(CHUNKS)
    reviews, texts, eval_info = load_test_exclusions(kb, paths['eval_meta'], paths['eval_text'])
    old = pd.read_parquet(paths['old_labels'])
    reference = pd.read_parquet(paths['reference_split'])
    validate_training_data(old, kb, reviews, texts)
    if set(old.chunk_id) != set(reference.chunk_id):
        raise ValueError('旧弱标签与参考切分编号不一致')
    joined = old.merge(reference[['chunk_id', 'review_id', 'split']], on=['chunk_id', 'review_id'],
                       how='left', validate='one_to_one')
    if joined.split.isna().any():
        raise ValueError('参考切分与旧父评论映射不一致')
    dev = joined.loc[joined.split.eq('dev')].drop(columns='split').sort_values('chunk_id').reset_index(drop=True)
    candidates = common_candidates(kb, reviews, texts)
    candidates, selection = assign_frozen_split(candidates, reference, dev)
    print('[2/5] 生成宽松评分下放与严格单方面子集', flush=True)
    broad = expand_broad(candidates)
    strict = select_strict(broad, candidates)
    if not set(strict.label_id).issubset(set(broad.label_id)):
        raise ValueError('严格标签不是宽松标签的子集')
    paired = strict.merge(broad[['label_id', 'aspect', 'polarity', 'rating', 'split']], on='label_id',
                          validate='one_to_one', suffixes=('_s', '_b'))
    if any(not paired[c+'_s'].eq(paired[c+'_b']).all() for c in ['aspect', 'polarity', 'rating', 'split']):
        raise ValueError('严格子集偷偷改变了评分或标签')
    report = {'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
              'evaluation_kind': 'fixed_old_weak_dev_not_human_gold',
              'previous_dev_used_for_rule_design': True,
              'inputs': inputs, 'evaluation_exclusion': eval_info, 'split_selection': selection,
              'raw_kb_sentences': len(kb), 'eligible_sentences': len(candidates), 'dev_sentences': len(dev),
              'dev_parent_reviews': int(dev.review_id.nunique()), 'hyperparameters': HYPERPARAMETERS,
              'tfidf_policy': 'fit separately on each strategy unique training chunk_id, never on dev',
              'duplicate_policy': 'keep within-training repeats and conflicting labels in both strategies',
              'loss_weighting': 'one per original label record; aggregate identical chunk/task target using sample_weight',
              'strategies': {}, 'runtime': {'sklearn': sklearn.__version__, 'pandas': pd.__version__},
              'checks': {'strict_is_broad_subset_with_unchanged_targets': True,
                         'same_frozen_dev_for_both_strategies': True,
                         'same_parent_assignment_before_label_generation': True,
                         'gold_answers_not_read': True},
              'limits': ['旧开发目标来自严格关键词弱标签，存在有利于严格方案的选择偏差。',
                         '本次严格子集不删除训练内重复与矛盾记录，与历史严格文件的额外去重策略不同。',
                         '开发错误已用于规则修正，本次不是独立测试。',
                         '不含 overall/other/主客观，不替换已有八类交付候选。',
                         '标签规模、主题覆盖和词表会随策略变化，不能只归因于标签质量。',
                         '宽松中同句跨方面/正负矛盾刻意保留；记录数不是唯一句数。',
                         '两边均只用旧训练父评论；未知父评论只生成标签，不参与本次训练。',
                         '人工金标准定稿及模型冻结后才能补正式同集评测，当前没有金标准成绩。']}
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.weak_strategy_', dir=dest.parent))
    try:
        candidates[['chunk_id', 'review_id', 'split']].to_parquet(temp / 'candidate_split.parquet', index=False)
        dev.to_parquet(temp / 'frozen_dev.parquet', index=False)
        for strategy, labels in [('broad', broad), ('strict', strict)]:
            folder = temp / strategy
            folder.mkdir()
            train = select_training_rows(labels)
            boundary = check_fit_boundary(train, dev, eval_reviews=reviews, eval_texts=texts)
            if train.is_eval.any() or set(train.review_id) & reviews or set(train.text.map(normalize_text)) & texts:
                raise ValueError('训练评测隔离失败')
            if set(train.review_id) & set(dev.review_id) or set(train.text.map(normalize_text)) & set(dev.text.map(normalize_text)):
                raise ValueError('训练开发隔离失败')
            print(f'[3/5] {strategy}: 全部 {len(labels):,} 记录；训练 {len(train):,}；共同开发 {len(dev):,} 句', flush=True)
            labels.to_parquet(folder / 'weak_labels.parquet', index=False, compression='zstd')
            # 保存紧凑的抽查表，不让未分配/开发记录被误当作训练样例。
            audit_sample(train).to_csv(folder / '人工抽查.csv', index=False, encoding='utf-8-sig')
            pd.testing.assert_frame_equal(labels, pd.read_parquet(folder / 'weak_labels.parquet'))
            bundle, metrics, predictions = fit_strategy(train, dev)
            bundle['training_inputs'] = inputs
            bundle['strategy'] = strategy
            bundle['train_review_ids'] = sorted(set(train.review_id))
            bundle['dev_review_ids'] = sorted(set(dev.review_id))
            joblib.dump(bundle, folder / 'model.joblib', compress=3)
            recovered = predict_texts(joblib.load(folder / 'model.joblib'), dev.text.tolist())
            for task in TASKS:
                if not np.array_equal(recovered[task], predictions['predicted_' + task]):
                    raise ValueError('模型保存后预测不一致')
            predictions.to_parquet(folder / 'dev_predictions.parquet', index=False)
            errors = []
            for task in TASKS:
                bad = predictions.loc[~predictions[task + '_matches_weak']].copy()
                bad = bad.groupby('weak_' + task, sort=True).head(20)
                bad['task'] = task
                errors.append(bad)
            pd.concat(errors, ignore_index=True).to_csv(folder / 'errors.csv', index=False, encoding='utf-8-sig')
            report['strategies'][strategy] = {'all_labels': describe_labels(labels),
                                               'training': describe_labels(train), 'metrics': metrics,
                                               'vocabulary_size': len(bundle['vectorizer'].vocabulary_),
                                               'audit_status': 'pending_human_review'}
            report['checks'][strategy + '_roundtrip_and_isolation'] = True
            report['checks'][strategy + '_shared_training_guard'] = all(boundary.values())
            print(f'  完成 {strategy}: ' + ', '.join(f'{t} F1={m["macro_f1"]:.4f}' for t, m in metrics.items()), flush=True)
            del bundle, recovered, train
        print('[4/5] 核验输入未改写，记录可复现来源', flush=True)
        report['checks']['inputs_unchanged'] = all(sha256(paths[k]) == inputs[k]['sha256'] for k in paths)
        if not all(report['checks'].values()):
            raise ValueError('最终核验失败')
        report['implementation_sha256'] = {str(p.relative_to(ROOT)): sha256(p) for p in
            [Path(__file__).resolve(), ROOT / 'src/classify.py', ROOT / 'src/weak_labels.py',
             ROOT / 'src/weak_label_rules.py', ROOT / 'src/config.py', ROOT / 'src/training_guards.py']}
        summary = []
        for strategy, rec in report['strategies'].items():
            for task, metrics in rec['metrics'].items():
                summary.append({'strategy': strategy, 'task': task,
                                'train_label_records': rec['training']['label_records'],
                                'train_unique_sentences': rec['training']['unique_sentences'],
                                'dev_sentences': len(dev),
                                **{key: metrics[key] for key in ['macro_precision', 'macro_recall', 'macro_f1', 'accuracy']},
                                'reference': 'old_weak_labels_not_human_gold'})
        pd.DataFrame(summary).to_csv(temp / 'comparison.csv', index=False, encoding='utf-8-sig')
        report['artifacts_sha256'] = {str(p.relative_to(temp)): sha256(p) for p in temp.rglob('*') if p.is_file()}
        report['elapsed_seconds'] = round(time.perf_counter() - start, 2)
        (temp / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        if dest.exists():
            raise ValueError('运行期间目标出现，拒绝覆盖')
        temp.rename(dest)
    except Exception:
        print(f'运行中断，诊断产物保留于 {temp}，未覆盖任何旧实验。', flush=True)
        raise
    print(f'[5/5] 完成：{dest}，耗时 {report["elapsed_seconds"]} 秒；不是金标准评测。', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
