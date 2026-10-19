"""C3 主客观弱监督基线：train 训练；predict 预测。金标准仅用于防泄漏。

python -m src.subjectivity train --output-dir outputs/c3_subjectivity_full
python -m src.subjectivity predict --model outputs/c3_subjectivity_full/model.joblib --text 'The seat was comfortable.'
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
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

from src.config import CHUNKS, DATA_EVAL, OUT_DIR, ROOT
from src.weak_labels import load_test_exclusions, normalize_text, sha256, validate_ids
from src.training_guards import check_fit_boundary, reject_role_annotated_input

VERSION = 'c3-subjectivity-tfidf-v1'
RULE_VERSION = 'c3-text-seeds-v1'
LABELS = ['objective', 'subjective']

# 明确评价信号；不是把 C1 有评分的句子一律判成主观。
OPINION = re.compile(
    r'\b(?:good|great|excellent|amazing|wonderful|fantastic|superb|outstanding|'
    r'bad|poor|terrible|awful|horrible|appalling|worst|best|disappoint(?:ed|ing|ment)|'
    r'comfortabl[ey]|uncomfortable|cramped|spacious|tasty|delicious|inedible|'
    r'friendly|unfriendly|rude|helpful|unhelpful|attentive|inattentive|'
    r'efficient|inefficient|professional|unprofessional|pleasant|unpleasant|'
    r'expensive|overpriced|affordable|worth|worthless|dirty|filthy|clean|'
    r'broken|unacceptable|acceptable|enjoy(?:ed|able)?|love[ds]?|hate[ds]?|'
    r'recommend(?:ed)?|satisfied|dissatisfied|impressed|disgusting|appreciat(?:e|ed)|'
    r'frustrat(?:ed|ing)|slow|rubbish|disaster|okay|fine)\b', re.I)
IMPLICIT_EVALUATION = re.compile(
    r'\b(?:delay(?:ed|s)?|cancel(?:led|ed|lation)|on[ -]time|punctual)\b|'
    r'\bno (?:issues?|problems?|complaints?)\b|'
    r'\b(?:no|without) (?:food|meals?|entertainment|legroom|drinks?)\b|'
    r'\b(?:should|must|need to) improve\b|\bnot (?:work|working|worth)\b', re.I)

# 只接受整句匹配的事实模板。没有评价词 ≠ 客观；没有命中模板则弃权。
# fullmatch 禁止将一个含评价的长句仅凭开头的事实从句收入 objective。
CITY = r"[a-z][a-z '-]{1,35}"
OBJECTIVE_PATTERNS = {
    'journey': re.compile(rf'(?:i|we) (?:flew|travelled|traveled) from {CITY} to {CITY}'
                          rf'(?: via {CITY})?(?: in (?:january|february|march|april|may|june|july|august|'
                          r'september|october|november|december)(?: \d{4})?)?[.!]?', re.I),
    'cabin': re.compile(r'(?:i|we) (?:flew|travelled|traveled|was seated|were seated) in '
                       r'(?:economy|premium economy|business|first) class[.!]?', re.I),
    'aircraft': re.compile(r'(?:the|our|this) (?:aircraft|plane|flight) (?:was|is) (?:an? )?'
                          r'(?:(?:airbus|boeing) ?)?[ab]?[2378]\d{2}(?:[ -]?\d{2,3})?(?:er|lr|neo)?[.!]?', re.I),
    'price': re.compile(r'(?:the|my|our) (?:return |one-way )?(?:ticket|fare) '
                       r'(?:was|is|cost|costs) (?:[£$€]\s?\d[\d,.]*|'
                       r'(?:sgd|usd|eur|gbp|aud) ?\d[\d,.]*|'
                       r'\d[\d,.]* (?:sgd|usd|eur|gbp|aud|dollars?|euros?|pounds?))[.!]?', re.I),
    'duration': re.compile(r'(?:the|our|this) (?:flight|journey) (?:took|lasted|was) '
                          r'(?:about |approximately )?\d+(?:\.\d+)? hours?'
                          r'(?: (?:and )?\d+ minutes?)?(?: long)?[.!]?', re.I),
    'clock_time': re.compile(r'(?:the|our) flight (?:departed|arrived|left|landed) at '
                            r'\d{1,2}(?::\d{2})? ?(?:am|pm|a\.m\.|p\.m\.)[.!]?', re.I),
    'seat_layout': re.compile(r'(?:the )?seats (?:were|are) (?:arranged )?in (?:a )?'
                             r'\d-\d(?:-\d)? (?:configuration|layout)[.!]?', re.I),
}


def weak_decision(text):
    text = normalize_text(text)
    if OPINION.search(text):
        return 'subjective', 'explicit_evaluation'
    if IMPLICIT_EVALUATION.search(text):
        return 'subjective', 'project_implicit_evaluation'
    for name, pattern in OBJECTIVE_PATTERNS.items():
        if pattern.fullmatch(text):
            return 'objective', 'fact_' + name
    return None, 'abstain'


def build_labels(frame, test_reviews, test_texts):
    reject_role_annotated_input(frame, '旧版 C3 输入')
    required = {'chunk_id', 'review_id', 'text', 'is_eval', 'is_route_only',
                'is_non_english', 'is_fragment'}
    if required - set(frame):
        raise ValueError(f'输入缺列：{sorted(required - set(frame))}')
    validate_ids(frame, 'C3 输入')
    d = frame.copy().reset_index(drop=True)
    for col in ['is_eval', 'is_route_only', 'is_non_english', 'is_fragment']:
        if not pd.api.types.is_bool_dtype(d[col]) or d[col].isna().any():
            raise ValueError(f'{col} 必须是无空值布尔值')
    if not d.text.map(lambda s: isinstance(s, str)).all():
        raise ValueError('原文必须是文本')
    reasons = pd.Series('', index=d.index)
    def exclude(mask, name):
        reasons.loc[reasons.eq('') & mask] = name
    exclude(d.is_eval, 'eval_sentence')
    exclude(d.review_id.isin(test_reviews), 'eval_parent')
    norm = d.text.map(normalize_text)
    exclude(norm.isin(test_texts), 'eval_duplicate_text')
    exclude(d.text.str.strip().eq(''), 'empty')
    for col in ['is_route_only', 'is_non_english', 'is_fragment']:
        exclude(d[col], col)
    candidates = d.loc[reasons.eq(''), ['chunk_id', 'review_id', 'text', 'is_eval']].copy()
    decisions = [weak_decision(t) for t in candidates.text]
    candidates['subjectivity'] = [t[0] for t in decisions]
    candidates['matched_rule'] = [t[1] for t in decisions]
    reasons.loc[candidates.index[candidates.subjectivity.isna()]] = 'rule_abstain'
    candidates = candidates.loc[candidates.subjectivity.notna()].sort_values('chunk_id')
    duplicate = candidates.text.map(normalize_text).duplicated()
    reasons.loc[candidates.index[duplicate]] = 'duplicate_text'
    result = candidates.loc[~duplicate].reset_index(drop=True)
    result['source'] = 'weak_subjectivity'
    result['rule_version'] = RULE_VERSION
    if set(result.subjectivity) != set(LABELS):
        raise ValueError('主客观弱标签不足两类，请扩大数据；不能把未命中词的句子默认为客观')
    dropped = reasons.loc[reasons.ne('')].value_counts().to_dict()
    assert sum(dropped.values()) + len(result) == len(d)
    return result, {'input_rows': len(d), 'output_rows': len(result),
                    'class_counts': result.subjectivity.value_counts().to_dict(),
                    'rule_counts': result.matched_rule.value_counts().to_dict(),
                    'excluded_by_first_reason': dropped, 'rule_version': RULE_VERSION,
                    'scores_used_for_labels': False}


def validate_labels(labels, kb, test_reviews, test_texts):
    validate_ids(labels, 'C3 弱标签')
    if labels.empty or not labels.subjectivity.isin(LABELS).all():
        raise ValueError('主客观标签非法或为空')
    if not labels.source.eq('weak_subjectivity').all():
        raise ValueError('C3 只接受主客观弱标签')
    if (not pd.api.types.is_bool_dtype(labels.is_eval) or labels.is_eval.isna().any()
            or labels.is_eval.any() or labels.review_id.isin(test_reviews).any()):
        raise ValueError('评测句或父评论泄漏')
    normalized = labels.text.map(normalize_text)
    if normalized.isin(test_texts).any() or not normalized.is_unique:
        raise ValueError('重复文本或评测文本泄漏')
    mapped = labels[['chunk_id', 'review_id', 'text']].merge(
        kb[['chunk_id', 'review_id', 'text', 'is_eval']], on='chunk_id', how='left',
        suffixes=('_weak', '_kb'), validate='one_to_one')
    if not (mapped.review_id_weak.eq(mapped.review_id_kb).all() and
            mapped.text_weak.eq(mapped.text_kb).all() and mapped.is_eval.eq(False).all()):
        raise ValueError('弱标签与参考库不一致')
    return {'zero_eval_id_parent_text_overlap': True, 'unique_normalized_texts': True,
            'matches_current_kb': True}


def split_labels(frame, seed=42):
    reject_role_annotated_input(frame, '旧版 C3 待切分标签')
    ordered = frame.sort_values('chunk_id').reset_index(drop=True)
    ti, di = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(
        ordered, groups=ordered.review_id))
    train, dev = ordered.iloc[ti].copy(), ordered.iloc[di].copy()
    if set(train.review_id) & set(dev.review_id) or set(train.text.map(normalize_text)) & set(dev.text.map(normalize_text)):
        raise ValueError('训练/验证评论或文本重叠')
    if any(set(part.subjectivity) != set(LABELS) for part in (train, dev)):
        raise ValueError('切分后类别不齐，请扩大样本，不要根据金标准调切分')
    return train, dev


def score(actual, predicted):
    r = classification_report(actual, predicted, labels=LABELS, output_dict=True, zero_division=0)
    return {'accuracy': float(accuracy_score(actual, predicted)),
            'macro_precision': r['macro avg']['precision'], 'macro_recall': r['macro avg']['recall'],
            'macro_f1': r['macro avg']['f1-score'], 'per_class': {k: r[k] for k in LABELS},
            'confusion_matrix': confusion_matrix(actual, predicted, labels=LABELS).tolist(),
            'confusion_labels': LABELS, 'confusion_axes': 'rows=weak reference; columns=prediction'}


def fit_model(train, dev, seed=42):
    check_fit_boundary(train, dev)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, stop_words=None)
    x, xd = vectorizer.fit_transform(train.text), None
    xd = vectorizer.transform(dev.text)
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, train.subjectivity)
    predictions = model.predict(xd)
    metrics = score(dev.subjectivity, predictions)
    dummy = DummyClassifier(strategy='most_frequent').fit(x, train.subjectivity)
    metrics['majority_reference'] = score(dev.subjectivity, dummy.predict(xd))
    metrics['iterations'] = model.n_iter_.tolist()
    bundle = {'version': VERSION, 'vectorizer': vectorizer, 'model': model,
              'labels': LABELS, 'input_features': ['text'], 'rule_version': RULE_VERSION}
    return bundle, metrics


def predict_subjectivity(bundle, texts):
    if bundle.get('version') != VERSION or list(bundle.get('labels', [])) != LABELS:
        raise ValueError('不是支持的 C3 模型文件')
    if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        raise ValueError('至少需要一个非空句子')
    x = bundle['vectorizer'].transform(texts)
    return pd.DataFrame({'text': texts, 'subjectivity': bundle['model'].predict(x)})


def train_command(args):
    start = time.perf_counter()
    dest = args.output_dir.resolve()
    if dest.exists():
        raise ValueError(f'输出目录已存在：{dest}，请换新目录，禁止覆盖旧实验')
    paths = {'input': args.input.resolve(), 'kb': CHUNKS,
             'eval_meta': DATA_EVAL / 'to_label_meta.csv', 'eval_text': DATA_EVAL / 'to_label.csv'}
    hashes = {k: {'path': str(p), 'sha256': sha256(p)} for k, p in paths.items()}
    print('[1/4] 核对评测隔离，生成文本规则弱标签（不使用评分或人工答案）...', flush=True)
    kb = pd.read_parquet(CHUNKS, columns=['chunk_id', 'review_id', 'text', 'is_eval'])
    reviews, texts, eval_info = load_test_exclusions(kb, paths['eval_meta'], paths['eval_text'])
    frame = pd.read_parquet(paths['input'])
    reject_role_annotated_input(frame, '旧版 C3 输入')
    # 包括未被规则选中的输入句，也逐条核对 ID、原文、父评论及 is_eval。
    validate_ids(frame, 'C3 输入')
    check = frame[['chunk_id', 'review_id', 'text', 'is_eval']].merge(
        kb, on='chunk_id', how='left', suffixes=('_input', '_kb'), validate='one_to_one')
    if any(not check[c + '_input'].eq(check[c + '_kb']).all() for c in ['review_id', 'text', 'is_eval']):
        raise ValueError('输入与当前知识库不一致')
    labels, generation = build_labels(frame, reviews, texts)
    checks = validate_labels(labels, kb, reviews, texts)
    train, dev = split_labels(labels, args.seed)
    checks.update(check_fit_boundary(train, dev, eval_reviews=reviews, eval_texts=texts))
    print(f'[2/4] 训练 {len(train):,} 句，弱标签验证 {len(dev):,} 句；类别：{generation["class_counts"]}', flush=True)
    model, metrics = fit_model(train, dev, args.seed)
    predicted = predict_subjectivity(model, dev.text.tolist())
    predictions = dev[['chunk_id', 'review_id', 'text', 'subjectivity', 'matched_rule']].rename(
        columns={'subjectivity': 'weak_subjectivity'}).reset_index(drop=True)
    predictions['predicted_subjectivity'] = predicted.subjectivity.to_numpy()
    predictions['matches_weak'] = predictions.weak_subjectivity.eq(predictions.predicted_subjectivity)
    checks.update(zero_train_dev_parent_text_overlap=True, tfidf_fitted_on_train_only=True,
                  model_input_text_only=True, all_rows_in_one_split=len(train) + len(dev) == len(labels))
    report = {'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
              'evaluation_kind': 'held_out_text_rule_weak_labels_not_human_gold',
              'generation': generation, 'evaluation_exclusion': eval_info,
              'split': {'method': 'GroupShuffleSplit by review_id', 'seed': args.seed,
                        'train_rows': len(train), 'dev_rows': len(dev),
                        'train_reviews': int(train.review_id.nunique()), 'dev_reviews': int(dev.review_id.nunique()),
                        'train_class_counts': train.subjectivity.value_counts().to_dict(),
                        'dev_class_counts': dev.subjectivity.value_counts().to_dict()},
              'metrics': metrics, 'checks': checks, 'inputs': hashes,
              'hyperparameters': {'ngram_range': [1, 2], 'max_features': 20000, 'stop_words': None,
                                  'class_weight': 'balanced', 'C': 1, 'max_iter': 1000, 'tuned_on_dev': False},
              'runtime': {'python': sys.version.split()[0], 'sklearn': sklearn.__version__, 'pandas': pd.__version__},
              'implementation_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                  [Path(__file__).resolve(), ROOT/'src/classify.py', ROOT/'src/weak_labels.py', ROOT/'src/config.py']},
              'limits': ['规则弱标签，尚未经人工确认；指标不能表示真实主客观准确率。',
                         'objective 仅覆盖少数事实模板，明显存在选择偏差；未知句未被默认标成 objective。',
                         'subjective 词表会误判引用、条件、无关话题、语境；未实现完整语义判断。',
                         'class_weight 平衡训练损失，不等于真实类别分布平衡。',
                         '最终评测答案未读取；尚未在人工金标准上评测。']}
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.c3_', dir=dest.parent))
    try:
        print('[3/4] 保存独立模型、标签、验证预测和人工抽查表...', flush=True)
        model['training_inputs'] = hashes
        joblib.dump(model, temp/'model.joblib', compress=3)
        pd.testing.assert_frame_equal(predicted, predict_subjectivity(joblib.load(temp/'model.joblib'), dev.text.tolist()))
        checks['model_reload_predictions_identical'] = True
        labels.to_parquet(temp/'weak_subjectivity.parquet', index=False)
        pd.testing.assert_frame_equal(labels, pd.read_parquet(temp/'weak_subjectivity.parquet'))
        checks['labels_roundtrip_identical'] = True
        pd.concat([train.assign(split='train'), dev.assign(split='dev')])[
            ['chunk_id', 'review_id', 'subjectivity', 'split']].to_parquet(temp/'split.parquet', index=False)
        predictions.to_parquet(temp/'dev_predictions.parquet', index=False)
        predictions.loc[~predictions.matches_weak].groupby('weak_subjectivity').head(20).to_csv(
            temp/'errors.csv', index=False, encoding='utf-8-sig')
        audit = pd.concat([g.sample(n=min(60, len(g)), random_state=args.seed)
                           for _, g in labels.groupby('subjectivity')])
        audit['human_subjectivity'] = ''
        audit['note'] = ''
        audit.to_csv(temp/'人工抽查.csv', index=False, encoding='utf-8-sig')
        report['audit'] = {'rows': len(audit), 'status': 'pending_human_review'}
        checks['input_files_unchanged'] = all(sha256(p) == hashes[k]['sha256'] for k, p in paths.items())
        if not all(checks.values()):
            raise ValueError(f'最终核验失败：{checks}')
        report['elapsed_seconds'] = round(time.perf_counter() - start, 2)
        report['artifacts_sha256'] = {p.name: sha256(p) for p in temp.iterdir()}
        (temp/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if dest.exists():
            raise ValueError('运行期间输出目录已出现，拒绝覆盖')
        temp.rename(dest)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    print(f'[4/4] 完成，弱标签验证 Macro-F1={metrics["macro_f1"]:.4f}（不是人工金标准成绩）\n{dest}', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--input', type=Path, default=CHUNKS)
    train.add_argument('--output-dir', type=Path, default=OUT_DIR/'c3_subjectivity_full')
    train.add_argument('--seed', type=int, default=42)
    pred = sub.add_parser('predict')
    pred.add_argument('--model', type=Path, required=True)
    pred.add_argument('--text', action='append', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'train':
            return train_command(args)
        print(predict_subjectivity(joblib.load(args.model), args.text).to_json(
            orient='records', force_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, ConvergenceWarning) as exc:
        print(f'错误：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
