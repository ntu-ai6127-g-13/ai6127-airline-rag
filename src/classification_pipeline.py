"""八方面、主客观、情感联合基线。各任务独立弱标签，共用父评论切分。

不把未知句当 other/objective，不为 other 伪造评分，不读取金标准答案。
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from src.config import ASPECTS, CHUNKS, DATA_EVAL, ROOT
from src.subjectivity import weak_decision
from src.subjectivity_legacy import weak_decision as legacy_weak_decision
from src.subjectivity_v3 import weak_decision as v3_weak_decision
from src.language_rules import choose_focus, polarity_evidence
from src.language_rules_v3 import choose_focus as v3_choose_focus, polarity_evidence as v3_polarity_evidence
from src.weak_label_rules import match_series
from src.weak_labels import load_test_exclusions, normalize_text, sha256, validate_ids
from src.training_guards import check_fit_boundary, reject_role_annotated_input

VERSION = 'eight-aspect-subjectivity-polarity-v4.1'
V3_VERSION = 'eight-aspect-subjectivity-polarity-v3'
SUPPORTED_VERSIONS = {VERSION, V3_VERSION, 'eight-aspect-subjectivity-polarity-v2', 'eight-aspect-subjectivity-polarity-v1'}
TASKS = {'aspect': [*ASPECTS, 'overall', 'other'],
         'subjectivity': ['objective', 'subjective'], 'polarity': ['negative', 'positive']}
QUALITY = ['is_route_only', 'is_non_english', 'is_fragment']
OTHER = re.compile(r'\b(?:lavator(?:y|ies)|toilets?|restrooms?|washrooms?|'
                   r'frequent[ -]flyer|loyalty (?:points|program(?:me)?)|air miles|mileage|'
                   r'upgrad(?:e|ed|es|ing)|airbus|boeing|[ab][2378]\d{2}(?:-\d{2,3})?)\b|'
                   r'\b(?:cabin|aircraft|plane) (?:was |is |were )?(?:very |really )?'
                   r'(?:clean|spotless|dirty|filthy)\b', re.I)
OVERALL = re.compile(r'\boverall\b|\b(?:whole|entire) (?:experience|trip|journey|flight)\b|'
                     r'\b(?:great|good|bad|terrible|awful|excellent|wonderful|amazing|disappointing) '
                     r'(?:experience|trip|journey|flight|airline)\b|'
                     r'\b(?:recommend|fly (?:with|them)|use this airline) again\b', re.I)
OFF_TOPIC = re.compile(r'(?:i|we) (?:went|was going|were going) to (?:a|my|the) '
                       r'(?:wedding|birthday party|conference)[.!]?', re.I)


def checked_input(path):
    """对整份输入（包括未选中的句子）核对当前 KB 与评测掩码。"""
    kb = pd.read_parquet(CHUNKS, columns=['chunk_id', 'review_id', 'text', 'is_eval'])
    reviews, texts, info = load_test_exclusions(kb, DATA_EVAL/'to_label_meta.csv', DATA_EVAL/'to_label.csv')
    frame = pd.read_parquet(path)
    validate_ids(frame, '八类输入')
    for col in ['is_eval', *QUALITY]:
        if col not in frame or not pd.api.types.is_bool_dtype(frame[col]) or frame[col].isna().any():
            raise ValueError(f'{col} 缺失或不是无空值布尔值')
    if not frame.text.map(lambda t: isinstance(t, str)).all():
        raise ValueError('输入原文必须是文本')
    joined = frame[['chunk_id', 'review_id', 'text', 'is_eval']].merge(
        kb, on='chunk_id', how='left', suffixes=('_input', '_kb'), validate='one_to_one')
    if any(not joined[c+'_input'].eq(joined[c+'_kb']).all() for c in ['review_id', 'text', 'is_eval']):
        raise ValueError('输入与当前 KB 的编号、原文、父评论或评测标记不一致')
    return frame, reviews, texts, info


def eligible_rows(frame, reviews, texts):
    for c in ['is_eval', *QUALITY]:
        if not pd.api.types.is_bool_dtype(frame[c]) or frame[c].isna().any():
            raise ValueError(f'{c} 必须是无空值布尔值')
    mask = ~frame.is_eval & ~frame.review_id.isin(reviews) & ~frame.text.map(normalize_text).isin(texts)
    mask &= frame.text.str.strip().ne('')
    for col in QUALITY:
        mask &= ~frame[col]
    return frame.loc[mask].copy().reset_index(drop=True)


def build_labels(frame, reviews, texts):
    reject_role_annotated_input(frame, '八类训练输入')
    validate_ids(frame, '八类输入')
    d = eligible_rows(frame, reviews, texts)
    matches, cleanliness = match_series(d.text)
    six_hits = matches.map(bool)
    count = six_hits.sum(axis=1)
    other = d.text.str.contains(OTHER)
    overall = d.text.str.contains(OVERALL)
    off_topic = d.text.map(lambda t: bool(OFF_TOPIC.fullmatch(normalize_text(t))))
    out = d[['chunk_id', 'review_id', 'text', 'is_eval']].copy()
    out['aspect'] = pd.Series(None, index=d.index, dtype=object)
    out['aspect_rule'] = 'abstain'
    # 泛泛 overall 与具体方面并存时保留具体方面；other 与具体方面混合则弃权。
    specific = count.eq(1) & ~other & ~cleanliness
    out.loc[specific, 'aspect'] = six_hits.idxmax(axis=1)[specific]
    out.loc[specific, 'aspect_rule'] = 'context_v4_single_aspect'
    overall_only = count.eq(0) & overall & ~other & ~cleanliness
    out.loc[overall_only, ['aspect', 'aspect_rule']] = ['overall', 'explicit_overall']
    other_only = count.eq(0) & other & ~overall
    out.loc[other_only, ['aspect', 'aspect_rule']] = ['other', 'explicit_other_topic']
    # 清洁规则可命中 food stuck to seats；有额外服务方面的混合句仍跳过。
    stuck_food = d.text.str.contains(r'\bfood\b[^.!?;]{0,25}\bstuck\b[^.!?;]{0,25}\bseats?\b', case=False, regex=True)
    clean_only = cleanliness & (count.eq(0) | (count.eq(1) & six_hits.seat) | stuck_food)
    clean_only &= ~six_hits[['staff', 'entertainment', 'ground', 'value']].any(axis=1)
    out.loc[clean_only, ['aspect', 'aspect_rule']] = ['other', 'cabin_cleanliness']
    out.loc[off_topic, ['aspect', 'aspect_rule']] = ['other', 'explicit_off_topic']
    decisions = [weak_decision(t) for t in d.text]
    out['subjectivity'] = [v[0] for v in decisions]
    out['subjectivity_rule'] = [v[1] for v in decisions]
    out.loc[out.subjectivity_rule.eq('fact_cleaning_event'), ['aspect','aspect_rule']] = ['other','cleaning_topic']
    out.loc[off_topic, ['subjectivity', 'subjectivity_rule']] = ['objective', 'project_off_topic']
    out['polarity'] = pd.Series(None, index=d.index, dtype=object)
    out['polarity_rule'] = 'abstain'
    # 情感训练只选明确主观且方面可判的句子。other 没有评分，保留为空。
    for aspect in ASPECTS:
        values = pd.to_numeric(d['r_'+aspect], errors='coerce')
        use = out.aspect.eq(aspect) & out.subjectivity.eq('subjective') & values.isin([1, 2, 4, 5])
        out.loc[use, 'polarity'] = values[use].map(lambda v: 'positive' if v >= 4 else 'negative')
        out.loc[use, 'polarity_rule'] = 'aspect_rating_1_2_negative_4_5_positive'
    rating = pd.to_numeric(d.overall_rating, errors='coerce')
    use = out.aspect.eq('overall') & out.subjectivity.eq('subjective') & rating.isin([1, 2, 3, 4, 7, 8, 9, 10])
    out.loc[use, 'polarity'] = rating[use].map(lambda v: 'positive' if v >= 7 else 'negative')
    out.loc[use, 'polarity_rule'] = 'overall_1_4_negative_7_10_positive'
    # 评分仍是原始弱监督来源。明确文本线索与评分冲突时弃权，不反写评分/人工答案。
    conflict_count=0
    for i in out.index[out.polarity.notna()]:
        cue,_=polarity_evidence(choose_focus(d.at[i,'text'])['text'])
        if cue is not None and cue != out.at[i,'polarity']:
            out.at[i,'polarity']=None
            out.at[i,'polarity_rule']='rating_text_conflict_abstain'
            conflict_count+=1
    unlabelled = out[list(TASKS)].isna().all(axis=1)
    out = out.loc[~unlabelled].copy()
    out['_norm'] = out.text.map(normalize_text)
    # 同文本评分冲突不能随意选一个；整组跳过，再去重。
    out['_signature'] = out[list(TASKS)].fillna('ABSTAIN').agg('|'.join, axis=1)
    conflict = out.groupby('_norm')._signature.transform('nunique').gt(1)
    conflicts = int(conflict.sum())
    out = out.loc[~conflict].sort_values('chunk_id')
    before = len(out)
    out = out.drop_duplicates('_norm').drop(columns=['_norm', '_signature']).reset_index(drop=True)
    out['source'] = 'weak_multitask'
    out['rule_version'] = VERSION
    # 显式可空字符串，避免含弃权空值的 object 列写入 Parquet 后隐式变类型。
    for task in TASKS:
        out[task] = out[task].astype('string')
    if out.empty:
        raise ValueError('没有可用弱标签')
    counts = {t: out[t].value_counts().to_dict() for t in TASKS}
    return out, {'input_rows': len(frame), 'eligible_rows': len(d), 'output_rows': len(out),
                 'all_tasks_abstained': int(unlabelled.sum()), 'conflicting_duplicate_rows': conflicts,
                 'duplicate_rows': before-len(out), 'class_counts': counts,
                 'rating_text_conflicts_abstained': conflict_count,
                 'polarity_rule_counts': out.polarity_rule.value_counts().to_dict()}


def split_labels(labels, seed=42, reference=None):
    reject_role_annotated_input(labels, '八类待切分标签')
    if labels.empty or not labels.chunk_id.is_unique or not labels.text.map(normalize_text).is_unique:
        raise ValueError('标签为空或编号/文本重复')
    for task, allowed in TASKS.items():
        if not labels[task].dropna().isin(allowed).all():
            raise ValueError(f'{task} 非法标签')
    if labels.loc[labels.polarity.notna(), 'subjectivity'].ne('subjective').any():
        raise ValueError('情感训练混入非主观句')
    ordered = labels.sort_values('chunk_id').reset_index(drop=True)
    if reference is None:
        ti, di = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(
            ordered, groups=ordered.review_id))
        train, dev = ordered.iloc[ti].copy(), ordered.iloc[di].copy()
    else:
        if (not reference.split.isin(['train','dev']).all() or
                reference.groupby('review_id').split.nunique().gt(1).any()):
            raise ValueError('参考切分非法或同一父评论跨集合')
        mapping=reference.drop_duplicates('review_id').set_index('review_id').split
        role=ordered.review_id.map(mapping)
        if role.isna().any():raise ValueError('有父评论不在参考切分中')
        train,dev=ordered.loc[role.eq('train')].copy(),ordered.loc[role.eq('dev')].copy()
    if set(train.review_id) & set(dev.review_id) or set(train.text.map(normalize_text)) & set(dev.text.map(normalize_text)):
        raise ValueError('训练/验证父评论或文本泄漏')
    for name, part in [('train', train), ('dev', dev)]:
        for task, allowed in TASKS.items():
            if set(part[task].dropna()) != set(allowed):
                raise ValueError(f'{name}/{task} 类别不足，请扩大开发样本；不查看金标准调切分')
    return train, dev


def score(actual, pred, labels):
    r = classification_report(actual, pred, labels=labels, output_dict=True, zero_division=0)
    return {'accuracy': float(accuracy_score(actual, pred)), 'macro_f1': r['macro avg']['f1-score'],
            'macro_precision': r['macro avg']['precision'], 'macro_recall': r['macro avg']['recall'],
            'per_class': {k: r[k] for k in labels},
            'confusion_matrix': confusion_matrix(actual, pred, labels=labels).tolist(),
            'confusion_labels': labels, 'axes': 'rows=weak reference; columns=prediction'}


def fit_models(train, dev, max_features=20000, max_iter=1000, seed=42):
    check_fit_boundary(train, dev)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, stop_words=None)
    x = vectorizer.fit_transform(train.text)
    xd = vectorizer.transform(dev.text)
    models, metrics = {}, {}
    for task, classes in TASKS.items():
        use, val = train[task].notna().to_numpy(), dev[task].notna().to_numpy()
        print(f'  {task}: 训练 {int(use.sum()):,} / 验证 {int(val.sum()):,}', flush=True)
        model = LogisticRegression(max_iter=max_iter, class_weight='balanced', random_state=seed)
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(x[use], train.loc[use, task])
        pred = model.predict(xd[val])
        metrics[task] = score(dev.loc[val, task], pred, classes)
        dummy = DummyClassifier(strategy='most_frequent').fit(x[use], train.loc[use, task])
        metrics[task]['majority_reference'] = score(dev.loc[val, task], dummy.predict(xd[val]), classes)
        metrics[task]['iterations'] = model.n_iter_.tolist()
        models[task] = model
    return {'version': VERSION, 'tasks': TASKS, 'vectorizer': vectorizer,
            'models': models, 'input_features': ['text']}, metrics


def predict_texts(bundle, texts, include_raw=False):
    if bundle.get('version') not in SUPPORTED_VERSIONS or bundle.get('tasks') != TASKS:
        raise ValueError('不是当前八类联合模型')
    if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        raise ValueError('输入至少一个非空句子')
    x = bundle['vectorizer'].transform(texts)
    raw_aspect = bundle['models']['aspect'].predict(x)
    raw_subjectivity = bundle['models']['subjectivity'].predict(x)
    out = pd.DataFrame({'text': texts, 'aspect': raw_aspect, 'subjectivity': raw_subjectivity})
    # v3 固定历史规则；v4.1 使用修正后的规则。中间 v4 候选已撤回，不接受它。
    v3 = bundle['version'] in {VERSION, V3_VERSION}
    focus_fn = choose_focus if bundle['version'] == VERSION else v3_choose_focus
    polarity_fn = polarity_evidence if bundle['version'] == VERSION else v3_polarity_evidence
    guarded = v3 or bundle['version'].endswith('-v2')
    focuses = [focus_fn(t) if v3 else {'text':t,'reason':'whole_sentence','aspect':None} for t in texts]
    focused_texts = [f['text'] for f in focuses]
    xf = bundle['vectorizer'].transform(focused_texts) if focused_texts != texts else x
    aspects = bundle['models']['aspect'].predict(xf).tolist() if v3 else raw_aspect.tolist()
    subjective = raw_subjectivity.tolist()
    subject_sources = ['model'] * len(texts)
    aspect_sources = ['model:focus' if f['text'] != t else 'model' for f,t in zip(focuses,texts)]
    if guarded:
        decision_fn = (weak_decision if bundle['version'] == VERSION else v3_weak_decision) if v3 else legacy_weak_decision
        for i,text in enumerate(texts):
            if OFF_TOPIC.fullmatch(normalize_text(text)):
                aspects[i],subjective[i],subject_sources[i] = 'other','objective','rule:off_topic'
                aspect_sources[i]='rule:off_topic'
                continue
            label,reason=decision_fn(focused_texts[i])
            if label is not None:
                subjective[i],subject_sources[i]=label,'rule:'+reason
            if v3 and focuses[i]['aspect']:
                aspects[i],aspect_sources[i]=focuses[i]['aspect'],'rule:'+focuses[i]['reason']
            if v3 and reason=='fact_cleaning_event':
                aspects[i],aspect_sources[i]='other','rule:cleaning_topic'
        out['subjectivity_source']=subject_sources
    out['aspect'],out['subjectivity']=aspects,subjective
    out['polarity']='NA'
    mask=out.subjectivity.eq('subjective').to_numpy()
    if mask.any():out.loc[mask,'polarity']=bundle['models']['polarity'].predict(xf[mask])
    if v3:
        polarities=out.polarity.tolist();sources=['model:focus' if f['text']!=t else 'model' for f,t in zip(focuses,texts)]
        for i,f in enumerate(focuses):
            if not mask[i]:sources[i]='objective_gate';continue
            cue,reason=polarity_fn(f['text'])
            if cue is not None:polarities[i],sources[i]=cue,'rule:'+reason
        out['polarity']=polarities
        out['aspect_source']=aspect_sources
        out['polarity_source']=sources
        out['focus_text']=focused_texts
        out['focus_reason']=[f['reason'] for f in focuses]
    if include_raw:
        out['raw_aspect']=raw_aspect
        out['raw_subjectivity']=raw_subjectivity
        out['raw_polarity']=bundle['models']['polarity'].predict(x)
    return out


def train_command(args):
    start = time.perf_counter()
    dest = args.output_dir.resolve()
    if dest.exists():
        raise ValueError(f'输出目录已存在：{dest}；请用新目录，禁止覆盖旧实验')
    if args.max_features < 1 or args.max_iter < 1:
        raise ValueError('训练参数必须大于零')
    paths = {'input': args.input.resolve(), 'kb': CHUNKS,
             'eval_meta': DATA_EVAL/'to_label_meta.csv', 'eval_text': DATA_EVAL/'to_label.csv'}
    reference_path=getattr(args,'split_reference',None)
    if reference_path is not None:paths['split_reference']=reference_path.resolve()
    inputs = {n: {'path': str(p), 'sha256': sha256(p)} for n, p in paths.items()}
    print('[1/4] 核验数据，生成独立的三个任务弱标签...', flush=True)
    frame, reviews, texts, eval_info = checked_input(paths['input'])
    labels, generation = build_labels(frame, reviews, texts)
    reference=None
    if reference_path is not None:
        reference=pd.read_parquet(reference_path)
        if (not reference.chunk_id.is_unique or not reference.split.isin(['train','dev']).all() or
                reference.groupby('review_id').split.nunique().gt(1).any()):
            raise ValueError('参考切分非法')
        before=len(labels)
        labels=labels.loc[labels.review_id.isin(reference.review_id)].copy()
        # 防止排重保留位置改变后，旧开发文本出现在新训练侧。
        kb_text=pd.read_parquet(CHUNKS,columns=['chunk_id','text'])
        old_dev_text=set(kb_text.loc[kb_text.chunk_id.isin(reference.loc[reference.split.eq('dev'),'chunk_id']),'text'].map(normalize_text))
        old_train_reviews=set(reference.loc[reference.split.eq('train'),'review_id'])
        bad=labels.review_id.isin(old_train_reviews) & labels.text.map(normalize_text).isin(old_dev_text)
        bad_reviews=set(labels.loc[bad,'review_id'])
        labels=labels.loc[~labels.review_id.isin(bad_reviews)].reset_index(drop=True)
        generation['fixed_split_removed_rows']=before-len(labels)
        generation['old_dev_duplicate_blocked_train_reviews']=len(bad_reviews)
        generation['output_rows']=len(labels)
        generation['class_counts']={t:labels[t].value_counts().to_dict() for t in TASKS}
    if labels.is_eval.any() or labels.review_id.isin(reviews).any() or labels.text.map(normalize_text).isin(texts).any():
        raise ValueError('评测数据泄漏，拒绝训练')
    train, dev = split_labels(labels, args.seed, reference)
    boundary_checks = check_fit_boundary(train, dev, eval_reviews=reviews, eval_texts=texts)
    print('[2/4] 共用训练/验证父评论划分，训练三个模型...', flush=True)
    bundle, metrics = fit_models(train, dev, args.max_features, args.max_iter, args.seed)
    predicted = predict_texts(bundle, dev.text.tolist(), include_raw=True)
    dev_output = dev[['chunk_id', 'review_id', 'text', *TASKS]].rename(
        columns={t: 'weak_'+t for t in TASKS}).reset_index(drop=True)
    for t in TASKS:
        dev_output['predicted_'+t] = predicted[t]
        dev_output['raw_'+t] = predicted['raw_'+t]
    for c in ['subjectivity_source','aspect_source','polarity_source','focus_text','focus_reason']:
        dev_output[c]=predicted[c]
    final_metrics={}
    for t,classes in TASKS.items():
        valid=dev[t].notna().to_numpy()
        # polarity 把 NA 作为可能的预测错误保留在混淆矩阵，不加入目标宏平均类别。
        actual=dev.loc[valid,t].to_numpy();actual_pred=predicted.loc[valid,t].to_numpy()
        final_metrics[t]=score(actual,actual_pred,classes)
        if t=='polarity':
            final_metrics[t]['gate_NA_errors']=int((actual_pred=='NA').sum())
            final_metrics[t]['confusion_matrix']=confusion_matrix(actual,actual_pred,labels=classes+['NA']).tolist()
            final_metrics[t]['confusion_labels']=classes+['NA']
    polar_mask = dev.polarity.notna().to_numpy()
    gated_accuracy = float((dev.loc[polar_mask, 'polarity'].to_numpy() == predicted.loc[polar_mask, 'polarity'].to_numpy()).mean())
    report = {'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
              'evaluation_kind': 'held_out_weak_labels_not_human_gold', 'generation': generation,
              'evaluation_exclusion': eval_info, 'inputs': inputs, 'metrics': metrics,
              'final_metrics': final_metrics,
              'weak_reference_warning': 'final rules overlap weak label construction; final weak scores are rule consistency, not independent accuracy',
              'gated_polarity_accuracy_on_weak_subjective_rows': gated_accuracy,
              'inference_policy': 'R16 focus + auditable bounded subjectivity/polarity rules, model fallback; original text preserved',
              'metrics_policy': 'metrics=raw models on original sentences; final_metrics=full pipeline; errors separated by view',
              'dev_subjectivity_sources': predicted.subjectivity_source.value_counts().to_dict(),
              'split': {'method': 'frozen reference review split' if reference is not None else 'shared GroupShuffleSplit by review_id', 'seed': args.seed,
                        'train_rows': len(train), 'dev_rows': len(dev),
                        'train_reviews': int(train.review_id.nunique()), 'dev_reviews': int(dev.review_id.nunique()),
                        'class_counts': {name: {t: part[t].value_counts().to_dict() for t in TASKS}
                                         for name, part in [('train', train), ('dev', dev)]}},
              'hyperparameters': {'max_features': args.max_features, 'max_iter': args.max_iter,
                                  'ngram_range': [1, 2], 'class_weight': 'balanced', 'C': 1, 'tuned_on_dev': False},
              'runtime': {'sklearn': sklearn.__version__, 'pandas': pd.__version__},
              'implementation_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                    [Path(__file__).resolve(), ROOT/'src/train_classifiers.py', ROOT/'src/subjectivity.py',
                     ROOT/'src/weak_labels.py', ROOT/'src/weak_label_rules.py', ROOT/'src/config.py',
                     ROOT/'src/language_rules.py', ROOT/'src/subjectivity_legacy.py',
                     ROOT/'src/training_guards.py']},
              'checks': {'zero_eval_parent_id_text_overlap': True, 'zero_train_dev_parent_text_overlap': True,
                         'shared_split_for_all_tasks': True, 'tfidf_train_only': True,
                         'no_polarity_targets_for_objective': not labels.loc[labels.subjectivity.eq('objective'), 'polarity'].notna().any(),
                         'no_fabricated_other_rating': not labels.loc[labels.aspect.eq('other'), 'polarity'].notna().any(),
                         'objective_predictions_have_NA': predicted.loc[predicted.subjectivity.eq('objective'), 'polarity'].eq('NA').all().item()},
              'limits': ['三个任务均用带噪声的规则/评分弱监督，开发指标不是人工金标准成绩。',
                         '未命中方面或主客观规则时弃权，绝不默认 other/objective。',
                         'overall 极性阈值为本地实验约定：1–4 负面、7–10 正面、5–6 弃权。',
                         'other 无极性训练标签；预测其情感依赖其他主题学到的模型泛化，未验证。',
                         '客观弱标签仅覆盖严格事实模板，分布偏差与覆盖不足待人工补标。',
                         '显式 other 只覆盖部分主题，不是完整的无关话题检测器。',
                         '多方面和复杂否定/转折没有完整语义处理；仍需人工金标准评测。']}
    report['checks'].update(boundary_checks)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.eightclass_', dir=dest.parent))
    try:
        print('[3/4] 保存模型并回读核验...', flush=True)
        bundle['training_inputs'] = inputs
        bundle['train_review_ids'] = sorted(train.review_id.unique())
        bundle['dev_review_ids'] = sorted(dev.review_id.unique())
        joblib.dump(bundle, temp/'model.joblib', compress=3)
        restored = joblib.load(temp/'model.joblib')
        pd.testing.assert_frame_equal(predicted, predict_texts(restored, dev.text.tolist(), include_raw=True))
        labels.to_parquet(temp/'weak_labels.parquet', index=False)
        pd.testing.assert_frame_equal(labels, pd.read_parquet(temp/'weak_labels.parquet'))
        pd.concat([train.assign(split='train'), dev.assign(split='dev')])[
            ['chunk_id', 'review_id', 'split']].to_parquet(temp/'split.parquet', index=False)
        dev_output.to_parquet(temp/'dev_predictions.parquet', index=False)
        audits = []
        for t in TASKS:
            for label, g in labels.loc[labels[t].notna()].groupby(t):
                a = g.sample(min(15, len(g)), random_state=args.seed).copy()
                a['audit_task'], a['weak_label'], a['human_label'], a['note'] = t, label, '', ''
                audits.append(a)
        pd.concat(audits, ignore_index=True).to_csv(temp/'人工抽查.csv', index=False, encoding='utf-8-sig')
        report['error_counts']={}
        for view,prefix in [('raw','raw_'),('final','predicted_')]:
            errors=[];report['error_counts'][view]={}
            for t in TASKS:
                bad=dev_output.loc[dev_output['weak_'+t].notna() & dev_output['weak_'+t].ne(dev_output[prefix+t])]
                report['error_counts'][view][t]=len(bad)
                errors.append(bad.groupby('weak_'+t).head(20).assign(task=t,view=view))
            pd.concat(errors,ignore_index=True).to_csv(temp/('errors_'+view+'.csv'),index=False,encoding='utf-8-sig')
        report['checks'].update(model_reload_identical=True, labels_roundtrip_identical=True,
                                inputs_unchanged=all(sha256(p) == inputs[n]['sha256'] for n, p in paths.items()))
        if not all(report['checks'].values()):
            raise ValueError('最终核验失败')
        report['elapsed_seconds'] = round(time.perf_counter()-start, 2)
        report['artifacts_sha256'] = {p.name: sha256(p) for p in temp.iterdir()}
        (temp/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if dest.exists():
            raise ValueError('输出目录在运行期间出现，拒绝覆盖')
        temp.rename(dest)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    print('[4/4] 完成（以下仅为弱标签验证分数）：', flush=True)
    for t, m in metrics.items():
        print(f'  {t}: macro-F1={m["macro_f1"]:.4f}', flush=True)
    print(dest, flush=True)
    return 0


def export_command(args):
    """给本地联调输出合格的非评测句，保留模型输出来源，绝不覆盖知识库。"""
    run_start = time.perf_counter()
    destination = args.output.resolve()
    report_path = destination.with_suffix('.report.json')
    if destination.exists() or report_path.exists():
        raise ValueError('预测输出或报告已存在，请换新路径')
    if args.batch_size < 1:
        raise ValueError('batch-size 必须大于零')
    inputs = {k: {'path': str(p.resolve()), 'sha256': sha256(p)} for k, p in
              {'model': args.model, 'input': args.input, 'kb': CHUNKS,
               'eval_meta': DATA_EVAL/'to_label_meta.csv', 'eval_text': DATA_EVAL/'to_label.csv'}.items()}
    bundle = joblib.load(args.model)
    for key in ['kb', 'eval_meta', 'eval_text']:
        if bundle.get('training_inputs', {}).get(key, {}).get('sha256') != inputs[key]['sha256']:
            raise ValueError('模型与当前知识库/评测清单版本不一致，拒绝批量预测')
    frame, reviews, texts, eval_info = checked_input(args.input)
    eligible = eligible_rows(frame, reviews, texts)
    if eligible.empty:
        raise ValueError('没有合格的非评测句可预测')
    parts = []
    for start in range(0, len(eligible), args.batch_size):
        batch = eligible.iloc[start:start+args.batch_size]
        pred = predict_texts(bundle, batch.text.tolist())
        pred.insert(0, 'review_id', batch.review_id.to_numpy())
        pred.insert(0, 'chunk_id', batch.chunk_id.to_numpy())
        parts.append(pred)
        if start % (args.batch_size * 8) == 0:
            print(f'预测 {min(start+len(batch),len(eligible)):,}/{len(eligible):,}', flush=True)
    out = pd.concat(parts, ignore_index=True)
    out['is_eval'], out['source'], out['model_version'] = False, 'hybrid_prediction', bundle['version']
    out['parent_split'] = 'unlabelled_parent'
    out.loc[out.review_id.isin(bundle['train_review_ids']), 'parent_split'] = 'training_parent'
    out.loc[out.review_id.isin(bundle['dev_review_ids']), 'parent_split'] = 'development_parent'
    checks = {'unique_ids': out.chunk_id.is_unique, 'row_count_matches': len(out) == len(eligible),
              'zero_eval_parent_text_overlap': not out.review_id.isin(reviews).any() and not out.text.map(normalize_text).isin(texts).any(),
              'legal_aspects': bool(out.aspect.isin(TASKS['aspect']).all()),
              'legal_subjectivity': bool(out.subjectivity.isin(TASKS['subjectivity']).all()),
              'objective_is_NA': bool(out.loc[out.subjectivity.eq('objective'), 'polarity'].eq('NA').all()),
              'subjective_has_binary_polarity': bool(out.loc[out.subjectivity.eq('subjective'), 'polarity'].isin(TASKS['polarity']).all()),
              'inputs_unchanged': all(sha256(Path(i['path'])) == i['sha256'] for i in inputs.values())}
    if not all(checks.values()):
        raise ValueError(f'批量预测校验失败：{checks}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.predictions_', dir=destination.parent))
    try:
        out.to_parquet(temp/'predictions.parquet', index=False)
        pd.testing.assert_frame_equal(out, pd.read_parquet(temp/'predictions.parquet'))
        checks['roundtrip_identical'] = True
        report = {'model_version': bundle['version'], 'created_at': datetime.now(timezone.utc).isoformat(),
                  'input_rows': len(frame), 'output_rows': len(out), 'excluded_rows': len(frame)-len(out),
                  'filters': ['eval sentences and entire eval parent reviews', 'duplicate eval text',
                              'empty, route-only, obvious non-English, fragments'],
                  'counts': {t: out[t].value_counts().to_dict() for t in TASKS},
                  'checks': checks, 'evaluation_exclusion': eval_info, 'inputs': inputs,
                  'elapsed_seconds': round(time.perf_counter()-run_start, 2),
                  'implementation_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                       [Path(__file__).resolve(), ROOT/'src/subjectivity.py', ROOT/'src/subjectivity_v3.py',
                        ROOT/'src/subjectivity_legacy.py', ROOT/'src/language_rules.py', ROOT/'src/language_rules_v3.py',
                        ROOT/'src/weak_label_rules.py', ROOT/'src/weak_label_rules_v3.py', ROOT/'src/weak_labels.py']},
                  'training_base_version': bundle.get('base_model_version', bundle['version']),
                  'model_retrained_for_rule_revision': bundle.get('model_retrained_for_rule_revision'),
                  'output_sha256': sha256(temp/'predictions.parquet'),
                  'purpose': 'local integration draft, not evaluation results or human labels'}
        (temp/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if destination.exists() or report_path.exists():
            raise ValueError('运行中目标出现，拒绝覆盖')
        (temp/'predictions.parquet').rename(destination)
        (temp/'report.json').rename(report_path)
    finally:
        shutil.rmtree(temp)
    print(f'完成 {len(out):,} 条本地联调预测：{destination}', flush=True)
    return 0
