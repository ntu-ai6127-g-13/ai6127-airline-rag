"""固定既有模型，仅补非金标准兄弟句推理并另存 v2 交接包；不训练、不改 GitHub。"""
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
from scripts.check_c_b_sync import audit_package as audit_prior_package
from scripts.prepare_c_b_sync import (b_counts, fitted_state_hash, verify_revision_provenance)
from scripts.prepare_c_handoff import (KB_COLUMNS, LABEL_COLUMNS as OLD_LABEL_COLUMNS,
                                      GUIDE_PATH, validate_kb)
from src.classification_pipeline import predict_texts
from src.c_label_reader_v2 import (BASE_METADATA_COLUMNS, COVERAGE_COLUMNS, HELDOUT_PARENT_ROLE,
    INFERENCE_ORIGINS, LABEL_COLUMNS, METADATA_COLUMNS, SCHEMA_VERSION, SKIP_REASONS, VERSION,
    _validate_table, load_b_labels)
from src.weak_labels import load_test_exclusions, normalize_text, sha256, validate_ids


def inference_reasons(kb, eval_texts):
    """推理单独选范围；不使用或修改训练 eligible_rows。"""
    validate_kb(kb)
    reason = pd.Series(pd.NA, index=kb.index, dtype='string', name='skip_reason')
    masks = [('eval_sentence', kb.is_eval),
             ('eval_duplicate_text', kb.text.map(normalize_text).isin(eval_texts)),
             ('empty', kb.text.str.strip().eq('')), ('route_only', kb.is_route_only),
             ('non_english', kb.is_non_english), ('fragment', kb.is_fragment)]
    for name, mask in masks:
        reason.loc[reason.isna() & mask] = name
    return reason


def select_supplement(kb, old_labels, eval_reviews, eval_texts, train_reviews, dev_reviews):
    reasons = inference_reasons(kb, eval_texts)
    validate_ids(old_labels, '原有标签')
    target_ids = set(kb.loc[reasons.isna(), 'chunk_id'])
    if not set(old_labels.chunk_id).issubset(target_ids):
        raise ValueError('原有标签不全在本次合法推理范围内')
    if set(train_reviews) & set(dev_reviews) or (set(train_reviews) | set(dev_reviews)) & set(eval_reviews):
        raise ValueError('模型训练/开发父评论名单与评测隔离冲突')
    added = kb.loc[reasons.isna() & ~kb.chunk_id.isin(old_labels.chunk_id)].copy().reset_index(drop=True)
    if added.empty:
        raise ValueError('没有缺失兄弟句，无需重复预测')
    if (not added.review_id.isin(eval_reviews).all() or added.is_eval.any()
            or added.text.map(normalize_text).isin(eval_texts).any()
            or added.review_id.isin(set(train_reviews) | set(dev_reviews)).any()):
        raise ValueError('补预测范围不是隔离父评论中的合格非金标准兄弟句')
    return added


def supplemental_labels(rows, predictions):
    if len(rows) != len(predictions) or not set(BASE_METADATA_COLUMNS[:3]).issubset(predictions):
        raise ValueError('预测行数或任务输出不一致')
    out = pd.DataFrame({'chunk_id': rows.chunk_id.to_numpy(), 'review_id': rows.review_id.to_numpy(),
                        'is_eval_parent': True})
    for column in ['aspect', 'subjectivity', 'polarity', 'aspect_source', 'subjectivity_source', 'polarity_source']:
        out[column] = predictions[column].to_numpy()
    out['model_version'], out['source'] = VERSION, 'hybrid_prediction'
    out['parent_split'], out['prediction_origin'] = HELDOUT_PARENT_ROLE, 'supplemental_eval_parent'
    for column in ['chunk_id', 'review_id', *METADATA_COLUMNS]:
        out[column] = out[column].astype('string')
    return out[LABEL_COLUMNS]


def make_extended_tables(kb, old_labels, added, eval_reviews, eval_texts):
    old = old_labels.copy()
    old['is_eval_parent'], old['prediction_origin'] = False, 'existing_v4_1'
    old['prediction_origin'] = old.prediction_origin.astype('string')
    labels = pd.concat([old[LABEL_COLUMNS], added[LABEL_COLUMNS]], ignore_index=True)
    reasons = inference_reasons(kb, eval_texts)
    coverage = kb[['chunk_id', 'review_id', 'is_eval']].copy()
    coverage['is_eval_parent'] = kb.review_id.isin(eval_reviews)
    coverage['prediction_status'] = reasons.isna().map({True: 'predicted', False: 'skipped'}).astype('string')
    coverage['skip_reason'] = reasons
    coverage = coverage.merge(labels, on=['chunk_id', 'review_id', 'is_eval_parent'], how='left', validate='one_to_one')
    for column in ['chunk_id', 'review_id', 'prediction_status', 'skip_reason', *METADATA_COLUMNS]:
        coverage[column] = coverage[column].astype('string')
    coverage = coverage[COVERAGE_COLUMNS]
    validate_extended_tables(kb, old_labels, labels, added, coverage, eval_reviews, eval_texts)
    b_coverage = coverage.loc[(~kb.is_route_only & ~kb.is_non_english).to_numpy()].reset_index(drop=True)
    non_eval = coverage.loc[~coverage.is_eval].reset_index(drop=True)
    return labels, coverage, b_coverage, non_eval


def validate_extended_tables(kb, old_labels, labels, added, coverage, eval_reviews, eval_texts):
    validate_kb(kb)
    validate_ids(labels, '扩展标签')
    validate_ids(added, '新增标签')
    _validate_table(coverage)
    if list(labels.columns) != LABEL_COLUMNS or list(added.columns) != LABEL_COLUMNS:
        raise ValueError('扩展标签字段不一致')
    if coverage.chunk_id.tolist() != kb.chunk_id.tolist() or len(coverage) != len(kb):
        raise ValueError('完整覆盖表未逐 ID 对齐 KB')
    if (not coverage.review_id.eq(kb.review_id.to_numpy()).all()
            or not coverage.is_eval.eq(kb.is_eval.to_numpy()).all()
            or not coverage.is_eval_parent.eq(kb.review_id.isin(eval_reviews).to_numpy()).all()):
        raise ValueError('覆盖表父评论、评测句或评测父评论标记变化')
    expected_reason = inference_reasons(kb, eval_texts)
    if not coverage.skip_reason.fillna('').eq(expected_reason.fillna('').to_numpy()).all():
        raise ValueError('覆盖表跳过理由与独立推理范围不一致')
    if not coverage.prediction_status.eq(expected_reason.isna().map({True: 'predicted', False: 'skipped'}).to_numpy()).all():
        raise ValueError('覆盖表状态不一致')
    if set(labels.chunk_id) != set(kb.loc[expected_reason.isna(), 'chunk_id']):
        raise ValueError('标签未精确覆盖合法非金标准推理范围')
    actual = coverage.loc[coverage.prediction_status.eq('predicted'), LABEL_COLUMNS]
    pd.testing.assert_frame_equal(actual.sort_values('chunk_id').reset_index(drop=True),
                                  labels.sort_values('chunk_id').reset_index(drop=True))
    previous = labels.loc[labels.prediction_origin.eq('existing_v4_1'), OLD_LABEL_COLUMNS]
    pd.testing.assert_frame_equal(previous.reset_index(drop=True), old_labels.reset_index(drop=True))
    actual_added = labels.loc[labels.prediction_origin.eq('supplemental_eval_parent'), LABEL_COLUMNS]
    pd.testing.assert_frame_equal(actual_added.reset_index(drop=True), added.reset_index(drop=True))
    expected_added = set(kb.loc[expected_reason.isna(), 'chunk_id']) - set(old_labels.chunk_id)
    if set(added.chunk_id) != expected_added:
        raise ValueError('新增标签不是缺失 ID 的精确差集')
    mapped = added[['chunk_id', 'review_id']].merge(kb[['chunk_id', 'review_id', 'is_eval', 'text']],
        on='chunk_id', how='left', suffixes=('_label', '_kb'), validate='one_to_one')
    if (not mapped.review_id_label.eq(mapped.review_id_kb).all() or mapped.is_eval.any()
            or not mapped.review_id_kb.isin(eval_reviews).all()
            or mapped.text.map(normalize_text).isin(eval_texts).any()):
        raise ValueError('新增标签的父评论或金标准隔离不正确')


def validate_jsonl(path, labels):
    """逐记录验证 JSONL 是同一标签镜像；不把 NA 转成 null，不编造 confidence。"""
    count = 0
    with Path(path).open(encoding='utf-8') as handle:
        for values in labels.itertuples(index=False, name=None):
            line = handle.readline()
            if not line:
                raise ValueError('JSONL 丢行')
            row = json.loads(line)
            expected = {**dict(zip(LABEL_COLUMNS, values)), 'confidence': None}
            if row != expected:
                raise ValueError('JSONL 与标签不一致或置信度被编造')
            count += 1
        if handle.readline():
            raise ValueError('JSONL 多行')
    return count


def repackage_existing(package, destination, historical_sources):
    """Copy verified prediction bytes with one guide; never load or run a model."""
    from scripts.check_c_b_extended import (ARTIFACTS, HISTORY_FILE, PACKAGING_SOURCES,
                                          audit_package)
    package, destination = Path(package).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('单指南输出目录已存在，拒绝覆盖；不重复推理或重打包')
    if not package.is_relative_to(ROOT) or not destination.is_relative_to(ROOT):
        raise ValueError('单指南整理来源与输出必须位于项目内')
    if historical_sources is None:
        raise ValueError('整理历史包必须显式指定 --historical-sources')
    historical_sources = Path(historical_sources).resolve()
    original_path = package/'manifest.json'
    original_hash = sha256(original_path)
    original = json.loads(original_path.read_text(encoding='utf-8'))
    if 'documentation_repack' in original:
        raise ValueError('请使用原始补预测包，不重复整理已合并的包')
    audit_package(package, historical_sources=historical_sources)
    history_hash = sha256(historical_sources)
    source_hashes = {name: sha256(ROOT/name) for name in PACKAGING_SOURCES}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.c_b_single_guide_', dir=destination.parent))
    try:
        for name in sorted(ARTIFACTS - {'README.md'}):
            shutil.copy2(package/name, temporary/name)
            if sha256(temporary/name) != original['artifacts_sha256'][name]:
                raise ValueError(f'复制后数据变化：{name}')
        shutil.copy2(ROOT/GUIDE_PATH, temporary/'README.md')
        shutil.copy2(historical_sources, temporary/HISTORY_FILE)
        manifest = json.loads(json.dumps(original))
        manifest['documentation_repack'] = {
            'version': 'single-guide-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
            'original_package': {'path': str(package), 'manifest_sha256': original_hash},
            'historical_sources_file': HISTORY_FILE, 'historical_sources_sha256': history_hash,
            'current_packaging_sources_sha256': source_hashes,
            'new_inference': False, 'model_training': False, 'data_files_unchanged': True,
        }
        manifest['artifacts_sha256'] = {name: sha256(temporary/name)
                                       for name in sorted(ARTIFACTS | {HISTORY_FILE})}
        (temporary/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n',
                                             encoding='utf-8')
        audit_package(temporary)
        if (sha256(original_path) != original_hash or sha256(historical_sources) != history_hash
                or any(sha256(ROOT/name) != digest for name, digest in source_hashes.items())):
            raise ValueError('整理期间原包、历史来源或当前包装程序变化')
        if destination.exists():
            raise ValueError('整理期间输出目录出现，拒绝覆盖')
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps({'package': str(destination), 'operation': 'documentation_only_repack',
                      'prediction_rows': original['jsonl_interface']['rows'],
                      'new_predictions_performed': False, 'model_training_performed': False},
                     ensure_ascii=False, indent=2))
    return manifest


def run(args):
    started = time.perf_counter()
    destination = args.output_dir.resolve()
    if destination.exists():
        raise ValueError(f'补预测包已存在，拒绝覆盖或重复推理：{destination}')
    if args.batch_size < 1:
        raise ValueError('batch-size 必须大于零')
    prior = args.prior_package.resolve()
    prior_audit = audit_prior_package(prior, historical_sources=getattr(args, 'historical_sources', None))
    prior_manifest = json.loads((prior/'manifest.json').read_text(encoding='utf-8'))
    inputs = dict(prior_manifest['inputs'])
    inputs['prior_manifest'] = {'path': str(prior/'manifest.json'), 'sha256': sha256(prior/'manifest.json')}
    paths = {key: Path(item['path']).resolve() for key, item in inputs.items()}
    hashes = {key: item['sha256'] for key, item in inputs.items()}
    prediction_report = json.loads(paths['prediction_report'].read_text(encoding='utf-8'))
    revision = json.loads(paths['revision_report'].read_text(encoding='utf-8'))
    training = json.loads(paths['training_report'].read_text(encoding='utf-8'))
    bundle, base = joblib.load(paths['model']), joblib.load(paths['base_model'])
    verify_revision_provenance(prediction_report, revision, training, bundle, base, hashes)
    for name, expected in prediction_report['implementation_sha256'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError(f'现行推理代码与已有 v4.1 预测来源不同：{name}')
    kb = pd.read_parquet(paths['kb'], columns=KB_COLUMNS)
    reviews, texts, evaluation = load_test_exclusions(kb, paths['eval_meta'], paths['eval_text'])
    old_labels = pd.read_parquet(prior/'labels.parquet')
    rows = select_supplement(kb, old_labels, reviews, texts, bundle['train_review_ids'], bundle['dev_review_ids'])
    before = fitted_state_hash(bundle)
    source_paths = [Path(__file__).resolve(), ROOT/'scripts/check_c_b_extended.py', ROOT/'src/c_label_reader_v2.py',
                    ROOT/GUIDE_PATH, *[ROOT/name for name in prediction_report['implementation_sha256']]]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    lock = {'locked_at': datetime.now(timezone.utc).isoformat(), 'model_sha256': hashes['model'],
            'fitted_state_hash': before, 'rule_source_sha256': prediction_report['implementation_sha256'],
            'parameters': {'batch_size': args.batch_size, 'model_version': VERSION, 'new_training': False},
            'scope': 'only eligible non-gold held-out parent sentences absent from prior predictions',
            'final_model_selection_or_w5_freeze': False}
    print(f'仅补 {len(rows):,} 句；原有 {len(old_labels):,} 句标签复用；金标准不预测。', flush=True)
    inference_started = time.perf_counter()
    parts = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows.iloc[start:start+args.batch_size]
        parts.append(supplemental_labels(batch, predict_texts(bundle, batch.text.tolist())))
        print(f'新增预测 {min(start+len(batch),len(rows)):,}/{len(rows):,}', flush=True)
    inference_seconds = round(time.perf_counter()-inference_started, 3)
    added = pd.concat(parts, ignore_index=True)
    if fitted_state_hash(bundle) != before:
        raise ValueError('推理改变模型权重或词表')
    labels, coverage, b_coverage, non_eval = make_extended_tables(kb, old_labels, added, reviews, texts)
    manifest = {
        'schema_version': SCHEMA_VERSION, 'status': 'candidate', 'model_version': VERSION,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'training_base_version': prior_manifest['training_base_version'], 'model_retrained': False,
        'new_inference': True, 'original_prediction_rows': len(old_labels), 'added_prediction_rows': len(added),
        'purpose': 'local extended inference handoff; not human gold, final acceptance or completed B integration',
        'source_package': {'path': str(prior), 'manifest_sha256': hashes['prior_manifest']}, 'inputs': inputs,
        'inference_lock': lock, 'inference_elapsed_seconds': inference_seconds,
        'inference_origin_counts': {origin: int(labels.prediction_origin.eq(origin).sum()) for origin in INFERENCE_ORIGINS},
        'b_pool': {**prior_manifest['b_pool'], **b_counts(b_coverage)},
        'coverage': {**b_counts(coverage), 'eval_sentence_rows': int(kb.is_eval.sum()),
                     'eval_parent_rows': int(kb.review_id.isin(reviews).sum())},
        'non_eval_coverage': b_counts(non_eval), 'evaluation_exclusion': evaluation,
        'schemas': {name: {column: str(frame[column].dtype) for column in frame} for name, frame in
                    [('labels.parquet', labels), ('supplemental_predictions.parquet', added),
                     ('coverage.parquet', coverage), ('b_coverage.parquet', b_coverage), ('non_eval_coverage.parquet', non_eval)]},
        'implementation_sha256_at_inference': source_hashes,
        'jsonl_interface': {'file': 'predictions.jsonl', 'rows': len(labels), 'fields': [*LABEL_COLUMNS, 'confidence'],
                            'confidence': 'null: no calibrated confidence provided; never a fabricated probability',
                            'objective_polarity': 'literal string NA'},
        'checks': {'prior_package_verified': True, 'existing_predictions_unchanged': True,
                   'supplement_is_exact_missing_id_set': True, 'supplement_only_non_gold_eval_parents': True,
                   'supplement_train_dev_parent_overlap_zero': True, 'supplement_normalized_gold_text_overlap_zero': True,
                   'model_fitted_state_unchanged': True, 'human_gold_answers_read': False,
                   'training_performed': False, 'original_text_or_ratings_copied': False},
        'acceptance': dict(prior_manifest['acceptance']),
        'limits': ['B 实际索引编号及标签读取/检索联调尚未验证；未发送本包。',
                   '527,912 句有预测，不声称全部 545,039 句均有标签；17,127 句保留质量跳过理由。',
                   '1,100 句金标准和 19 句碎片仍未预测；金标准答案未读取。',
                   '所有评测父评论及本次补预测严禁进入训练、开发调参、规则修正或模型选择。',
                   '本次锁定来源仅为复现这一轮候选推理，不代表正式最终冻结；弱监督噪声及原有模型局限仍保留。',
                   'JSONL/Parquet、八方面、confidence 空值和缺失策略仍需 B 确认；没有人工准确率结论。'],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.c_b_supplement_', dir=destination.parent))
    try:
        for name, frame in [('labels.parquet', labels), ('supplemental_predictions.parquet', added),
                            ('coverage.parquet', coverage), ('b_coverage.parquet', b_coverage), ('non_eval_coverage.parquet', non_eval)]:
            frame.to_parquet(temporary/name, index=False)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(temporary/name))
        labels.assign(confidence=None).to_json(temporary/'predictions.jsonl', orient='records', lines=True, force_ascii=False)
        validate_jsonl(temporary/'predictions.jsonl', labels)
        shutil.copy2(ROOT/'src/c_label_reader_v2.py', temporary/'read_labels.py')
        shutil.copy2(ROOT/GUIDE_PATH, temporary/'README.md')
        shutil.copy2(prior/'b_retriever_source.py', temporary/'b_retriever_source.py')
        manifest['checks'].update(parquet_roundtrip_identical=True, jsonl_exact_label_mirror=True)
        if (any(sha256(path) != hashes[key] for key, path in paths.items())
                or any(sha256(ROOT/name) != value for name, value in source_hashes.items())):
            raise ValueError('推理期间模型、来源或代码变化，拒绝保存')
        manifest['checks']['all_inputs_and_inference_sources_unchanged'] = True
        manifest['artifacts_sha256'] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
        (temporary/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        b_ids = kb.loc[~kb.is_route_only & ~kb.is_non_english, ['chunk_id', 'review_id', 'is_eval']].copy()
        b_ids['is_eval_parent'] = b_ids.review_id.isin(reviews)
        pd.testing.assert_frame_equal(b_coverage, load_b_labels(temporary, b_pool=b_ids), check_dtype=False)
        manifest['checks']['standalone_b_reader_compatible'] = True
        manifest['elapsed_seconds'] = round(time.perf_counter()-started, 3)
        (temporary/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        if destination.exists():
            raise ValueError('输出目录在运行期间出现，拒绝覆盖')
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps({'package': str(destination), 'added': len(added), 'b_pool': manifest['b_pool'],
                      'non_eval': manifest['non_eval_coverage'], 'inference_seconds': inference_seconds}, ensure_ascii=False, indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-package', type=Path, default=ROOT/'outputs/c_b_sync_v4_1')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/c_b_sync_v4_1_extended')
    parser.add_argument('--reuse-package', type=Path,
                        help='只整理已经核验的补预测包，不加载模型、不推理')
    parser.add_argument('--historical-sources', type=Path,
                        help='显式提供旧包装脚本及说明的历史 ZIP；不允许分类规则回退')
    parser.add_argument('--batch-size', type=int, default=8192)
    args = parser.parse_args(argv)
    if args.reuse_package is not None:
        repackage_existing(args.reuse_package, args.output_dir, args.historical_sources)
    else:
        run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
