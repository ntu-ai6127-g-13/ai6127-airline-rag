"""只读核验补预测 v2 交接包及 JSONL；不加载模型、不训练、不推理。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_c_b_sync import audit_package as audit_prior_package
from scripts.prepare_c_b_sync import b_counts
from scripts.prepare_c_handoff import KB_COLUMNS, GUIDE_PATH, verify_packaging_sources
from scripts.supplement_c_b_predictions import validate_extended_tables, validate_jsonl
from src.c_label_reader_v2 import SCHEMA_VERSION, VERSION, load_b_labels
from src.weak_labels import load_test_exclusions, sha256

ARTIFACTS = {'labels.parquet', 'supplemental_predictions.parquet', 'coverage.parquet',
             'b_coverage.parquet', 'non_eval_coverage.parquet', 'predictions.jsonl',
             'read_labels.py', 'README.md', 'b_retriever_source.py'}
INPUTS = {'kb', 'input', 'model', 'base_model', 'predictions', 'prediction_report',
          'training_report', 'revision_report', 'eval_meta', 'eval_text', 'prior_manifest'}
HISTORY_FILE = 'historical_packaging_sources.zip'
PACKAGING_SOURCES = {
    'scripts/prepare_c_handoff.py', 'scripts/prepare_c_b_sync.py',
    'scripts/check_c_b_sync.py', 'scripts/supplement_c_b_predictions.py',
    'scripts/check_c_b_extended.py', GUIDE_PATH,
}


def validate_documentation_repack(package, manifest, historical_sources=None):
    """Bind a documentation-only copy to the original package and exact data bytes."""
    package = Path(package).resolve()
    metadata = manifest.get('documentation_repack')
    if metadata is None:
        return historical_sources
    keys = {'version', 'created_at', 'original_package', 'historical_sources_file',
            'historical_sources_sha256', 'current_packaging_sources_sha256',
            'new_inference', 'model_training', 'data_files_unchanged'}
    if (not isinstance(metadata, dict) or set(metadata) != keys
            or metadata['version'] != 'single-guide-v1'
            or metadata['new_inference'] is not False
            or metadata['model_training'] is not False
            or metadata['data_files_unchanged'] is not True
            or metadata['historical_sources_file'] != HISTORY_FILE):
        raise ValueError('单指南重打包记录不完整或越过仅整理说明的范围')
    archive = package/HISTORY_FILE
    if (archive.resolve().parent != package or not archive.is_file()
            or sha256(archive) != metadata['historical_sources_sha256']
            or manifest['artifacts_sha256'].get(HISTORY_FILE) != metadata['historical_sources_sha256']):
        raise ValueError('历史包装来源 ZIP 哈希或路径不一致')
    if historical_sources is not None and sha256(Path(historical_sources)) != sha256(archive):
        raise ValueError('外部历史来源与新包内记录不一致')
    current = metadata['current_packaging_sources_sha256']
    if not isinstance(current, dict) or set(current) != PACKAGING_SOURCES:
        raise ValueError('当前包装程序及统一指南来源不完整')
    verify_packaging_sources(current, root=ROOT)
    if sha256(package/'README.md') != current[GUIDE_PATH]:
        raise ValueError('包内 README 不是当前统一指南')
    original = metadata['original_package']
    if not isinstance(original, dict) or set(original) != {'path', 'manifest_sha256'}:
        raise ValueError('原补预测包来源不完整')
    source = Path(original['path']).resolve()
    if (source == package or not source.is_relative_to(Path(ROOT).resolve())
            or sha256(source/'manifest.json') != original['manifest_sha256']):
        raise ValueError('原补预测包路径或清单哈希变化')
    old = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if 'documentation_repack' in old or set(old.get('artifacts_sha256', {})) != ARTIFACTS:
        raise ValueError('只接受原始补预测包作为说明整理来源')
    old_metadata = {key: value for key, value in old.items() if key != 'artifacts_sha256'}
    new_metadata = {key: value for key, value in manifest.items()
                    if key not in {'artifacts_sha256', 'documentation_repack'}}
    if old_metadata != new_metadata:
        raise ValueError('重打包改变了原推理、训练、范围或候选验收记录')
    for name, expected in old['artifacts_sha256'].items():
        if (source/name).resolve().parent != source or sha256(source/name) != expected:
            raise ValueError(f'原补预测包产物改变：{name}')
        if name != 'README.md' and manifest['artifacts_sha256'].get(name) != expected:
            raise ValueError(f'重打包改变了原标签或覆盖数据：{name}')
    return archive


def audit_package(package, historical_sources=None):
    package = Path(package).resolve()
    manifest_path = package/'manifest.json'
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('schema_version') != SCHEMA_VERSION or manifest.get('model_version') != VERSION
            or manifest.get('status') != 'candidate' or manifest.get('model_retrained') is not False
            or manifest.get('new_inference') is not True):
        raise ValueError('不是当前补预测候选 v2 包')
    expected_artifacts = ARTIFACTS | ({HISTORY_FILE} if 'documentation_repack' in manifest else set())
    if set(manifest.get('inputs', {})) != INPUTS or set(manifest.get('artifacts_sha256', {})) != expected_artifacts:
        raise ValueError('来源或产物记录不完整')
    historical_sources = validate_documentation_repack(package, manifest, historical_sources)
    paths = {}
    for name, item in manifest['inputs'].items():
        path = Path(item['path']).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != item['sha256']:
            raise ValueError(f'来源 hash 变化或路径非法：{name}')
        paths[name] = path
    for name, expected in manifest['artifacts_sha256'].items():
        if sha256(package/name) != expected:
            raise ValueError(f'产物 hash 变化：{name}')
    source_validation = verify_packaging_sources(
        manifest['implementation_sha256_at_inference'], root=ROOT,
        historical_sources=historical_sources)
    prior = Path(manifest['source_package']['path']).resolve()
    if (not prior.is_relative_to(ROOT) or paths['prior_manifest'] != prior/'manifest.json'
            or manifest['source_package']['manifest_sha256'] != sha256(prior/'manifest.json')):
        raise ValueError('旧交接包来源不一致')
    audit_prior_package(prior, historical_sources=historical_sources)
    old_manifest = json.loads((prior/'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('training_base_version') != old_manifest['training_base_version']
            or manifest['inputs']['model'] != old_manifest['inputs']['model']):
        raise ValueError('补推理没有沿用原模型')
    lock = manifest['inference_lock']
    prediction_report = json.loads(paths['prediction_report'].read_text(encoding='utf-8'))
    if (lock['model_sha256'] != manifest['inputs']['model']['sha256']
            or lock['fitted_state_hash'] != old_manifest['inherited_fitted_state_hash']
            or lock['rule_source_sha256'] != prediction_report['implementation_sha256']
            or lock.get('final_model_selection_or_w5_freeze') is not False
            or lock['parameters'].get('new_training') is not False):
        raise ValueError('本次固定模型与规则来源不一致')
    for key, value in old_manifest['b_pool'].items():
        if key not in {'rows', 'predicted_rows', 'skipped_rows', 'excluded_by_first_reason', 'ordered_chunk_ids_sha256'}:
            if manifest['b_pool'].get(key) != value:
                raise ValueError('B 索引来源版本变化')
    if sha256(package/'b_retriever_source.py') != sha256(prior/'b_retriever_source.py'):
        raise ValueError('B 源码快照未保持原样')
    kb = pd.read_parquet(paths['kb'], columns=KB_COLUMNS)
    reviews, texts, evaluation = load_test_exclusions(kb, paths['eval_meta'], paths['eval_text'])
    if evaluation != manifest['evaluation_exclusion']:
        raise ValueError('评测隔离映射不一致')
    old_labels = pd.read_parquet(prior/'labels.parquet')
    labels = pd.read_parquet(package/'labels.parquet')
    added = pd.read_parquet(package/'supplemental_predictions.parquet')
    coverage = pd.read_parquet(package/'coverage.parquet')
    validate_extended_tables(kb, old_labels, labels, added, coverage, reviews, texts)
    pool = kb.loc[~kb.is_route_only & ~kb.is_non_english, ['chunk_id', 'review_id', 'is_eval']].copy()
    pool['is_eval_parent'] = pool.review_id.isin(reviews)
    b_coverage = load_b_labels(package, b_pool=pool)
    expected_b = coverage.loc[(~kb.is_route_only & ~kb.is_non_english).to_numpy()].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected_b, b_coverage)
    non_eval = pd.read_parquet(package/'non_eval_coverage.parquet')
    pd.testing.assert_frame_equal(coverage.loc[~coverage.is_eval].reset_index(drop=True), non_eval)
    if (manifest['original_prediction_rows'] != len(old_labels) or manifest['added_prediction_rows'] != len(added)
            or manifest['coverage'] != {**b_counts(coverage), 'eval_sentence_rows': int(kb.is_eval.sum()),
                                       'eval_parent_rows': int(kb.review_id.isin(reviews).sum())}
            or manifest['non_eval_coverage'] != b_counts(non_eval)
            or any(manifest['b_pool'].get(key) != value for key, value in b_counts(b_coverage).items())):
        raise ValueError('覆盖/新增计数记录不一致')
    for name, frame in [('labels.parquet', labels), ('supplemental_predictions.parquet', added),
                        ('coverage.parquet', coverage), ('b_coverage.parquet', b_coverage), ('non_eval_coverage.parquet', non_eval)]:
        if manifest['schemas'].get(name) != {column: str(frame[column].dtype) for column in frame}:
            raise ValueError(f'{name} schema 不一致')
    jsonl_rows = validate_jsonl(package/'predictions.jsonl', labels)
    if manifest['jsonl_interface']['rows'] != jsonl_rows:
        raise ValueError('JSONL 数量记录不一致')
    if sha256(package/'read_labels.py') != manifest['implementation_sha256_at_inference']['src/c_label_reader_v2.py']:
        raise ValueError('独立读取工具不是本次 v2 接口')
    required_checks = ['prior_package_verified', 'existing_predictions_unchanged', 'supplement_is_exact_missing_id_set',
                       'supplement_only_non_gold_eval_parents', 'supplement_train_dev_parent_overlap_zero',
                       'supplement_normalized_gold_text_overlap_zero', 'model_fitted_state_unchanged',
                       'parquet_roundtrip_identical', 'jsonl_exact_label_mirror', 'all_inputs_and_inference_sources_unchanged',
                       'standalone_b_reader_compatible']
    if any(manifest['checks'].get(key) is not True for key in required_checks):
        raise ValueError('推理运行核验记录未通过或不完整')
    if any(manifest['checks'].get(key) is not False for key in ['human_gold_answers_read', 'training_performed', 'original_text_or_ratings_copied']):
        raise ValueError('补推理越过声明范围')
    if sha256(manifest_path) != manifest_hash:
        raise ValueError('核验期间 manifest 变化')
    return {'created_at': datetime.now(timezone.utc).isoformat(), 'kind': 'read_only_extended_handoff_check',
            'package': str(package), 'manifest_sha256': manifest_hash, 'b_pool': b_counts(b_coverage),
            'non_eval_coverage': b_counts(non_eval), 'supplemental_rows': len(added), 'jsonl_rows': jsonl_rows,
            'checks': {'all_sources_and_artifacts_hashes_match': True, 'existing_labels_identical': True,
                       'added_labels_exactly_match_missing_non_gold_siblings': True, 'heldout_roles_explicit': True,
                       'gold_sentences_not_predicted': True, 'full_non_eval_ids_covered_once': True,
                       'b_pool_ids_parents_and_flags_match': True, 'skipped_labels_null': True,
                       'jsonl_matches_parquet_and_confidence_is_null': True, 'old_package_verified_unchanged': True},
            'models_loaded': False, 'new_predictions_performed': False, 'model_training_performed': False,
            'source_validation': source_validation,
            'documentation_repack_verified': 'documentation_repack' in manifest,
            'human_gold_answers_read': False, 'actual_b_index_artifact_checked': False,
            'acceptance': manifest['acceptance']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=ROOT/'outputs/c_b_sync_v4_1_pr_ready')
    parser.add_argument('--historical-sources', type=Path,
                        help='旧包仅可显式使用包装脚本/旧说明的历史 ZIP；不允许分类规则回退')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        raise ValueError('核验报告已存在，拒绝覆盖；省略 --output 即只读检查')
    report = audit_package(args.package, historical_sources=args.historical_sources)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
