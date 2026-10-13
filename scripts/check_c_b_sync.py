"""只读核验 v4.1 本地 C→B 同步包；不加载模型、不运行训练或预测。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_c_b_sync import (SCHEMA_VERSION, VERSION, BASE_VERSION, b_counts,
                                     b_source_at_commit, make_b_coverage)
from scripts.prepare_c_handoff import (KB_COLUMNS, LABEL_COLUMNS, SKIP_REASONS,
                                      validate_handoff, verify_packaging_sources)
from src.c_label_reader import load_b_labels
from src.weak_labels import load_test_exclusions, sha256

INPUT_KEYS = {'kb', 'input', 'model', 'base_model', 'predictions', 'prediction_report',
              'training_report', 'revision_report', 'eval_meta', 'eval_text'}
ARTIFACTS = {'labels.parquet', 'coverage.parquet', 'b_coverage.parquet',
             'README.md', 'read_labels.py', 'b_retriever_source.py'}


def audit_package(package, historical_sources=None):
    package = Path(package).resolve()
    manifest_path = package/'manifest.json'
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('schema_version') != SCHEMA_VERSION or manifest.get('model_version') != VERSION
            or manifest.get('status') != 'candidate'):
        raise ValueError('不是当前 v4.1 同步候选包')
    if (manifest.get('training_base_version') != BASE_VERSION or manifest.get('model_retrained') is not False
            or manifest.get('new_inference') is not False):
        raise ValueError('同步包的基础训练/规则修订来源不一致')
    if set(manifest.get('inputs', {})) != INPUT_KEYS:
        raise ValueError('同步输入记录不完整')
    paths = {}
    for name, item in manifest['inputs'].items():
        path = Path(item['path']).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != item['sha256']:
            raise ValueError(f'原输入 hash 变化或路径非法：{name}')
        paths[name] = path
    if set(manifest.get('artifacts_sha256', {})) != ARTIFACTS:
        raise ValueError('同步产物记录不完整')
    for name, digest in manifest['artifacts_sha256'].items():
        if sha256(package/name) != digest:
            raise ValueError(f'同步产物 hash 变化：{name}')
    source_validation = verify_packaging_sources(
        manifest['implementation_sha256_at_preparation'], root=ROOT,
        historical_sources=historical_sources)
    if sha256(package/'read_labels.py') != manifest['implementation_sha256_at_preparation']['src/c_label_reader.py']:
        raise ValueError('独立读取工具与本次接口源码不一致')
    source, source_info = b_source_at_commit(manifest['b_pool']['source_commit'], package/'b_retriever_source.py')
    if any(manifest['b_pool'].get(key) != value for key, value in source_info.items()):
        raise ValueError('B 索引来源代码或筛选策略记录不一致')
    kb = pd.read_parquet(paths['kb'], columns=KB_COLUMNS)
    labels = pd.read_parquet(package/'labels.parquet')
    coverage = pd.read_parquet(package/'coverage.parquet')
    reviews, texts, evaluation = load_test_exclusions(kb, paths['eval_meta'], paths['eval_text'])
    if manifest.get('evaluation_exclusion') != evaluation:
        raise ValueError('评测隔离映射记录不一致')
    validate_handoff(kb, labels, coverage, reviews, texts, version=VERSION)
    original = pd.read_parquet(paths['predictions'], columns=LABEL_COLUMNS).astype('string')
    pd.testing.assert_frame_equal(labels.sort_values('chunk_id').reset_index(drop=True),
                                  original.sort_values('chunk_id').reset_index(drop=True))
    expected_b = make_b_coverage(kb, coverage)
    # 轻量 B 端只读取编号/父评论/状态；从 B 的池输入验证连接而非按行拼接。
    pool = kb.loc[~kb.is_route_only & ~kb.is_non_english, ['chunk_id', 'review_id', 'is_eval']]
    actual_b = load_b_labels(package, b_pool=pool)
    pd.testing.assert_frame_equal(expected_b, actual_b)
    actual_coverage = {'kb_rows': len(kb), 'predicted_rows': len(labels),
                       'skipped_rows': len(coverage)-len(labels), 'eval_sentence_rows': int(kb.is_eval.sum()),
                       'excluded_by_first_reason': {name: int(coverage.skip_reason.eq(name).sum()) for name in SKIP_REASONS}}
    if manifest['coverage'] != actual_coverage:
        raise ValueError('完整覆盖计数与 manifest 不一致')
    actual_b_counts = b_counts(actual_b)
    if any(manifest['b_pool'].get(key) != value for key, value in actual_b_counts.items()):
        raise ValueError('B 范围覆盖计数与 manifest 不一致')
    for name, frame in [('labels.parquet', labels), ('coverage.parquet', coverage), ('b_coverage.parquet', actual_b)]:
        if manifest['schemas'].get(name) != {column: str(frame[column].dtype) for column in frame}:
            raise ValueError(f'{name} 字段类型与 manifest 不一致')
    if sha256(manifest_path) != manifest_hash:
        raise ValueError('读取期间 manifest 变化')
    return {'created_at': datetime.now(timezone.utc).isoformat(), 'kind': 'read_only_local_c_b_sync_check',
            'package': str(package), 'manifest_sha256': manifest_hash,
            'coverage': actual_coverage, 'b_pool': actual_b_counts,
            'checks': {'all_source_and_artifact_hashes_match': True, 'all_kb_ids_once': True,
                       'all_b_pool_ids_and_parents_match': True, 'labels_equal_existing_v4_1_predictions': True,
                       'skipped_labels_null_with_exact_reasons': True, 'no_predicted_gold_or_eval_parents': True,
                       'independent_b_reader_passed': True, 'fixed_b_code_pool_definition_verified': True},
            'models_loaded': False, 'model_fitting_performed': False, 'new_predictions_performed': False,
            'source_validation': source_validation,
            'human_gold_answers_read': False, 'actual_b_index_artifact_checked': False,
            'acceptance': manifest['acceptance']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=ROOT/'outputs/c_b_sync_v4_1')
    parser.add_argument('--historical-sources', type=Path,
                        help='显式核对包装脚本/旧说明的历史 ZIP；不允许分类规则回退')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise ValueError('报告已存在，拒绝覆盖；省略 --output 可只读核验')
    report = audit_package(args.package, historical_sources=args.historical_sources)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
