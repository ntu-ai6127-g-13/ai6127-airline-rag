"""将既有 v4.1 标签整理成 B 索引范围的本地同步包；不训练、不预测。"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_c_handoff import (ASPECTS, GUIDE_PATH, KB_COLUMNS, LABEL_COLUMNS, METADATA_COLUMNS,
                                      SKIP_REASONS, make_tables, validate_handoff, validate_kb)
from src.weak_labels import load_test_exclusions, sha256

VERSION = 'eight-aspect-subjectivity-polarity-v4.1'
BASE_VERSION = 'eight-aspect-subjectivity-polarity-v3'
B_BASELINE_COMMIT = '61fc85d46557bcd4034fe42fd257f58f92c58fda'
SCHEMA_VERSION = 'c-b-sync-v1'
EXPECTED_TASKS = {'aspect': ASPECTS, 'subjectivity': ['objective', 'subjective'],
                  'polarity': ['negative', 'positive']}


def ordered_ids_hash(values):
    return hashlib.sha256(('\n'.join(values) + '\n').encode('utf-8')).hexdigest()


def fitted_state_hash(bundle):
    vectorizer = bundle['vectorizer']
    if isinstance(vectorizer, TfidfVectorizer):
        # sklearn 缓存的是 stop_words 对象的进程内 id，transform 可更新它。
        # 它不是词表/IDF/参数；除这一缓存外比较全部向量器状态及三个模型。
        vectorizer = {'class': type(vectorizer).__qualname__,
                      'state': {key: value for key, value in vars(vectorizer).items() if key != '_stop_words_id'}}
    return joblib.hash({'vectorizer': vectorizer, 'models': bundle['models']})


def b_source_at_commit(commit, source_file=None):
    """只读取已知 Git 对象或代码快照；不 fetch，不导入/运行 B 的代码。"""
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('B 来源必须是完整的固定提交编号')
    baseline = subprocess.check_output(
        ['git', 'show', B_BASELINE_COMMIT + ':src/retriever.py'], cwd=ROOT).decode('utf-8')
    source = (Path(source_file).read_text(encoding='utf-8') if source_file else
              subprocess.check_output(['git', 'show', commit + ':src/retriever.py'], cwd=ROOT).decode('utf-8'))

    def pool_definition(code):
        tree = ast.parse(code)
        function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name == '_load_index_pool'), None)
        if function is None:
            raise ValueError('B 代码缺少已核验的索引池函数')
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {'_REQUIRED_COLUMNS', '_NOISE_COLUMNS'}:
                        constants[target.id] = ast.literal_eval(node.value)
        return ast.dump(function, include_attributes=False), constants

    definition, constants = pool_definition(source)
    expected, expected_constants = pool_definition(baseline)
    if (definition != expected or constants != expected_constants
            or constants.get('_NOISE_COLUMNS') != ('is_route_only', 'is_non_english')
            or constants.get('_REQUIRED_COLUMNS') != ('chunk_id', 'text')):
        raise ValueError('B 索引范围与已核验基线不同，需先重新分析；拒绝按旧范围同步')
    return source, {'source_commit': commit, 'source_path': 'src/retriever.py',
                    'source_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
                    'reference_baseline_commit': B_BASELINE_COMMIT,
                    'pool_function_identical_to_verified_baseline': True,
                    'policy': 'exclude is_route_only and is_non_english; retain is_eval; no index-time text deduplication',
                    'actual_b_index_artifact_checked': False}


def verify_revision_provenance(report, revision, training, bundle, base, hashes):
    """分别核验 v3 基础训练与 v4.1 规则修订，不伪造 v4.1 训练报告。"""
    if (report.get('model_version') != VERSION or revision.get('version') != VERSION
            or bundle.get('version') != VERSION or bundle.get('tasks') != EXPECTED_TASKS):
        raise ValueError('预测、规则报告与模型版本不是目标 v4.1')
    if (base.get('version') != BASE_VERSION or base.get('tasks') != EXPECTED_TASKS
            or training.get('version') != BASE_VERSION
            or bundle.get('base_model_version') != BASE_VERSION
            or revision.get('training_base_version') != BASE_VERSION):
        raise ValueError('基础训练版本不是 v3')
    if (bundle.get('model_retrained_for_rule_revision') is not False
            or report.get('model_retrained_for_rule_revision') is not False
            or revision.get('model_retrained') is not False):
        raise ValueError('本包仅允许已有权重的规则修订')
    if (bundle.get('training_inputs') != base.get('training_inputs')
            or training.get('inputs') != bundle.get('training_inputs')):
        raise ValueError('基础训练输入来源不一致')
    for key in ['model', 'input', 'kb', 'eval_meta', 'eval_text']:
        if report.get('inputs', {}).get(key, {}).get('sha256') != hashes[key]:
            raise ValueError(f'预测报告 {key} hash 与当前来源不一致')
    if report.get('output_sha256') != hashes['predictions']:
        raise ValueError('既有预测文件 hash 与报告不一致')
    if (bundle.get('base_model_sha256') != hashes['base_model']
            or bundle.get('base_training_report_sha256') != hashes['training_report']
            or training.get('artifacts_sha256', {}).get('model.joblib') != hashes['base_model']
            or revision.get('artifacts_sha256', {}).get('model.joblib') != hashes['model']):
        raise ValueError('规则修订、基础模型或训练报告 hash 不一致')
    for name, key in [('base_model', 'base_model'), ('base_training_report', 'training_report'),
                      ('kb', 'kb'), ('eval_meta', 'eval_meta'), ('eval_text', 'eval_text')]:
        if revision.get('inputs', {}).get(name, {}).get('sha256') != hashes[key]:
            raise ValueError(f'规则修订 {name} hash 不一致')
    for key in ['input', 'kb', 'eval_meta', 'eval_text']:
        if bundle['training_inputs'].get(key, {}).get('sha256') != hashes[key]:
            raise ValueError(f'模型 training_inputs/{key} 与当前文件不一致')
    for key in ['train_review_ids', 'dev_review_ids']:
        if bundle.get(key) != base.get(key):
            raise ValueError('规则修订改变了基础训练/开发名单')
    if fitted_state_hash(bundle) != fitted_state_hash(base):
        raise ValueError('继承的词表或模型权重与 v3 不一致')
    required = ['unique_ids', 'row_count_matches', 'zero_eval_parent_text_overlap', 'legal_aspects',
                'legal_subjectivity', 'objective_is_NA', 'subjective_has_binary_polarity',
                'inputs_unchanged', 'roundtrip_identical']
    if any(report.get('checks', {}).get(key) is not True for key in required):
        raise ValueError('原预测核验未通过或不完整')
    revision_checks = ['historical_v3_dev_exactly_reproduced', 'raw_model_predictions_unchanged',
                       'fitted_weights_and_vocabulary_unchanged', 'zero_dev_eval_id_parent_text_overlap',
                       'inputs_unchanged', 'model_reload_identical']
    if any(revision.get('checks', {}).get(key) is not True for key in revision_checks):
        raise ValueError('规则修订来源核验未通过')
    if (revision.get('human_gold_answers_read') is not False
            or revision.get('evaluation_exclusion', {}).get('human_labels_read') is not False):
        raise ValueError('规则修订不得读取人工答案')


def make_b_coverage(kb, coverage):
    validate_kb(kb)
    if (not kb.chunk_id.reset_index(drop=True).eq(coverage.chunk_id.reset_index(drop=True)).all()
            or len(kb) != len(coverage)):
        raise ValueError('完整覆盖表未按 KB 编号精确对齐')
    keep = ~kb.is_route_only & ~kb.is_non_english
    result = coverage.loc[keep.to_numpy()].copy().reset_index(drop=True)
    if (not result.chunk_id.is_unique
            or int(result.prediction_status.eq('predicted').sum()) != int(coverage.prediction_status.eq('predicted').sum())):
        raise ValueError('B 池重复 ID 或遗漏已有预测')
    return result


def b_counts(coverage):
    return {'rows': len(coverage), 'predicted_rows': int(coverage.prediction_status.eq('predicted').sum()),
            'skipped_rows': int(coverage.prediction_status.eq('skipped').sum()),
            'excluded_by_first_reason': {reason: int(coverage.skip_reason.eq(reason).sum()) for reason in SKIP_REASONS},
            'ordered_chunk_ids_sha256': ordered_ids_hash(coverage.chunk_id.tolist())}


def prepare(args):
    destination = args.output_dir.resolve()
    if destination.exists():
        raise ValueError(f'同步包已存在，拒绝覆盖：{destination}')
    paths = {key: getattr(args, key).resolve() for key in
             ['kb', 'model', 'base_model', 'predictions', 'prediction_report', 'training_report',
              'revision_report', 'eval_meta', 'eval_text']}
    paths['input'] = paths['kb']
    hashes = {key: sha256(path) for key, path in paths.items()}
    report = json.loads(paths['prediction_report'].read_text(encoding='utf-8'))
    revision = json.loads(paths['revision_report'].read_text(encoding='utf-8'))
    training = json.loads(paths['training_report'].read_text(encoding='utf-8'))
    bundle, base = joblib.load(paths['model']), joblib.load(paths['base_model'])
    verify_revision_provenance(report, revision, training, bundle, base, hashes)
    kb = pd.read_parquet(paths['kb'], columns=KB_COLUMNS)
    validate_kb(kb)
    reviews, texts, evaluation = load_test_exclusions(kb, paths['eval_meta'], paths['eval_text'])
    if evaluation != report.get('evaluation_exclusion') or evaluation != revision.get('evaluation_exclusion'):
        raise ValueError('评测隔离映射版本与既有报告不一致')
    if (set(bundle['train_review_ids']) & reviews or set(bundle['dev_review_ids']) & reviews):
        raise ValueError('已有模型训练/开发父评论与正式评测父评论重叠')
    predictions = pd.read_parquet(paths['predictions'], columns=[*LABEL_COLUMNS, 'text', 'is_eval'])
    if (len(kb) != report.get('input_rows') or len(predictions) != report.get('output_rows')
            or len(kb) - len(predictions) != report.get('excluded_rows')):
        raise ValueError('既有预测/输入数量与报告不一致')
    counts = {task: predictions[task].value_counts().to_dict() for task in EXPECTED_TASKS}
    if counts != report.get('counts'):
        raise ValueError('既有预测标签计数与报告不一致')
    labels, coverage = make_tables(kb, predictions, reviews, texts, version=VERSION, bundle=bundle)
    b_coverage = make_b_coverage(kb, coverage)
    source, b_provenance = b_source_at_commit(args.b_commit, args.b_source_file)
    sources = [Path(__file__).resolve(), ROOT/'scripts/check_c_b_sync.py',
               ROOT/'src/c_label_reader.py', ROOT/GUIDE_PATH]
    source_now = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    manifest = {
        'schema_version': SCHEMA_VERSION, 'status': 'candidate', 'model_version': VERSION,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'purpose': 'local B-index coverage synchronization using existing C predictions; not completed B integration',
        'acceptance': {'human_accepted': False, 'human_gold_evaluated': False,
                       'final_version_selected': False, 'b_integration_verified': False},
        'inputs': {key: {'path': str(path), 'sha256': hashes[key]} for key, path in paths.items()},
        'training_base_version': BASE_VERSION, 'model_retrained': False, 'new_inference': False,
        'inherited_fitted_state_hash': fitted_state_hash(bundle),
        'fitted_state_hash_method': 'joblib hash of all estimator states; ignore only TfidfVectorizer runtime _stop_words_id cache',
        'training_inputs_recorded_in_model': bundle['training_inputs'],
        'hyperparameters_recorded_at_base_training': training.get('hyperparameters', {}),
        'implementation_sha256_recorded_at_base_training': training.get('implementation_sha256', {}),
        'implementation_sha256_recorded_at_revision': revision.get('implementation_sha256', {}),
        'implementation_sha256_recorded_at_prediction': report.get('implementation_sha256', {}),
        'implementation_sha256_at_preparation': source_now,
        'b_pool': {**b_provenance, **b_counts(b_coverage)},
        'evaluation_exclusion': evaluation,
        'coverage': {'kb_rows': len(kb), 'predicted_rows': len(labels), 'skipped_rows': len(coverage)-len(labels),
                     'eval_sentence_rows': int(kb.is_eval.sum()),
                     'excluded_by_first_reason': {name: int(coverage.skip_reason.eq(name).sum()) for name in SKIP_REASONS}},
        'prediction_counts': counts,
        'schemas': {name: {column: str(frame[column].dtype) for column in frame}
                    for name, frame in [('labels.parquet', labels), ('coverage.parquet', coverage),
                                        ('b_coverage.parquet', b_coverage)]},
        'null_policy': {'objective_polarity': 'literal string NA',
                        'skipped': 'all label/source fields null; preserve prediction_status and skip_reason',
                        'confidence': 'not provided; do not invent values',
                        'unknown': 'missing external IDs are errors, not fabricated other/negative/objective'},
        'checks': {'model_weights_inherited_from_v3': True, 'source_hashes_match_reports': True,
                   'b_pool_definition_verified': True, 'all_kb_ids_once': True, 'all_b_pool_ids_once': True,
                   'existing_labels_unchanged': True, 'all_existing_predictions_in_b_pool': True,
                   'zero_prediction_eval_id_parent_text_overlap': True, 'skipped_labels_null': True,
                   'human_gold_answers_read': False, 'original_text_or_ratings_copied': False,
                   'training_or_prediction_called': False},
        'limits': ['本地同步包，不代表 B 组实际读取或接受接口。',
                   'B 范围来自固定提交的源码，在本地同版本 KB 上重建；未拿到 B 组机器上的实际索引 ID 清单。',
                   '仅复用已有 v4.1 标签；1,100 句评测题、11,535 句评测父评论兄弟句、19 句碎片暂未预测。',
                   '评测父评论及其句子仍不得用于训练、开发调参或选择模型。',
                   '八方面、Parquet 与缺少 confidence 的接口需 B 组确认；分类质量尚未经过正式金标准评估。'],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.c_b_sync_', dir=destination.parent))
    try:
        for name, frame in [('labels.parquet', labels), ('coverage.parquet', coverage), ('b_coverage.parquet', b_coverage)]:
            frame.to_parquet(temporary/name, index=False)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(temporary/name))
        validate_handoff(kb, pd.read_parquet(temporary/'labels.parquet'),
                         pd.read_parquet(temporary/'coverage.parquet'), reviews, texts, version=VERSION)
        shutil.copy2(ROOT/'src/c_label_reader.py', temporary/'read_labels.py')
        package_note = (
            '# 本包范围说明\n\n'
            f'本包接口为 `{SCHEMA_VERSION}`，模型版本为 `{VERSION}`，'
            f'实际提供 {len(labels):,} 条预测，B 候选范围为 {len(b_coverage):,} 条记录。\n\n'
            '以下统一指南介绍当前交接流程；其中补预测后 v2 包的数量不替代本包数量。'
            '本 v1 包没有新增兄弟句预测，读取时必须以本目录 manifest.json 的版本、'
            '覆盖范围和字段为准。\n\n---\n\n'
        )
        (temporary/'README.md').write_text(
            package_note + (ROOT/GUIDE_PATH).read_text(encoding='utf-8'), encoding='utf-8')
        (temporary/'b_retriever_source.py').write_text(source, encoding='utf-8')
        manifest['checks']['parquet_roundtrip_identical'] = True
        manifest['checks']['inputs_unchanged'] = all(sha256(path) == hashes[key] for key, path in paths.items())
        if not manifest['checks']['inputs_unchanged']:
            raise ValueError('同步期间来源文件变化，拒绝保存')
        manifest['artifacts_sha256'] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
        (temporary/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        from src.c_label_reader import load_b_labels
        restored = load_b_labels(temporary, b_pool=kb.loc[~kb.is_route_only & ~kb.is_non_english,
                                                       ['chunk_id', 'review_id', 'is_eval']])
        pd.testing.assert_frame_equal(b_coverage, restored, check_dtype=False)
        manifest['checks']['standalone_b_reader_compatible'] = True
        (temporary/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        if destination.exists():
            raise ValueError('同步期间目标目录出现，拒绝覆盖')
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps({'package': str(destination), 'model_version': VERSION, 'b_pool': manifest['b_pool']},
                     ensure_ascii=False, indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kb', type=Path, default=ROOT/'data/processed/chunks.parquet')
    parser.add_argument('--model', type=Path, default=ROOT/'outputs/classification_eight_rules_v4_1/model.joblib')
    parser.add_argument('--base-model', type=Path, default=ROOT/'outputs/classification_eight_full_v3/model.joblib')
    parser.add_argument('--predictions', type=Path, default=ROOT/'data/processed/chunk_predictions_eight_rules_v4_1.parquet')
    parser.add_argument('--prediction-report', type=Path, default=ROOT/'data/processed/chunk_predictions_eight_rules_v4_1.report.json')
    parser.add_argument('--training-report', type=Path, default=ROOT/'outputs/classification_eight_full_v3/metrics.json')
    parser.add_argument('--revision-report', type=Path, default=ROOT/'outputs/classification_eight_rules_v4_1/revision.json')
    parser.add_argument('--eval-meta', type=Path, default=ROOT/'data/eval/to_label_meta.csv')
    parser.add_argument('--eval-text', type=Path, default=ROOT/'data/eval/to_label.csv')
    parser.add_argument('--b-commit', default=B_BASELINE_COMMIT)
    parser.add_argument('--b-source-file', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/c_b_sync_v4_1')
    prepare(parser.parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
