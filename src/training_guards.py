"""分类训练共用的角色、数据来源和隔离检查；失败时在拟合词表前停止。"""
import pandas as pd

from src.weak_labels import normalize_text

ROLES = {'train', 'dev', 'unassigned', 'blocked_dev_duplicate'}
WEAK_SOURCES = {'weak', 'weak_multitask', 'weak_rating_broadcast', 'weak_single_aspect',
                'weak_subjectivity'}


def reject_role_annotated_input(frame, name='输入'):
    """原始句子可生成新标签；已有角色不能在生成或重新切分时被丢弃。"""
    if 'split' in frame:
        raise ValueError(f'{name} 已有 split 角色；禁止丢弃角色后重新生成标签或切分，'
                         '请使用无 split 的原始句子输入并单独指定合法参考切分')


def _check_identifiers(frame):
    for column in ['chunk_id', 'review_id']:
        if column not in frame or not frame[column].map(
                lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f'{column} 缺失或存在空/非法编号')


def validate_roles(frame):
    _check_identifiers(frame)
    if 'split' not in frame:
        raise ValueError('标签缺少 split；不能猜测哪些记录属于训练')
    if not frame.split.isin(ROLES).all():
        raise ValueError('split 存在空值或非法角色')
    if frame.groupby('review_id').split.nunique().gt(1).any():
        raise ValueError('同一父评论跨训练/开发角色')
    if frame.groupby('chunk_id').split.nunique().gt(1).any():
        raise ValueError('同一句子被分配到多个角色')


def select_training_rows(frame):
    """唯一的显式角色筛选入口；不改源文件，也不清洗训练内重复噪声。"""
    validate_roles(frame)
    train = frame.loc[frame.split.eq('train')].copy()
    if train.empty:
        raise ValueError('没有 split=train 的标签')
    return train


def check_eval_exclusion(frame, eval_reviews, eval_texts):
    if 'is_eval' not in frame or not pd.api.types.is_bool_dtype(frame.is_eval) or frame.is_eval.isna().any():
        raise ValueError('is_eval 必须是无空值的布尔列')
    if frame.is_eval.any():
        raise ValueError('评测句进入分类训练/开发材料')
    if frame.review_id.isin(eval_reviews).any():
        raise ValueError('评测父评论进入分类训练/开发材料')
    if frame.text.map(normalize_text).isin(eval_texts).any():
        raise ValueError('重复评测文本进入分类训练/开发材料')
    return {'zero_eval_sentence_overlap': True, 'zero_eval_parent_overlap': True,
            'zero_eval_text_overlap': True}


def check_fit_boundary(train, dev, *, eval_reviews=None, eval_texts=None):
    """合成测试可不带角色；真实含角色材料必须精确符合传入集合。"""
    for name, frame in [('train', train), ('dev', dev)]:
        if frame.empty or not {'chunk_id', 'review_id', 'text'}.issubset(frame.columns):
            raise ValueError(f'{name} 为空或缺少句子、父评论、原文')
        if not frame.text.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f'{name} 原文为空或不是文本')
        _check_identifiers(frame)
        if frame.groupby('chunk_id')[['review_id', 'text']].nunique().gt(1).any().any():
            raise ValueError('同一句子编号对应不同父评论或原文')
        if 'split' in frame:
            validate_roles(frame)
            if not frame.split.eq(name).all():
                raise ValueError(f'{name} 混入其他 split 角色；禁止整文件直接训练')
        if 'source' in frame and not frame.source.isin(WEAK_SOURCES).all():
            raise ValueError('预测结果或未知来源不能当作弱标签训练真值')
        if 'is_eval' in frame:
            if not pd.api.types.is_bool_dtype(frame.is_eval) or frame.is_eval.isna().any() or frame.is_eval.any():
                raise ValueError(f'{name} 的 is_eval 非法或包含评测句')
    if set(train.chunk_id) & set(dev.chunk_id):
        raise ValueError('训练和开发句子编号重叠')
    if set(train.review_id) & set(dev.review_id):
        raise ValueError('训练和开发父评论重叠')
    if set(train.text.map(normalize_text)) & set(dev.text.map(normalize_text)):
        raise ValueError('训练和开发规范化文本重叠')
    if (eval_reviews is None) != (eval_texts is None):
        raise ValueError('评测父评论和原文排除名单必须一起提供')
    checks = {'training_roles_checked': True, 'prediction_targets_rejected': True,
              'zero_train_dev_id_parent_text_overlap': True}
    if eval_reviews is not None:
        for frame in [train, dev]:
            check_eval_exclusion(frame, eval_reviews, eval_texts)
        checks['zero_eval_overlap_in_both_parts'] = True
    return checks
