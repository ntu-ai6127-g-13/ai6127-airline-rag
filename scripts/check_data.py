"""三秒确认你的 chunks.parquet 和评测集还是同一套。

用法：python scripts/check_data.py

为什么需要它
  is_eval 是**建库时烤进 parquet 的一列**，之后不会自己更新。
  评测集扩充过（1,000 → 1,100，2026-09-17），谁的 parquet 建于那之前，
  is_eval 就只标了 1,000 句 —— 剩下 100 句金标准会静默进训练集，
  P/R/F1 系统性虚高，而且全程不报错。

  build_kb 里的自检只在建库时跑。拉了新 meta 却没重建 parquet 的人
  得不到任何提示 —— 这个脚本补的就是那个缺口。

  跑它只要三秒，重建要 14 分钟。开工前、以及每次 git pull 之后跑一下。
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / 'data/processed/chunks.parquet'
META = ROOT / 'data/eval/to_label_meta.csv'


def main():
    if not CHUNKS.exists():
        sys.exit(f'❌ 找不到 {CHUNKS.relative_to(ROOT)}\n'
                 '   先跑：python -m src.build_kb')
    if not META.exists():
        sys.exit(f'❌ 找不到 {META.relative_to(ROOT)}\n   先 git pull')

    df = pd.read_parquet(CHUNKS, columns=['chunk_id', 'review_id', 'is_eval'])
    ids = set(pd.read_csv(META, usecols=['chunk_id'])['chunk_id'])
    marked = int(df.is_eval.sum())

    print(f'  语料          {len(df):,} 句')
    print(f'  评测集        {len(ids):,} 句')
    print(f'  parquet 标出   {marked:,} 句')

    if marked != len(ids):
        sys.exit(
            f'\n❌ 对不上 —— 你的 chunks.parquet 和评测集不是同一套。\n'
            f'   差 {abs(len(ids) - marked):,} 句。最常见的两个原因：\n'
            f'   1. parquet 建于评测集扩充之前 → 重建：python -m src.build_kb\n'
            f'   2. spaCy 版本不对导致 chunk_id 漂移 → 必须 spacy==3.8.16 '
            f'且 en_core_web_sm 3.8.0\n'
            f'   在修好之前**不要训练任何模型**：金标准会混进训练集，\n'
            f'   指标虚高而且不报错。')

    # 真正逐条核对，不只是比个数
    hit = df.chunk_id.isin(ids).sum()
    if hit != len(ids):
        sys.exit(f'\n❌ 数量对得上但 chunk_id 对不上（只匹配到 {hit:,}）—— '
                 'chunk_id 已漂移，重建语料。')

    eval_reviews = df[df.is_eval].review_id.nunique()
    same_rev = int(df.review_id.isin(set(df[df.is_eval].review_id)).sum()) - marked
    print(f'\n✅ 咬合正确，可以训练')
    print(f'   训练可用     {len(df) - marked:,} 句   →  df[~df.is_eval]')
    print(f'\n   补充：金标准来自 {eval_reviews:,} 条评论，这些评论里另有 '
          f'{same_rev:,} 句\n'
          f'   仍在训练集中。它们和金标准共享同一套评分（弱标签相同），\n'
          f'   属于更弱的关联泄漏。要收紧就再加一层：\n'
          f'     train = df[~df.review_id.isin(set(df[df.is_eval].review_id))]')


if __name__ == '__main__':
    main()
