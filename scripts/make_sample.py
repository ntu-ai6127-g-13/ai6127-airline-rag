"""从建好的 chunks.parquet 里切一小块出来，供三个组并行开发用。

用法：
    python -m src.build_kb                 # 先建全量（约 14 分钟）
    python scripts/make_sample.py          # 再抽样（几秒）
    python scripts/make_sample.py --n 500  # 抽 500 条评论

为什么要抽样而不是直接用全量：
  全量 546,139 句，建索引要几十分钟。开发阶段每改一行代码就等这么久没法干活。
  抽样后只有几千句，**几秒钟跑完，但数据是真的**。

------------------------------------------------------------------------
2026-09-08 改写：以前这个脚本**自己重新实现了一遍清洗+切句**，
结果 chunks.parquet 从 17 列加到 28 列之后，样本文件还停在 17 列，
数据浏览界面一切到样本就崩（缺 route_from / aircraft 等列）。

现在改成**直接从 chunks.parquet 里按 review_id 抽**：
  · schema 永远和全量一致，从结构上不可能再漂移
  · 抽的是完整评论（不是随机句子），parent-document retrieval 才有意义
  · 快得多 —— 不用再读 23 MB 的 xlsx、不用再跑 spaCy
------------------------------------------------------------------------
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CHUNKS, CHUNKS_SAMPLE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="抽多少条评论（默认 300）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not CHUNKS.exists():
        sys.exit(f"❌ 找不到 {CHUNKS}\n   先跑：python -m src.build_kb")

    print(f"读取 {CHUNKS.name} ...")
    chunks = pd.read_parquet(CHUNKS)
    print(f"   全量 {len(chunks):,} 句 / {chunks['review_id'].nunique():,} 条评论 "
          f"/ {len(chunks.columns)} 列")

    # 按评论的总体评分分层抽，保证好评差评大致各半 ——
    # 否则随机抽出来可能全是差评（原始数据里差评偏多）。
    per_review = (chunks.groupby("review_id")["overall_rating"]
                  .first().dropna())
    pos = per_review[per_review >= 7].index
    neg = per_review[per_review <= 4].index
    half = args.n // 2
    rng = pd.Series(list(pos)).sample(min(half, len(pos)), random_state=args.seed)
    rng2 = pd.Series(list(neg)).sample(min(args.n - half, len(neg)),
                                       random_state=args.seed)
    keep = set(rng) | set(rng2)

    sample = chunks[chunks["review_id"].isin(keep)].copy()
    sample = sample.sort_values("chunk_id").reset_index(drop=True)

    CHUNKS_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    # 和全量用同一套压缩参数，读的一侧行为完全一致
    sample.to_parquet(CHUNKS_SAMPLE, compression="zstd", compression_level=9)

    size_kb = CHUNKS_SAMPLE.stat().st_size / 1024
    print(f"\n✅ 已生成 {CHUNKS_SAMPLE}")
    print(f"   {len(sample):,} 句 / {sample['review_id'].nunique():,} 条评论 "
          f"/ {len(sample.columns)} 列 / {size_kb:.0f} KB")
    print(f"   好评(≥7) {len(rng):,} 条 · 差评(≤4) {len(rng2):,} 条")
    assert list(sample.columns) == list(chunks.columns), "列不一致！"
    print("   schema 与全量完全一致 ✅")


if __name__ == "__main__":
    main()
