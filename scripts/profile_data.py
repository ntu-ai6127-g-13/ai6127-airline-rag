"""对原始数据做一次体检。第一天先跑这个，把真实数字记进报告。

用法：python scripts/profile_data.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (RAW_XLSX, COL_TEXT, COL_AIRLINE, COL_DATE,
                        COL_CABIN, COL_TRAVELLER, COL_OVERALL, ASPECTS)


def main() -> None:
    df = pd.read_excel(RAW_XLSX)
    print(f"原始行数: {len(df):,}")

    # 这个文件有大量全空的填充行，必须先滤掉
    blank = df[COL_TEXT].isna().sum()
    print(f"无正文的空行: {blank:,}  ({blank/len(df)*100:.1f}%)")

    clean = df[df[COL_TEXT].notna()].copy()
    print(f"有正文的评论: {len(clean):,}")

    before = len(clean)
    clean = clean.drop_duplicates(subset=[COL_TEXT])
    print(f"去重后: {len(clean):,}  (删掉 {before - len(clean):,} 条重复)")

    words = clean[COL_TEXT].astype(str).str.split().str.len()
    print(f"\n总词数: {words.sum():,}")
    print(f"平均每条 {words.mean():.0f} 词，中位数 {words.median():.0f}")
    print(f"航司数: {clean[COL_AIRLINE].nunique()}")

    d = pd.to_datetime(clean[COL_DATE], errors="coerce")
    print(f"时间范围: {d.min():%Y-%m} ~ {d.max():%Y-%m}")

    print("\n=== 各维度有评分的数量（决定弱标签规模）===")
    for aspect, (col, _) in ASPECTS.items():
        s = clean[col].dropna()
        pos, neg = (s >= 4).sum(), (s <= 2).sum()
        print(f"  {aspect:14s} 有分 {len(s):>6,}  正 {pos:>6,}  负 {neg:>6,}")

    print("\n=== 舱位分布 ===")
    print(clean[COL_CABIN].value_counts(dropna=False).to_string())
    print("\n=== 旅客类型分布 ===")
    print(clean[COL_TRAVELLER].value_counts(dropna=False).to_string())

    print("\n=== 作业门槛核对 ===")
    print(f"  文档数 ≥10,000 : {len(clean):,}  {'✅' if len(clean) >= 10000 else '❌'}")
    print(f"  词数 ≥100,000  : {words.sum():,}  {'✅' if words.sum() >= 100000 else '❌'}")


if __name__ == "__main__":
    main()
