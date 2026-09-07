"""对原始数据做一次体检 + 自检是否拿对了文件。

用法：python scripts/profile_data.py

数据来源：https://www.kaggle.com/datasets/efehandanisman/skytrax-airline-reviews
下载后放到 data/raw/ 即可（csv 或 xlsx 都行）。
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (DATA_RAW, COL_TEXT, COL_AIRLINE, COL_DATE,
                        COL_CABIN, COL_TRAVELLER, ASPECTS)

# 自检基准：无论从哪个副本来，清洗后都应该是这些数
EXPECTED = {
    "有正文的评论": 65_947,
    "去重后": 61_172,
    "总词数": 8_402_547,
    "航司数": 81,
}


def find_raw_file() -> Path:
    """data/raw/ 下找 csv 或 xlsx，忽略 .gitkeep。"""
    files = [f for f in DATA_RAW.iterdir()
             if f.suffix.lower() in {".csv", ".xlsx", ".xls"}]
    if not files:
        print(f"❌ {DATA_RAW} 里没有数据文件。")
        print("   请从 https://www.kaggle.com/datasets/efehandanisman/skytrax-airline-reviews")
        print("   下载后放进去。")
        sys.exit(1)
    if len(files) > 1:
        print(f"⚠️  找到多个文件，使用第一个：{files[0].name}")
    return files[0]


def main() -> None:
    path = find_raw_file()
    print(f"读取 {path.name} ...")
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} \
        else pd.read_csv(path)

    print(f"\n原始行数: {len(df):,}")
    if len(df) not in (65_948, 131_895):
        print("  ⚠️  行数不是 65,948 或 131,895——可能是别的版本，继续看下面的自检")

    blank = df[COL_TEXT].isna().sum()
    if blank:
        print(f"无正文的空行: {blank:,}  (会被滤掉，属正常)")

    clean = df[df[COL_TEXT].notna()].copy()
    n_text = len(clean)
    before = n_text
    clean = clean.drop_duplicates(subset=[COL_TEXT])
    words = clean[COL_TEXT].astype(str).str.split().str.len()

    actual = {
        "有正文的评论": n_text,
        "去重后": len(clean),
        "总词数": int(words.sum()),
        "航司数": clean[COL_AIRLINE].nunique(),
    }

    print(f"\n=== 自检：是否拿对了文件 ===")
    all_ok = True
    for k, exp in EXPECTED.items():
        got = actual[k]
        ok = got == exp
        all_ok &= ok
        print(f"  {'✅' if ok else '❌'} {k:12s} {got:>11,}   (应为 {exp:,})")

    d = pd.to_datetime(clean[COL_DATE], errors="coerce")
    print(f"\n时间范围: {d.min():%Y-%m} ~ {d.max():%Y-%m}   (应为 2002-08 ~ 2019-06)")
    print(f"删掉重复: {before - len(clean):,} 条")
    print(f"平均每条 {words.mean():.0f} 词，中位数 {words.median():.0f}")

    print("\n=== 各维度有评分的数量（决定弱标签规模）===")
    for aspect, (col, _) in ASPECTS.items():
        s = clean[col].dropna()
        print(f"  {aspect:14s} 有分 {len(s):>6,}   正 {(s >= 4).sum():>6,}   负 {(s <= 2).sum():>6,}")

    print("\n=== 舱位分布 ===")
    print(clean[COL_CABIN].value_counts(dropna=False).to_string())
    print("\n=== 旅客类型分布 ===")
    print(clean[COL_TRAVELLER].value_counts(dropna=False).to_string())

    print("\n=== 作业门槛 ===")
    print(f"  文档数 ≥10,000 : {len(clean):,}  {'✅' if len(clean) >= 10_000 else '❌'}")
    print(f"  词数 ≥100,000  : {words.sum():,}  {'✅' if words.sum() >= 100_000 else '❌'}")

    print("\n" + ("✅ 文件正确，可以开始了" if all_ok else
                  "❌ 数字对不上，可能下错了数据集——网上同名的有好几个版本"))


if __name__ == "__main__":
    main()
