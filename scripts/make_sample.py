"""从真实数据里切一小块出来，供三个组并行开发用。

用法：python scripts/make_sample.py [--n 300]

为什么要抽样而不是直接用全量：
  全量 61,172 条评论切句后约 56.6 万句，建索引要几十分钟。
  开发阶段每改一行代码就等这么久没法干活。
  抽样后只有几千句，**几秒钟跑完，但数据是真的**。

产出 data/processed/chunks_sample.parquet，
schema 与将来的全量 chunks.parquet **完全一致**，
所以代码写好后换个文件路径就能跑全量。
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (DATA_RAW, CHUNKS_SAMPLE, ASPECTS,
                        COL_TEXT, COL_AIRLINE, COL_OVERALL, COL_DATE,
                        COL_CABIN, COL_TRAVELLER, COL_ROUTE, COL_RECOMMENDED)
from src.clean_text import clean_review

MIN_WORDS = 3          # 少于这么多词的句子丢掉（"Ok." "Never again."）


def load_clean() -> pd.DataFrame:
    files = [f for f in DATA_RAW.iterdir()
             if f.suffix.lower() in {".csv", ".xlsx", ".xls"}]
    if not files:
        print("❌ data/raw/ 里没有数据。先跑 python scripts/download_data.py")
        sys.exit(1)
    path = files[0]
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} \
        else pd.read_csv(path)

    # 这个文件有一半是全空的填充行
    df = df[df[COL_TEXT].notna()]
    df = df.drop_duplicates(subset=[COL_TEXT])

    # 修乱码（42% 的评论有）+ 去掉 "Trip Verified |" 模板前缀（43% 有）
    df[COL_TEXT] = df[COL_TEXT].astype(str).map(clean_review)
    df = df[df[COL_TEXT].str.len() > 20]          # 清洗后过短的丢掉

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    return df


def sample_reviews(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """按总体评分分层抽样，保证好评差评大致各半。"""
    pos = df[df[COL_OVERALL] >= 7]
    neg = df[df[COL_OVERALL] <= 4]
    half = n // 2
    take = lambda pool, k: pool.sample(min(k, len(pool)), random_state=42)
    out = pd.concat([take(pos, half), take(neg, n - half)])
    return out.sample(frac=1, random_state=42)      # 打乱


def to_chunks(reviews: pd.DataFrame) -> pd.DataFrame:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])

    rating_cols = {aspect: col for aspect, (col, _) in ASPECTS.items()}
    rows = []
    texts = reviews[COL_TEXT].astype(str).tolist()

    for i, doc in enumerate(nlp.pipe(texts, batch_size=100)):
        r = reviews.iloc[i]
        review_id = f"R{i:06d}"
        for j, sent in enumerate(doc.sents):
            s = sent.text.strip()
            if len(s.split()) < MIN_WORDS:
                continue
            row = {
                "chunk_id": f"{review_id}_s{j:02d}",
                "review_id": review_id,
                "text": s,
                "review_text": r[COL_TEXT],
                "airline": r[COL_AIRLINE],
                "cabin": r.get(COL_CABIN),
                "traveller_type": r.get(COL_TRAVELLER),
                "route": r.get(COL_ROUTE),
                "date": r[COL_DATE],
                "overall_rating": r.get(COL_OVERALL),
                "recommended": r.get(COL_RECOMMENDED),
            }
            for aspect, col in rating_cols.items():
                row[f"r_{aspect}"] = r.get(col)
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="抽多少条评论（默认 300）")
    args = ap.parse_args()

    print("读取并清洗原始数据 ...")
    df = load_clean()
    print(f"  可用评论 {len(df):,} 条")

    reviews = sample_reviews(df, args.n)
    print(f"  抽样 {len(reviews)} 条（好评差评各半）")

    print("切句中 ...")
    chunks = to_chunks(reviews)

    CHUNKS_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(CHUNKS_SAMPLE)

    print(f"\n✅ 已生成 {len(chunks):,} 句 -> {CHUNKS_SAMPLE}")
    print(f"   来自 {chunks['review_id'].nunique()} 条真实评论")
    print(f"   平均每条 {len(chunks)/chunks['review_id'].nunique():.1f} 句")
    print(f"\n随机看 3 句：")
    for t in chunks["text"].sample(3, random_state=1):
        print(f"   · {t[:90]}")


if __name__ == "__main__":
    main()
