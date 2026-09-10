"""构建全组共用的数据契约 chunks.parquet。

负责人：A 知识库组
对应手册第 13–15 节

用法：
    python -m src.build_kb              # 全量
    python -m src.build_kb --limit 500  # 只处理前 500 条评论（调试用）

流程：
    原始 xlsx → 滤空行 → 去重 → 清洗（修乱码 + 去模板）→ 切句 → chunks.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clean_text import (clean_field, clean_review, extract_route_from_text,
                            is_route_only, parse_date_flown, parse_review_date,
                            parse_route)
from src.config import (ASPECTS, CHUNKS, COL_AIRCRAFT, COL_AIRLINE, COL_CABIN,
                        COL_DATE, COL_DATE_FLOWN, COL_OVERALL, COL_RECOMMENDED,
                        COL_ROUTE, COL_TEXT, COL_TRAVELLER, DATA_PROC, DATA_RAW,
                        REVIEWS)

MIN_SENT_WORDS = 3        # 少于这么多词的句子丢弃（"Ok." "Never again."）
MIN_SENT_ALPHA = 5        # 字母少于这么多的句子丢弃（"1/21/14 - 2/1/14." 这种日期行）
MIN_REVIEW_CHARS = 20     # 清洗后过短的评论丢弃

# parquet 压缩方式。实测同样的数据：
#   SNAPPY（pandas 默认）              96 MB
#   ZSTD 默认等级(1)                   43 MB
#   ZSTD 等级 9                        31 MB   ← 用这个
# review_text 在每一句上都重复了一份（平均 8.9 次），zstd 对这种重复特别有效。
# 等级调高只是**写**的时候慢几秒，读取速度和等级无关 —— 这个文件建一次读无数次，
# 所以直接拉到 9。
# 另外试过「去掉 review_text 另存 parent 表」，两个文件加起来反而更大（见
# docs/实验发现记录.md B4），所以契约保持不变，只换压缩方式。
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 9


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """写 parquet。用 DuckDB 而不是 pandas —— 同样是 zstd-9，DuckDB 的字典编码
    和页布局更好，实测小约 16%（36.8 MB -> 31.0 MB）。读的一侧毫无差别，
    pd.read_parquet() 照常能读。duckdb 本来就是项目依赖（数据浏览界面在用）。"""
    import duckdb
    con = duckdb.connect()
    con.register("_df", df)
    con.execute(
        f"COPY (SELECT * FROM _df) TO '{path.as_posix()}' "
        f"(FORMAT parquet, COMPRESSION {COMPRESSION}, "
        f"COMPRESSION_LEVEL {COMPRESSION_LEVEL})")
    con.close()


def find_raw() -> Path:
    files = [f for f in DATA_RAW.iterdir()
             if f.suffix.lower() in {".csv", ".xlsx", ".xls"}]
    if not files:
        sys.exit("❌ data/raw/ 里没有数据。先跑 python scripts/download_data.py")
    return files[0]


def load_and_clean(limit: int | None = None) -> pd.DataFrame:
    """读取 → 滤空行 → 去重 → 清洗文本 → 解析日期。"""
    path = find_raw()
    print(f"[1/3] 读取 {path.name} ...")
    t0 = time.perf_counter()
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} \
        else pd.read_csv(path)
    print(f"      原始 {len(df):,} 行  ({time.perf_counter()-t0:.1f}s)")

    # 这个数据集有一半是全空的填充行
    blank = df[COL_TEXT].isna().sum()
    df = df[df[COL_TEXT].notna()].copy()
    print(f"      滤掉空行 {blank:,} → 剩 {len(df):,}")

    before = len(df)
    df = df.drop_duplicates(subset=[COL_TEXT])
    print(f"      去重 -{before - len(df):,} → 剩 {len(df):,}")

    # 修乱码（42% 的评论有）+ 去 "Trip Verified |" 模板前缀（43%）
    t0 = time.perf_counter()
    df[COL_TEXT] = df[COL_TEXT].astype(str).map(clean_review)
    df = df[df[COL_TEXT].str.len() > MIN_REVIEW_CHARS]

    # 元数据列同样有乱码（route 里有 'Phuket to DÃ¼sseldorf'），之前漏了。
    # 航司名要是带乱码，按航司聚合就会分裂成两个不同的航司。
    for col in (COL_AIRLINE, COL_ROUTE, COL_CABIN, COL_TRAVELLER):
        if col in df.columns:
            df[col] = df[col].map(clean_field)
    print(f"      清洗完成，剩 {len(df):,} 条  ({time.perf_counter()-t0:.1f}s)")

    # 日期。**别用 pd.to_datetime() 直接解析** —— review_date 是 "2nd May 2019"
    # 这种带序数后缀的字符串，默认解析器认得 th 却认不了 st/nd/rd，会静默失败
    # 21.6%（13,219 条），看起来像"这些评论没日期"，其实原始数据 100% 有。
    before_dt = df[COL_DATE].notna().sum()
    df[COL_DATE] = parse_review_date(df[COL_DATE])
    print(f"      日期解析 {df[COL_DATE].notna().sum():,}/{before_dt:,} "
          f"({df[COL_DATE].notna().sum()/before_dt:.1%})")

    # date_flown：实际乘机月份，**只有月精度**。当初因为混合类型（datetime + 'April 2019'）
    # 存不进 parquet 被整列丢掉，统一成 datetime 就能留下了。
    if COL_DATE_FLOWN in df.columns:
        df[COL_DATE_FLOWN] = parse_date_flown(df[COL_DATE_FLOWN])
        print(f"      date_flown 有值 {df[COL_DATE_FLOWN].notna().sum():,} "
              f"({df[COL_DATE_FLOWN].notna().mean():.1%})")

    # aircraft：也是当初因为混合类型（有 'A320' 也有裸数字 787）被丢掉的
    if COL_AIRCRAFT in df.columns:
        df[COL_AIRCRAFT] = df[COL_AIRCRAFT].map(
            lambda v: None if pd.isna(v) else clean_field(str(v)))
        print(f"      aircraft 有值 {df[COL_AIRCRAFT].notna().sum():,} "
              f"({df[COL_AIRCRAFT].notna().mean():.1%})，"
              f"{df[COL_AIRCRAFT].nunique():,} 种机型")

    # route 有 38.7% 是空的。评论正文开头常常就写着航线，捞回来一部分。
    miss = df[COL_ROUTE].isna()
    recovered = df.loc[miss, COL_TEXT].map(extract_route_from_text)
    df.loc[miss, COL_ROUTE] = recovered
    print(f"      route 从正文回收 {recovered.notna().sum():,} 条 → "
          f"覆盖率 {df[COL_ROUTE].notna().mean():.1%}")

    if limit:
        df = df.head(limit)
        print(f"      ⚠️  --limit {limit}，只处理前 {len(df):,} 条")

    df = df.reset_index(drop=True)

    # 只保留契约需要的列。原始表里 date_flown / aircraft 等列类型混杂
    # （同一列既有字符串又有数字），parquet 存不了，而且我们也用不到。
    keep = [COL_TEXT, COL_AIRLINE, COL_CABIN, COL_TRAVELLER, COL_ROUTE,
            COL_DATE, COL_DATE_FLOWN, COL_AIRCRAFT,
            COL_OVERALL, COL_RECOMMENDED] + \
           [col for _, (col, _) in ASPECTS.items()]
    df = df[[c for c in keep if c in df.columns]]

    # review_id 在这里就定下来，chunks 里用同一套编号，两张表可以直接 join。
    # （以前 reviews.parquet 没有这一列，只能靠行号去猜，很脆。）
    df.insert(0, "review_id", [f"R{i:06d}" for i in range(len(df))])

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    out = DATA_PROC / f"reviews_debug{limit}.parquet" if limit else REVIEWS
    write_parquet(df, out)
    print(f"      已存 {out.name}（保留 {len(df.columns)} 列）")
    return df


def to_chunks(reviews: pd.DataFrame) -> pd.DataFrame:
    """按句切分，组装成契约规定的 28 个字段。"""
    import spacy
    from tqdm import tqdm

    # 只需要切句，关掉其余组件加速
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])

    print(f"[2/3] 切句 {len(reviews):,} 条评论 ...")
    rating_cols = {a: col for a, (col, _) in ASPECTS.items()}
    rows = []
    texts = reviews[COL_TEXT].astype(str).tolist()

    t0 = time.perf_counter()
    for i, doc in enumerate(tqdm(nlp.pipe(texts, batch_size=200),
                                 total=len(texts), unit="条", ncols=80)):
        r = reviews.iloc[i]
        review_id = r["review_id"]
        r_from, r_to, r_via = parse_route(r.get(COL_ROUTE))
        for j, sent in enumerate(doc.sents):
            s = sent.text.strip()
            if len(s.split()) < MIN_SENT_WORDS:
                continue
            # 丢掉没有实际文字的碎片，例如 "1/21/14 - 2/1/14." "(5:55, 6:15, 7:30)"
            if sum(ch.isalpha() for ch in s) < MIN_SENT_ALPHA:
                continue
            row = {
                "chunk_id": f"{review_id}_s{j:02d}",
                "review_id": review_id,
                "sent_idx": j,
                "text": s,
                "review_text": r[COL_TEXT],
                "airline": r[COL_AIRLINE],
                "cabin": r.get(COL_CABIN),
                "traveller_type": r.get(COL_TRAVELLER),
                "route": r.get(COL_ROUTE),
                "route_from": r_from,
                "route_to": r_to,
                "route_via": r_via,
                "aircraft": r.get(COL_AIRCRAFT),
                "date": r[COL_DATE],
                "date_flown": r.get(COL_DATE_FLOWN),
                "overall_rating": r.get(COL_OVERALL),
                "recommended": r.get(COL_RECOMMENDED),
            }
            for aspect, col in rating_cols.items():
                row[f"r_{aspect}"] = r.get(col)
            rows.append(row)

    print(f"      切出 {len(rows):,} 句  ({time.perf_counter()-t0:.0f}s)")
    return add_derived(pd.DataFrame(rows))


def add_derived(chunks: pd.DataFrame) -> pd.DataFrame:
    """加派生列。这些都是**语料级**统计，必须等所有句子都切完才能算。"""
    t0 = time.perf_counter()

    # 结构列：有了 sent_idx + n_sents，取上下文窗口就是一个窗口函数的事，
    # 不用再去解析 chunk_id 字符串。首句/末句往往是总结句，是有用的检索信号。
    chunks["n_sents"] = chunks.groupby("review_id")["sent_idx"].transform("size")
    chunks["n_words"] = chunks["text"].str.split().str.len()

    # ⚠️ sent_idx 是**切句时的原始序号**，碎片句被丢掉之后会留下空洞
    #    （实测 11.5% 的评论有空洞）。所以**不能**用 sent_idx == n_sents-1
    #    判断末句 —— 那样会漏掉一部分。这里直接给一个显式的 is_last。
    #    末句常常是总结句（"Overall I would not fly them again."），
    #    对检索和情感聚合都是有用信号。
    #    保留 sent_idx 原样是为了守住 chunk_id == review_id_s{sent_idx} 这个不变式。
    chunks["is_last"] = (
        chunks["sent_idx"] == chunks.groupby("review_id")["sent_idx"].transform("max"))

    # 纯航线元数据句（3.03%），只打标记不删
    chunks["is_route_only"] = chunks["text"].map(is_route_only)

    # 这句话在全语料里出现几次。检索时同一句话可能来自几十条不同评论，
    # 不去重会把 top-K 占满（见 docs/实验发现记录.md B6）。
    chunks["dup_count"] = chunks.groupby("text")["text"].transform("size").astype("int32")

    # 数值列瘦身：1–10 / 1–5 用不着 8 字节浮点。省空间是次要的（才 0.5 MB），
    # 主要是语义正确 —— 评分是整数，不该显示成 7.0。Int8 是 pandas 的可空整数。
    for col in ["overall_rating"] + [f"r_{a}" for a in ASPECTS]:
        if col in chunks.columns:
            chunks[col] = chunks[col].astype("Int8")

    print(f"      派生列完成  ({time.perf_counter()-t0:.0f}s)")
    return chunks


def report(chunks: pd.DataFrame) -> None:
    """打印语料统计（报告 Q1 需要这些数字）。"""
    print("\n[3/3] 语料统计（报告 Q1 用）")
    words = chunks["text"].str.split().str.len()
    all_words = " ".join(chunks["text"]).lower().split()

    print(f"      评论数（文档）  {chunks['review_id'].nunique():>10,}")
    print(f"      句子数（chunk） {len(chunks):>10,}")
    print(f"      总词数          {len(all_words):>10,}")
    print(f"      不重复词 type   {len(set(all_words)):>10,}")
    print(f"      每条平均 {len(chunks)/chunks['review_id'].nunique():.1f} 句，"
          f"每句平均 {words.mean():.1f} 词")

    print("\n      六个维度的评分覆盖率：")
    for aspect in ASPECTS:
        col = f"r_{aspect}"
        n = chunks[col].notna().sum()
        print(f"        {aspect:14s} {n:>8,} 句 ({n/len(chunks)*100:.0f}%)")

    print("\n      v3 新增列：")
    ro = int(chunks["is_route_only"].sum())
    dup = int((chunks["dup_count"] > 1).sum())
    print(f"        纯航线元数据句  {ro:>8,} 句 ({ro/len(chunks)*100:.2f}%)  <- 建索引时建议排除")
    print(f"        属于重复组的句  {dup:>8,} 句 ({dup/len(chunks)*100:.2f}%)  <- 检索后建议去重")
    print(f"        每句平均 {chunks['n_words'].mean():.1f} 词，"
          f"每条评论平均 {chunks['n_sents'].mean():.1f} 句")

    print(f"\n      作业门槛：")
    n_docs = chunks["review_id"].nunique()
    print(f"        文档 ≥10,000 : {n_docs:,}  {'✅' if n_docs >= 10_000 else '❌'}")
    print(f"        词数 ≥100,000: {len(all_words):,}  "
          f"{'✅' if len(all_words) >= 100_000 else '❌'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="只处理前 N 条评论（调试用）。产物写到 chunks_debug<N>.parquet，"
                         "不会覆盖正式的 chunks.parquet")
    args = ap.parse_args()

    t_start = time.perf_counter()
    reviews = load_and_clean(args.limit)
    chunks = to_chunks(reviews)

    # --limit 是调试用的，**绝不能覆盖正式文件** ——
    # 否则跑一次 --limit 200 就把全组共用的 54 万行契约悄悄换成 2,100 行，
    # 而且不会有任何报错。（我自己就踩过这个坑。）
    out = (CHUNKS.parent / f"chunks_debug{args.limit}.parquet") if args.limit else CHUNKS
    out.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(chunks, out)
    report(chunks)

    size_mb = out.stat().st_size / 1024**2
    print(f"\n✅ 已生成 {out}")
    if args.limit:
        print(f"   ⚠️  这是 --limit {args.limit} 的调试产物，**没有**覆盖 {CHUNKS.name}")
    print(f"   {len(chunks):,} 行 × {len(chunks.columns)} 列，{size_mb:.0f} MB")
    print(f"   总耗时 {(time.perf_counter()-t_start)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
