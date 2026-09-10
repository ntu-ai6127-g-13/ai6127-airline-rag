"""抽 1,000 句给标注组（作业任务 A6）。

用法：
    python -m src.build_kb                 # 先有 chunks.parquet
    python scripts/make_eval_set.py        # 抽样，几秒
    python scripts/make_eval_set.py --n 1200 --seed 7

产出两个文件：

    data/eval/to_label.csv        ← 发给三位标注员的，只有 id + text + 四个空列
    data/eval/to_label_meta.csv   ← 元数据，标注**完成之后**再按 id join 回来

------------------------------------------------------------------------
为什么拆成两个文件 ⭐ 这是方法论问题，不是洁癖

手册第 16 节的示例代码把 `r_seat`、`overall_rating` 这些**评分列一起导给
标注员**。但弱标签（训练用的那套）正是从这些评分推出来的：

    评分 4-5 星 -> positive      评分 1-2 星 -> negative

标注员一旦看到「这条评论 overall 给了 9 分」，判断就会被锚定。于是：

    人工金标准  ←被评分影响→  弱标签  ←推出→  分类器

金标准和被评测对象共享同一个信息源，**评测就等于自己考自己**，
算出来的 P/R/F1 会系统性偏高，而且这个偏差无法在报告里辩护。

所以 `to_label.csv` 里**只有句子本身**。航司、日期、评分、原评论全部
放到 meta 文件，标注完再按 `id` join 回去做分析。
------------------------------------------------------------------------

抽样规则：
  1. 排除 is_route_only（纯航线元数据，没有情感极性，标了浪费人力）
  2. 排除 dup_count > 1（同一句标两遍没有意义，作业也要求评测集无重复）
  3. 排除太短的句子（< 5 词，多是 "Never again." 这类，标注价值低）
  4. 按总体评分分层：>=7 偏正面、<=4 偏负面，各占一半；5-6 分丢弃
  5. 每层内部**按六个维度尽量均衡**，避免全是 value（它覆盖率 98% 最高，
     不控制的话会挤掉 entertainment 和 ground）
  6. 打乱顺序，避免标注员形成心理惯性
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ASPECTS, CHUNKS, DATA_EVAL

MIN_WORDS = 5           # 少于这么多词的不进标注集
POS_MIN = 7             # overall >= 7 算偏正面
NEG_MAX = 4             # overall <= 4 算偏负面

# 标注员要填的四列，导出时留空
BLANK_COLS = ["subjectivity", "polarity", "aspect", "note"]


def build_aspect_matcher():
    """把 config 里的关键词编译成正则。用词边界，避免 'seat' 匹配到 'seated'。"""
    out = {}
    for name, (_col, kws) in ASPECTS.items():
        pat = "|".join(re.escape(k) for k in sorted(kws, key=len, reverse=True))
        out[name] = re.compile(rf"(?<![a-z])({pat})(?![a-z])", re.IGNORECASE)
    return out


def tag_aspect(text: str, matchers) -> str:
    """这句大概在讲哪个维度。命中多个就取命中次数最多的；都不命中返回 none。

    ⚠️ 这只是**抽样时用来配平**的粗标记，不写进 to_label.csv。
    真正的 aspect 由标注员独立判断——把它给标注员看会造成锚定。
    """
    best, best_n = "none", 0
    for name, pat in matchers.items():
        n = len(pat.findall(text))
        if n > best_n:
            best, best_n = name, n
    return best


def take_balanced(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """从一层里取 n 条，六个维度尽量均分，不够的用 none 补。"""
    names = list(ASPECTS)
    per = int(n * 0.72 / len(names))        # 72% 给六个维度，其余留给 none
    picked = []
    for a in names:
        sub = pool[pool["_aspect"] == a]
        k = min(per, len(sub))
        if k:
            picked.append(sub.sample(k, random_state=seed))
    got = pd.concat(picked) if picked else pool.iloc[:0]

    # 剩下的名额从还没被选中的里面补（优先 none，让语料更像真实分布）
    rest = pool.drop(index=got.index)
    need = n - len(got)
    if need > 0:
        none_pool = rest[rest["_aspect"] == "none"]
        k = min(need, len(none_pool))
        if k:
            got = pd.concat([got, none_pool.sample(k, random_state=seed)])
            rest = rest.drop(index=got.index.intersection(rest.index))
        need = n - len(got)
        if need > 0 and len(rest):
            got = pd.concat([got, rest.sample(min(need, len(rest)), random_state=seed)])
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="总共抽多少句（默认 1000）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not CHUNKS.exists():
        sys.exit(f"❌ 找不到 {CHUNKS}\n   先跑：python -m src.build_kb")

    print(f"读取 {CHUNKS.name} ...")
    cols = ["chunk_id", "review_id", "text", "airline", "cabin", "date",
            "overall_rating", "n_words", "is_route_only", "dup_count"] + \
           [f"r_{a}" for a in ASPECTS]
    df = pd.read_parquet(CHUNKS, columns=cols)
    print(f"   全量 {len(df):,} 句")

    # ---- 过滤 ----
    before = len(df)
    df = df[~df["is_route_only"]]
    print(f"   排除纯航线元数据  -{before - len(df):,}")
    before = len(df)
    df = df[df["dup_count"] == 1]
    print(f"   排除重复句        -{before - len(df):,}")
    before = len(df)
    df = df[df["n_words"] >= MIN_WORDS]
    print(f"   排除 <{MIN_WORDS} 词的短句   -{before - len(df):,}")
    before = len(df)
    df = df[df["overall_rating"].notna()]
    print(f"   排除无总体评分    -{before - len(df):,}  → 候选池 {len(df):,}")

    # ---- 打维度标记（只用于配平，不导出）----
    matchers = build_aspect_matcher()
    df = df.copy()
    df["_aspect"] = df["text"].map(lambda t: tag_aspect(t, matchers))

    # ---- 分层 ----
    pos = df[df["overall_rating"] >= POS_MIN]
    neg = df[df["overall_rating"] <= NEG_MAX]
    half = args.n // 2
    print(f"\n分层：偏正面 {len(pos):,} 句 / 偏负面 {len(neg):,} 句"
          f"（5–6 分的 {len(df) - len(pos) - len(neg):,} 句按规范丢弃）")
    if len(pos) < half or len(neg) < half:
        sys.exit(f"❌ 某一层不够 {half} 句")

    sample = pd.concat([
        take_balanced(pos, half, args.seed),
        take_balanced(neg, args.n - half, args.seed + 1),
    ])

    # ---- 去重（作业硬性要求评测集无重复）----
    before = len(sample)
    sample = sample.drop_duplicates(subset=["text"])
    if before != len(sample):
        print(f"   ⚠️ 去掉 {before - len(sample)} 条重复文本")

    # ---- 打乱 ----
    sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    sample.insert(0, "id", [f"E{i:04d}" for i in range(1, len(sample) + 1)])

    DATA_EVAL.mkdir(parents=True, exist_ok=True)

    # ---- ① 发给标注员的：只有 id + text + 四个空列 ----
    to_label = sample[["id", "text"]].copy()
    for c in BLANK_COLS:
        to_label[c] = ""
    p1 = DATA_EVAL / "to_label.csv"
    to_label.to_csv(p1, index=False, encoding="utf-8-sig")   # -sig 让 Excel 不乱码

    # ---- ② 元数据：标注完成之后再 join 回来 ----
    meta = sample[["id", "chunk_id", "review_id", "airline", "cabin", "date",
                   "overall_rating"] + [f"r_{a}" for a in ASPECTS]].copy()
    meta["weak_polarity"] = sample["overall_rating"].map(
        lambda v: "positive" if v >= POS_MIN else "negative")
    meta["aspect_keyword_guess"] = sample["_aspect"]
    p2 = DATA_EVAL / "to_label_meta.csv"
    meta.to_csv(p2, index=False, encoding="utf-8-sig")

    # ---- 报告 ----
    print(f"\n✅ 已生成 {p1}   {len(to_label):,} 行 × {len(to_label.columns)} 列")
    print(f"✅ 已生成 {p2}   {len(meta):,} 行 × {len(meta.columns)} 列")
    print("\n   分层平衡：")
    vc = meta["weak_polarity"].value_counts()
    for k, v in vc.items():
        print(f"     {k:9} {v:>4} 句 ({v/len(meta):.1%})")
    print("\n   维度分布（关键词粗标，仅供参考）：")
    for k, v in sample["_aspect"].value_counts().items():
        print(f"     {k:14} {v:>4} 句 ({v/len(sample):.1%})")
    print(f"\n   文本唯一: {'✅' if sample['text'].nunique() == len(sample) else '❌'}"
          f"   来自 {sample['review_id'].nunique():,} 条不同评论"
          f"   {sample['airline'].nunique()} 家航司")
    print("\n⚠️  to_label.csv 里**故意不含**航司/日期/评分——"
          "标注员看到评分会被锚定，\n"
          "    金标准就和弱标签相关了，评测会失去意义。元数据在 meta 文件里，"
          "标注完再 join。")


if __name__ == "__main__":
    main()
