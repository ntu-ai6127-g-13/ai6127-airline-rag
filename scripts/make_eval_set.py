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
  3. 排除 is_non_english（让人标法语句子的英文情感，纯属浪费人工；
     而且分类器跑在英文上，非英语进金标准会把评测搅浑）
  4. 排除 is_fragment（以标点开头的残片，没有完整语义可判）
  5. 排除太短的句子（< 5 词，多是 "Never again." 这类，标注价值低）
  4. 按总体评分分层：>=7 偏正面、<=4 偏负面，各占一半；5-6 分丢弃
  5. 每层内部**按六个维度尽量均衡**，避免全是 value（它覆盖率 98% 最高，
     不控制的话会挤掉 entertainment 和 ground）
  6. 打乱顺序，避免标注员形成心理惯性
"""
import argparse
import hashlib
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


def stable_pick(pool: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """从 pool 里稳定地取 k 条。

    **不要用 df.sample(random_state=...)** —— 它按行位置抽，
    池子里少掉任何一行，后面所有行的抽中结果都会错位。
    实测：过滤规则多排掉 355 句，1,000 句标注集就被换掉 752 句（75%），
    已经标过的部分全部作废。

    改成按 chunk_id 的哈希排序取前 k 条：每条句子中不中只取决于它自己的 id，
    和池子里有没有别的句子无关。以后再加过滤规则，只有被排掉的那几句会被替换。
    """
    if k <= 0 or pool.empty:
        return pool.iloc[:0]
    h = pool["chunk_id"].map(
        lambda c: hashlib.md5(f"{c}|{seed}".encode()).hexdigest())
    # 用 sort_values 而不是 nsmallest —— 后者不支持字符串 dtype
    return pool.loc[h.sort_values().index[:min(k, len(pool))]]


def take_balanced(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """从一层里取 n 条，六个维度尽量均分，不够的用 none 补。"""
    names = list(ASPECTS)
    per = int(n * 0.72 / len(names))        # 72% 给六个维度，其余留给 none
    picked = []
    for a in names:
        sub = pool[pool["_aspect"] == a]
        k = min(per, len(sub))
        if k:
            picked.append(stable_pick(sub, k, seed))
    got = pd.concat(picked) if picked else pool.iloc[:0]

    # 剩下的名额从还没被选中的里面补（优先 none，让语料更像真实分布）
    rest = pool.drop(index=got.index)
    need = n - len(got)
    if need > 0:
        none_pool = rest[rest["_aspect"] == "none"]
        k = min(need, len(none_pool))
        if k:
            got = pd.concat([got, stable_pick(none_pool, k, seed)])
            rest = rest.drop(index=got.index.intersection(rest.index))
        need = n - len(got)
        if need > 0 and len(rest):
            got = pd.concat([got, stable_pick(rest, need, seed)])
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="总共抽多少句（默认 1000）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="覆盖已存在的评测集。⚠️ 评测集一旦开标就等于冻结了，"
                         "重抽会让已有的人工标注全部对不上号")
    args = ap.parse_args()

    if not CHUNKS.exists():
        sys.exit(f"❌ 找不到 {CHUNKS}\n   先跑：python -m src.build_kb")

    # ---- 覆盖防护 ----
    # 评测集一旦发给标注员就是冻结的：句子的 id 已经和数据库里的人工标注绑死，
    # 重抽一次，E0001 就指向另一句话，而按 id 做的 join 不会报错，只会静默出错。
    # 实测过一次：过滤规则一变，1,000 句里 765 句被换掉，三个人标的东西全部作废。
    # 所以这里 fail closed —— 要重抽，先自己确认标注可以作废，再加 --force。
    existing = [p for p in (DATA_EVAL / "to_label.csv", DATA_EVAL / "to_label_meta.csv")
                if p.exists()]
    if existing and not args.force:
        listing = "\n".join(f"     {p}  （{sum(1 for _ in p.open(encoding='utf-8-sig')) - 1:,} 行）"
                            for p in existing)
        sys.exit(
            "❌ 评测集已存在，拒绝覆盖：\n"
            f"{listing}\n\n"
            "   这些文件和标注数据库里的句子是按 id 绑定的。重新抽样会把 id 指向\n"
            "   别的句子，已有的人工标注会静默错位 —— 不报错，但全错。\n\n"
            "   · 只是想加预留句 → python scripts/make_reserve.py（不碰现有文件）\n"
            "   · 确实要重抽且接受已有标注作废 → 加 --force")

    print(f"读取 {CHUNKS.name} ...")
    cols = ["chunk_id", "review_id", "text", "airline", "cabin", "date",
            "overall_rating", "n_words", "is_route_only", "dup_count",
            "is_non_english", "is_fragment"] + \
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
    # 这两列是 2026-09-13 加的质量标记。非英语句尤其要排 ——
    # 让标注员去判断一句法语的英文情感，既浪费人工又会把金标准搅浑。
    for col, label in [("is_non_english", "排除非英语句"), ("is_fragment", "排除残片句")]:
        if col in df.columns:
            before = len(df)
            df = df[~df[col].fillna(False)]
            print(f"   {label}      -{before - len(df):,}")
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
    # 打乱顺序（避免标注员形成心理惯性）。同样用哈希，保证顺序也稳定。
    order = sample["chunk_id"].map(
        lambda c: hashlib.md5(f"shuffle|{c}|{args.seed}".encode()).hexdigest())
    sample = sample.loc[order.sort_values().index].reset_index(drop=True)
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
