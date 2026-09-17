"""生成预留句池（E1001 起），供仲裁剔除句子后补足金标准用。

用法：
    python scripts/make_reserve.py            # 默认 100 句
    python scripts/make_reserve.py --k 150

产出（都不进标注系统，先躺在仓库里）：

    data/eval/reserve.csv        ← 真要补足时，按需取前几条发给标注员
    data/eval/reserve_meta.csv   ← 元数据，同样标注完成之后才 join

------------------------------------------------------------------------
原理：预留 = 同一套抽样规则跑 --n (1000+k)，减去已冻结的那 1,000 句

已实测：`--n` 变大时产出是**超集**——原来的 1,000 句一条不少地保留
（候选池不变是前提）。所以差集就是干净的、符合同样分层和过滤规则的新句。

⚠️ 为什么不直接跑 `make_eval_set.py --n 1100`

那样会把 `data/eval/to_label.csv` 覆盖掉。虽然句子是超集，但 **id 是洗牌后
按位置编的**——实测 `--n 1100` 之后，仍指向同一句的 id 是 0/1000。而标注库里
1,000 句的 id 已经和三个人的标注绑定了，文件一变就和库失配，按 id 做的任何
join 都会静默出错。

所以本脚本把抽样的输出重定向到临时目录，`data/eval/to_label.csv` 和
`to_label_meta.csv` 一个字节都不动。

⚠️ 前提：`chunks.parquet` 不能重建

超集性质依赖候选池不变。当前抽样用的是 `df.sample(random_state=)`，按行位置
抽——池子少掉几百句，1,000 句里能活下来的只有个位数（实测排掉 355 句 → 存活
5/1000）。真要重建语料，先把 stash 里的 `stable_pick`（按 chunk_id 哈希排序）
apply 了再重建，那个版本同样情况能活 999/1000。
------------------------------------------------------------------------
"""
import argparse
import runpy
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHUNKS, DATA_EVAL  # noqa: E402

FROZEN = DATA_EVAL / "to_label.csv"
SAMPLER = ROOT / "scripts" / "make_eval_set.py"
BLANK_COLS = ["subjectivity", "polarity", "aspect", "note"]


def run_sampler(n: int, seed: int, out_dir: Path) -> pd.DataFrame:
    """跑一遍抽样，但把产出写到 out_dir，不碰 data/eval/。"""
    import src.config as cfg

    original = cfg.DATA_EVAL
    cfg.DATA_EVAL = out_dir
    argv = sys.argv
    sys.argv = ["make_eval_set.py", "--n", str(n), "--seed", str(seed)]
    try:
        runpy.run_path(str(SAMPLER), run_name="__main__")
    except SystemExit as exc:
        if exc.code:
            raise
    finally:
        cfg.DATA_EVAL = original
        sys.argv = argv
    return (pd.read_csv(out_dir / "to_label.csv"),
            pd.read_csv(out_dir / "reserve_meta.csv"
                        if (out_dir / "reserve_meta.csv").exists()
                        else out_dir / "to_label_meta.csv"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=100, help="预留多少句（默认 100）")
    ap.add_argument("--seed", type=int, default=42, help="必须和当初生成金标准时一致")
    args = ap.parse_args()

    if not CHUNKS.exists():
        sys.exit(f"❌ 找不到 {CHUNKS}\n   先跑：python -m src.build_kb")
    if not FROZEN.exists():
        sys.exit(f"❌ 找不到 {FROZEN}")

    frozen = pd.read_csv(FROZEN)
    n_frozen = len(frozen)
    frozen_text = set(frozen["text"].str.strip())
    print(f"已冻结的金标准：{n_frozen:,} 句（{FROZEN.name}）")
    print(f"目标：再抽 {args.k} 句作预留\n")

    with tempfile.TemporaryDirectory() as tmp:
        wide, wide_meta = run_sampler(n_frozen + args.k, args.seed, Path(tmp))

    # ---- 差集就是预留句 ----
    keep = ~wide["text"].str.strip().isin(frozen_text)
    extra = wide[keep].copy()
    survived = len(wide) - len(extra)
    print(f"\n超集校验：已冻结的 {n_frozen:,} 句中，{survived:,} 句仍在 "
          f"--n {n_frozen + args.k} 的产出里")
    if survived != n_frozen:
        sys.exit(
            f"❌ 超集性质被破坏（只剩 {survived:,}/{n_frozen:,}）。\n"
            "   多半是 chunks.parquet 被重建过，候选池变了。\n"
            "   这种情况下不要继续——先确认语料版本，或改用 stable_pick 抽样。")
    if len(extra) != args.k:
        sys.exit(f"❌ 差集是 {len(extra)} 句，期望 {args.k} 句")

    # ---- 编号：按 chunk_id 排序后顺序编，和抽样时的洗牌无关，可复现 ----
    extra = (extra.merge(wide_meta[["id", "chunk_id"]], on="id", how="left")
                  .sort_values("chunk_id")
                  .reset_index(drop=True))
    extra["new_id"] = [f"E{n_frozen + i:04d}" for i in range(1, len(extra) + 1)]

    meta = wide_meta[wide_meta["id"].isin(extra["id"])].copy()
    meta = meta.merge(extra[["id", "new_id"]], on="id")
    meta["id"] = meta["new_id"]
    meta = meta.drop(columns=["new_id"]).sort_values("id")

    out_label = pd.DataFrame({"id": extra["new_id"], "text": extra["text"]})
    for c in BLANK_COLS:
        out_label[c] = ""
    out_label = out_label.sort_values("id")

    p1 = DATA_EVAL / "reserve.csv"
    p2 = DATA_EVAL / "reserve_meta.csv"
    out_label.to_csv(p1, index=False, encoding="utf-8-sig")
    meta.to_csv(p2, index=False, encoding="utf-8-sig")

    print(f"\n✅ 已生成 {p1}   {len(out_label)} 句　{out_label['id'].iloc[0]}–{out_label['id'].iloc[-1]}")
    print(f"✅ 已生成 {p2}   {len(meta)} 行 × {len(meta.columns)} 列")
    print(f"\n   与金标准文本重复: {len(set(out_label['text'].str.strip()) & frozen_text)}（应为 0）")
    print(f"   正负平衡: {meta['weak_polarity'].value_counts().to_dict()}")
    print(f"   来自 {meta['review_id'].nunique()} 条不同评论，{meta['airline'].nunique()} 家航司")
    print(f"\n⚠️  {FROZEN.name} 和 to_label_meta.csv 未被修改——它们和标注库严格对齐，"
          "动了会让\n    三个人已标的数据对不上号。预留句要启用时，"
          "从 reserve.csv 按需取，走追加流程。")


if __name__ == "__main__":
    main()
