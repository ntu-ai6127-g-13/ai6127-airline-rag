"""C1: 从单方面句及对应评分构造可审计弱标签，不训练模型。

运行 ``python -m src.weak_labels --help``；说明见 docs/C1弱标签使用说明.md。
不修改输入知识库或人工标注。必须使用带有正确 is_eval 标记的新版知识库。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import unicodedata

import pandas as pd
import pyarrow.parquet as pq

from src.clean_text import is_clearly_non_english, is_obvious_fragment
from src.config import ASPECTS, CHUNKS, DATA_EVAL, OUT_DIR, ROOT, WEAK_LABELS
from src.weak_label_rules import VERSIONS, match_series, rule_description

RULE_VERSION = VERSIONS["context_v4"]
RATING_COLUMNS = [f"r_{a}" for a in ASPECTS]
REQUIRED = ["chunk_id", "review_id", "text", "is_eval", "is_route_only", *RATING_COLUMNS]
QUALITY_RULES = {"is_non_english": is_clearly_non_english,
                 "is_fragment": is_obvious_fragment}
REASONS = ["test_sentence", "test_review", "test_text", "empty_text", "route_only",
           "non_english", "fragment", "unsupported_aspect", "no_aspect", "multiple_aspects",
           "rating_missing", "rating_invalid", "rating_three",
           "duplicate_label_conflict", "duplicate_text"]


def normalize_text(text: str) -> str:
    """仅用于查重；原句不改写，保留否定词及标点。"""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def validate_ids(frame: pd.DataFrame, name: str, key: str = "chunk_id") -> None:
    for col in [key, "review_id"]:
        if col not in frame:
            raise ValueError(f"{name} 缺少 {col}")
        if not frame[col].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"{name} 的 {col} 有空值或非文本编号")
    if not frame[key].is_unique:
        raise ValueError(f"{name} 的 {key} 重复，拒绝按行顺序拼接")


def validate_eval_flags(kb: pd.DataFrame, test_ids: set[str]) -> dict:
    """逐行比较实际标记与评测 ID；仅比较计数无法发现标错行。"""
    if "is_eval" not in kb:
        raise ValueError("知识库缺少 is_eval，请用最新版 python -m src.build_kb 重建")
    if not pd.api.types.is_bool_dtype(kb.is_eval) or kb.is_eval.isna().any():
        raise ValueError("is_eval 必须是无空值的布尔列，请重建知识库")
    if not test_ids.issubset(set(kb.chunk_id)):
        raise ValueError("评测 ID 未全部出现在参考知识库中，请先统一版本")
    expected = kb.chunk_id.isin(test_ids)
    mismatches = int(kb.is_eval.ne(expected).sum())
    if mismatches:
        raise ValueError(f"is_eval 与评测清单有 {mismatches} 行不一致；即使数量相同也不能训练，请重建")
    return {"marked_sentences": int(kb.is_eval.sum()), "exact_id_match": True}


def load_test_exclusions(kb: pd.DataFrame, meta_path: Path, text_path: Path):
    """读取 ID/文本而非人工答案，核验所有测试句在参考知识库中的映射。"""
    meta = pd.read_csv(meta_path, usecols=["id", "chunk_id", "review_id"],
                       keep_default_na=False, dtype=str)
    gold = pd.read_csv(text_path, usecols=["id", "text"], keep_default_na=False, dtype=str)
    validate_ids(kb, "参考知识库")
    validate_ids(meta, "测试映射表")
    if (not meta.id.is_unique or not gold.id.is_unique or
            set(meta.id) != set(gold.id) or len(gold) < 1000):
        raise ValueError("测试集需要至少 1000 条，且两表 id 唯一、完全一致")
    if not gold.text.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("测试集存在无效文本")
    if not gold.text.map(normalize_text).is_unique:
        raise ValueError("测试集存在规范化重复文本，先统一评测数据")
    joined = meta.merge(gold, on="id", validate="one_to_one").merge(
        kb[["chunk_id", "review_id", "text"]], on="chunk_id", how="left",
        suffixes=("_eval", "_kb"), validate="one_to_one", indicator=True)
    if not (joined._merge.eq("both").all() and
            joined.review_id_eval.eq(joined.review_id_kb).all() and
            joined.text_eval.eq(joined.text_kb).all()):
        raise ValueError("测试句与参考知识库的 ID、父评论或原文不一致，请先统一版本")
    flag_info = validate_eval_flags(kb, set(meta.chunk_id))
    return set(meta.review_id), set(gold.text.map(normalize_text)), {
        "records": len(gold), "parent_reviews": int(meta.review_id.nunique()),
        "mapping_verified": True, "human_labels_read": False, **flag_info}


def build_weak_labels(frame: pd.DataFrame, test_reviews: set[str],
                      test_texts: set[str], ruleset: str = "context_v4"):
    """纯内存转换；互斥过滤计数可以加回输入行数。"""
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"输入缺少必需列：{sorted(missing)}")
    validate_ids(frame, "输入")
    d = frame.copy().reset_index(drop=True)
    if d.empty:
        raise ValueError("输入为空，没有符合规则的弱标签")
    if not d.text.dropna().map(lambda x: isinstance(x, str)).all():
        raise ValueError("text 中存在非文本值")
    for col in ["is_eval", "is_route_only", *QUALITY_RULES]:
        if col in d and (not pd.api.types.is_bool_dtype(d[col]) or d[col].isna().any()):
            raise ValueError(f"{col} 必须是无空值的布尔列")
    text = d.text.fillna("")
    norm = text.map(normalize_text)
    reason = pd.Series("", index=d.index)

    def exclude(mask, name):
        reason.loc[reason.eq("") & mask] = name

    # 先满足团队的句子级契约，再收紧到整条父评论及跨评论重复测试文本。
    exclude(d.is_eval, "test_sentence")
    exclude(d.review_id.isin(test_reviews), "test_review")
    exclude(norm.isin(test_texts), "test_text")
    exclude(text.str.strip().eq(""), "empty_text")
    exclude(d.is_route_only, "route_only")
    derived = []
    for col, rule in QUALITY_RULES.items():
        if col not in d:
            # 复用 build_kb 的同一函数；显式报告，绝不把缺列默认为 False。
            d[col] = text.map(rule)
            derived.append(col)
        exclude(d[col], "non_english" if col == "is_non_english" else "fragment")

    matches, unsupported = match_series(text, ruleset)
    exclude(unsupported, "unsupported_aspect")
    hits = matches.map(bool)
    count = hits.sum(axis=1)
    exclude(count.eq(0), "no_aspect")
    exclude(count.gt(1), "multiple_aspects")
    chosen = hits.idxmax(axis=1)
    scores = pd.Series(float("nan"), index=d.index)
    for aspect in ASPECTS:
        mask = reason.eq("") & chosen.eq(aspect)
        raw = d.loc[mask, f"r_{aspect}"]
        numeric = pd.to_numeric(raw, errors="coerce")
        scores.loc[mask] = numeric.astype(float)
        exclude(mask & d[f"r_{aspect}"].isna(), "rating_missing")
        exclude(mask & ~scores.isin([1, 2, 3, 4, 5]), "rating_invalid")
        exclude(mask & scores.eq(3), "rating_three")

    selected = reason.eq("")
    if not selected.any():
        raise ValueError("没有符合规则的弱标签；检查输入和过滤条件，未保存空文件")
    out = d.loc[selected, ["chunk_id", "review_id", "text", "is_eval"]].copy()
    out["aspect"] = chosen[selected]
    out["rating"] = scores[selected].astype("int8")
    out["polarity"] = out.rating.map(lambda v: "positive" if v >= 4 else "negative")
    out["rating_column"] = "r_" + out.aspect
    out["source"] = "weak"
    out["rule_version"] = VERSIONS[ruleset]
    out["matched_keywords"] = [json.dumps(sorted(set(matches.at[i, a])))
                                for i, a in zip(out.index, out.aspect)]
    out["_normalized_text"] = norm[selected]
    # 同文本出现相反标签或不同方面时不强选；其余重复只保留最小 chunk_id。
    out["_signature"] = out.aspect + "/" + out.polarity
    conflict = out.groupby("_normalized_text")["_signature"].transform("nunique").gt(1)
    reason.loc[out.index[conflict]] = "duplicate_label_conflict"
    out = out.loc[~conflict].sort_values("chunk_id")
    duplicate = out._normalized_text.duplicated()
    reason.loc[out.index[duplicate]] = "duplicate_text"
    out = out.loc[~duplicate].drop(columns=["_normalized_text", "_signature"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("没有符合规则的弱标签；检查输入和过滤条件，未保存空文件")

    checks = {
        "unique_chunk_ids": out.chunk_id.is_unique,
        "unique_normalized_texts": out.text.map(normalize_text).is_unique,
        "zero_test_review_overlap": not set(out.review_id) & test_reviews,
        "zero_test_text_overlap": not set(out.text.map(normalize_text)) & test_texts,
        "zero_is_eval_true": not out.is_eval.any(),
        "legal_aspects": out.aspect.isin(ASPECTS).all(),
        "legal_polarity": out.polarity.isin(["positive", "negative"]).all(),
        "rating_label_consistent": ((out.rating.isin([4, 5]) & out.polarity.eq("positive")) |
                                    (out.rating.isin([1, 2]) & out.polarity.eq("negative"))).all(),
    }
    dropped = {r: int(reason.eq(r).sum()) for r in REASONS}
    checks["all_input_rows_accounted_for"] = sum(dropped.values()) + len(out) == len(d)
    if not all(checks.values()):
        raise ValueError(f"输出校验失败：{checks}")
    distribution = {a: {p: int((out.aspect.eq(a) & out.polarity.eq(p)).sum())
                         for p in ["positive", "negative"]} for a in ASPECTS}
    report = {"rule_version": VERSIONS[ruleset], "rules": rule_description(ruleset), "input_rows": len(d), "output_rows": len(out),
              "output_reviews": int(out.review_id.nunique()), "excluded_by_first_reason": dropped,
              "aspect_polarity_counts": distribution,
              "polarity_counts": {str(k): int(v) for k, v in out.polarity.value_counts().items()},
              "quality_flags_derived_in_memory": derived,
              "checks": {k: bool(v) for k, v in checks.items()},
              "limits": ["关键词和评论评分是带噪声的弱监督，不是人工真值。",
                         "只产生六方面极性标签，不产生主观性或 overall/other 标签。",
                         "尚未划分 train/dev；C2 必须按 review_id 分组切分。",
                         "规则过滤和排重可能改变分布；不要求凑到文档示例数量。"]}
    return out, report


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_input(path: Path) -> pd.DataFrame:
    columns = pq.read_schema(path).names
    missing = set(REQUIRED) - set(columns)
    if missing:
        raise ValueError(f"{path.name} 缺少 {sorted(missing)}；请重建知识库和小样本，不要默认填 False")
    return pd.read_parquet(path, columns=REQUIRED + [c for c in QUALITY_RULES if c in columns])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=CHUNKS)
    ap.add_argument("--ruleset", choices=list(VERSIONS), default="context_v4", help="默认修正版；legacy/context_v3 用于复现旧结果")
    ap.add_argument("--kb", type=Path, default=CHUNKS, help="核对全部测试句的全量参考库")
    ap.add_argument("--eval-meta", type=Path, default=DATA_EVAL / "to_label_meta.csv")
    ap.add_argument("--eval-text", type=Path, default=DATA_EVAL / "to_label.csv")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--audit", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-per-class", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true", help="明确允许覆盖这次指定的产物")
    args = ap.parse_args(argv)
    start = time.perf_counter()
    try:
        if args.sample_per_class < 1:
            raise ValueError("--sample-per-class 必须至少为 1")
        args.input = args.input.resolve()
        default_output = WEAK_LABELS if args.ruleset == "legacy" else WEAK_LABELS.with_name(f"weak_labels_{args.ruleset}.parquet")
        output = (args.output or (default_output if args.input == CHUNKS.resolve() else
                                 args.input.with_name(args.input.stem + "_weak_labels_" + args.ruleset + ".parquet"))).resolve()
        report_path = (args.report or OUT_DIR / (output.stem + "_report.json")).resolve()
        audit_path = (args.audit or OUT_DIR / (output.stem + "_audit.csv")).resolve()
        inputs = {"input": args.input, "reference_kb": args.kb.resolve(),
                  "eval_meta": args.eval_meta.resolve(), "eval_text": args.eval_text.resolve()}
        destinations = [output, report_path, audit_path]
        if len(set(destinations)) != 3 or set(destinations) & set(inputs.values()):
            raise ValueError("输出路径彼此重合或覆盖输入，拒绝运行")
        if [p.suffix.lower() for p in destinations] != [".parquet", ".json", ".csv"]:
            raise ValueError("输出、报告、抽查文件扩展名应分别为 .parquet、.json、.csv")
        for p in destinations:
            if p.exists() and not args.overwrite:
                raise ValueError(f"产物已存在：{p}；请换输出名，或明确使用 --overwrite")
        print("[1/4] 读取并核对数据、测试映射 ...", flush=True)
        frame = read_input(args.input)
        kb = frame if args.input == args.kb.resolve() else read_input(args.kb)
        test_reviews, test_texts, eval_info = load_test_exclusions(kb, args.eval_meta, args.eval_text)
        validate_ids(frame, "输入")
        joined = frame[REQUIRED].merge(
            kb[REQUIRED], on="chunk_id", how="left",
            suffixes=("_input", "_kb"), validate="one_to_one", indicator=True)
        if not (joined._merge.eq("both").all() and joined.text_input.eq(joined.text_kb).all() and
                joined.review_id_input.eq(joined.review_id_kb).all()):
            raise ValueError("输入句子与参考知识库不一致")
        for col in ["is_eval", "is_route_only", *RATING_COLUMNS]:
            a, b = joined[col + "_input"], joined[col + "_kb"]
            if not (a.eq(b) | (a.isna() & b.isna())).fillna(False).all():
                raise ValueError(f"输入 {col} 与参考知识库不一致")
        print("[2/4] 过滤、单方面匹配、评分转标签及排重 ...", flush=True)
        out, report = build_weak_labels(frame, test_reviews, test_texts, ruleset=args.ruleset)
        report["evaluation_exclusion"] = eval_info
        report["inputs"] = {k: {"path": str(p), "sha256": sha256(p)} for k, p in inputs.items()}
        report["implementation_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in
                                           [Path(__file__), ROOT / "src/weak_label_rules.py", ROOT / "src/config.py", ROOT / "src/clean_text.py"]}
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        report["base_commit"] = git.stdout.strip() if git.returncode == 0 else None
        report["keyword_config"] = rule_description(args.ruleset)
        audit = pd.concat([g.sample(n=min(args.sample_per_class, len(g)), random_state=args.seed)
                           for _, g in out.groupby(["aspect", "polarity"], sort=True)], ignore_index=True)
        audit["human_aspect_correct"] = ""
        audit["human_polarity_correct"] = ""
        audit["note"] = ""
        report["audit"] = {"rows": len(audit), "seed": args.seed,
                           "sample_per_class": args.sample_per_class,
                           "status": "pending_human_review", "path": str(audit_path)}
        report["outputs"] = {"weak_labels": str(output), "report": str(report_path), "audit": str(audit_path)}
        print("[3/4] 输出校验通过，写出本地产物 ...", flush=True)
        temporary = []
        try:
            for p in destinations:
                p.parent.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(prefix=".weak_labels_", suffix=p.suffix, dir=p.parent)
                os.close(fd)
                temporary.append(Path(name))
            out.to_parquet(temporary[0], index=False, compression="zstd")
            pd.testing.assert_frame_equal(out, pd.read_parquet(temporary[0]))
            report["checks"]["parquet_roundtrip"] = True
            report["elapsed_seconds"] = round(time.perf_counter() - start, 2)
            temporary[1].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            audit.to_csv(temporary[2], index=False, encoding="utf-8-sig")
            for tmp, dest in zip(temporary, destinations):
                if args.overwrite:
                    os.replace(tmp, dest)
                else:
                    os.link(tmp, dest)  # 若运行期间出现同名产物，也不静默覆盖。
                    tmp.unlink()
        finally:
            for p in temporary:
                p.unlink(missing_ok=True)
        print(f"[4/4] 完成：{len(out):,} 条弱标签，{out.review_id.nunique():,} 条父评论")
        print(pd.crosstab(out.aspect, out.polarity).reindex(list(ASPECTS), fill_value=0).to_string())
        for p in destinations:
            print(p)
        print("自动检查通过；抽查表仍需人工确认，不代表弱标签全部正确。")
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"错误：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
