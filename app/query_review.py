"""Human development judgments for paired retrieval results; never model-generated scores."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import evaluate_query_modes

st.set_page_config(page_title="查询结果评审", layout="wide")
st.title("查询结果评审")
st.caption("比较同一查询的两组结果。请根据评论原文判断；系统不会代填相关性或回答成功分数。")
directory = Path(os.environ.get("QUERY_REVIEW_DIR", ROOT / "outputs/query_quality_benchmark_20261018_paired"))
seed = directory / "judgments_to_review.jsonl"
saved = directory / "judgments_reviewed.jsonl"
if not seed.exists():
    st.info("尚无配对查询结果，请先运行查询评测。"); st.stop()
rows = [json.loads(line) for line in (saved if saved.exists() else seed).read_text().splitlines() if line.strip()]
mapping = {row["query_id"]: row for row in rows}
query_ids = sorted({row["base_query_id"] for row in rows})
finished = sum(all(mapping[q + "_" + mode].get("human_reviewed") for mode in ["baseline", "labels"]) for q in query_ids)
st.sidebar.write(f"已保存 {finished}/{len(query_ids)} 个查询的评审")
qid = st.sidebar.selectbox("查询编号", query_ids)
reviewer = st.sidebar.text_input("评审人标识", help="例如姓名缩写，用于记录评分来源。")
doc = json.loads((directory / (qid + ".json")).read_text())
case, result = doc["case"], doc["result"]
aspect_names = {"seat":"座椅舒适度","staff":"空乘服务","food":"餐饮","entertainment":"娱乐与 Wi-Fi",
                "ground":"值机、行李与准点","value":"价格、退款与客服","overall":"总体体验","other":"清洁及其他"}
st.subheader(f"{qid} · {aspect_names.get(case['aspect'], case['aspect'])}")
st.write("观点要求：" + {None:"好评和差评均可", "positive":"好评", "negative":"差评"}[case["polarity"]])
st.write("筛选条件：", case["filters"])
if case.get("query"): st.write("补充关键词：", case["query"])
if case.get("comparison_group"):
    st.info("这是航司比较题中的一个航司组件。这里评审该组件，完整比较结论仍需结合另一航司单独验收。")
with st.expander("判定口径", expanded=False):
    st.write("相关原文应满足筛选条件，并直接支持所问方面；指定好评或差评时，还应符合对应观点。多方面句可以相关，不需要照抄主标签。")
    st.write("回答成功需要实际解决问题。仅有引用、数量统计或格式正确不代表成功。将实质性论断逐项与引用核对，再填写论断总数和有证据支持的数量。")
    st.write("能否回答取决于语料是否有足够证据，不能因为当前结果没找到就判为无答案。若语料确实无答案，再判断系统是否正确说明不足。")
    st.write("这些查询已用于开发，保存时固定为开发评审，不会标成独立测试。")

options = ["未判断", "是", "否"]
def ask_bool(label, value, key):
    answer = st.selectbox(label, options, index=options.index({None:"未判断", True:"是", False:"否"}.get(value,"未判断")), key=key)
    return {"未判断":None,"是":True,"否":False}[answer]

answerable = ask_bool("语料中是否有足够证据回答这个查询？", mapping[qid + "_baseline"].get("expected_answerable"), qid + "_answerable")
modes = ["baseline", "labels"]
if int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 2: modes.reverse()
updates = []
for column, mode, letter in zip(st.columns(2), modes, ["A", "B"]):
    with column:
        st.subheader("结果 " + letter)
        old = mapping[qid + "_" + mode]
        evidence = result["baseline_evidence" if mode == "baseline" else "evidence"]
        answer = result["baseline_answer" if mode == "baseline" else "answer"]
        st.write(answer["text"])
        judgments = {}
        for rank, item in enumerate(evidence, 1):
            st.markdown(f"**[{rank}] {item['airline']} · {item.get('cabin') or '舱位未知'}**")
            st.write(item["text"])
            with st.expander(f"完整评论与来源 [{rank}]"):
                st.write(item.get("review_text") or item["text"])
                st.caption(f"{item['chunk_id']} · {item.get('date') or '日期未知'} · {item.get('route') or '航线未知'}")
            judgments[item["chunk_id"]] = ask_bool("这条原文是否相关？", old.get("relevance", {}).get(item["chunk_id"]), qid + mode + item["chunk_id"])
        success = ask_bool("回答是否实际解决了问题？", old.get("answer_successful"), qid + mode + "success")
        total = st.number_input("实质性论断总数（-1 表示未判断）", min_value=-1, value=old.get("total_claims") if old.get("total_claims") is not None else -1, key=qid + mode + "total")
        supported = st.number_input("有原文支持的论断数（-1 表示未判断）", min_value=-1, value=old.get("supported_claims") if old.get("supported_claims") is not None else -1, key=qid + mode + "supported")
        abstention = ask_bool("如果语料无答案，是否正确说明了证据不足？", old.get("correct_abstention"), qid + mode + "abstain")
        updates.append({**old, "expected_answerable":answerable, "relevance":judgments,
            "answer_successful":success, "total_claims":None if total < 0 else total,
            "supported_claims":None if supported < 0 else supported, "correct_abstention":abstention})

if st.button("保存该查询的人工评审", type="primary"):
    errors = []
    if not reviewer.strip(): errors.append("请填写评审人标识。")
    if answerable is None: errors.append("请先判断语料是否能回答。")
    for row in updates:
        if answerable is True:
            if any(value is None for value in row["relevance"].values()): errors.append("请判定两组结果的全部原文相关性。")
            if row["answer_successful"] is None: errors.append("请判定两组回答是否成功。")
            total, supported = row["total_claims"], row["supported_claims"]
            if total is None or supported is None or supported > total: errors.append("请填写有效的论断计数，支持数不能超过总数。")
        elif answerable is False and row["correct_abstention"] is None:
            errors.append("请判断两组结果是否正确说明证据不足。")
    if errors:
        for error in dict.fromkeys(errors): st.error(error)
    else:
        for row in updates:
            row.update(human_reviewed=True, independent_test=False, judgment_provenance="human:" + reviewer.strip())
            mapping[row["query_id"]] = row
        updated = [mapping[row["query_id"]] for row in rows]
        temporary = saved.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in updated)); temporary.replace(saved)
        (directory / "evaluation_reviewed.json").write_text(json.dumps(evaluate_query_modes(updated), ensure_ascii=False, indent=2))
        st.success("已保存人工判定，并更新两种模式各自的指标。尚未评审的查询仍保持空值。")
        st.download_button("下载已保存的判定", saved.read_bytes(), file_name=saved.name, mime="application/jsonl")
