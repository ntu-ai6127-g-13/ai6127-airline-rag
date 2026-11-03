"""streamlit run experiments/downstream_evaluation/review_app.py -- path/to/packet.json"""
import copy
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.downstream_evaluation.common import read, write
from experiments.downstream_evaluation.review import summarize, validate_review

st.set_page_config(page_title="C乙回答人工评审", layout="wide")
st.title("C乙回答人工评审")
if len(sys.argv) < 2:
    st.error("启动时在 -- 后提供 human_review.json 文件路径。")
    st.stop()
source = Path(sys.argv[1]).resolve()
target = source.with_name(source.stem + "_annotations.json")
packet = read(target if target.exists() else source)
st.caption(f"保存到本地：{target}；记录 {len(packet['records'])} 条。")
with st.expander("评审规则"):
    st.json(packet["rubric"])
reviewer = st.sidebar.text_input("评审人 ID")
ident = st.sidebar.selectbox("记录", [r["id"] for r in packet["records"]])
index = next(i for i, r in enumerate(packet["records"]) if r["id"] == ident)
row = copy.deepcopy(packet["records"][index])
st.subheader(row["question"])
st.markdown(row["answer"])
with st.expander("可用于核验的程序汇总"):
    st.json(row["verified_aggregates"])
relevance = {}
choices = [None, True, False]
for item in row["evidence"]:
    eid = str(item["citation_id"])
    st.markdown(f"**[{eid}] {item.get('airline', '')}**")
    st.write(item.get("text", ""))
    if item.get("display_text") != item.get("text"):
        st.caption(item.get("display_text", ""))
    old = row["evidence_relevance"].get(eid)
    relevance[eid] = st.selectbox("此证据与问题相关？", choices, index=choices.index(old),
                                  format_func=lambda x: {None:"待判断",True:"相关",False:"不相关"}[x], key=ident+eid)
answerable = st.selectbox("这些证据能回答问题吗？", choices,
                          index=choices.index(row.get("answerable_from_evidence")), key=ident+"answerable")
score_choices = [None, 0, 1, 2, 3, 4]
score = st.selectbox("回答相关性（0–4；定义见规则）", score_choices,
                     index=score_choices.index(row.get("answer_relevance_0_to_4")), key=ident+"score")
st.write("逐条拆分可核查论断；supported 勾选表示有依据，未勾选表示无依据。reason 写引用编号或程序汇总字段及理由。")
claims = pd.DataFrame(row["claims"], columns=["claim", "supported", "reason"])
if claims.empty:
    claims = pd.DataFrame({"claim": pd.Series(dtype=str), "supported": pd.Series(dtype=bool),
                           "reason": pd.Series(dtype=str)})
edited = st.data_editor(claims, num_rows="dynamic", key=ident+"claims")
none = st.checkbox("回答没有可核查的事实论断", value=row.get("no_checkable_claims", False), key=ident+"none")
abstention = st.selectbox("证据不足时是否正确拒答（可回答时留空）", choices,
                          index=choices.index(row.get("correct_abstention")), key=ident+"abstention")
notes = st.text_area("备注", value=row.get("notes", ""), key=ident+"notes")
if st.button("保存该条完整评审"):
    row.update(reviewer_id=reviewer, review_complete=True, evidence_relevance=relevance,
               answerable_from_evidence=answerable, answer_relevance_0_to_4=score,
               claims=edited.to_dict("records"), no_checkable_claims=none,
               correct_abstention=abstention, notes=notes)
    try:
        validate_review(row)
        packet["records"][index] = row
        write(target, packet)
        st.success("已保存")
    except ValueError as error:
        st.error(str(error))
st.sidebar.json({k:v for k,v in summarize(packet).items() if k in {"records", "reviewed", "evaluation_complete"}})
