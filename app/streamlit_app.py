"""Local evidence query UI: streamlit run app/streamlit_app.py."""
import sys
from datetime import date
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CHUNKS, ROOT
from src.query_service import query_reviews, compare_airlines
from src.question_parser import parse_question
from src.grounded_qa import answer_question, model_health

st.set_page_config(page_title="航空服务评价检索", layout="wide")
st.title("✈️ 航空服务评价检索")

st.caption("按航司和评价方面查阅真实旅客评论。数据覆盖 2002–2019 年，不能代表航司当前服务。")

@st.cache_data
def filter_options():
    frame=pd.read_parquet(CHUNKS,columns=["airline","cabin","route_from","route_to"])
    return tuple(sorted(frame[column].dropna().astype(str).unique()) for column in frame.columns)

if not CHUNKS.exists():
    st.error("评论知识库尚未生成，请先完成数据构建。"); st.stop()
airlines,cabins,origins,destinations=filter_options()
aspect_names={"seat":"座椅舒适度","staff":"空乘服务","food":"餐饮","entertainment":"娱乐与 Wi-Fi",
              "ground":"值机、行李与准点","value":"价格、退款与客服","overall":"总体体验","other":"清洁及其他"}
manifest=ROOT/"outputs/query_r36_experiment_20261019/manifest.json"
label_index=ROOT/"outputs/label_index_20261019/labels.parquet"
with st.sidebar:
    st.subheader("查询条件")
    airline=st.selectbox("航司",["全部"]+airlines)
    comparison=st.multiselect("比较航司（可选，选择 2–3 家）",airlines,max_selections=3)
    cabin=st.selectbox("舱位",["全部"]+cabins)
    aspect_choice=st.selectbox("评价方面",["auto"]+list(aspect_names),
        format_func=lambda value: "从问题自动识别" if value=="auto" else aspect_names[value])
    sentiment=st.radio("查看哪些评价",["自动/好评和差评","好评","差评"])
    evidence_limit=st.select_slider("每家展示证据数",options=[10,20,30],value=10,
                                    help="全量匹配数会单独显示；这里只控制页面展示的高相关原文数量。")
    labels_available=label_index.exists() or manifest.exists()
    use_labels=st.checkbox("分类标签辅助排序",value=labels_available,disabled=not labels_available,
                           help="自动标签用于辅助排序和展示，可能有误；可关闭以查看原检索顺序，原文证据始终保留。")
    if use_labels and label_index.exists():
        st.caption("当前使用全量离线基础标签；标签只做软排序，不删除评论。")
    elif use_labels:
        st.caption("离线标签不可用，当前回退到本地实时分类。")
    qa_health=model_health()
    use_local_qa=st.checkbox("使用本地 Qwen 生成回答",value=qa_health.get("status")=="ready")
    if qa_health.get("status")=="ready":
        st.caption("Qwen3 4B 已就绪；回答会经过引用校验。")
    elif use_local_qa:
        st.caption("模型服务未连接；查询仍会返回程序生成的有据回答。")
    with st.expander("航线与日期条件"):
        origin=st.selectbox("出发地",["不限"]+origins)
        destination=st.selectbox("目的地",["不限"]+destinations)
        limit_dates=st.checkbox("限定评论日期",value=False)
        dates=st.date_input("评论日期范围",value=(date(2010,1,1),date(2019,12,31)),disabled=not limit_dates)
        st.caption("启用这些条件后，仅检索含对应航线或日期记录的评论。")
with st.form("query"):
    question=st.text_input("请输入问题",placeholder="例如：新航经济舱的座位舒适度怎么样？请给出评论依据")
    submitted=st.form_submit_button("生成有据回答",type="primary")
if submitted:
    if len(comparison)==1:
        st.error("比较航司时请选择至少两家；单航司查询请使用上方航司选项。"); st.stop()
    filters={}
    if airline!="全部": filters["airline"]=airline
    if cabin!="全部": filters["cabin"]=cabin
    if origin!="不限": filters["route_from"]=origin
    if destination!="不限": filters["route_to"]=destination
    if limit_dates:
        if len(dates)!=2 or dates[0]>dates[1]:
            st.error("请选择完整的起止日期。"); st.stop()
        filters["date"]=[value.isoformat() for value in dates]
    polarity={"自动/好评和差评":None,"好评":"positive","差评":"negative"}[sentiment]
    parsed=parse_question(question,known_airlines=airlines,origins=origins,destinations=destinations,
        default_aspect=None if aspect_choice=="auto" else aspect_choice,
        default_polarity=polarity,default_filters=filters)
    aspect=parsed.aspect
    polarity=parsed.polarity
    filters=parsed.filters
    requested_comparison=list(comparison) or (parsed.airlines if len(parsed.airlines)>=2 else [])
    if aspect is None:
        st.error("还不能确定你想问哪个评价方面。请在问题中写明座位、空乘、餐饮、娱乐、地勤、性价比或总体体验，或在侧栏选择。"); st.stop()
    if not question.strip():
        question=f"请根据评论说明{aspect_names[aspect]}的评价情况。"
    try:
        if use_labels and label_index.exists():
            spinner_text="正在检索、读取离线标签并生成有据回答；首次查询通常约 30–45 秒，后续约 10–15 秒……"
        elif use_labels:
            spinner_text="正在检索并进行本地实时分类；首次运行可能需要约一分钟……"
        else:
            spinner_text="正在检索并生成有据回答；服务启动后的首次查询需要加载检索模型……"
        with st.spinner(spinner_text):
            if len(requested_comparison)>=2:
                result=compare_airlines(requested_comparison,query=parsed.retrieval_keywords,aspect=aspect,polarity=polarity,
                    filters=filters,classifier_manifest=manifest if use_labels and not label_index.exists() else None,
                    label_index_path=label_index if use_labels and label_index.exists() else None,
                    top_k=evidence_limit)
            else:
                result=query_reviews(parsed.retrieval_keywords,aspect=aspect,polarity=polarity,filters=filters,
                    classifier_manifest=manifest if use_labels and not label_index.exists() else None,
                    label_index_path=label_index if use_labels and label_index.exists() else None,
                    top_k=evidence_limit,candidate_k=max(60,evidence_limit))
            result["question"]=question
            result["parsed_question"]=parsed.to_dict()
            result["grounded_answer"]=answer_question(question,result,use_model=use_local_qa)
            st.session_state["result"]=result
    except Exception as error:
        st.error(str(error))
        st.session_state.pop("result",None)
def display_result(result,citation_prefix=""):
    answer=result["answer"]
    st.write(answer["text"])
    full=result.get("corpus_label_matches")
    if full:
        st.info(f"全库离线标签中，符合当前结构化条件的评价有 {full['sentence_count']:,} 句，"
                f"来自 {full['review_count']:,} 条不同评论；当前展示相关性最高的 {len(result['evidence'])} 条。")
        if result.get("query"):
            st.caption("全量数按航司、方面、正负、舱位、航线和日期计算，不代表全部都与补充关键词同等相关。")
    label_mode={"offline_index":"离线标签","realtime_classifier":"实时分类","none":"未使用标签"}.get(
        result.get("label_lookup_mode"),"未使用标签")
    st.caption(f"耗时 {result['total_seconds']:.1f} 秒 · {len(result['evidence'])} 条证据 · "
               f"{answer['review_count']} 条不同评论 · {label_mode}")
    for i,item in enumerate(result["evidence"],1):
        st.markdown(f"**[{citation_prefix}{i}] {item['airline']} · {item.get('cabin') or '舱位未知'}**")
        st.write(item["text"])
        if "pred_aspect" in item:
            st.caption(f"模型标签：{aspect_names.get(item['pred_aspect'],item['pred_aspect'])} / {item['pred_polarity']}")
        with st.expander("查看完整评论与来源"):
            st.write(item.get("review_text") or item["text"])
            st.caption(f"来源 {item['chunk_id']} · 日期 {item.get('date') or '未知'} · 航线 {item.get('route') or '未知'}")
if "result" in st.session_state:
    result=st.session_state["result"]
    if result.get("grounded_answer"):
        grounded=result["grounded_answer"]
        st.subheader("回答")
        st.markdown(grounded["text"])
        mode={"local_qwen":"本地 Qwen3 4B · 引用校验通过",
              "validation_fallback":"模型回答未通过校验 · 已使用程序回答",
              "service_fallback":"本地模型未连接 · 已使用程序回答",
              "program_fallback":"程序回答"}.get(grounded.get("mode"),grounded.get("mode",""))
        timing=f" · 生成 {grounded['model_seconds']:.1f} 秒" if grounded.get("model_seconds") else ""
        st.caption(f"回答模式：{mode}{timing}")
        st.divider()
    if result.get("type")=="comparison":
        st.subheader("检索证据")
        st.write(result["text"])
        tabs=st.tabs(list(result["results"]))
        for index,(tab,part) in enumerate(zip(tabs,result["results"].values())):
            with tab: display_result(part,chr(ord("A")+index))
    else: display_result(result)
    import json
    st.download_button("下载本次证据",json.dumps(result,ensure_ascii=False,indent=2),file_name="query_evidence.json",mime="application/json")
