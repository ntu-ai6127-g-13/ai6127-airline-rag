"""Local evidence query UI: streamlit run app/streamlit_app.py."""
import sys
import html
from datetime import date
from pathlib import Path
import time

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CHUNKS, ROOT
from src.query_service import (compare_airlines, compare_available_airlines, query_aspects,
                               query_reviews, query_session)
from src.question_parser import SUBTOPIC_NAMES, parse_question
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
        st.caption("Qwen3 4B 已就绪；回答会实时生成，并在完成后经过引用校验。")
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
live_answer_slot=st.empty()
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
    aspects=parsed.aspects or ([aspect] if aspect else [])
    polarity=parsed.polarity
    filters=parsed.filters
    requested_comparison=list(comparison) or (parsed.airlines if len(parsed.airlines)>=2 else [])
    open_comparison=parsed.comparison and not requested_comparison and "airline" not in filters
    if aspect is None:
        st.error("还不能确定你想问哪个评价方面。请在问题中写明座位、空乘、餐饮、娱乐、地勤、性价比或总体体验，或在侧栏选择。"); st.stop()
    if not question.strip():
        question=f"请根据评论说明{aspect_names[aspect]}的评价情况。"
    query_status=None
    try:
        if (len(requested_comparison)>=2 or open_comparison) and len(aspects)>1:
            st.error("当前一次查询可以比较多家航司的一个方面，或查询一家航司的多个方面；请先选择其中一种。")
            st.stop()
        if use_labels and label_index.exists():
            spinner_text="正在检索、读取离线标签并生成有据回答；首次查询通常约 30–45 秒，后续约 10–15 秒……"
        elif use_labels:
            spinner_text="正在检索并进行本地实时分类；首次运行可能需要约一分钟……"
        else:
            spinner_text="正在检索并生成有据回答；服务启动后的首次查询需要加载检索模型……"
        with st.status(spinner_text,expanded=True) as query_status, query_session():
            if len(requested_comparison)>=2:
                result=compare_airlines(requested_comparison,query=parsed.retrieval_keywords,aspect=aspect,polarity=polarity,
                    filters=filters,classifier_manifest=manifest if use_labels and not label_index.exists() else None,
                    label_index_path=label_index if use_labels and label_index.exists() else None,
                    subtopics=parsed.subtopics,
                    top_k=evidence_limit)
            elif open_comparison:
                result=compare_available_airlines(query=parsed.retrieval_keywords,aspect=aspect,polarity=polarity,
                    filters=filters,classifier_manifest=manifest if use_labels and not label_index.exists() else None,
                    label_index_path=label_index if use_labels and label_index.exists() else None,
                    subtopics=parsed.subtopics,top_k=evidence_limit)
            elif len(aspects)>1:
                result=query_aspects(aspects,query=parsed.retrieval_keywords,polarity=polarity,filters=filters,
                    classifier_manifest=manifest if use_labels and not label_index.exists() else None,
                    label_index_path=label_index if use_labels and label_index.exists() else None,
                    subtopics_by_aspect=parsed.subtopics_by_aspect,
                    top_k=evidence_limit,candidate_k=max(60,evidence_limit))
            else:
                result=query_reviews(parsed.retrieval_keywords,aspect=aspect,polarity=polarity,filters=filters,
                    classifier_manifest=manifest if use_labels and not label_index.exists() else None,
                    label_index_path=label_index if use_labels and label_index.exists() else None,
                    subtopics=parsed.subtopics,
                    top_k=evidence_limit,candidate_k=max(60,evidence_limit))
            evidence_count=(sum(len(part['evidence']) for part in result['results'].values())
                            if result.get("results") else len(result["evidence"]))
            query_status.update(label=f"已找到 {evidence_count} 条证据，正在生成并校验回答……",
                                state="running",expanded=True)
            result["question"]=question
            result["parsed_question"]=parsed.to_dict()
            stream_state={"updated":0.0}
            def stream_answer(text,event):
                now=time.monotonic()
                if event=="delta" and now-stream_state["updated"]<0.06 and not text.endswith("\n"):
                    return
                stream_state["updated"]=now
                if event=="start":
                    live_answer_slot.info("已完成检索，本地 Qwen 正在生成回答……")
                elif event=="reset":
                    live_answer_slot.info("初稿未通过引用或数字校验，正在自动修正……")
                else:
                    cursor=" ▌" if event=="delta" else ""
                    live_answer_slot.markdown("### 回答\n\n"+text+cursor)
            result["grounded_answer"]=answer_question(
                question,result,use_model=use_local_qa,
                on_stream=stream_answer if use_local_qa else None)
            st.session_state["result"]=result
            query_status.update(label="问答完成",state="complete",expanded=False)
    except Exception as error:
        if query_status is not None:
            query_status.update(label="本次查询未完成",state="error",expanded=True)
        st.error(str(error))
        st.session_state.pop("result",None)
if submitted and "result" in st.session_state:
    live_answer_slot.empty()
def display_result(result,citation_prefix=""):
    answer=result["answer"]
    st.write(answer["text"])
    full=result.get("corpus_label_matches")
    if full:
        st.info(f"全库离线标签中，符合当前结构化条件的评价有 {full['sentence_count']:,} 句，"
                f"来自 {full['review_count']:,} 条不同评论；当前展示相关性最高的 {len(result['evidence'])} 条。")
        if result.get("subtopics"):
            st.caption("全量数按上位评价方面统计，不是各细分问题的匹配数；细分问题使用各自的检索式选取下方原文。")
        elif result.get("query"):
            st.caption("全量数按航司、方面、正负、舱位、航线和日期计算，不代表全部都与补充关键词同等相关。")
    label_mode={"offline_index":"离线标签","realtime_classifier":"实时分类","none":"未使用标签"}.get(
        result.get("label_lookup_mode"),"未使用标签")
    st.caption(f"耗时 {result['total_seconds']:.1f} 秒 · {len(result['evidence'])} 条证据 · "
               f"{answer['review_count']} 条不同评论 · {label_mode}")
    if result.get("subtopics"):
        direct_counts=result.get("direct_subtopic_evidence_counts",{})
        analysis=result.get("subtopic_analysis",{})
        st.caption("细分证据覆盖："+" · ".join(
            (f"{SUBTOPIC_NAMES.get(topic,topic)}：当前条件下直接匹配 "
             f"{analysis.get(topic,{}).get('direct_sentence_count',direct_counts.get(topic,0)):,} 句/"
             f"{analysis.get(topic,{}).get('direct_review_count',direct_counts.get(topic,0)):,} 条评论，"
             f"本页展示 {direct_counts.get(topic,0)} 句")
            for topic in result["subtopics"]))
        st.caption("系统先扫描符合航司、舱位、航线和日期条件的完整语料切片，再把直接提及细分属性的原文送入重排；"
                   "本地 Qwen 根据本页展示的代表性原文作答，完整直接匹配数量用于判断证据是否充足。")
    for i,item in enumerate(result["evidence"],1):
        st.markdown(f"**[{citation_prefix}{i}] {item['airline']} · {item.get('cabin') or '舱位未知'}**")
        st.write(item["text"])
        matched=item.get("matched_subtopics",[])
        direct=item.get("direct_subtopic_matches",[])
        if direct:
            st.caption("原文直接匹配："+"、".join(SUBTOPIC_NAMES.get(topic,topic) for topic in direct))
        elif matched:
            st.caption("语义检索匹配："+"、".join(SUBTOPIC_NAMES.get(topic,topic) for topic in matched))
        if "pred_aspect" in item:
            st.caption(f"模型标签：{aspect_names.get(item['pred_aspect'],item['pred_aspect'])} / {item['pred_polarity']}")
        with st.expander("查看完整评论与来源"):
            st.write(item.get("review_text") or item["text"])
            st.caption(f"来源 {item['chunk_id']} · 日期 {item.get('date') or '未知'} · 航线 {item.get('route') or '未知'}")

def display_rating_summary(result):
    if result.get("type")=="multi_aspect":
        summaries=[(aspect,summary) for aspect,summary in result.get("rating_summaries",{}).items()]
    else:
        summaries=[(result.get("aspect"),result.get("rating_summary"))]
    available=[(aspect,summary) for aspect,summary in summaries if summary and summary.get("available")]
    if not available:
        st.info("原始评论表没有这个方面的独立评分列，本题只能依据评论原文回答。")
        return
    st.subheader("原始评分汇总（按评价方面）")
    if result.get("subtopics"):
        requested="、".join(SUBTOPIC_NAMES.get(topic,topic) for topic in result["subtopics"])
        st.info(f"原始表只有“{aspect_names.get(result.get('aspect'), result.get('aspect'))}”总体评分，没有“{requested}”的独立评分。"
                "该分数只作背景，细分结论以下方对应原文为依据。")
    rows=[]
    for aspect_key,summary in available:
        scale=summary["scale"]
        for name,data in summary["airlines"].items():
            rows.append({"评价方面":aspect_names.get(aspect_key,aspect_key),
                         "原始评分字段":summary.get("rating_field","—"),"航司":name,
                         "平均分":f"{data['mean']:.2f} / {scale}" if data['mean'] is not None else "无数据",
                         "评分数":data["n"],"中位数":data["median"],
                         "高分占比":f"{data['high_share']*100:.1f}%" if data['high_share'] is not None else "—",
                         "低分占比":f"{data['low_share']*100:.1f}%" if data['low_share'] is not None else "—"})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
    for aspect_key,summary in available:
        comparison_summary=summary.get("comparison",{})
        label=aspect_names.get(aspect_key,aspect_key)
        if comparison_summary.get("status")=="clear_difference":
            st.success(f"{label}评分结论：{comparison_summary['winner']} 领先 {comparison_summary['runner_up']} "
                       f"{comparison_summary['difference']:.2f} 分；差值超过预设的实际意义门槛，且 95% 差值区间不跨 0。")
        elif comparison_summary.get("status")=="no_clear_difference":
            st.warning(f"{label}评分差为 {comparison_summary['difference']:.2f} 分，目前不足以判定两者存在有实际意义的差异。")
        elif comparison_summary.get("status")=="insufficient_ratings":
            st.warning(f"{label}方面至少一家航司的同口径评分少于 30 份，暂不做评分优劣判断。")
    st.caption(f"评分来自原始 review-level 汇总表，并应用当前航司、舱位、旅客类型、航线和日期条件；"
               "表中已显式列出评价方面和对应的原始评分字段。方面评分为 1–5 分，总体评分为 1–10 分；"
               "方面高分为 4–5、低分为 1–2，总体高分为 8–10、低分为 1–4。好评/差评选项只控制下方原文，"
               "不会从评分汇总中删掉另一类评分。")

def display_sentiment_overview(result):
    """Render review-level positive/negative cards from the complete matching pool."""
    groups=[]
    if result.get("subtopics"):
        for topic in result["subtopics"]:
            if result.get("type")=="comparison":
                summaries={
                    airline:(part.get("subtopic_analysis",{}).get(topic,{})
                             .get("polarity_summary"))
                    for airline,part in result.get("results",{}).items()
                }
                direct_counts={
                    airline:part.get("subtopic_analysis",{}).get(topic,{}).get("direct_review_count")
                    for airline,part in result.get("results",{}).items()
                }
            else:
                summary=(result.get("subtopic_analysis",{}).get(topic,{})
                         .get("polarity_summary"))
                airline=result.get("filters",{}).get("airline") or next(
                    (row.get("airline") for row in result.get("evidence",[]) if row.get("airline")),"所选航司")
                summaries={str(airline):summary}
                direct_counts={str(airline):result.get("subtopic_analysis",{}).get(topic,{}).get("direct_review_count")}
            groups.append((SUBTOPIC_NAMES.get(topic,topic),summaries,direct_counts,True))
    elif result.get("type")=="comparison":
        summaries=result.get("label_distributions") or {
            airline:part.get("corpus_label_matches")
            for airline,part in result.get("results",{}).items()
        }
        groups.append((aspect_names.get(result.get("aspect"),result.get("aspect")),summaries,{},False))
    elif result.get("corpus_label_matches"):
        airline=result.get("filters",{}).get("airline") or next(
            (row.get("airline") for row in result.get("evidence",[]) if row.get("airline")),"所选航司")
        groups.append((aspect_names.get(result.get("aspect"),result.get("aspect")),
                       {str(airline):result["corpus_label_matches"]},{},False))
    groups=[(name,{airline:data for airline,data in summaries.items() if data},direct_counts,direct)
            for name,summaries,direct_counts,direct in groups]
    groups=[group for group in groups if group[1]]
    if not groups:
        return
    st.subheader("评论倾向概览")
    st.caption("按完整匹配池统计，每条评论只计一次；百分比的分母为能明确归入正向或负向的评论。")
    for group_index,(name,summaries,direct_counts,direct) in enumerate(groups):
        if len(groups)>1 or result.get("subtopics"):
            st.markdown(f"**{name}**")
        columns=st.columns(min(3,len(summaries)))
        for index,(airline,data) in enumerate(summaries.items()):
            total=int(data.get("labeled_review_count") or 0)
            positive=int(data.get("positive_review_count") or 0)
            negative=int(data.get("negative_review_count") or 0)
            if not total:
                continue
            positive_pct=positive/total*100
            negative_pct=negative/total*100
            ambiguous=int(data.get("ambiguous_review_count") or 0)
            scope_total=(direct_counts.get(airline) if direct else data.get("review_count"))
            scope=(f"相关评论总数 {int(scope_total):,} 条 · " if scope_total is not None else "")
            tie=f" · 另有 {ambiguous:,} 条正负票数持平" if ambiguous else ""
            with columns[index % len(columns)]:
                st.markdown(f"**{airline}**")
                st.markdown(
                    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">'
                    '<div style="border:1px solid #86d6a3;background:#edf9f1;border-radius:12px;padding:14px;text-align:center">'
                    '<div style="color:#18723a;font-weight:700">正向</div>'
                    f'<div style="color:#14532d;font-size:1.7rem;font-weight:800">{positive_pct:.1f}%</div>'
                    f'<div style="color:#3f5f49">{positive:,} 条</div></div>'
                    '<div style="border:1px solid #ef9a9a;background:#fff1f1;border-radius:12px;padding:14px;text-align:center">'
                    '<div style="color:#b42318;font-weight:700">负向</div>'
                    f'<div style="color:#7f1d1d;font-size:1.7rem;font-weight:800">{negative_pct:.1f}%</div>'
                    f'<div style="color:#6f4545">{negative:,} 条</div></div></div>'
                    f'<div style="text-align:center;margin-top:8px;color:#667085;font-size:.88rem">'
                    f'{html.escape(scope)}纳入正负比例 {total:,} 条{html.escape(tie)}</div>',
                    unsafe_allow_html=True)
        if group_index<len(groups)-1:
            st.divider()

if "result" in st.session_state:
    result=st.session_state["result"]
    if result.get("grounded_answer"):
        grounded=result["grounded_answer"]
        st.subheader("回答")
        st.markdown(grounded["text"])
        mode={"local_qwen":"本地 Qwen3 4B · 引用校验通过",
              "local_qwen_repaired":"本地 Qwen3 4B · 自动修正后校验通过",
              "rating_decision":"原始评分汇总 + 评论原文",
              "validation_fallback":"模型回答未通过校验 · 已使用程序回答",
              "service_fallback":"本地模型未连接 · 已使用程序回答",
              "program_fallback":"程序回答"}.get(grounded.get("mode"),grounded.get("mode",""))
        timing=f" · 生成 {grounded['model_seconds']:.1f} 秒" if grounded.get("model_seconds") else ""
        st.caption(f"回答模式：{mode}{timing}")
        display_sentiment_overview(result)
        st.divider()
    display_rating_summary(result)
    if result.get("type")=="comparison":
        st.subheader("检索证据")
        st.write(result["text"])
        if result.get("candidate_selection"):
            selection=result["candidate_selection"]
            st.info("自动比较候选："+"、".join(
                f"{row['airline']}（{aspect_names.get(result.get('aspect'),result.get('aspect'))}原始评分 "
                f"{row['rating_mean']:.2f}/{selection['scale']}，{row['rating_count']} 份）"
                for row in selection["candidates"]))
            st.caption(f"候选航司先按结构化路线和舱位筛选，再对小样本评分进行收缩排序；"
                       f"进入候选的最低同口径评分数为 {selection['minimum_candidate_ratings']}。"
                       "细分属性没有独立评分，最终回答还会核对各航司的对应原文。")
        tabs=st.tabs(list(result["results"]))
        for index,(tab,part) in enumerate(zip(tabs,result["results"].values())):
            with tab: display_result(part,chr(ord("A")+index))
    elif result.get("type")=="multi_aspect":
        st.subheader("分方面检索证据")
        tab_names=[aspect_names.get(aspect,aspect) for aspect in result["results"]]
        tabs=st.tabs(tab_names)
        for index,(tab,part) in enumerate(zip(tabs,result["results"].values())):
            with tab: display_result(part,chr(ord("A")+index))
    else: display_result(result)
    import json
    st.download_button("下载本次证据",json.dumps(result,ensure_ascii=False,indent=2),file_name="query_evidence.json",mime="application/json")
