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

initial_language=st.session_state.get("ui_language","zh")
st.set_page_config(page_title="航空服务评价检索" if initial_language=="zh" else "Airline Review Search",layout="wide")
language=st.sidebar.selectbox("界面语言" if initial_language=="zh" else "Interface language",
                              ["zh","en"],format_func=lambda value:"中文" if value=="zh" else "English",
                              key="ui_language")
def tr(chinese,english): return chinese if language=="zh" else english
if "result" in st.session_state and st.session_state["result"].get("language","zh")!=language:
    st.session_state.pop("result",None)

st.title(tr("✈️ 航空服务评价检索","✈️ Airline Review Search"))

st.caption(tr("按航司和评价方面查阅真实旅客评论。数据覆盖 2002–2019 年，不能代表航司当前服务。",
              "Explore real passenger reviews by airline and service aspect. The data cover 2002–2019 and do not represent current service."))

@st.cache_data
def filter_options():
    frame=pd.read_parquet(CHUNKS,columns=["airline","cabin","route_from","route_to"])
    return tuple(sorted(frame[column].dropna().astype(str).unique()) for column in frame.columns)

if not CHUNKS.exists():
    st.error(tr("评论知识库尚未生成，请先完成数据构建。",
                "The review knowledge base has not been built yet.")); st.stop()
airlines,cabins,origins,destinations=filter_options()
aspect_names_zh={"seat":"座椅舒适度","staff":"空乘服务","food":"餐饮","entertainment":"娱乐与 Wi-Fi",
                 "ground":"值机、行李与准点","value":"价格、退款与客服","overall":"总体体验","other":"清洁及其他"}
aspect_names_en={"seat":"Seat comfort","staff":"Cabin crew service","food":"Food and beverages",
                 "entertainment":"Entertainment and Wi-Fi","ground":"Check-in, baggage and punctuality",
                 "value":"Price, refunds and support","overall":"Overall experience","other":"Cleanliness and other services"}
aspect_names=aspect_names_zh if language=="zh" else aspect_names_en
subtopic_names_en={
    "quality":"Food quality and taste","portion":"Meal portion size","variety":"Meal variety and choice",
    "drinks":"Onboard drinks","special_meals":"Special meals","temperature":"Meal temperature",
    "content":"Entertainment content","screen":"Entertainment screen","wifi":"Onboard Wi-Fi","audio":"Headphones and audio quality",
    "usability":"System usability","legroom":"Seat legroom","width":"Seat width","recline":"Seat recline and lie-flat capability",
    "comfort":"Seat comfort","condition":"Seat condition","seat_cleanliness":"Seat cleanliness","power":"Seat power and charging",
    "attitude":"Crew attitude","efficiency":"Crew efficiency","professionalism":"Crew professionalism",
    "availability":"Crew availability","punctuality":"Punctuality","delay_handling":"Delay and cancellation handling",
    "baggage":"Baggage service","checkin":"Check-in service","gate_service":"Gate and boarding service","lounge":"Airport lounge",
    "price":"Ticket price","refund":"Refunds and changes","support":"Customer support","booking":"Booking system",
    "cleanliness":"Cabin cleanliness","toilet":"Onboard lavatory","cabin_temperature":"Cabin temperature"}
def subtopic_name(topic): return SUBTOPIC_NAMES.get(topic,topic) if language=="zh" else subtopic_names_en.get(topic,topic)
airline_names_zh={"Singapore Airlines":"新加坡航空","United Airlines":"美国联合航空","Qatar Airways":"卡塔尔航空",
                  "Cathay Pacific Airways":"国泰航空","British Airways":"英国航空","Emirates":"阿联酋航空"}
def airline_name(name): return airline_names_zh.get(name,name) if language=="zh" else name
cabin_names_zh={"Economy Class":"经济舱","Premium Economy":"超级经济舱","Business Class":"商务舱","First Class":"头等舱"}
def cabin_name(name): return cabin_names_zh.get(name,name) if language=="zh" else name
rating_fields_zh={"seat_comfort":"座椅舒适度评分","cabin_service":"空乘服务评分","food_bev":"餐饮评分",
                  "entertainment":"娱乐评分","ground_service":"地面服务评分","value_for_money":"性价比评分","overall":"总体评分"}
def rating_field_name(name): return rating_fields_zh.get(name,name) if language=="zh" else name
manifest=ROOT/"outputs/query_r36_experiment_20261019/manifest.json"
label_index=ROOT/"outputs/label_index_20261019/labels.parquet"
with st.sidebar:
    st.subheader(tr("查询条件","Search filters"))
    all_option=tr("全部","All")
    airline=st.selectbox(tr("航司","Airline"),[all_option]+airlines,format_func=lambda value:airline_name(value))
    comparison=st.multiselect(tr("比较航司（可选，选择 2–3 家）","Compare airlines (optional, select 2–3)"),airlines,max_selections=3,
                              format_func=airline_name)
    cabin=st.selectbox(tr("舱位","Cabin"),[all_option]+cabins,format_func=lambda value:cabin_name(value))
    aspect_choice=st.selectbox(tr("评价方面","Service aspect"),["auto"]+list(aspect_names),
        format_func=lambda value: tr("从问题自动识别","Detect from question") if value=="auto" else aspect_names[value])
    sentiment_options=["both","positive","negative"]
    sentiment=st.radio(tr("查看哪些评价","Evidence sentiment"),sentiment_options,
                       format_func=lambda value:{"both":tr("自动/好评和差评","Both positive and negative"),
                                                 "positive":tr("好评","Positive"),"negative":tr("差评","Negative")}[value])
    evidence_limit=st.select_slider(tr("每家展示证据数","Evidence shown per airline"),options=[10,20,30],value=10,
                                    help=tr("全量匹配数会单独显示；这里只控制页面展示的高相关评论数量。",
                                            "Full-match totals are shown separately; this only controls the number of highly relevant excerpts displayed."))
    labels_available=label_index.exists() or manifest.exists()
    use_labels=st.checkbox(tr("分类标签辅助排序","Use classifier labels for ranking"),value=labels_available,disabled=not labels_available,
                           help=tr("自动标签用于辅助排序和展示，可能有误；关闭后使用原检索顺序。",
                                   "Automatic labels assist ranking and display and may be imperfect; disable them to use the original retrieval order."))
    if use_labels and label_index.exists():
        st.caption(tr("当前使用全量离线基础标签；标签只做软排序，不删除评论。",
                      "Using the full offline label index; labels rerank evidence without deleting reviews."))
    elif use_labels:
        st.caption(tr("离线标签不可用，当前回退到本地实时分类。",
                      "The offline index is unavailable; using the local real-time classifier."))
    qa_health=model_health()
    use_local_qa=st.checkbox(tr("使用本地 Qwen 生成回答","Use local Qwen to generate the answer"),value=qa_health.get("status")=="ready")
    if qa_health.get("status")=="ready":
        st.caption(tr("Qwen3 4B 已就绪；回答会实时生成，并在完成后经过引用校验。",
                      "Qwen3 4B is ready; the answer streams live and is citation-checked on completion."))
    elif use_local_qa:
        st.caption(tr("模型服务未连接；查询仍会返回程序生成的有据回答。",
                      "The model service is unavailable; the system will still return a program-generated grounded answer."))
    with st.expander(tr("航线与日期条件","Route and date filters")):
        any_option=tr("不限","Any")
        origin=st.selectbox(tr("出发地","Origin"),[any_option]+origins)
        destination=st.selectbox(tr("目的地","Destination"),[any_option]+destinations)
        limit_dates=st.checkbox(tr("限定评论日期","Limit review dates"),value=False)
        dates=st.date_input(tr("评论日期范围","Review date range"),value=(date(2010,1,1),date(2019,12,31)),disabled=not limit_dates)
        st.caption(tr("启用这些条件后，仅检索含对应航线或日期记录的评论。",
                      "When enabled, only reviews with matching route or date records are searched."))
with st.form("query"):
    question=st.text_input(tr("请输入问题","Enter your question"),
        placeholder=tr("例如：新航经济舱的座位舒适度怎么样？请给出评论依据",
                       "Example: How is seat comfort in Singapore Airlines economy class? Cite the reviews."))
    submitted=st.form_submit_button(tr("生成有据回答","Generate grounded answer"),type="primary")
live_answer_slot=st.empty()
if submitted:
    if len(comparison)==1:
        st.error(tr("比较航司时请选择至少两家；单航司查询请使用上方航司选项。",
                    "Select at least two airlines for a comparison; use the airline field for a single-airline query.")); st.stop()
    filters={}
    if airline!=all_option: filters["airline"]=airline
    if cabin!=all_option: filters["cabin"]=cabin
    if origin!=any_option: filters["route_from"]=origin
    if destination!=any_option: filters["route_to"]=destination
    if limit_dates:
        if len(dates)!=2 or dates[0]>dates[1]:
            st.error(tr("请选择完整的起止日期。","Select a complete start and end date.")); st.stop()
        filters["date"]=[value.isoformat() for value in dates]
    polarity={"both":None,"positive":"positive","negative":"negative"}[sentiment]
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
        st.error(tr("还不能确定你想问哪个评价方面。请在问题中写明座位、空乘、餐饮、娱乐、地勤、性价比或总体体验，或在侧栏选择。",
                    "The service aspect is unclear. Mention seats, crew, food, entertainment, ground service, value, or overall experience, or select an aspect in the sidebar.")); st.stop()
    if not question.strip():
        question=(f"请根据评论说明{aspect_names[aspect]}的评价情况。" if language=="zh" else
                  f"What do the reviews say about {aspect_names[aspect]}?")
    query_status=None
    try:
        if (len(requested_comparison)>=2 or open_comparison) and len(aspects)>1:
            st.error(tr("当前一次查询可以比较多家航司的一个方面，或查询一家航司的多个方面；请先选择其中一种。",
                        "A query can compare several airlines on one aspect or examine several aspects for one airline. Choose one of these query types."))
            st.stop()
        if use_labels and label_index.exists():
            spinner_text=tr("正在检索、读取离线标签并生成有据回答；首次查询通常约 30–45 秒，后续约 10–15 秒……",
                            "Searching, reading offline labels, translating when needed, and generating a grounded answer…")
        elif use_labels:
            spinner_text=tr("正在检索并进行本地实时分类；首次运行可能需要约一分钟……",
                            "Searching and running the local classifier; the first run may take about a minute…")
        else:
            spinner_text=tr("正在检索并生成有据回答；服务启动后的首次查询需要加载检索模型……",
                            "Searching and generating a grounded answer; the first query loads the retrieval models…")
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
            query_status.update(label=(f"已找到 {evidence_count} 条证据，正在翻译、生成并校验回答……" if language=="zh" else
                                       f"Found {evidence_count} evidence excerpts; generating and validating the answer…"),
                                state="running",expanded=True)
            result["question"]=question
            result["language"]=language
            result["parsed_question"]=parsed.to_dict()
            stream_state={"updated":0.0}
            def stream_answer(text,event):
                now=time.monotonic()
                if event=="delta" and now-stream_state["updated"]<0.06 and not text.endswith("\n"):
                    return
                stream_state["updated"]=now
                if event=="start":
                    live_answer_slot.info(tr("已完成检索和评论翻译，本地 Qwen 正在生成回答……",
                                             "Retrieval is complete; local Qwen is generating the answer…"))
                elif event=="reset":
                    live_answer_slot.info(tr("初稿未通过引用或数字校验，正在自动修正……",
                                             "The draft failed citation or numeric validation and is being corrected…"))
                else:
                    cursor=" ▌" if event=="delta" else ""
                    live_answer_slot.markdown(("### 回答\n\n" if language=="zh" else "### Answer\n\n")+text+cursor)
            result["grounded_answer"]=answer_question(
                question,result,use_model=use_local_qa,
                on_stream=stream_answer if use_local_qa else None)
            st.session_state["result"]=result
            query_status.update(label=tr("问答完成","Answer complete"),state="complete",expanded=False)
    except Exception as error:
        if query_status is not None:
            query_status.update(label=tr("本次查询未完成","Query failed"),state="error",expanded=True)
        st.error("查询失败，请检查本地模型服务和筛选条件后重试。" if language=="zh" else f"Query failed: {error}")
        st.session_state.pop("result",None)
if submitted and "result" in st.session_state:
    live_answer_slot.empty()
def display_result(result,citation_prefix=""):
    answer=result["answer"]
    full=result.get("corpus_label_matches")
    if full:
        st.info((f"全库离线标签中，符合当前结构化条件的评价有 {full['sentence_count']:,} 句，"
                 f"来自 {full['review_count']:,} 条不同评论；当前展示相关性最高的 {len(result['evidence'])} 条。")
                if language=="zh" else
                (f"The offline label index contains {full['sentence_count']:,} matching sentences from "
                 f"{full['review_count']:,} distinct reviews under the current structured filters; "
                 f"the {len(result['evidence'])} most relevant excerpts are displayed."))
        if result.get("subtopics"):
            st.caption(tr("全量数按上位评价方面统计，不是各细分问题的匹配数；细分问题使用各自的检索式选取下方评论。",
                          "The full count is for the parent aspect rather than each subtopic; each subtopic uses its own retrieval query for the evidence below."))
        elif result.get("query"):
            st.caption(tr("全量数按航司、方面、舱位、航线和日期计算，不代表全部都与补充关键词同等相关。",
                          "The full count applies airline, aspect, cabin, route, and date filters; free-text relevance can still vary."))
    label_modes_zh={"offline_index":"离线标签","realtime_classifier":"实时分类","none":"未使用标签"}
    label_modes_en={"offline_index":"Offline labels","realtime_classifier":"Real-time classifier","none":"No labels"}
    label_mode=(label_modes_zh if language=="zh" else label_modes_en).get(
        result.get("label_lookup_mode"),tr("未使用标签","No labels"))
    st.caption((f"耗时 {result['total_seconds']:.1f} 秒 · {len(result['evidence'])} 条证据 · "
                f"{answer['review_count']} 条不同评论 · {label_mode}") if language=="zh" else
               (f"{result['total_seconds']:.1f} seconds · {len(result['evidence'])} evidence excerpts · "
                f"{answer['review_count']} distinct reviews · {label_mode}"))
    if result.get("subtopics"):
        direct_counts=result.get("direct_subtopic_evidence_counts",{})
        analysis=result.get("subtopic_analysis",{})
        if language=="zh":
            coverage="细分证据覆盖："+" · ".join(
                f"{subtopic_name(topic)}：直接匹配 {analysis.get(topic,{}).get('direct_sentence_count',direct_counts.get(topic,0)):,} 句/"
                f"{analysis.get(topic,{}).get('direct_review_count',direct_counts.get(topic,0)):,} 条评论，本页展示 {direct_counts.get(topic,0)} 句"
                for topic in result["subtopics"])
        else:
            coverage="Subtopic coverage: "+" · ".join(
                f"{subtopic_name(topic)}: {analysis.get(topic,{}).get('direct_sentence_count',direct_counts.get(topic,0)):,} direct sentences/"
                f"{analysis.get(topic,{}).get('direct_review_count',direct_counts.get(topic,0)):,} reviews; {direct_counts.get(topic,0)} shown"
                for topic in result["subtopics"])
        st.caption(coverage)
        st.caption(tr("系统先扫描符合结构化条件的完整语料，再重排直接提及细分属性的评论；完整直接匹配数量用于判断证据是否充足。",
                      "The system scans the full structured corpus slice before reranking reviews that directly mention the subtopic; full direct-match counts determine evidence sufficiency."))
    for i,item in enumerate(result["evidence"],1):
        st.markdown(f"**[{citation_prefix}{i}] {airline_name(item['airline'])} · {cabin_name(item.get('cabin')) if item.get('cabin') else tr('舱位未知','Cabin unknown')}**")
        st.write(item.get("display_text") or item.get("text_zh") or item["text"])
        matched=item.get("matched_subtopics",[])
        direct=item.get("direct_subtopic_matches",[])
        if direct:
            st.caption(tr("评论直接匹配：","Direct match: ")+("、" if language=="zh" else ", ").join(subtopic_name(topic) for topic in direct))
        elif matched:
            st.caption(tr("语义检索匹配：","Semantic match: ")+("、" if language=="zh" else ", ").join(subtopic_name(topic) for topic in matched))
        if "pred_aspect" in item:
            polarity_name={"positive":tr("正向","Positive"),"negative":tr("负向","Negative"),"NA":tr("不适用","Not applicable")}.get(item['pred_polarity'],item['pred_polarity'])
            st.caption(f"{tr('模型标签：','Model label: ')}{aspect_names.get(item['pred_aspect'],item['pred_aspect'])} / {polarity_name}")
        with st.expander(tr("查看评论来源","View review source")):
            if language=="en": st.write(item.get("review_text") or item["text"])
            else: st.write(item.get("display_text") or item.get("text_zh"))
            st.caption((f"来源 {item['chunk_id']} · 日期 {item.get('date') or '未知'} · 航线 {item.get('route') or '未知'}") if language=="zh" else
                       (f"Source {item['chunk_id']} · Date {item.get('date') or 'Unknown'} · Route {item.get('route') or 'Unknown'}"))

def display_rating_summary(result):
    if result.get("type")=="multi_aspect":
        summaries=[(aspect,summary) for aspect,summary in result.get("rating_summaries",{}).items()]
    else:
        summaries=[(result.get("aspect"),result.get("rating_summary"))]
    available=[(aspect,summary) for aspect,summary in summaries if summary and summary.get("available")]
    if not available:
        st.info(tr("原始评论表没有这个方面的独立评分列，本题只能依据评论内容回答。",
                   "The source review table has no independent score for this aspect, so the answer relies on review evidence."))
        return
    st.subheader(tr("原始评分汇总（按评价方面）","Original rating summary by aspect"))
    if result.get("subtopics"):
        requested=("、" if language=="zh" else ", ").join(subtopic_name(topic) for topic in result["subtopics"])
        st.info((f"原始表只有“{aspect_names.get(result.get('aspect'), result.get('aspect'))}”总体评分，没有“{requested}”的独立评分。"
                 "该分数只作背景，细分结论以下方对应评论为依据。") if language=="zh" else
                (f"The source table has only a parent {aspect_names.get(result.get('aspect'), result.get('aspect'))} score, "
                 f"not independent scores for {requested}. Treat it as context; subtopic conclusions rely on the matching reviews."))
    rows=[]
    for aspect_key,summary in available:
        scale=summary["scale"]
        for name,data in summary["airlines"].items():
            rows.append({tr("评价方面","Aspect"):aspect_names.get(aspect_key,aspect_key),
                         tr("原始评分字段","Source rating field"):rating_field_name(summary.get("rating_field","—")),
                         tr("航司","Airline"):airline_name(name),
                         tr("平均分","Mean score"):f"{data['mean']:.2f} / {scale}" if data['mean'] is not None else tr("无数据","No data"),
                         tr("评分数","Ratings"):data["n"],tr("中位数","Median"):data["median"],
                         tr("高分占比","High-score share"):f"{data['high_share']*100:.1f}%" if data['high_share'] is not None else "—",
                         tr("低分占比","Low-score share"):f"{data['low_share']*100:.1f}%" if data['low_share'] is not None else "—"})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
    for aspect_key,summary in available:
        comparison_summary=summary.get("comparison",{})
        label=aspect_names.get(aspect_key,aspect_key)
        if comparison_summary.get("status")=="clear_difference":
            st.success((f"{label}评分结论：{airline_name(comparison_summary['winner'])} 领先 {airline_name(comparison_summary['runner_up'])} "
                        f"{comparison_summary['difference']:.2f} 分；差值超过预设的实际意义门槛，且 95% 差值区间不跨 0。") if language=="zh" else
                       (f"{label} rating result: {comparison_summary['winner']} leads {comparison_summary['runner_up']} by "
                        f"{comparison_summary['difference']:.2f} points; the gap exceeds the practical threshold and its 95% interval excludes zero."))
        elif comparison_summary.get("status")=="no_clear_difference":
            st.warning((f"{label}评分差为 {comparison_summary['difference']:.2f} 分，目前不足以判定两者存在有实际意义的差异。") if language=="zh" else
                       (f"The {label} rating gap is {comparison_summary['difference']:.2f} points, which is not enough to establish a practically meaningful difference."))
        elif comparison_summary.get("status")=="insufficient_ratings":
            st.warning((f"{label}方面至少一家航司的同口径评分少于 30 份，暂不做评分优劣判断。") if language=="zh" else
                       (f"At least one airline has fewer than 30 comparable {label} ratings, so no rating winner is declared."))
    st.caption(tr("评分来自原始评论级汇总表，并应用当前航司、舱位、旅客类型、航线和日期条件。方面评分为 1–5 分，总体评分为 1–10 分；好评/差评选项只控制下方证据，不改变评分汇总。",
                  "Ratings come from the original review-level table and apply the current airline, cabin, traveller, route, and date filters. Aspect ratings use a 1–5 scale and overall ratings use 1–10; the evidence-sentiment control does not alter this summary."))

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
            groups.append((subtopic_name(topic),summaries,direct_counts,True))
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
    st.subheader(tr("评论倾向概览","Review sentiment overview"))
    st.caption(tr("按完整匹配池统计，每条评论只计一次；百分比的分母为能明确归入正向或负向的评论。",
                  "Calculated from the complete matching pool with one vote per review; percentages use reviews with a clear positive or negative direction."))
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
            scope=((f"相关评论总数 {int(scope_total):,} 条 · " if language=="zh" else
                    f"{int(scope_total):,} related reviews · ") if scope_total is not None else "")
            tie=((f" · 另有 {ambiguous:,} 条正负票数持平" if language=="zh" else
                  f" · {ambiguous:,} tied reviews excluded") if ambiguous else "")
            with columns[index % len(columns)]:
                st.markdown(f"**{airline_name(airline)}**")
                st.markdown(
                    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">'
                    '<div style="border:1px solid #86d6a3;background:#edf9f1;border-radius:12px;padding:14px;text-align:center">'
                    f'<div style="color:#18723a;font-weight:700">{tr("正向","Positive")}</div>'
                    f'<div style="color:#14532d;font-size:1.7rem;font-weight:800">{positive_pct:.1f}%</div>'
                    f'<div style="color:#3f5f49">{positive:,} {tr("条","reviews")}</div></div>'
                    '<div style="border:1px solid #ef9a9a;background:#fff1f1;border-radius:12px;padding:14px;text-align:center">'
                    f'<div style="color:#b42318;font-weight:700">{tr("负向","Negative")}</div>'
                    f'<div style="color:#7f1d1d;font-size:1.7rem;font-weight:800">{negative_pct:.1f}%</div>'
                    f'<div style="color:#6f4545">{negative:,} {tr("条","reviews")}</div></div></div>'
                    f'<div style="text-align:center;margin-top:8px;color:#667085;font-size:.88rem">'
                    f'{html.escape(scope)}{tr("纳入正负比例", "used for percentages")} {total:,} {tr("条","reviews")}{html.escape(tie)}</div>',
                    unsafe_allow_html=True)
        if group_index<len(groups)-1:
            st.divider()

if "result" in st.session_state:
    result=st.session_state["result"]
    if result.get("grounded_answer"):
        grounded=result["grounded_answer"]
        st.subheader(tr("回答","Answer"))
        st.markdown(grounded["text"])
        mode_zh={"local_qwen":"本地 Qwen3 4B · 引用校验通过","local_qwen_repaired":"本地 Qwen3 4B · 自动修正后校验通过",
                 "rating_decision":"原始评分汇总 + 评论证据","validation_fallback":"模型回答未通过校验 · 已使用程序回答",
                 "service_fallback":"本地模型未连接 · 已使用程序回答","program_fallback":"程序回答"}
        mode_en={"local_qwen":"Local Qwen3 4B · citations validated","local_qwen_repaired":"Local Qwen3 4B · corrected answer validated",
                 "rating_decision":"Original ratings + review evidence","validation_fallback":"Model answer failed validation · program answer used",
                 "service_fallback":"Local model unavailable · program answer used","program_fallback":"Program-generated answer"}
        mode=(mode_zh if language=="zh" else mode_en).get(grounded.get("mode"),grounded.get("mode",""))
        timing=((f" · 生成 {grounded['model_seconds']:.1f} 秒" if language=="zh" else f" · generated in {grounded['model_seconds']:.1f}s")
                if grounded.get("model_seconds") else "")
        st.caption(f"{tr('回答模式：','Answer mode: ')}{mode}{timing}")
        display_sentiment_overview(result)
        st.divider()
    display_rating_summary(result)
    if result.get("type")=="comparison":
        st.subheader(tr("检索证据","Retrieved evidence"))
        st.write(tr("以下按相同方面和筛选条件展示各航司的评论证据。检索样本不代表所有旅客。",
                    "The reviews below use the same aspect and filters for every airline. Retrieved evidence does not represent every passenger."))
        if result.get("candidate_selection"):
            selection=result["candidate_selection"]
            if language=="zh":
                candidate_text="自动比较候选："+"、".join(
                    f"{airline_name(row['airline'])}（{aspect_names.get(result.get('aspect'),result.get('aspect'))}原始评分 "
                    f"{row['rating_mean']:.2f}/{selection['scale']}，{row['rating_count']} 份）" for row in selection["candidates"])
            else:
                candidate_text="Automatically selected candidates: "+"; ".join(
                    f"{row['airline']} ({aspect_names.get(result.get('aspect'),result.get('aspect'))} rating "
                    f"{row['rating_mean']:.2f}/{selection['scale']}, {row['rating_count']} ratings)" for row in selection["candidates"])
            st.info(candidate_text)
            st.caption(tr(f"候选航司先按航线和舱位筛选，再对小样本评分进行收缩排序；最低同口径评分数为 {selection['minimum_candidate_ratings']}。细分属性没有独立评分。",
                          f"Candidates are filtered by route and cabin before shrinkage ranking; at least {selection['minimum_candidate_ratings']} comparable ratings are required. Subtopics have no independent scores."))
        tabs=st.tabs([airline_name(name) for name in result["results"]])
        for index,(tab,part) in enumerate(zip(tabs,result["results"].values())):
            with tab: display_result(part,chr(ord("A")+index))
    elif result.get("type")=="multi_aspect":
        st.subheader(tr("分方面检索证据","Evidence by aspect"))
        tab_names=[aspect_names.get(aspect,aspect) for aspect in result["results"]]
        tabs=st.tabs(tab_names)
        for index,(tab,part) in enumerate(zip(tabs,result["results"].values())):
            with tab: display_result(part,chr(ord("A")+index))
    else: display_result(result)
    import json
    st.download_button(tr("下载本次证据","Download this evidence"),json.dumps(result,ensure_ascii=False,indent=2),file_name="query_evidence.json",mime="application/json")
