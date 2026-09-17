"""数据浏览界面 —— 直观地看 data/processed/*.parquet 里到底有什么。

运行：
    cd ~/Desktop/NLP
    .venv/bin/streamlit run app/data_explorer.py

说明：
* 这个文件**不属于任何一组的交付物**，也不参与评分，谁都可以改，随便加东西。
* RAG 的查询界面是 `app/streamlit_app.py`（检索组负责），两个别混：
  那个是"提问→检索→生成答案"，这个只是"看表"。
* 底层用 DuckDB 直接查 parquet，不把 92 MB 全读进内存，所以 54 万行也不卡。
"""
from __future__ import annotations

import html
import re
import time
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

# ---------------------------------------------------------------- 数据源
# reviews.parquet 用的是**原始列名**，chunks.parquet 用的是**契约列名**。
# 这里统一 alias 成契约列名，后面所有代码只认契约列名一套。
SOURCES = {
    "句子表 chunks.parquet（一行 = 一句）": {
        "path": PROCESSED / "chunks.parquet",
        "sql": "SELECT * FROM read_parquet('{p}')",
        "grain": "句",
    },
    "评论表 reviews.parquet（一行 = 一条评论）": {
        "path": PROCESSED / "reviews.parquet",
        "sql": """
            SELECT
                review_id AS chunk_id,
                review_id,
                customer_review  AS text,
                customer_review  AS review_text,
                airline, cabin, traveller_type, route, aircraft,
                -- chunks 里 route 已经拆好了，这张表没有，这里现算一份保持一致
                nullif(regexp_extract(route, '^(.*?) [Tt]o (.*?)( via .*)?$', 1), '') AS route_from,
                nullif(regexp_extract(route, '^(.*?) [Tt]o (.*?)( via .*)?$', 2), '') AS route_to,
                nullif(trim(replace(regexp_extract(route, '^(.*?) [Tt]o (.*?)( via .*)?$', 3),
                                    ' via ', '')), '') AS route_via,
                review_date      AS date,
                date_flown,
                overall          AS overall_rating,
                recommended,
                seat_comfort     AS r_seat,
                cabin_service    AS r_staff,
                food_bev         AS r_food,
                entertainment    AS r_entertainment,
                ground_service   AS r_ground,
                value_for_money  AS r_value
            FROM read_parquet('{p}')
        """,
        "grain": "条",
    },
    "小样本 chunks_sample.parquet（快，用来试手）": {
        "path": PROCESSED / "chunks_sample.parquet",
        "sql": "SELECT * FROM read_parquet('{p}')",
        "grain": "句",
    },
}

# 六个方面，和 src/config.py 的 ASPECTS 一一对应（没有 wifi）
ASPECTS = {
    "r_seat": "座椅 seat",
    "r_staff": "服务 staff",
    "r_food": "餐食 food",
    "r_entertainment": "娱乐 entertainment",
    "r_ground": "地服 ground",
    "r_value": "性价比 value",
}

# ---------------------------------------------------------------- 配色
# 取自 dataviz 参考调色板，两种模式都过了校验脚本。
# 图表基本都是单序列，所以只用 accent 一个色；热力图用同一个蓝色的深浅。
PALETTE = {
    "light": {
        "accent": "#2a78d6",
        "ramp": ["#cde2fb", "#0d366b"],   # 浅 → 深
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e6e5e1",
        "surface": "#ffffff",
        # 热力图格子里的数字：色块越深越要用白字。3.5 是算出来的翻转点
        # （在那个点上白字对比度 4.67、黑字 4.21，再往深走白字才占优）。
        "heat_thr": 3.5, "heat_hi": "#ffffff", "heat_lo": "#0b0b0b",
    },
    "dark": {
        "accent": "#3987e5",
        "ramp": ["#184f95", "#cde2fb"],   # 深 → 浅（深色底上低值要贴近底色）
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#333330",
        "surface": "#0e1117",
        # 深色模式下色阶是反的，所以数字颜色也得反过来，翻转点在 2.2。
        "heat_thr": 2.2, "heat_hi": "#0b0b0b", "heat_lo": "#ffffff",
    },
}


def theme() -> dict:
    """跟随 Streamlit 当前的明暗模式，取对应的一组颜色。"""
    try:
        mode = st.context.theme.type
    except Exception:
        mode = None
    return PALETTE["dark" if mode == "dark" else "light"]


# ---------------------------------------------------------------- DuckDB
@st.cache_resource(show_spinner=False)
def _conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(":memory:")


def q(sql: str, params: list | None = None) -> pd.DataFrame:
    """跑一条查询，返回 DataFrame。每次用独立 cursor，避免并发串味。"""
    return _conn().cursor().execute(sql, params or []).df()


@st.cache_data(show_spinner=False)
def distinct_values(base: str, col: str) -> list[str]:
    df = q(f"SELECT DISTINCT {col} AS v FROM ({base}) WHERE {col} IS NOT NULL ORDER BY 1")
    return df["v"].tolist()


@st.cache_data(show_spinner=False)
def date_bounds(base: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    df = q(f"SELECT min(date) AS lo, max(date) AS hi FROM ({base})")
    return df["lo"].iloc[0], df["hi"].iloc[0]


# ---------------------------------------------------------------- 筛选条件
def build_where(f: dict) -> tuple[str, list]:
    """把侧边栏的筛选条件拼成 WHERE。值一律走 ? 占位符，不做字符串拼接，
    否则航司名里的单引号（比如 Sun d'Or）会直接把 SQL 语法搞崩。"""
    clauses: list[str] = []
    params: list = []

    if f["airlines"]:
        ph = ",".join(["?"] * len(f["airlines"]))
        clauses.append(f"airline IN ({ph})")
        params += f["airlines"]
    if f["cabins"]:
        ph = ",".join(["?"] * len(f["cabins"]))
        clauses.append(f"cabin IN ({ph})")
        params += f["cabins"]
    if f["travellers"]:
        ph = ",".join(["?"] * len(f["travellers"]))
        clauses.append(f"traveller_type IN ({ph})")
        params += f["travellers"]
    if f["recommended"] != "全部":
        clauses.append("recommended = ?")
        params.append("yes" if f["recommended"] == "推荐 yes" else "no")
    if f["route"].strip():
        clauses.append("route ILIKE ?")
        params.append(f"%{f['route'].strip()}%")
    # 起降点单独筛。作业的范例问题「新加坡飞伦敦经济舱」就是这种约束，
    # 用拆好的 route_from/route_to 比在整条 route 里模糊匹配准得多：
    # 搜 'London' 会把 'London to Dubai'（从伦敦出发）也算进目的地。
    if f["route_from"].strip() and "route_from" in f["cols"]:
        clauses.append("route_from ILIKE ?")
        params.append(f"%{f['route_from'].strip()}%")
    if f["route_to"].strip() and "route_to" in f["cols"]:
        clauses.append("route_to ILIKE ?")
        params.append(f"%{f['route_to'].strip()}%")
    if f["aircraft"].strip() and "aircraft" in f["cols"]:
        clauses.append("aircraft ILIKE ?")
        params.append(f"%{f['aircraft'].strip()}%")
    if f["drop_route_only"] and "is_route_only" in f["cols"]:
        clauses.append("NOT is_route_only")
    if f["drop_non_en"] and "is_non_english" in f["cols"]:
        clauses.append("NOT is_non_english")
    if f["drop_frag"] and "is_fragment" in f["cols"]:
        clauses.append("NOT is_fragment")

    # 日期：21.6% 的记录 date 是空的。只有用户真的把区间收窄了才加这个条件，
    # 否则默认视图会悄悄少掉 11 万句 —— 很容易让人把总量报错。
    lo, hi = f["date_range"]
    if (lo, hi) != f["date_full"]:
        cl = "(date >= ? AND date < ?"
        params += [pd.Timestamp(lo), pd.Timestamp(hi) + pd.Timedelta(days=1)]
        cl += " OR date IS NULL)" if f["keep_null_date"] else ")"
        clauses.append(cl)
    elif not f["keep_null_date"]:
        clauses.append("date IS NOT NULL")

    o_lo, o_hi = f["overall"]
    if (o_lo, o_hi) != (1, 10):
        clauses.append("overall_rating BETWEEN ? AND ?")
        params += [o_lo, o_hi]

    if f["aspect_col"] != "不限":
        col = f["aspect_col"]
        a_lo, a_hi = f["aspect_range"]
        clauses.append(f"{col} BETWEEN ? AND ?")
        params += [a_lo, a_hi]

    kws = [k for k in re.split(r"\s+", f["search"].strip()) if k]
    if kws:
        joiner = " AND " if f["search_mode"].startswith("全部") else " OR "
        subs = []
        for k in kws:
            if f["whole_word"]:
                subs.append("regexp_matches(text, ?, 'i')")
                params.append(rf"\b{re.escape(k)}\b")
            else:
                subs.append("text ILIKE ?")
                params.append(f"%{k}%")
        clauses.append("(" + joiner.join(subs) + ")")

    return ("WHERE " + " AND ".join(clauses)) if clauses else "", params


# ---------------------------------------------------------------- 图表样式
def style(chart: alt.Chart, height: int = 260) -> alt.Chart:
    """统一的图表外观：网格线弱化、背景透明（跟随 Streamlit 底色）。"""
    c = theme()
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0, fill=None)
        .configure_axis(
            grid=True,
            gridColor=c["grid"],
            gridWidth=1,
            domainColor=c["grid"],
            tickColor=c["grid"],
            labelColor=c["muted"],
            titleColor=c["muted"],
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
        )
        .configure_legend(labelColor=c["muted"], titleColor=c["muted"])
        .configure(background="transparent")
    )


def show(chart: alt.Chart, height: int = 260) -> None:
    st.altair_chart(style(chart, height), width="stretch", theme=None)


def bar_with_labels(
    df: pd.DataFrame,
    value: str,
    label: str,
    value_title: str,
    fmt: str = ".2f",
    hide_axis: bool = False,
    pad: float = 1.14,
) -> alt.Chart:
    """横向条形图 + 直接标数。单序列，所以不需要图例，标题已经说明是什么。

    每根条右边都直接标了数字，所以坐标轴刻度只要几个做参考就行 —— 默认给 6 个。
    刻度密密麻麻反而看不清（Altair 默认会按像素宽度铺满，1–5 分的图能铺出 36 个）。
    pad 是给右边的数字标签留的余量，不留的话最长那根的数字会被裁掉。
    """
    c = theme()
    hi = float(df[value].max()) * pad
    base = alt.Chart(df).encode(
        y=alt.Y(f"{label}:N", sort="-x", title=None),
        x=alt.X(
            f"{value}:Q",
            title=None if hide_axis else value_title,
            scale=alt.Scale(domain=[0, hi], nice=False),
            axis=None if hide_axis else alt.Axis(tickCount=6),
        ),
    )
    bars = base.mark_bar(
        color=c["accent"], cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=16
    ).encode(tooltip=list(df.columns))
    text = base.mark_text(align="left", dx=6, color=c["muted"], fontSize=12).encode(
        text=alt.Text(f"{value}:Q", format=fmt)
    )
    return alt.layer(bars, text)


# ---------------------------------------------------------------- 页面
st.set_page_config(page_title="航空评价数据浏览", page_icon="🔎", layout="wide")

st.markdown(
    """
    <style>
      [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
      .parent-review { line-height: 1.9; font-size: 0.95rem; }
      .hit { background: rgba(42,120,214,.22); border-radius: 3px; padding: 0 2px; }
      .kw  { background: rgba(235,104,52,.28); border-radius: 3px; padding: 0 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔎 航空评价数据浏览")

# ---- 侧边栏：选数据源 ----
with st.sidebar:
    st.header("数据源")
    missing = [n for n, s in SOURCES.items() if not s["path"].exists()]
    available = [n for n in SOURCES if n not in missing]
    if not available:
        st.error("data/processed/ 下没有 parquet 文件。先跑 `python -m src.build_kb`。")
        st.stop()
    src_name = st.selectbox("选一个表", available, index=0)
    if missing:
        st.caption("缺失：" + "、".join(m.split("（")[0] for m in missing))

src = SOURCES[src_name]
BASE = src["sql"].format(p=src["path"].as_posix())
GRAIN = src["grain"]
# 当前数据源实际有哪些列。样本文件可能是旧版本建的（列少），
# 界面里凡是引用新列的地方都要先查这个集合，否则一切过去整页就崩。
SRC_COLS = set(q(f"SELECT * FROM ({BASE}) LIMIT 0").columns)

# ---- 侧边栏：筛选 ----
with st.sidebar:
    st.header("筛选")
    with st.form("filters"):
        airlines = st.multiselect("航司 airline", distinct_values(BASE, "airline"))
        cabins = st.multiselect("舱位 cabin", distinct_values(BASE, "cabin"))
        travellers = st.multiselect("出行类型 traveller_type", distinct_values(BASE, "traveller_type"))
        aircraft = st.text_input("机型 aircraft", placeholder="A380 / Boeing 777")
        recommended = st.radio("是否推荐", ["全部", "推荐 yes", "不推荐 no"], horizontal=True)
        rc1, rc2 = st.columns(2)
        route_from = rc1.text_input("出发地", placeholder="Singapore")
        route_to = rc2.text_input("目的地", placeholder="London")
        route = st.text_input("整条航线包含", placeholder="例如 via Dubai")

        lo, hi = date_bounds(BASE)
        date_range = st.date_input(
            "评论日期", value=(lo.date(), hi.date()), min_value=lo.date(), max_value=hi.date()
        )
        keep_null_date = st.checkbox(
            "包含无日期的记录", value=True,
            help="原始数据里约 21.6% 的评论没有日期。默认保留，"
                 "否则一上来就看不到五分之一的数据。",
        )

        overall = st.slider("总分 overall_rating（1–10）", 1, 10, (1, 10))

        aspect_col = st.selectbox(
            "限定某个维度评分", ["不限"] + list(ASPECTS),
            format_func=lambda k: k if k == "不限" else ASPECTS[k],
        )
        aspect_range = st.slider("该维度评分（1–5）", 1, 5, (1, 5))

        st.divider()
        st.caption("**句子质量**（v3 新增的标记）")
        drop_route_only = st.checkbox(
            "排除纯航线句", value=False,
            help="像 'Guangzhou to New York.' 这种整句就是航线的元数据句，"
                 "占 3.07%。和 route 列完全重复，没有情感极性 —— 建索引时建议排除。")
        drop_non_en = st.checkbox(
            "排除非英语句", value=False,
            help="336 句法语/西语/意语等。分类器跑在英文上，这些进索引是纯噪声。")
        drop_frag = st.checkbox(
            "排除残片句", value=False,
            help="19 句以标点开头的切句残片，没有完整语义。")
        drop_dups = st.checkbox(
            "只保留每句的第一条（去重）", value=False,
            help="'All flights on time.' 这句话在语料里出现 83 次，来自 37 家航司。"
                 "检索时不去重会把 top-K 占满。")

        st.divider()
        search = st.text_input("关键词（空格分隔多个）", placeholder="legroom seat")
        search_mode = st.radio("多个关键词", ["任一 OR", "全部 AND"], horizontal=True)
        whole_word = st.checkbox(
            "整词匹配", value=False,
            help="勾上以后 seat 不会匹配到 seated / seating。用正则 \\b 边界。",
        )

        st.form_submit_button("应用筛选", width="stretch", type="primary")

# date_input 只选了一个日期时返回单元素元组
if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    d = date_range[0] if isinstance(date_range, (tuple, list)) else date_range
    date_range = (d, d)

F = dict(
    airlines=airlines, cabins=cabins, travellers=travellers, recommended=recommended,
    route=route, route_from=route_from, route_to=route_to, aircraft=aircraft,
    drop_route_only=drop_route_only, drop_dups=drop_dups,
    drop_non_en=drop_non_en, drop_frag=drop_frag, cols=SRC_COLS,
    date_range=date_range, date_full=(lo.date(), hi.date()),
    keep_null_date=keep_null_date, overall=overall, aspect_col=aspect_col,
    aspect_range=aspect_range, search=search, search_mode=search_mode, whole_word=whole_word,
)
WHERE, PARAMS = build_where(F)
FILTERED = f"SELECT * FROM ({BASE}) {WHERE}"
if drop_dups:
    # 同样的句子只留 chunk_id 最小的那条。用 QUALIFY 比 GROUP BY 省事，
    # 而且能保住所有列。
    FILTERED = (f"SELECT * FROM ({BASE}) {WHERE} "
                "QUALIFY row_number() OVER (PARTITION BY text ORDER BY chunk_id) = 1")

# ---- 概览数字 ----
t0 = time.perf_counter()
stats = q(
    f"""SELECT count(*) AS n_rows,
               count(DISTINCT review_id) AS n_reviews,
               count(DISTINCT airline)   AS n_airlines,
               avg(overall_rating)       AS avg_overall,
               sum(CASE WHEN recommended='yes' THEN 1 ELSE 0 END)::DOUBLE
                 / nullif(count(recommended),0) AS rec_rate
        FROM ({FILTERED})""",
    PARAMS,
).iloc[0]
elapsed = (time.perf_counter() - t0) * 1000

total_rows = q(f"SELECT count(*) AS n FROM ({BASE})")["n"].iloc[0]
n_rows = int(stats["n_rows"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"命中{GRAIN}数", f"{n_rows:,}")
c2.metric("涉及评论数", f"{int(stats['n_reviews']):,}")
c3.metric("涉及航司", f"{int(stats['n_airlines']):,}")
c4.metric("平均总分", "—" if pd.isna(stats["avg_overall"]) else f"{stats['avg_overall']:.2f}")
c5.metric("推荐率", "—" if pd.isna(stats["rec_rate"]) else f"{stats['rec_rate']:.1%}")
st.caption(
    f"占全表 **{n_rows / total_rows:.1%}**（全表 {total_rows:,} {GRAIN}）· "
    f"查询耗时 {elapsed:.0f} ms · 数据源 `{src['path'].name}`"
)

if n_rows == 0:
    st.warning("当前筛选条件下没有任何数据。把条件放宽一点。")
    st.stop()

with st.expander("各列的空值情况（先看这个，再下结论）"):
    # 只统计当前数据源真有的列 —— 样本文件可能是旧版本建的，缺新列。
    # 以前这里硬写列名，切到旧样本就整页崩掉。
    want = ["cabin", "traveller_type", "route", "route_from", "route_to",
            "route_via", "aircraft", "date", "date_flown", "overall_rating",
            "recommended", "is_non_english", "is_fragment"] + list(ASPECTS)
    null_cols = [c_ for c_ in want if c_ in SRC_COLS]
    nulls = q(
        "SELECT " + ", ".join(f"count({c_}) AS {c_}" for c_ in null_cols)
        + f", count(*) AS _all FROM ({FILTERED})",
        PARAMS,
    ).iloc[0]
    nd = pd.DataFrame(
        {
            "列": [ASPECTS.get(c_, c_) for c_ in null_cols],
            "有值": [int(nulls[c_]) for c_ in null_cols],
            "覆盖率": [nulls[c_] / nulls["_all"] for c_ in null_cols],
        }
    ).sort_values("覆盖率", ascending=False)
    st.dataframe(
        nd, hide_index=True, width="stretch",
        column_config={
            "有值": st.column_config.NumberColumn(format="%d"),
            "覆盖率": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.1f%%"),
        },
    )
    st.caption(
        "六个维度的评分是**很稀疏**的：value 覆盖最好，ground 最差。"
        "弱标签只能从有评分的那部分拿，这决定了每个维度能训练多少数据。"
    )

tab_table, tab_one, tab_chart, tab_sql = st.tabs(["📋 表格", "📖 逐条精读", "📊 图表", "🛠 自己写 SQL"])

# ================================================================ 表格
with tab_table:
    left, right = st.columns([3, 1])
    with right:
        page_size = st.selectbox("每页", [50, 100, 200, 500], index=1)
        n_pages = max(1, -(-n_rows // page_size))
        page = st.number_input("第几页", 1, n_pages, 1, step=1)
        st.caption(f"共 {n_pages:,} 页")
        order_by = st.selectbox(
            "排序",
            [c_ for c_ in ["chunk_id", "date", "overall_rating", "airline",
                           "n_words", "dup_count"] + list(ASPECTS) if c_ in SRC_COLS],
            format_func=lambda k: ASPECTS.get(k, k),
        )
        desc = st.checkbox("倒序", value=False)

    page_df = q(
        f"""SELECT * FROM ({FILTERED})
            ORDER BY {order_by} {'DESC' if desc else 'ASC'} NULLS LAST, chunk_id
            LIMIT {page_size} OFFSET {(page - 1) * page_size}""",
        PARAMS,
    )

    with left:
        st.dataframe(
            page_df,
            hide_index=True,
            height=560,
            column_config={
                "text": st.column_config.TextColumn("句子 text", width="large"),
                "review_text": st.column_config.TextColumn("完整评论 review_text", width="medium"),
                "date": st.column_config.DateColumn("评论日期", format="YYYY-MM-DD"),
                "date_flown": st.column_config.DateColumn("乘机月份", format="YYYY-MM"),
                "is_route_only": st.column_config.CheckboxColumn("纯航线句"),
                "is_non_english": st.column_config.CheckboxColumn("非英语"),
                "is_fragment": st.column_config.CheckboxColumn("残片"),
                "is_last": st.column_config.CheckboxColumn("末句"),
                "dup_count": st.column_config.NumberColumn("重复次数", format="%d"),
                "overall_rating": st.column_config.NumberColumn("总分", format="%d"),
                **{
                    c: st.column_config.ProgressColumn(
                        n, min_value=0, max_value=5, format="%.0f"
                    )
                    for c, n in ASPECTS.items()
                },
            },
        )

    st.download_button(
        "下载当前页 CSV",
        page_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"page{page}.csv",
        mime="text/csv",
    )

# ================================================================ 逐条精读
with tab_one:
    st.caption(
        "从命中结果里随机抽几条，把**句子**放回**它所在的整条评论**里看 —— "
        "这正是 parent-document retrieval 的效果：检索按句子，展示按评论。"
    )
    top = st.columns([1, 1, 4])
    n_sample = top[0].selectbox("抽几条", [5, 10, 20], index=0)
    if top[1].button("换一批", width="stretch"):
        st.session_state["seed"] = st.session_state.get("seed", 0) + 1
    seed = st.session_state.get("seed", 0)

    sample = q(
        f"""SELECT * FROM ({FILTERED})
            ORDER BY hash(chunk_id || '{seed}') LIMIT {n_sample}""",
        PARAMS,
    )

    kws = [k for k in re.split(r"\s+", search.strip()) if k]

    def mark(parent: str, sentence: str, keywords: list[str]) -> str:
        """把整条评论转成 HTML：命中句子加蓝底，关键词加橙底。先转义再插标签。"""
        esc = html.escape(parent)
        sent = html.escape(sentence)
        if sent and sent in esc:
            esc = esc.replace(sent, f'<span class="hit">{sent}</span>', 1)
        for k in keywords:
            esc = re.sub(
                f"({re.escape(html.escape(k))})",
                r'<span class="kw">\1</span>',
                esc,
                flags=re.IGNORECASE,
            )
        return esc

    for _, r in sample.iterrows():
        head = f"**{r['airline']}** · {r['cabin'] or '—'} · {r['traveller_type'] or '—'}"
        with st.expander(f"{r['chunk_id']} — {str(r['text'])[:90]}", expanded=False):
            m1, m2 = st.columns([2, 3])
            with m1:
                st.markdown(head)
                st.markdown(
                    f"航线 `{r['route'] or '—'}` · {pd.Timestamp(r['date']).date()} · "
                    f"总分 **{r['overall_rating']:.0f}** · "
                    f"{'👍 推荐' if r['recommended'] == 'yes' else '👎 不推荐'}"
                )
            with m2:
                asp = pd.DataFrame(
                    {"维度": list(ASPECTS.values()), "评分": [r[c] for c in ASPECTS]}
                ).dropna()
                if not asp.empty:
                    show(
                        bar_with_labels(asp, "评分", "维度", "1–5", fmt=".0f", hide_axis=True),
                        height=22 * len(asp) + 30,
                    )
                else:
                    st.caption("这条评论六个维度都没打分。")
            st.markdown("**完整评论**（蓝底 = 当前这句，橙底 = 你搜的关键词）")
            st.markdown(
                f'<div class="parent-review">{mark(str(r["review_text"]), str(r["text"]), kws)}</div>',
                unsafe_allow_html=True,
            )

# ================================================================ 图表
with tab_chart:
    c = theme()

    # --- 六个维度的平均分（单序列 → 无需图例，直接标数）---
    st.subheader("六个维度的平均分")
    means = q(
        f"SELECT {', '.join(f'avg({c_}) AS {c_}' for c_ in ASPECTS)}, "
        f"{', '.join(f'count({c_}) AS n_{c_}' for c_ in ASPECTS)} FROM ({FILTERED})",
        PARAMS,
    ).iloc[0]
    asp_df = pd.DataFrame(
        {
            "维度": list(ASPECTS.values()),
            "平均分": [means[k] for k in ASPECTS],
            "有评分条数": [int(means[f"n_{k}"]) for k in ASPECTS],
        }
    ).dropna(subset=["平均分"])
    if asp_df.empty:
        st.info("当前筛选下六个维度都没有评分。")
    else:
        show(bar_with_labels(asp_df, "平均分", "维度", "平均分（1–5）"), height=26 * len(asp_df) + 50)
        st.caption(
            "注意每个维度的**有评分条数**差别很大（见 tooltip）—— "
            "value 覆盖率最高，ground 最低。稀疏程度会直接影响弱标签的数量。"
        )

    st.divider()

    # --- 航司排名（就是报告里 C10 用的那个参考排名）---
    st.subheader("航司排名")
    r1, r2, r3 = st.columns([2, 1, 1])
    rank_col = r1.selectbox("按哪个维度排", list(ASPECTS), format_func=ASPECTS.get)
    min_n = r2.number_input(f"至少多少{GRAIN}", 10, 5000, 200, step=10)
    top_n = r3.number_input("显示前几名", 5, 40, 15, step=5)

    rank = q(
        f"""SELECT airline AS 航司,
                   avg({rank_col}) AS 平均分,
                   count({rank_col}) AS 有评分{GRAIN}数
            FROM ({FILTERED})
            GROUP BY 1 HAVING count({rank_col}) >= {min_n}
            ORDER BY 2 DESC LIMIT {top_n}""",
        PARAMS,
    )
    if rank.empty:
        st.info(f"没有航司达到「至少 {min_n} {GRAIN}」的门槛，把门槛调低点。")
    else:
        show(bar_with_labels(rank, "平均分", "航司", f"{ASPECTS[rank_col]} 平均分（1–5）"),
             height=24 * len(rank) + 50)
        with st.expander("表格视图（同样的数字）"):
            st.dataframe(rank, hide_index=True, width="stretch")
        st.caption(
            "把舱位设成 Economy Class、维度选 seat，得到的就是报告 C10 里"
            "用来对照系统输出的那份**参考排名**（整条流水线不许读这一列）。"
        )

    st.divider()
    g1, g2 = st.columns(2)

    # --- 总分分布 ---
    with g1:
        st.subheader("总分分布")
        dist = q(
            f"""SELECT overall_rating AS 总分, count(*) AS 数量 FROM ({FILTERED})
                WHERE overall_rating IS NOT NULL GROUP BY 1 ORDER BY 1""",
            PARAMS,
        )
        ch = (
            alt.Chart(dist)
            .mark_bar(color=c["accent"], cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=22)
            .encode(
                x=alt.X("总分:O", title="overall_rating（1–10）",
                        axis=alt.Axis(labelAngle=0)),
                y=alt.Y("数量:Q", title=f"{GRAIN}数"),
                tooltip=["总分", "数量"],
            )
        )
        show(ch, height=240)

    # --- 每年数量 ---
    with g2:
        st.subheader("每年评论量")
        yr = q(
            f"""SELECT date_trunc('year', date) AS 年份, count(DISTINCT review_id) AS 评论数
                FROM ({FILTERED}) GROUP BY 1 ORDER BY 1""",
            PARAMS,
        )
        line = alt.Chart(yr).encode(
            x=alt.X("年份:T", title=None),
            y=alt.Y("评论数:Q", title="评论数"),
            tooltip=[alt.Tooltip("年份:T", format="%Y"), "评论数"],
        )
        show(
            alt.layer(
                line.mark_line(color=c["accent"], strokeWidth=2),
                line.mark_point(color=c["accent"], size=70, filled=True),
            ),
            height=240,
        )
        st.caption("2015 年前样本很少 —— 划分训练/测试集时别按时间切。")

    st.divider()

    # --- 舱位 × 维度 热力图（连续量 → 单一色相由浅到深）---
    st.subheader("舱位 × 维度 平均分")
    heat_raw = q(
        f"""SELECT coalesce(cabin,'（未填）') AS 舱位,
                   {', '.join(f'avg({k}) AS {k}' for k in ASPECTS)},
                   count(*) AS n
            FROM ({FILTERED}) GROUP BY 1 ORDER BY n DESC""",
        PARAMS,
    )
    heat = heat_raw.melt(
        id_vars=["舱位"], value_vars=list(ASPECTS), var_name="维度", value_name="平均分"
    ).dropna()
    heat["维度"] = heat["维度"].map(ASPECTS)
    if heat.empty:
        st.info("当前筛选下没有可比的舱位数据。")
    else:
        cells = alt.Chart(heat).encode(
            # labelAngle=0：不加的话 Altair 会把中文标签竖起来转 90°，没法看
            x=alt.X("维度:N", title=None, sort=list(ASPECTS.values()),
                    axis=alt.Axis(labelAngle=0, labelPadding=6)),
            y=alt.Y("舱位:N", title=None),
        )
        show(
            alt.layer(
                # stroke 用底色画 2px，相邻色块之间就有一条缝，不会糊成一片
                cells.mark_rect(stroke=c["surface"], strokeWidth=2).encode(
                    color=alt.Color(
                        "平均分:Q",
                        scale=alt.Scale(range=c["ramp"], domain=[1, 5]),
                        legend=alt.Legend(title="平均分", gradientLength=160, gradientThickness=12),
                    ),
                    tooltip=["舱位", "维度", alt.Tooltip("平均分:Q", format=".2f")],
                ),
                cells.mark_text(fontSize=13, fontWeight="bold").encode(
                    text=alt.Text("平均分:Q", format=".2f"),
                    color=alt.condition(
                        alt.datum["平均分"] >= c["heat_thr"],
                        alt.value(c["heat_hi"]),
                        alt.value(c["heat_lo"]),
                    ),
                ),
            ),
            height=46 * heat["舱位"].nunique() + 40,
        )

# ================================================================ SQL
with tab_sql:
    st.caption(
        "直接写 DuckDB SQL。当前表已经注册成 `t`（**未**套用左边的筛选），"
        "所以可以随便查。只允许 SELECT / WITH / DESCRIBE / SUMMARIZE。"
    )
    presets = {
        "看看有哪些列": "DESCRIBE SELECT * FROM t",
        "整表体检": "SUMMARIZE SELECT * FROM t",
        "经济舱座椅排名（C10 参考排名）": (
            "SELECT airline, avg(r_seat) AS mean_seat, count(*) AS n\n"
            "FROM t WHERE cabin = 'Economy Class' AND r_seat IS NOT NULL\n"
            "GROUP BY 1 HAVING count(*) >= 200\nORDER BY 2 DESC LIMIT 20"
        ),
        "六个维度的覆盖率": (
            "SELECT\n"
            + ",\n".join(
                f"  count({k}) * 100.0 / count(*) AS {k}_pct" for k in ASPECTS
            )
            + "\nFROM t"
        ),
        "新加坡飞伦敦、经济舱、按座椅排": (
            "SELECT airline, avg(r_seat) AS mean_seat, count(*) AS n\n"
            "FROM t\n"
            "WHERE route_from ILIKE '%Singapore%' AND route_to ILIKE '%London%'\n"
            "  AND cabin = 'Economy Class' AND r_seat IS NOT NULL\n"
            "GROUP BY 1 ORDER BY 2 DESC"
        ),
        "v3：干净语料下的航司排名（排除元数据句+去重）": (
            "-- 先去重再聚合，所以 QUALIFY 要放在子查询里\n"
            "-- （DuckDB 的 QUALIFY 在 GROUP BY 之后执行，直接写会语法错）\n"
            "SELECT airline, avg(r_seat) AS mean_seat, count(*) AS n\n"
            "FROM (\n"
            "  SELECT * FROM t\n"
            "  WHERE cabin = 'Economy Class' AND r_seat IS NOT NULL\n"
            "    AND NOT is_route_only AND NOT is_non_english\n"
            "  QUALIFY row_number() OVER (PARTITION BY text ORDER BY chunk_id) = 1\n"
            ")\n"
            "GROUP BY 1 HAVING count(*) >= 200\nORDER BY 2 DESC LIMIT 15"
        ),
        "v2：找回来的日期长什么样（按年份）": (
            "SELECT date_part('year', date) AS 年份,\n"
            "       count(DISTINCT review_id) AS 评论数\n"
            "FROM t GROUP BY 1 ORDER BY 1"
        ),
        "还有多少乱码没洗掉": (
            "SELECT count(*) FILTER (WHERE text LIKE '%â%' OR text LIKE '%Ã%') AS bad,\n"
            "       count(*) AS total\nFROM t"
        ),
    }
    pick = st.selectbox("先套一个模板", list(presets))
    sql = st.text_area("SQL", value=presets[pick], height=180, key=f"sql_{pick}")

    if st.button("运行", type="primary"):
        # 先剥掉开头的注释行和括号，否则以 "-- 说明" 开头的查询会被误拒
        probe = re.sub(r"^\s*(--[^\n]*\n|/\*.*?\*/|\s)+", "", sql, flags=re.S)
        head = probe.lstrip("(").split(None, 1)[0].upper() if probe.strip() else ""
        if head not in {"SELECT", "WITH", "DESCRIBE", "SUMMARIZE", "FROM", "PIVOT", "UNPIVOT"}:
            st.error("只读界面，只允许 SELECT / WITH / DESCRIBE / SUMMARIZE 开头的语句。")
        else:
            try:
                cur = _conn().cursor()
                cur.execute(f"CREATE OR REPLACE TEMP VIEW t AS {BASE}")
                t0 = time.perf_counter()
                out = cur.execute(sql).df()
                st.success(f"{len(out):,} 行 · {(time.perf_counter() - t0) * 1000:.0f} ms")
                st.dataframe(out.head(2000), hide_index=True, width="stretch", height=460)
                if len(out) > 2000:
                    st.caption("只显示前 2000 行。")
            except Exception as e:  # noqa: BLE001 — 语法错误是常态，原样显示给用户看
                st.error(f"{type(e).__name__}: {e}")
