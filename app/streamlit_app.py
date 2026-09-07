"""网页查询界面（检索组负责）。见手册第 23 节。

运行：streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="航空服务评价检索", layout="wide")
st.title("✈️ 航空服务评价检索")

st.info(
    "占位页面。检索组按手册第 23 节实现：\n\n"
    "1. 侧边栏放筛选条件（舱位、航司、日期）\n"
    "2. 主区域放查询框\n"
    "3. 结果要显示**耗时**（报告 Q2 需要）\n"
    "4. 每条结果可展开查看完整评论（parent-document retrieval）"
)
