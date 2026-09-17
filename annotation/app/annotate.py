"""Run from repository root: streamlit run app/annotate.py"""
import streamlit as st
from annot.ui import annotation, restore_session
from annot.config import USERS

st.set_page_config(page_title='双语标注系统 · AI6127', page_icon='✈️', layout='centered')
restore_session()
if st.session_state.get('user'):
    st.sidebar.write(f"当前标注员：{USERS[st.session_state.user]}")
    if st.sidebar.button('退出登录'):
        st.session_state.clear()
        st.session_state.clear_login = True
        st.rerun()
page = st.navigation([
    st.Page(annotation, title='标注页', icon='📝', default=True),
    st.Page('pages/1_进度看板.py', title='进度看板', icon='📊'),
    st.Page('pages/2_登录.py', title='登录', icon='🔐'),
    st.Page('pages/3_仲裁.py', title='仲裁', icon='⚖️'),
    st.Page('pages/4_总览.py', title='总览', icon='📋'),
])
page.run()
