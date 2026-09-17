"""总览页：全部语料的全景表，三个人的答案并排看。

为什么要有这一页
  仲裁页一次只看一条，只管分歧；看板只给聚合数字。中间缺一个能一眼扫完
  全局的地方——谁标到哪、哪些句子三人吵起来了、某个方面是不是被系统性
  标歪了。导出 CSV 再开 Excel 也能看，但那就脱离了系统，改完还得再导一次。

为什么跟仲裁页共用同一道闸门
  这一页同样会露出三个人各自的答案。标注期间打开，一致率就没有意义了。
"""
import pandas as pd
import streamlit as st

from annot.config import USERS
from annot.export import (AGREE, DIFFER, DROP, PARTIAL, needs_arbitration,
                          sentence_status, triple)
from annot.ui import config, md_safe, reveal_gate, sentences_for, store

st.set_page_config(page_title='标注总览', layout='wide')
st.title('总览')
try:
    db = store()
    sentences = sentences_for(db, config())
    annotations = db.all_annotations()
    finals = db.finals()
except Exception:
    st.error('读不到数据，请检查存储配置。')
    st.stop()

by_sent = {}
for r in annotations:
    by_sent.setdefault(r['sentence_id'], {})[r['username']] = r

st.caption(f'{len(sentences):,} 句 · 各人答案并排 · 支持筛选与检索')

if not reveal_gate(by_sent, len(sentences), key='overview'):
    st.stop()

ACTION_CN = {'adopt': '采纳多数', 'manual': '手工裁定', 'drop': '剔除'}


def cell(record):
    """一个人的答案压成一格。空着表示这句他还没标。"""
    if not record:
        return ''
    subj, pol, asp = triple(record)
    if subj == 'unclear':
        return '标不了'
    if subj == 'objective':
        return f'客观 · {asp}'
    return f'{"正" if pol == "positive" else "负"} · {asp}'


STATUS_CN = {AGREE: '一致', DIFFER: '分歧', DROP: '待剔除'}


def status_of(got):
    """状态的中文名。判定本身在 export.sentence_status()，跟仲裁页共用一份。"""
    code = sentence_status(got)
    if code == PARTIAL:
        return f'{sum(1 for u in USERS if u in got)}/{len(USERS)} 标注中'
    return STATUS_CN.get(code, '未开始')


rows = []
for i, s in enumerate(sentences, 1):
    got = by_sent.get(s['id'], {})
    f = finals.get(s['id'])
    row = {'#': i, 'ID': s['id'], '状态': status_of(got)}
    for user, name in USERS.items():
        row[name] = cell(got.get(user))
    row['裁定'] = ACTION_CN.get(f['action'], '') if f else ''
    row['英文原句'] = s['text_en']
    row['中文'] = s.get('text_zh', '')
    rows.append(row)

df = pd.DataFrame(rows)

# ── 概览数字 ────────────────────────────────────────────────
counts = df['状态'].value_counts()
n_agree = int(counts.get('一致', 0))
n_differ = int(counts.get('分歧', 0))
n_drop = int(counts.get('待剔除', 0))
done = n_agree + n_differ + n_drop
# 待仲裁 = 分歧 + 待剔除，跟仲裁页的清单口径一致。两页数字对不上最容易让人以为漏了句子。
n_arb = n_differ + n_drop
n_decided = int((df['裁定'] != '').sum())

cols = st.columns(5)
cols[0].metric('全员已标', f'{done} / {len(df)}')
cols[1].metric('一致', n_agree)
cols[2].metric('分歧', n_differ, help='含只有方面不同的情况')
cols[3].metric('待剔除', n_drop,
               help='≥2 人标「标不了」，按规范 R11 不进金标准')
cols[4].metric('已裁定', f'{n_decided} / {n_arb}',
               help='待仲裁 = 分歧 + 待剔除，与仲裁页同一口径')

# 一致率的分母不含「待剔除」—— 那些句子会从金标准里拿掉，不该稀释这个指标。
pool = n_agree + n_differ
if pool:
    st.progress(n_agree / pool,
                text=f"一致率 {n_agree / pool:.1%}　"
                     f"（{n_agree}/{pool}，已排除 {n_drop} 句待剔除；"
                     f"作业要求 ≥80%，以仲裁前的原始一致率为准）")

# ── 筛选 ───────────────────────────────────────────────────
f1, f2, f3 = st.columns([2, 2, 3])
order = ['分歧', '待剔除', '一致', '未开始']
present = [s for s in order if s in set(df['状态'])] + \
          sorted({s for s in df['状态'] if s not in order})
pick = f1.multiselect('状态', present, placeholder='全部状态')
who = f2.multiselect('这些人已标', list(USERS.values()), placeholder='不限')
q = f3.text_input('检索原句 / 译文 / ID', placeholder='输入关键词，如 seat、延误、E0041')

view = df
if pick:
    view = view[view['状态'].isin(pick)]
for name in who:
    view = view[view[name] != '']
if q:
    kw = q.strip().lower()
    hit = (view['英文原句'].str.lower().str.contains(kw, regex=False)
           | view['中文'].str.lower().str.contains(kw, regex=False)
           | view['ID'].str.lower().str.contains(kw, regex=False))
    view = view[hit]

st.caption(f'筛出 {len(view)} 句' + ('　·　点任意一行看详情' if len(view) else ''))

STATUS_HELP = '一致 / 分歧 / 待剔除 / n·3 标注中 / 未开始'
picked = st.dataframe(
    view, use_container_width=True, hide_index=True, height=520,
    on_select='rerun', selection_mode='single-row',
    column_config={
        '#': st.column_config.NumberColumn(width='small'),
        'ID': st.column_config.TextColumn(width='small'),
        '状态': st.column_config.TextColumn(width='small', help=STATUS_HELP),
        '裁定': st.column_config.TextColumn(width='small'),
        '英文原句': st.column_config.TextColumn(width='large'),
        '中文': st.column_config.TextColumn(width='medium'),
    })

# ── 选中行的详情 ────────────────────────────────────────────
sel = picked.selection.rows if picked and picked.selection else []
if sel:
    r = view.iloc[sel[0]]
    got = by_sent.get(r['ID'], {})
    st.divider()
    with st.container(border=True):
        st.markdown(f"**{r['ID']}**　{md_safe(r['英文原句'])}")
        if r['中文']:
            st.caption(md_safe(r['中文']))
    detail = st.columns(len(USERS))
    for col, (user, name) in zip(detail, USERS.items()):
        rec = got.get(user)
        with col:
            st.markdown(f'**{name}**')
            if rec:
                st.code('\n'.join(triple(rec)), language=None)
                if rec.get('note'):
                    st.caption(f"备注：{rec['note']}")
            else:
                st.caption('尚未标注')
    if needs_arbitration(got):
        if r['ID'] in finals:
            f = finals[r['ID']]
            st.info(f"已裁定：**{ACTION_CN.get(f['action'], f['action'])}**"
                    + (f"　{f.get('note')}" if f.get('note') else ''))
        else:
            st.warning('这句还没裁定 —— 到「仲裁」页处理。')

st.download_button('下载当前筛选结果 CSV',
                   view.to_csv(index=False).encode('utf-8-sig'),
                   file_name='overview.csv', mime='text/csv')
