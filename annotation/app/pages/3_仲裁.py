"""仲裁页：三人分歧的句子在这里定夺，结论直接产出 eval.xls。

为什么要有这一页
  全部标完会剩下几十上百条分歧，原来的流程是导出 CSV → Excel 里人工比对
  → 决定 → 手工拼回去。既慢又容易出错，而这一步**直接产出要交的 eval.xls**。

为什么要等标完才开放
  这一页会显示三个人各自的答案。标注期间打开就破坏了独立性，
  一致率也就失去意义了。所以默认锁着，要显式确认才能进。
"""
import streamlit as st

from annot.config import USERS
from annot.export import merged_label, needs_arbitration, triple
from annot.ui import config, md_safe, reveal_gate, sentences_for, store

ASPECT_OPTIONS = ['seat', 'staff', 'food', 'entertainment',
                  'ground', 'value', 'overall', 'other']

st.title('仲裁')
st.caption('标注有分歧的句子在这里定夺 · 结论用于生成 eval.xls')

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

if not reveal_gate(by_sent, len(sentences), key='arb'):
    st.stop()

# ── 待仲裁清单 ──────────────────────────────────────────────
# 两类需要全组过一遍：
#   1. 三人答案不完全相同（含只有方面不同的情况）
#   2. 三人一致但 ≥2 人标了 unclear —— 这类「一致」恰恰说明该句不可标注，
#      要从金标准剔除。不单独捞出来的话它永远不会出现在这一页。
rows = []
for s in sentences:
    got = by_sent.get(s['id'], {})
    if needs_arbitration(got):
        rows.append((s, got, [merged_label(got[u]) for u in USERS]))

pending = [r for r in rows if r[0]['id'] not in finals]
st.metric('待仲裁', f'{len(pending)} / {len(rows)}',
          f'已裁定 {len(rows) - len(pending)}', delta_color='off')
if not rows:
    st.success('没有分歧，或标注员尚未标完同一批句子。')
    st.stop()
if not pending:
    st.success('全部分歧已裁定。到进度看板导出。')

only_pending = st.checkbox('只看待裁定的', value=True)
show = pending if only_pending else rows
if not show:
    st.stop()

idx = st.session_state.get('arb_idx', 0) % max(len(show), 1)
s, got, labels = show[idx]

nav = st.columns([1, 4, 1])
if nav[0].button('← 上一条', use_container_width=True, disabled=idx == 0):
    st.session_state.arb_idx = idx - 1
    st.rerun()
nav[1].markdown(f"<div style='text-align:center'><b>{s['id']}</b>　"
                f"<span style='color:#888'>{idx+1} / {len(show)}</span></div>",
                unsafe_allow_html=True)
if nav[2].button('下一条 →', use_container_width=True, disabled=idx >= len(show) - 1):
    st.session_state.arb_idx = idx + 1
    st.rerun()

with st.container(border=True):
    st.markdown(f"### {md_safe(s['text_en'])}")
    if s.get('text_zh'):
        st.caption(md_safe(s['text_zh']))

st.markdown('**各人的答案**')
cols = st.columns(len(USERS))
for col, (u, name) in zip(cols, USERS.items()):
    r = got[u]
    with col:
        st.markdown(f"**{name}**")
        st.code(f"{r['subjectivity']}\n{r['polarity']}\n{r['aspect']}", language=None)
        if r.get('note'):
            st.caption(f"备注：{r['note']}")

# 多数票按完整三元组算：三人都是 negative 但方面不同时，不存在真正的多数
triples = [triple(got[u]) for u in USERS]
tcounts = {t: triples.count(t) for t in set(triples)}
majority_triple = (max(tcounts, key=tcounts.get)
                   if max(tcounts.values()) >= 2 else None)
unclear_n = labels.count('unclear')

st.divider()
st.markdown('**裁定**')
a, b, c = st.columns(3)


def save(action, rec=None):
    payload = dict(sentence_id=s['id'], action=action, decided_by='group',
                   note=(rec or {}).get('note', ''),
                   subjectivity=(rec or {}).get('subjectivity', ''),
                   polarity=(rec or {}).get('polarity', ''),
                   aspect=(rec or {}).get('aspect', ''))
    try:
        db.save_final(payload)
        st.toast(f"{s['id']} 已裁定：{action}")
        st.rerun()
    except Exception as exc:
        st.error(f'保存失败：{exc}')


# ≥2 人 unclear 时不给「采纳」这条路：采纳等于把 unclear/NA/unclear 写进金标准，
# 而 R11 要求的恰恰是剔除。三人一致标 unclear 是最容易误点的一种——
# 多数票算出来是「unclear」，按钮却长得像个正常的采纳动作。
if unclear_n >= 2:
    a.button('不可采纳（应剔除）', disabled=True, use_container_width=True,
             help='≥2 人标 unclear，按 R11 只能剔除，不能写进金标准')
elif majority_triple:
    winner = next(u for u in USERS if triple(got[u]) == majority_triple)
    label = '/'.join(x for x in majority_triple if x)
    if a.button(f'采纳多数（{label}）', type='primary', use_container_width=True):
        save('adopt', got[winner])
else:
    a.button('各人答案互异，无多数', disabled=True, use_container_width=True)

drop_hint = ('≥2 人标了 unclear，按规范 R11 应从金标准剔除'
             if unclear_n >= 2 else '全组都认为不该进金标准时使用')
if c.button('剔除此句', use_container_width=True,
            type='primary' if unclear_n >= 2 else 'secondary', help=drop_hint):
    save('drop')
if unclear_n >= 2:
    st.warning(f'{unclear_n} 人标了 **unclear** —— 按规范 R11，这句应从金标准剔除。')

with b.popover('手工裁定', use_container_width=True):
    subj = st.radio('subjectivity', ['objective', 'subjective', 'unclear'],
                    horizontal=True, key=f'm_s_{s["id"]}')
    pol = st.radio('polarity', ['NA', 'positive', 'negative'], horizontal=True,
                   key=f'm_p_{s["id"]}',
                   disabled=subj in ('objective', 'unclear'))
    asp = st.selectbox('aspect', ASPECT_OPTIONS, key=f'm_a_{s["id"]}',
                       disabled=subj == 'unclear')
    note = st.text_input('备注', key=f'm_n_{s["id"]}')
    if st.button('保存裁定', type='primary', key=f'm_b_{s["id"]}'):
        if subj == 'unclear':
            pol, asp = 'NA', 'unclear'
        elif subj == 'objective':
            pol = 'NA'
        save('manual', dict(subjectivity=subj, polarity=pol, aspect=asp, note=note))

ACTION_CN = {'adopt': '采纳多数', 'manual': '手工裁定', 'drop': '剔除'}

if s['id'] in finals:
    f = finals[s['id']]
    detail = ' / '.join(x for x in (f.get('subjectivity'), f.get('polarity'),
                                    f.get('aspect')) if x)
    left, right = st.columns([4, 1])
    left.info(f"已裁定：**{ACTION_CN.get(f['action'], f['action'])}**"
              + (f"　{detail}" if detail else '')
              + (f"　{f['note']}" if f.get('note') else '')
              + (f"　·　{f['decided_at'][:16].replace('T', ' ')}"
                 if f.get('decided_at') else ''))
    # 剔除是破坏性的：点错了那句就从金标准消失，而看板只报「已裁定」的计数，
    # 不会提示少了一句。没有撤销入口的话只能去手改 CSV。
    if right.button('撤销裁定', use_container_width=True,
                    help='这句回到待裁定清单，各人的原始标注不受影响'):
        try:
            db.delete_final(s['id'])
            st.toast(f"{s['id']} 已撤销裁定")
            st.rerun()
        except Exception as exc:
            st.error(f'撤销失败：{exc}')
