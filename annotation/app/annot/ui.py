import hmac
import os
import re
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import streamlit as st
from .auth import issue_token, password_ok, verify_token
from .cache import Cache
from .config import ROOT, USERS, cn_count, settings
from .db import LocalStore, make_store, speed, validate
from .export import deviation_alerts, exports
from .keyboard import session_bridge, workbench


def config():
    try:
        return settings(dict(st.secrets))
    except FileNotFoundError:
        return settings({})


@st.cache_resource
def backend(storage, dsn, local_dir):
    return make_store({'storage': storage, 'dsn': dsn, 'local_dir': Path(local_dir)})


def store():
    cfg = config()
    return backend(cfg['storage'], cfg['dsn'], str(cfg['local_dir']))


def restore_session():
    cfg = config()
    clear = st.session_state.get('clear_login', False)
    value = session_bridge(st.session_state.get('login_token', ''), clear)
    if value and not clear and not st.session_state.get('user'):
        user = verify_token(value.get('token', ''), cfg['annotators'])
        if user:
            st.session_state.user = user
            st.session_state.login_token = value['token']
            st.rerun()


_MD_META = re.compile(r'([$*_`~\[\]])')


def md_safe(text: str) -> str:
    """把句子原文交给 st.markdown 之前转义。

    航司评论里满是金额（"$3 for a seat selection, $25 for a carry-on bag"），
    Streamlit 会把成对的 $ 当 LaTeX 公式：美元符号消失、中间文字变成数学字体。
    而 value（性价比）这个方面，判断依据恰恰就是那些金额。
    标注页走自定义组件（已 HTML 转义），不受影响；这里只管走 Markdown 的地方。
    """
    return _MD_META.sub(r'\\\1', text or '')


def storage_error(headline, exc):
    """连不上存储时，把真实报错摆出来。

    原来这里只 st.info 一句「请检查数据库连接」，异常被吞掉 —— 而这正是
    部署当天最可能出事、也最需要看清原因的一处。连接串写错、IPv4/IPv6
    走不通、prepared statement 冲突，三种报错完全不同，修法也完全不同，
    光看「请检查连接」只能靠猜。
    """
    st.info(headline)
    with st.expander('技术细节（给统筹排查用）'):
        st.code(f'{type(exc).__name__}: {exc}', language=None)
        st.caption(
            '常见原因：\n'
            '· `PoolTimeout` —— 这只是连接池的外层超时，**真实原因在应用日志里**，'
            '搜 `error connecting in` 那一行。\n'
            '· `invalid URI query parameter` —— 连接串里塞了非 libpq 参数。'
            '`prepare_threshold` 不能写在连接串里，它由代码按端口自动设置。\n'
            '· `password authentication failed` —— 密码里的特殊字符要 URL 编码'
            '（`@` 写成 `%40`）。\n'
            '· `no tenant` / `ENOIDENTIFIER` —— 用户名不对，Supabase 的格式是 '
            '`postgres.<项目ref>`。\n'
            '· `relation ... does not exist` —— 库是空的，先跑 '
            '`scripts/init_annot_db.py --storage supabase`。')


def login():
    cfg = config()
    st.title('双语标注系统')
    st.caption(f'AI6127 · 组 13 · {cn_count(len(USERS))}位标注员独立标注同一套语料')
    if st.session_state.get('user'):
        st.success(f"已登录：{USERS[st.session_state.user]}")
        st.page_link(st.Page(annotation, title='标注页', default=True), label='进入标注页', icon='📝')
        return
    if not cfg['annotators']:
        st.info('请先运行 scripts/init_annot_db.py 建立三个账号，再启动标注。详见 docs/标注系统运行.md。')
        return
    with st.form('login_form'):
        username = st.selectbox('选择标注员', list(USERS), format_func=USERS.get)
        password = st.text_input('密码', type='password')
        submitted = st.form_submit_button('登录并继续标注', type='primary', use_container_width=True)
    if submitted:
        now = time.time()
        if now < st.session_state.get('retry_login_at', 0):
            st.error('请稍候再试。')
        elif password_ok(username, password, cfg['annotators']):
            st.session_state.user = username
            st.session_state.login_token = issue_token(username, cfg['annotators'])
            st.session_state.clear_login = False
            st.session_state.pop('position', None)
            st.rerun()
        else:
            st.session_state.retry_login_at = now + 2
            st.error('账号或密码不正确。')
    st.divider()
    st.subheader('标注进度')
    progress_rows(compact=True)


def backoff_seconds(attempts, base=2, cap=60):
    """连续输错的等待时间：2, 4, 8, 16, 32, 60, 60…

    固定 2 秒对 6 位数字几乎没有意义。递增之后，试到第 10 次要等一分钟，
    「随手多试几下」这条路就堵死了 —— 这也是这道闸唯一要挡的东西。
    真要防爆破得靠服务端限流，Streamlit 的 session_state 换个会话就清零。
    """
    return min(base * 2 ** max(attempts - 1, 0), cap) if attempts > 0 else 0


def reveal_gate(by_sent, total, key='reveal'):
    """会露出三个人答案的页面（仲裁 / 总览），进来之前先过这道闸。

    为什么要加口令
      标注期间看到别人的答案，一致率就失去意义了——那是作业的硬指标（≥80%）。
      原来只有一个复选框，任何一个标注员都能自己勾开来偷看。换成口令之后，
      只有拿着口令的统筹能开。

    为什么没配口令就锁死
      口令存在 gitignored 的 secrets.toml 里，不随代码分发。组员克隆下来的
      副本没有这个键——这时如果放行，等于这道闸从来没存在过。所以 fail closed。

    返回 True 表示可以往下渲染；返回 False 时调用方应当 st.stop()。
    """
    cfg = config()
    # 解锁是会话级的，不按页分。两页露出的是同一批答案，开了一页还要再输
    # 一次没有任何安全收益，只会让人把口令写在便签上。
    # key 只用来区分控件，防止 Streamlit 的控件键在两页之间撞车。
    if st.session_state.get('reveal_unlocked'):
        return True

    done = {u: sum(1 for v in by_sent.values() if u in v) for u in USERS}
    finished = all(n >= total for n in done.values())

    if not finished:
        st.warning('**标注尚未全部完成。**')
        for user, name in USERS.items():
            st.write(f"　{name}　{done[user]}/{total}")
        st.error(
            '这一页会显示**每位标注员各自的答案**。标注期间打开就破坏了独立性，'
            '一致率会失去意义——那是作业的硬指标（≥80%）。\n\n'
            '只有一种情况可以提前进来：**试标对齐**（各人都标完前 100 句之后碰头）。')
    else:
        st.info('全员已标注完毕。这一页会显示各自的答案，输入口令进入。')

    secret = cfg['reveal_password']
    if not secret:
        st.error('**本页未配置口令，已锁定。**\n\n'
                 '在 `.streamlit/secrets.toml` 里加一行 '
                 '`reveal_password = "……"` 再刷新。'
                 '口令请向统筹私下索取，不要发在群里。')
        return False

    with st.form(f'{key}_gate'):
        entered = st.text_input('口令', type='password', key=f'{key}_pw',
                                placeholder='向统筹索取')
        ok = st.form_submit_button('进入本页', type='primary',
                                   use_container_width=True)
    if ok:
        now = time.time()
        if now < st.session_state.get('reveal_retry_at', 0):
            left = int(st.session_state['reveal_retry_at'] - now) + 1
            st.error(f'输错太多次，请等 {left} 秒。')
        elif hmac.compare_digest(entered.strip(), secret):
            st.session_state['reveal_unlocked'] = True
            st.session_state.pop('reveal_tries', None)
            st.rerun()
        else:
            n = st.session_state.get('reveal_tries', 0) + 1
            wait = backoff_seconds(n)
            st.session_state['reveal_tries'] = n
            st.session_state['reveal_retry_at'] = now + wait
            st.error(f'口令不正确。{wait} 秒后可再试。')
    return False


def sentences_for(backend_store, cfg):
    try:
        return backend_store.sentences()
    except Exception:
        rows = LocalStore(cfg['local_dir']).sentences()
        if not rows:
            raise
        return rows


def count_for(backend_store, cfg):
    """句子总数。看板每 15 秒要一次，所以走 count(*) 而不是拉全表。"""
    try:
        return backend_store.count_sentences()
    except Exception:
        n = LocalStore(cfg['local_dir']).count_sentences()
        if not n:
            raise
        return n


def own_records(backend_store, cache, username):
    try:
        rows = backend_store.own(username)
        cache.remember(rows)
    except Exception:
        rows = cache.snapshot()
    index = {r['sentence_id']: r for r in rows}
    for row in cache.pending():
        existing = index.get(row['sentence_id'])
        if not existing or str(row['updated_at']) >= str(existing['updated_at']):
            index[row['sentence_id']] = row
    return index


PAGE = 50


def on_ephemeral_host():
    """是不是跑在会被重置的云容器上。

    Streamlit Community Cloud 的容器重新部署、休眠唤醒、崩溃重启都会重建，
    写在容器磁盘上的 CSV 会连同容器一起消失 —— 而且不会有任何提示。
    """
    return any(os.getenv(k) for k in
               ('STREAMLIT_SHARING_MODE', 'STREAMLIT_SERVER_ADDRESS_IS_CLOUD',
                'HOSTNAME_IS_STREAMLIT_CLOUD')) or Path('/mount/src').exists()


def storage_panel():
    """侧边栏：数据存在哪、存了多少、有没有风险。

    原来界面上只有一句「当前为本地 CSV 模式」，看不出数据落在哪、
    也看不出云端部署时会丢。标注是 9 小时人工，值得把这件事说清楚。
    """
    with st.sidebar:
        st.divider()
        try:
            info = store().stats()
        except Exception:
            st.error('⚠ 读不到存储状态，请检查配置')
            return
        icon = '🟢' if info['shared'] else '🟡'
        st.markdown(f"{icon} **{info['kind']}**　已存 **{info['rows']}** 条")
        st.caption(info['where'])
        if not info['shared']:
            if on_ephemeral_host():
                st.error('🔴 **数据会丢失**\n\n'
                         '这是云端容器上的临时磁盘，重新部署或休眠唤醒后 CSV 会消失。'
                         '请改用共享数据库（见 docs/标注系统运行.md），'
                         '或**立刻到进度看板导出备份**。')
            else:
                st.caption('数据只在这台机器上，别人看不到。'
                           '每天收工请到进度看板导出一次。')


def overview_drawer(sentences, own, position):
    """总览抽屉：一页 50 题，✓ 已标 / ✗ 未标，可前后翻，点一下跳过去。

    侧边栏的「漏标列表」只管当前位置之前遗漏的；这个抽屉是**全量**视图——
    1,000 题分 20 页，随时能看到整体标到哪了、哪一段有空洞。
    默认收起（st.popover），不占标注页的视觉空间。
    """
    pages = -(-len(sentences) // PAGE)
    key = 'ov_page'
    # 第一次打开时定位到当前句所在的那一页
    if key not in st.session_state:
        st.session_state[key] = position // PAGE
    page = max(0, min(st.session_state[key], pages - 1))
    lo, hi = page * PAGE, min((page + 1) * PAGE, len(sentences))
    chunk = sentences[lo:hi]
    done = sum(1 for x in chunk if x['id'] in own)

    with st.sidebar:
        # ⚠️ 50 个 st.button 每次 rerun 都会重建，而标注时每答一题就 rerun 一次。
        #    抽屉一直展开的话，每题都要多渲染 50 个组件，会明显拖慢节奏。
        #    所以用一个 checkbox 控制：不展开就一个组件都不渲染。
        opened = st.checkbox(f'📋 总览　{len(answered_ids(own))}/{len(sentences)}',
                             key='ov_open',
                             help='展开后每次标注会多渲染 50 个按钮，标得慢时收起来')
        if not opened:
            return
        with st.container(border=True):
            nav = st.columns([1, 3, 1])
            if nav[0].button('←', key='ov_prev', disabled=page == 0,
                             use_container_width=True):
                st.session_state[key] = page - 1
                st.rerun()
            nav[1].markdown(
                f"<div style='text-align:center'>"
                f"<b>{chunk[0]['id']} – {chunk[-1]['id']}</b><br>"
                f"<span style='font-size:.8rem;color:#888'>"
                f"第 {page+1}/{pages} 页　本页已标 {done}/{len(chunk)}</span></div>",
                unsafe_allow_html=True)
            if nav[2].button('→', key='ov_next', disabled=page >= pages - 1,
                             use_container_width=True):
                st.session_state[key] = page + 1
                st.rerun()

            st.divider()
            # 5 列 × 10 行；标号用页内序号，省地方
            ids = {x['id'] for x in own} if isinstance(own, (set, list)) else set(own)
            for start in range(0, len(chunk), 5):
                cols = st.columns(5)
                for offset, (col, item) in enumerate(zip(cols, chunk[start:start + 5])):
                    idx = lo + start + offset
                    mark = '✓' if item['id'] in ids else '✗'
                    here = '▸' if idx == position else ''
                    if col.button(f"{here}{idx+1}{mark}", key=f'ov_{item["id"]}',
                                  use_container_width=True,
                                  help=md_safe(item['text_en'][:70])):
                        st.session_state.position = idx
                        st.rerun()
            st.caption('✓ 已标　✗ 未标　▸ 当前　·　点一下跳过去')


def answered_ids(own):
    """真正标完的句子 id。「待定」（只有备注、没选答案）不算 ——
    它是卡住时留的便签，进度和漏标提醒都必须把它当成还没标。"""
    return {sid for sid, rec in own.items() if rec.get('subjectivity')}


def skipped_panel(sentences, own, position):
    """侧边栏：列出【当前位置之前】还没标的句子，可以点回去补。

    往下翻的时候很容易跳过几句（← → 是自由导航的），不列出来就会漏。
    只列前面的：后面的本来就还没轮到，列出来只是噪声。
    """
    done = answered_ids(own)
    skipped = [(i, s) for i, s in enumerate(sentences[:position]) if s['id'] not in done]
    with st.sidebar:
        st.divider()
        if not skipped:
            st.caption('✅ 前面没有漏标的句子')
            return
        st.markdown(f"**⚠ 前面漏标 {len(skipped)} 句**")
        st.caption('点一下跳回去补。不补也能继续，但导出时这些会是空的。')
        for i, s in skipped[:40]:
            label = md_safe(s['text_en'][:26]) + ('…' if len(s['text_en']) > 26 else '')
            # 暂存过的加个记号：那是「卡住了待处理」，不是「翻页翻漏了」
            mark = '⏸ ' if s['id'] in own else ''
            if st.button(f"{mark}{s['id']}　{label}", key=f'skip_{s["id"]}',
                         use_container_width=True):
                st.session_state.position = i
                st.rerun()
        if len(skipped) > 40:
            st.caption(f'另有 {len(skipped) - 40} 句未列出')
        if st.button('跳到第一条漏标', key='goto_first_skip',
                     use_container_width=True, type='primary'):
            st.session_state.position = skipped[0][0]
            st.rerun()


def annotation():
    if not st.session_state.get('user'):
        login()
        return
    cfg, user = config(), st.session_state.user
    st.title('双语标注')
    st.caption('先独立完成相同的前 100 句，再碰头对齐标准。')
    try:
        db = store()
        sentences = sentences_for(db, cfg)
        if not sentences:
            st.info('语料尚未导入，请运行 scripts/init_annot_db.py。')
            return
        cache = Cache(cfg['cache_dir'], user)
        pending = cache.flush(db)
        own = own_records(db, cache, user)
    except Exception as exc:
        storage_error('无法读取标注数据。', exc)
        return
    if 'position' not in st.session_state:
        st.session_state.position = next((i for i,s in enumerate(sentences) if s['id'] not in own), 0)
    position = max(0, min(st.session_state.position, len(sentences)-1))
    sentence = sentences[position]
    overview_drawer(sentences, own, position)
    skipped_panel(sentences, own, position)
    storage_panel()
    seconds = speed([r['updated_at'] for r in own.values()])
    guideline = (ROOT / 'docs/选题与标注方案.md').read_text(encoding='utf-8')
    guideline = guideline[guideline.index('## 7.'):guideline.index('## 10.')]
    if 'work_session' not in st.session_state:
        st.session_state.work_session = uuid.uuid4().hex
    payload = workbench(username=user, display_name=USERS[user], sentence=sentence, session_id=st.session_state.work_session,
        answer={k:v for k,v in own.get(sentence['id'],{}).items() if k in ('subjectivity','polarity','aspect','note')},
        total=len(sentences), completed=len(answered_ids(own)), index=position, storage=cfg['storage'],
        remaining_hours=round((len(sentences)-len(answered_ids(own)))*seconds/3600,1) if seconds else None,
        pending=pending, guide=guideline, ack=st.session_state.get('ack',''),
        error=st.session_state.get('save_error',False), saved_at=st.session_state.get('saved_at',''))
    if payload and payload.get('nonce') != st.session_state.get('ack'):
        st.session_state.ack = payload.get('nonce')
        st.session_state.save_error = False
        if payload.get('sentence_id') != sentence['id'] or payload.get('session_id') != st.session_state.work_session:
            st.session_state.save_error = True
            st.rerun()
        direction = payload.get('direction', 'stay')
        if payload.get('action') == 'save':
            answer = payload.get('answer') or {}
            record = {k: answer.get(k, '') for k in ('subjectivity','polarity','aspect','note')}
            record.update(username=user, sentence_id=sentence['id'], updated_at=datetime.now(timezone.utc).isoformat())
            try:
                validate(record)
                cache.enqueue(record)
                remaining = cache.flush(db)
                if not remaining:
                    st.session_state.saved_at = datetime.now().strftime('%H:%M:%S')
                direction = payload.get('next', 'next')
            except Exception:
                st.session_state.save_error = True
                direction = 'stay'
        elif payload.get('action') == 'clear':
            # C 键：清掉本题作答。先丢队列再删存储，顺序反了会被 flush 写回去。
            try:
                cache.drop(sentence['id'])
                db.delete(user, sentence['id'])
                st.session_state.saved_at = ''
                st.toast(f"已清除 {sentence['id']} 的作答")
            except Exception:
                st.session_state.save_error = True
            direction = 'stay'
        elif payload.get('action') != 'navigate':
            direction = 'stay'
        if direction == 'next':
            st.session_state.position = min(position+1, len(sentences)-1)
        elif direction == 'prev':
            st.session_state.position = max(position-1, 0)
        st.rerun()
    if len(answered_ids(own)) == len(sentences):
        st.success('已完成全部标注。可用 ← 回看修改，或到进度看板导出备份。')
    if pending:
        retry_cache(cache, db)


@st.fragment(run_every=15)
def retry_cache(cache, db):
    remaining = cache.flush(db)
    if remaining:
        st.caption(f'本机缓存 {remaining} 次写入；每 15 秒自动重试。缓存位于应用运行机器。')
    else:
        st.rerun()


def progress_rows(compact=False, result=None):
    if result is None:
        try:
            db = store()
            rows = db.summary()
            total = count_for(db, config())
        except Exception as exc:
            storage_error('暂时无法取得进度。', exc)
            return None
    else:
        rows, total = result
    for user, name in USERS.items():
        row = rows[user]
        st.write(f"**{name}**　{row['count']}/{total}")
        st.progress(min(row['count']/max(total,1),1.0))
        if not compact and row['seconds']:
            duration = (total-row['count'])*row['seconds']
            days = int(-(-duration//3600))
            end = date.today()+timedelta(days=days)
            st.caption(f"{row['seconds']:.1f} 秒/句　·　预计 {end:%m/%d} 完成（每天 1 小时）")
    return rows, total


def dashboard():
    st.title('进度看板')
    st.caption('实时聚合进度 · 无需登录即可查看')
    dashboard_live()
    st.divider()
    st.subheader('导出全部为 CSV')
    st.caption('按设计提供各人原始长表和并排宽表；一致率与仲裁在导出后进行。')
    try:
        db = store()
        raw, wide = exports(db.sentences(), db.all_annotations())
        a,b = st.columns(2)
        a.download_button('下载 annotations_raw.csv', raw, 'annotations_raw.csv', 'text/csv', use_container_width=True)
        b.download_button('下载 annotations_wide.csv', wide, 'annotations_wide.csv', 'text/csv', use_container_width=True)
        st.caption('每天收工导出一次，并将备份提交进 Git。')
    except Exception:
        st.error('导出暂不可用。请恢复数据库连接后重试，未同步缓存不会冒充云端记录。')


@st.fragment(run_every=15)
def dashboard_live():
    try:
        db = store()
        summary, total = db.summary(), count_for(db, config())
    except Exception as exc:
        storage_error('暂时无法取得进度。', exc)
        return
    completed = sum(r['count'] for r in summary.values())
    # 总量 = 句数 × 人数，不是句数 × 3。名册现在从 secrets 读，人数会变。
    overall = total * len(USERS)
    st.metric('总进度', f'{completed}/{overall}', f'{completed/max(overall,1):.0%}', delta_color='off')
    st.progress(completed/max(overall,1))
    deadline = date.fromisoformat(config()['deadline'])
    st.caption(f'距 {deadline:%m/%d} 还有 {(deadline-date.today()).days} 天　·　每 15 秒刷新')
    progress_rows(result=(summary,total))
    st.subheader('标签分布（全员合计）')
    labels = {label: sum(r['labels'].get(label, 0) for r in summary.values())
              for label in ('objective', 'positive', 'negative', 'unclear')}
    for label, count in labels.items():
        st.write(f'{label}　{count/max(completed,1):.0%}　({count})')
        st.progress(count/max(completed,1))
    if labels['unclear']:
        st.caption(f"unclear = 脱离上下文无法判断（规范 R11）。仲裁时 ≥2 人标 unclear 的句子"
                   f"从金标准剔除，只有 1 人标的全组过一遍。当前共 {labels['unclear']} 条。")
    for user,label,own,average in deviation_alerts(summary):
        st.warning(f'{USERS[user]}的 {label} 占比 {own:.0%}，另两人均值 {average:.0%}，相差超过 15 个百分点。建议碰头对齐判定标准。')
