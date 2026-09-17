import os
from pathlib import Path
try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]

ASPECTS = {'seat': '座位', 'staff': '服务', 'food': '餐饮', 'entertainment': '娱乐',
           'ground': '地勤', 'value': '性价比', 'overall': '总体', 'other': '其他'}


def read_secrets():
    """读配置。云端优先走 st.secrets —— Streamlit Cloud 的 Secrets 面板不会
    在仓库里落一个 secrets.toml，只认文件的话云端永远读不到名册。"""
    try:
        import streamlit as st
        data = dict(st.secrets)
        if data:
            return data
    except Exception:
        pass
    path = ROOT / '.streamlit/secrets.toml'
    if path.exists():
        return tomllib.loads(path.read_text(encoding='utf-8'))
    return {}


def roster(secrets=None):
    """标注员名册 {username: 显示名}，按 secrets 里的书写顺序。

    为什么不写死在代码里
      名册一变就要改代码、重新部署。而「谁来标」这件事到临门一脚还在动
      （现在就只定了一个人，另外两位待定）。放进 secrets 之后，加人 =
      在 Streamlit Cloud 的 Secrets 面板贴一段，重启就生效。

    没配名册时退回 DEFAULT_ROSTER，好让 --generate-passwords 有东西可生成。
    """
    cfg = secrets if secrets is not None else read_secrets()
    people = cfg.get('annotators') or {}
    names = {u: (dict(v).get('display_name') or u) for u, v in people.items()}
    return names or dict(DEFAULT_ROSTER)


#: 一个都没配置时的兜底。真正生效的名册在 secrets 的 [annotators.*] 里。
#: 现在只定了孙思洁，另外两位确定后用 scripts/add_annotator.py 生成配置段。
DEFAULT_ROSTER = {'sunsijie': '孙思洁'}


def cn_count(n):
    """人数的中文写法。文案里原来写死「三人」，名册变成可配置之后就会说谎
    —— 现在只有一位标注员，页面却还在说「三人分歧」。"""
    return '〇一二三四五六七八九十'[n] if 0 <= n <= 10 else str(n)


def settings(secrets=None):
    config = dict(secrets if secrets is not None else read_secrets())
    return {
        'storage': os.getenv('STORAGE', config.get('storage', 'supabase')),
        'dsn': os.getenv('DATABASE_URL', config.get('database_url', '')),
        'local_dir': Path(os.getenv('ANNOT_LOCAL_DIR', str(ROOT / 'outputs/annotation-local'))),
        'cache_dir': Path(os.getenv('ANNOT_CACHE_DIR', str(Path.home() / '.annot_cache'))),
        'annotators': dict(config.get('annotators', {})),
        # 仲裁页 / 总览页的口令。这两页会露出三个人各自的答案，只有统筹该开。
        # 明文存在 gitignored 的 secrets.toml 里 —— 6 位数字做成哈希也是分钟级
        # 爆破，加密只会给人虚假的安全感。真正起作用的是它不进仓库。
        'reveal_password': os.getenv('REVEAL_PASSWORD',
                                     config.get('reveal_password', '')),
        'deadline': config.get('deadline', '2026-10-25'),
    }


#: 模块级快照。改名册要重启应用（Streamlit Cloud 改完 Secrets 会自动重启）。
USERS = roster()
