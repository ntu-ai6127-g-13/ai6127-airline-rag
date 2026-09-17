"""给名册加一个人：生成密码 + bcrypt 哈希 + 可直接粘贴的配置段。

用法：
    python scripts/add_annotator.py zhuJindou 朱津豆
    python scripts/add_annotator.py zhuJindou 朱津豆 --write   # 顺便写进本地 secrets.toml

为什么要有这个脚本
  名册在 secrets 里，加人本身只要贴一段 TOML。但 pw_hash 得算，
  手搓 bcrypt 容易出错。这个脚本只管生成，贴哪儿由你决定：
    · 云端 → Streamlit Cloud 的 Secrets 面板（改完会自动重启，名册立刻生效）
    · 本地 → 加 --write 直接追加到 .streamlit/secrets.toml

加完之后要跑一次 init_annot_db.py，把账号写进数据库（否则登录查不到）。
"""
import argparse
import re
import secrets as pysecrets
import sys
from pathlib import Path

import bcrypt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from annot.config import roster        # noqa: E402

# username 会进数据库主键、CSV 列名和登录下拉框，限制成安全字符
VALID = re.compile(r'^[a-zA-Z][a-zA-Z0-9_]{1,30}$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('username', help='登录名，字母开头，只能用字母数字下划线')
    ap.add_argument('display_name', help='显示名，例如 朱津豆')
    ap.add_argument('--write', action='store_true',
                    help='追加到本地 .streamlit/secrets.toml')
    args = ap.parse_args()

    if not VALID.match(args.username):
        sys.exit(f'❌ username「{args.username}」不合法：字母开头，2–31 位字母数字下划线')
    existing = roster()
    if args.username in existing:
        sys.exit(f'❌ 名册里已经有 {args.username}（{existing[args.username]}）。'
                 f'\n   改密码请直接换掉 secrets 里那一段的 pw_hash。')

    password = pysecrets.token_urlsafe(12)
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    block = (f'[annotators.{args.username}]\n'
             f'display_name = "{args.display_name}"\n'
             f'pw_hash = "{pw_hash}"\n')

    print(f'\n{args.display_name} / {args.username}')
    print(f'  密码：{password}      ← 私下单独发给本人，不要发群里\n')
    print('把下面这段贴进 Streamlit Cloud → Settings → Secrets：\n')
    print(block)

    if args.write:
        path = ROOT / '.streamlit/secrets.toml'
        if not path.exists():
            sys.exit('❌ 本地没有 .streamlit/secrets.toml，先跑 init_annot_db.py --generate-passwords')
        with path.open('a', encoding='utf-8') as f:
            f.write('\n' + block)
        print(f'✅ 已追加到 {path.relative_to(ROOT)}')

    print(f'\n下一步：把账号写进数据库，否则登录查不到\n'
          f'  python scripts/init_annot_db.py --storage '
          f'{"supabase" if not args.write else "local"}')


if __name__ == '__main__':
    main()
