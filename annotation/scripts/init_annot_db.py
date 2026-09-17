"""Build tables, import the fixed corpus/translations, and create three accounts."""
import argparse
import csv
import json
import re
import secrets
import sys
from pathlib import Path
import bcrypt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from annot.config import USERS, read_secrets, settings
from annot.db import LocalStore, make_store


def load_sentences(input_path, translation_path):
    with input_path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sentence IDs')
    if len({r['text'].strip() for r in rows}) != len(rows):
        raise ValueError('Duplicate sentence texts')
    translations = {}
    if translation_path.exists():
        with translation_path.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                if row['id'] in translations:
                    raise ValueError('Duplicate translation ID')
                translations[row['id']] = row['text_zh']
    result = []
    for i, row in enumerate(rows, 1):
        text = row['text'].strip()
        if not text:
            raise ValueError('Empty sentence')
        risk = row.get('risk_flag', '')
        if not risk:
            if re.search(r'\b(yeah right|so much for|wonderful[,!].*delay)\b', text, re.I):
                risk = 'sarcasm'
            elif re.search(r'\b(IFE|AVOD|PNR|ETOPS|codeshare|bulkhead|pitch|red-eye)\b', text, re.I):
                risk = 'jargon'
            elif len(text.split()) < 5:
                risk = 'short'
        result.append({'id': row['id'], 'text_en': text, 'text_zh': translations.get(row['id'], ''),
                       'risk_flag': risk, 'seq': i})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', choices=['local','supabase'])
    parser.add_argument('--generate-passwords', action='store_true', help='Create secrets only if absent; never rotate silently')
    parser.add_argument('--annotators', default='',
                        help='首次生成名册，形如 sunsijie=孙思洁,zhu=朱津豆。'
                             '不填就用 config.DEFAULT_ROSTER。人没定齐时先建已定的，'
                             '其余用 scripts/add_annotator.py 随时追加。')
    parser.add_argument('--input', type=Path, default=ROOT/'data/eval/to_label.csv')
    parser.add_argument('--translations', type=Path, default=ROOT/'data/eval/translations.csv')
    args = parser.parse_args()
    if args.generate_passwords:
        path = ROOT / '.streamlit/secrets.toml'
        if path.exists():
            raise SystemExit('secrets.toml already exists; existing credentials were preserved')
        people = USERS
        if args.annotators:
            people = {}
            for pair in args.annotators.split(','):
                key, _, name = pair.partition('=')
                if not key.strip() or not name.strip():
                    raise SystemExit(f'--annotators 格式错误：{pair!r}，应为 key=名字')
                people[key.strip()] = name.strip()
        passwords = {u: secrets.token_urlsafe(12) for u in people}
        mode = args.storage or 'supabase'
        content = [f'storage = "{mode}"', 'deadline = "2026-10-25"', 'database_url = ""', '']
        for user, name in people.items():
            pw_hash = bcrypt.hashpw(passwords[user].encode(), bcrypt.gensalt()).decode()
            content.extend([f'[annotators.{user}]', f'display_name = "{name}"', f'pw_hash = "{pw_hash}"', ''])
        path.parent.mkdir(exist_ok=True)
        with path.open('x', encoding='utf-8') as f:
            f.write('\n'.join(content))
        credentials = ROOT / 'outputs/annotation-credentials.txt'
        credentials.parent.mkdir(exist_ok=True)
        with credentials.open('x', encoding='utf-8') as f:
            f.write('标注系统账号（仅本机保存，请线下分别给本人；不要提交 Git）\n\n')
            for user, name in people.items():
                f.write(f'{name} / {user}: {passwords[user]}\n')
        print('Created .streamlit/secrets.toml; passwords saved to outputs/annotation-credentials.txt')
    # secrets 刚写完，config 里 USERS 的模块级快照已经过期了 —— 重新读一次，
    # 否则 --annotators 指定的名册不会生效（会用 DEFAULT_ROSTER 去建账号）。
    import annot.config as _cfgmod
    _cfgmod.USERS = _cfgmod.roster()
    people_now = _cfgmod.USERS
    cfg = settings()
    if args.storage:
        cfg['storage'] = args.storage
    annotators = [{'username': u, 'display_name': people_now[u],
                   'pw_hash': cfg['annotators'][u]['pw_hash']} for u in people_now]
    sentences = load_sentences(args.input, args.translations)
    store = make_store(cfg)
    store.initialize(sentences, annotators)
    if cfg['storage'] != 'local':
        # Keep a read-only corpus snapshot for a database outage.
        LocalStore(cfg['local_dir']).initialize(sentences, annotators)
    missing = sum(not s['text_zh'].strip() for s in sentences)
    print(f'Initialized {len(sentences)} sentences, {len(annotators)} accounts, {missing} missing translations; storage={cfg["storage"]}')


if __name__ == '__main__':
    main()
