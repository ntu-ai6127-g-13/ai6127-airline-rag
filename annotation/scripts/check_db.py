"""检查标注数据存在哪、存了多少、有没有风险。接 Supabase 之前先跑这个。

用法：
    python scripts/check_db.py                    # 看当前配置的存储
    python scripts/check_db.py --dsn "postgres://…"  # 试一个连接串，不写进配置
    python scripts/check_db.py --backup            # 顺便导出一份备份

------------------------------------------------------------------------
为什么需要这个脚本

标注是 3 人 × 3–4 小时的人工成本，**丢了没法重建**。而出问题的方式
往往是安静的：连错库、云容器把 CSV 冲掉、以为在写共享库其实写的是本机。
每次开工前跑一次，10 秒就知道数据落在哪。
------------------------------------------------------------------------
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from annot.config import settings          # noqa: E402
from annot.db import make_store            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dsn', help='试一个 Postgres 连接串（不写进配置文件）')
    ap.add_argument('--backup', action='store_true', help='顺便导出一份 CSV 备份')
    args = ap.parse_args()

    cfg = settings()
    if args.dsn:
        cfg = {**cfg, 'storage': 'supabase', 'dsn': args.dsn}
        print('用命令行给的连接串测试（不会写进 secrets.toml）\n')

    print(f"  storage   = {cfg['storage']}")
    print(f"  local_dir = {cfg['local_dir']}")
    print(f"  dsn       = {'已配置' if cfg['dsn'] else '(空)'}")

    try:
        store = make_store(cfg)
    except Exception as exc:
        sys.exit(f"\n❌ 连不上：{type(exc).__name__}: {exc}")

    try:
        info = store.stats()
        n_sent = len(store.sentences())
    except Exception as exc:
        msg = str(exc)
        if 'connection' in msg.lower() or 'timeout' in msg.lower():
            sys.exit(f"\n❌ 连不上数据库：{msg}\n"
                     "   检查：连接串对不对、密码有没有写错、"
                     "Supabase 项目是不是被暂停了、本机网络能不能出去。")
        sys.exit(f"\n❌ 连上了但读不了数据（表可能还没建）：{msg}\n"
                 f"   先跑：python scripts/init_annot_db.py --storage {cfg['storage']}")

    print(f"\n  ✅ {info['kind']}")
    print(f"     位置   {info['where']}")
    print(f"     语料   {n_sent:,} 句")
    print(f"     标注   {info['rows']:,} 条")

    if info['shared']:
        print("\n  🟢 共享存储，三个人写同一份数据。")
    else:
        print("\n  🟡 本机存储，数据只在这台机器上，三人无法共享。")
        print("     部署到 Streamlit Cloud 时**不要**用这个模式 ——")
        print("     容器重新部署或休眠唤醒会把 CSV 一起清掉。")
        backups = Path(cfg['local_dir']) / 'backups'
        if backups.exists():
            files = sorted(backups.glob('annotations-*.csv'))
            if files:
                print(f"     每日备份 {len(files)} 份，最新 {files[-1].name}")

    if args.backup:
        rows = store.all_annotations()
        out = ROOT / 'outputs' / f'annotations-backup-{datetime.now():%Y%m%d-%H%M%S}.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with out.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['username', 'sentence_id', 'subjectivity',
                                              'polarity', 'aspect', 'note', 'updated_at'],
                               extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f"\n  💾 已备份 {len(rows):,} 条 → {out.relative_to(ROOT)}")


if __name__ == '__main__':
    main()
