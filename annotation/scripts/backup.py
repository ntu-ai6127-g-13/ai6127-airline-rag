"""把标注和仲裁一起导出成带日期的备份。

用法：
    DATABASE_URL='...' python scripts/backup.py            # 存进 backups/
    DATABASE_URL='...' python scripts/backup.py --out ~/Dropbox/airlabel

为什么要有这个脚本
  Supabase 免费版**不含自动备份，也没有时间点恢复**。数据没了就是没了，
  而这是 3 人 × 3–4 小时的人工，只能重标。

  看板上那个「下载 annotations_raw.csv」只导标注，**不含仲裁结论**。
  仲裁是全组碰头讨论出来的，比原始标注更难重建 —— 原始标注至少还有
  三个人的记忆，讨论结论过两周没人说得清。所以备份必须两张表都带。

  语料和账号不用备份：语料在仓库里（data/eval/），账号在 secrets 里。
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from annot.config import settings                      # noqa: E402
from annot.db import ARB_FIELDS, FIELDS, atomic_csv, make_store   # noqa: E402
from annot.export import exports                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'backups')
    args = ap.parse_args()

    cfg = settings()
    store = make_store(cfg)
    annotations = store.all_annotations()
    finals = list(store.finals().values())
    sentences = store.sentences()

    # 空备份不落盘。脚本已经知道这份是空的，还建个带时间戳的目录，
    # 三周后回头看 backups/ 一堆目录分不清哪个是真的 —— 那比没有备份更危险。
    if not annotations:
        sys.exit('❌ 一条标注都没有，不生成备份。\n'
                 f'   当前存储：{cfg["storage"]}'
                 + ('（本机空目录）\n   备份线上库要先：export DATABASE_URL=...'
                    if cfg['storage'] == 'local' else '\n   确认连的是正确的库。'))

    # 目录名带上存储类型，本地快照不会被误当成云端备份
    stamp = datetime.now().strftime('%Y%m%d-%H%M')
    out = args.out / (stamp if cfg['storage'] != 'local' else f'{stamp}-local')
    out.mkdir(parents=True, exist_ok=True)

    atomic_csv(out / 'annotations.csv', FIELDS, annotations)
    atomic_csv(out / 'arbitration.csv', ARB_FIELDS, finals)
    raw, wide = exports(sentences, annotations)
    (out / 'annotations_wide.csv').write_bytes(wide)

    per_user = {}
    for r in annotations:
        per_user[r['username']] = per_user.get(r['username'], 0) + 1

    print(f'✅ 备份到 {out}')
    print(f'   存储     {cfg["storage"]}')
    print(f'   标注     {len(annotations):,} 条', end='  ')
    print('（' + '  '.join(f'{u} {n}' for u, n in sorted(per_user.items())) + '）'
          if per_user else '')
    print(f'   仲裁     {len(finals):,} 条')
    print(f'   语料     {len(sentences):,} 句')

    if cfg['storage'] == 'local':
        print('\n⚠️  当前是本地存储，备份的不是云端数据。'
              '备份线上库要先 export DATABASE_URL=...')


if __name__ == '__main__':
    main()
