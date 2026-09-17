"""造测试标注数据，用来验收仲裁页和进度看板。

用法：
    STORAGE=local python scripts/seed_test_data.py           # 造 50 题
    STORAGE=local python scripts/seed_test_data.py --n 120   # 造 120 题（够触发偏离预警）
    STORAGE=local python scripts/seed_test_data.py --clear   # 清空

⚠️ 只允许在 STORAGE=local 下运行。往共享库里塞假数据会污染三个人的真实标注，
   而标注是 9 小时人工、丢了没法重建。
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from annot.config import USERS, settings      # noqa: E402
from annot.db import make_store               # noqa: E402

A, B, C = list(USERS)

# (subjectivity, polarity, aspect)
AGREE = ('subjective', 'negative', 'seat')
POS = ('subjective', 'positive', 'staff')
OBJ = ('objective', 'NA', 'other')
UNC = ('unclear', 'NA', 'unclear')

# 十种分歧，覆盖仲裁页要处理的每一类情况
DISAGREE = [
    # (说明, A 的答案, B 的答案, C 的答案)
    ('2:1 极性分歧（多数 negative）', AGREE, AGREE, POS),
    ('2:1 极性分歧（多数 positive）', POS, POS, AGREE),
    ('2:1 主客观分歧', OBJ, AGREE, AGREE),
    ('2:1 方面分歧', AGREE, AGREE, ('subjective', 'negative', 'food')),
    ('2:1 方面分歧（staff vs ground）',
     ('subjective', 'negative', 'ground'), ('subjective', 'negative', 'ground'),
     ('subjective', 'negative', 'staff')),
    ('2:1 含 unclear（多数 negative）', AGREE, AGREE, UNC),
    ('三人各异，无多数', AGREE, POS, OBJ),
    ('三人各异，含 unclear', AGREE, POS, UNC),
    ('≥2 人标 unclear → 应剔除', UNC, UNC, AGREE),
    ('三人全标 unclear → 应剔除', UNC, UNC, UNC),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50, help='造多少题（默认 50）')
    ap.add_argument('--clear', action='store_true', help='清空所有标注和仲裁')
    args = ap.parse_args()

    cfg = settings()
    if cfg['storage'] != 'local':
        sys.exit('❌ 只允许 STORAGE=local。往共享库塞假数据会污染真实标注。\n'
                 '   用法：STORAGE=local python scripts/seed_test_data.py')
    store = make_store(cfg)
    sentences = store.sentences()
    if not sentences:
        sys.exit('❌ 语料未导入，先跑 scripts/init_annot_db.py --storage local')

    path = Path(cfg['local_dir'])
    if args.clear:
        for f in ('annotations.csv', 'arbitration.csv'):
            (path / f).unlink(missing_ok=True)
        print('✅ 已清空标注和仲裁记录')
        return

    n = min(args.n, len(sentences))
    n_dis = len(DISAGREE)
    if n < n_dis:
        sys.exit(f'❌ 至少要 {n_dis} 题才放得下 {n_dis} 种分歧')

    # 分歧均匀分布在这 n 题里，别全挤在开头
    step = n // n_dis
    dis_at = {i * step: DISAGREE[i] for i in range(n_dis)}

    t0 = datetime.now(timezone.utc) - timedelta(hours=2)
    written = 0
    for idx in range(n):
        s = sentences[idx]
        combo = dis_at.get(idx)
        for k, user in enumerate((A, B, C)):
            ans = combo[k + 1] if combo else (AGREE if idx % 3 else POS if idx % 2 else OBJ)
            # 每人节奏不同：A 快、B 中、C 慢，这样看板的「秒/句」能看出差别
            pace = (12, 18, 25)[k]
            ts = t0 + timedelta(seconds=idx * pace + k * 3)
            store.upsert(dict(username=user, sentence_id=s['id'],
                              subjectivity=ans[0], polarity=ans[1], aspect=ans[2],
                              note='#sarcasm' if combo and k == 0 else '',
                              updated_at=ts.isoformat()))
            written += 1

    print(f'✅ 已写入 {written} 条标注（{n} 题 × 3 人）')
    print(f'   其中 {n_dis} 题有分歧，分布在第 '
          f'{", ".join(str(i+1) for i in sorted(dis_at))} 题：')
    for i in sorted(dis_at):
        print(f'     第 {i+1:>3} 题（{sentences[i]["id"]}）  {dis_at[i][0]}')
    print(f'\n   三人速度不同（12/18/25 秒每题），可在看板上对比。')


if __name__ == '__main__':
    main()
