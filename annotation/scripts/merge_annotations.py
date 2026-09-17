"""把三个人各自本地标的结果合并成一份，并算一致率。

用法：
    # 三个人把各自的 annotations.csv 交上来，改名后放进一个目录
    python scripts/merge_annotations.py --dir 收上来的/

    # 或者直接点名文件
    python scripts/merge_annotations.py zhang.csv guo.csv sun.csv

产出（默认写进 outputs/merged/）：
    annotations_raw.csv    三人原始长表，算一致率用
    annotations_wide.csv   一句一行三人并排，仲裁分歧用
    disagreements.csv      只有分歧的句子，按严重程度排序
    agreement_report.txt   一致率、Kappa、各人标签分布

------------------------------------------------------------------------
为什么合并是安全的

每条记录的主键是 (username, sentence_id)，三个人的 username 不同，
所以**天然不会冲突**——合并就是拼起来。唯一要处理的是同一人同一句
出现两条（比如他从两台机器上传），这时取 updated_at 更新的那条。
------------------------------------------------------------------------
"""
import argparse
import csv
import itertools
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from annot.config import USERS                        # noqa: E402
from annot.db import FIELDS, atomic_csv, read_csv     # noqa: E402
from annot.export import DROP, sentence_status, triple  # noqa: E402

FIELDS3 = ('subjectivity', 'polarity', 'aspect')


def collect(paths):
    """读入所有文件，同一人同一句取时间最新的一条。"""
    best, sources = {}, defaultdict(set)
    for path in paths:
        rows = read_csv(path)
        if not rows:
            print(f'  ⚠ {Path(path).name} 里没有数据，跳过')
            continue
        for r in rows:
            user = r.get('username')
            if user not in USERS:
                print(f'  ⚠ {Path(path).name} 出现未知标注员 {user!r}，跳过该行')
                continue
            key = (user, r['sentence_id'])
            old = best.get(key)
            if old is None or str(r['updated_at']) > str(old['updated_at']):
                best[key] = r
            sources[user].add(Path(path).name)
        print(f'  读入 {Path(path).name:28} {len(rows):>5,} 条')
    for user, files in sources.items():
        if len(files) > 1:
            print(f'  ℹ {USERS[user]} 的记录来自 {len(files)} 个文件，已按时间取最新')
    return best


def label3(rec):
    """把两步标注合成 benchmark 惯例的单一标签。"""
    if rec['subjectivity'] == 'unclear':
        return 'unclear'
    if rec['subjectivity'] == 'objective':
        return 'objective'
    return rec['polarity']


def kappa(a, b):
    """两人之间的 Cohen's Kappa。"""
    pairs = [(x, y) for x, y in zip(a, b)]
    if not pairs:
        return None
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] / n * cb[k] / n for k in set(ca) | set(cb))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', help='三个人的 annotations.csv')
    ap.add_argument('--dir', type=Path, help='目录，自动收集里面所有 csv')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/merged')
    ap.add_argument('--sentences', type=Path,
                    default=ROOT / 'data/eval/to_label.csv')
    args = ap.parse_args()

    paths = list(args.files)
    if args.dir:
        paths += sorted(str(p) for p in args.dir.glob('*.csv'))
    if not paths:
        sys.exit('❌ 没有输入文件。用 --dir 指定目录，或直接列出文件。')

    print('读取：')
    best = collect(paths)
    if not best:
        sys.exit('❌ 一条有效记录都没有。')

    sentences = read_csv(args.sentences)
    order = {s['id']: i for i, s in enumerate(sentences)}
    text = {s['id']: s['text'] if 'text' in s else s.get('text_en', '')
            for s in sentences}

    args.out.mkdir(parents=True, exist_ok=True)

    # ── 长表 ──
    rows = sorted(best.values(),
                  key=lambda r: (order.get(r['sentence_id'], 10**9), r['username']))
    atomic_csv(args.out / 'annotations_raw.csv', FIELDS, rows)

    # ── 宽表 ──
    by_sent = defaultdict(dict)
    for (user, sid), r in best.items():
        by_sent[sid][user] = r
    users = list(USERS)
    wide_fields = (['id', 'text'] +
                   [f'{f}_{u}' for u in users for f in FIELDS3] +
                   [f'label_{u}' for u in users] + ['agree', 'note'])
    wide, disagree = [], []
    for sid in sorted(by_sent, key=lambda s: order.get(s, 10**9)):
        got = by_sent[sid]
        row = {'id': sid, 'text': text.get(sid, '')}
        labels = []
        notes = []
        for u in users:
            r = got.get(u)
            for f in FIELDS3:
                row[f'{f}_{u}'] = r[f] if r else ''
            row[f'label_{u}'] = label3(r) if r else ''
            if r:
                labels.append(label3(r))
                if r.get('note'):
                    notes.append(f"{USERS[u]}:{r['note']}")
        row['note'] = ' | '.join(notes)
        # 一致要看完整三元组，不能只看三分类标签 —— 三人都标 negative 但方面
        # 一个 seat 一个 food，只比标签会算成「全一致」，一致率因此虚高。
        # 而方面正是要被评测的三个分类器之一。
        triples = [triple(got[u]) for u in users if got.get(u)]
        uniq = len(set(triples))
        row['agree'] = (f'仅 {len(labels)} 人已标' if len(labels) < len(users)
                        else '全一致' if uniq == 1
                        else '三人各异' if uniq == 3
                        else '两人一致')
        wide.append(row)
        if len(labels) == len(users) and uniq > 1:
            disagree.append(row)
    atomic_csv(args.out / 'annotations_wide.csv', wide_fields, wide)
    # 分歧按严重程度排：三人各异 > 两人一致
    disagree.sort(key=lambda r: (r['agree'] != '三人各异', order.get(r['id'], 0)))
    atomic_csv(args.out / 'disagreements.csv', wide_fields, disagree)

    # ── 一致率报告 ──
    complete = [r for r in wide if all(r[f'label_{u}'] for u in users)]
    for r in complete:
        r['_status'] = sentence_status(
            {u: {f: r[f'{f}_{u}'] for f in FIELDS3} for u in users})
    lines = ['标注一致率报告', f'生成时间 {datetime.now():%Y-%m-%d %H:%M}', '']
    lines.append(f'三人都标过的句子：{len(complete):,} / {len(sentences):,}')
    for u in users:
        n = sum(1 for r in wide if r[f'label_{u}'])
        lines.append(f'  {USERS[u]:6} {n:>5,} 条')
    if complete:
        # 待剔除的句子不进金标准，不该算进一致率的分母。
        drop = [r for r in complete if r['_status'] == DROP]
        pool = [r for r in complete if r['_status'] != DROP]
        full = sum(1 for r in pool if r['agree'] == '全一致')
        rate = full / len(pool) if pool else 0
        lines += ['', f'三人完全一致：{full:,} / {len(pool):,} = {rate:.1%}'
                      f"   {'✅ 达标（作业要求 ≥80%）' if rate >= .8 else '❌ 未达标（作业要求 ≥80%）'}",
                  f'  口径：比完整三元组（主客观 + 极性 + 方面），'
                  f'已从分母排除 {len(drop)} 句待剔除']

        # 分维度看，才知道分歧到底出在哪一环。方面有 8 个类目，
        # 一致率天然低于只有 2–3 个类目的另外两维，报告里要分开讲。
        lines += ['', '各维度一致率（三人完全相同的比例）：']
        for field, cn in (('subjectivity', '主客观'), ('polarity', '极性'),
                          ('aspect', '方面')):
            n = sum(1 for r in pool
                    if len({r[f'{field}_{u}'] for u in users}) == 1)
            lines.append(f'  {cn:4} {n:>5,} / {len(pool):,} = {n/len(pool):.1%}'
                         if pool else f'  {cn:4} —')
        lines.append('  三维全对 = 上面那个总一致率；单看某一维会高于它。')

        lines.append('')
        lines.append('两两 Cohen Kappa（三分类标签，chance-corrected）：')
        for a, b in itertools.combinations(users, 2):
            k = kappa([r[f'label_{a}'] for r in pool],
                      [r[f'label_{b}'] for r in pool])
            lines.append(f'  {USERS[a]} × {USERS[b]}   {k:.3f}')
        lines.append('  Kappa 比原始一致率更该写进报告：标签分布偏斜时'
                     '（negative 占多数），光靠瞎猜也能撞对不少。')
        lines.append('')
        lines.append('方面维度的 Kappa（8 类，最容易分歧）：')
        for a, b in itertools.combinations(users, 2):
            k = kappa([r[f'aspect_{a}'] for r in pool],
                      [r[f'aspect_{b}'] for r in pool])
            lines.append(f'  {USERS[a]} × {USERS[b]}   {k:.3f}')
        lines += ['', '各人标签分布：']
        for u in users:
            c = Counter(r[f'label_{u}'] for r in complete)
            total = sum(c.values())
            lines.append(f'  {USERS[u]}: ' + '  '.join(
                f'{k} {v/total:.0%}' for k, v in sorted(c.items())))
        lines += ['', f'≥2 人标 unclear（从金标准剔除）：{len(drop)} 句',
                  f'需仲裁（分歧 + 待剔除）：'
                  f'{sum(1 for r in complete if r["agree"] != "全一致") + len(drop) - sum(1 for r in drop if r["agree"] != "全一致"):,} 句',
                  '',
                  '仲裁之后这些句子都会有唯一答案，金标准规模 = '
                  f'{len(complete) - len(drop):,} 句。',
                  '报告里要写的是**仲裁前**的这个原始一致率 —— '
                  '仲裁后必然 100%，那个数字没有意义。']
    report = '\n'.join(lines)
    (args.out / 'agreement_report.txt').write_text(report + '\n', encoding='utf-8')

    print(f'\n产出 → {args.out}')
    for f in ('annotations_raw.csv', 'annotations_wide.csv',
              'disagreements.csv', 'agreement_report.txt'):
        print(f'  {f}')
    print('\n' + report)


if __name__ == '__main__':
    main()
