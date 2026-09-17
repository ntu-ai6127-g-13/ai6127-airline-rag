"""The two exports specified in the Claude design."""
import csv
import io
from .config import USERS
from .db import FIELDS


def csv_bytes(fields, rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        # Keep exported notes/text from becoming executable spreadsheet formulas.
        writer.writerow({k: ("'" + str(v) if str(v).lstrip().startswith(('=', '+', '-', '@')) else v)
                         for k, v in row.items()})
    return buffer.getvalue().encode('utf-8-sig')


def exports(sentences, annotations):
    raw = csv_bytes(FIELDS, annotations)
    index = {(r['sentence_id'], r['username']): r for r in annotations}
    fields = ['id', 'text_en', 'text_zh']
    # 列后缀按名册长度生成，不写死 abc —— 名册是 secrets 里配的，
    # 写死的话多一个人 zip 会静默截断（那个人的标注就这么丢了），
    # 少一个人则多出两列空的。
    letters = [chr(ord('a') + i) for i in range(len(USERS))]
    for letter in letters:
        fields.extend(f'{name}_{letter}' for name in ('subj','polarity','aspect','note'))
        fields.append(f'label_{letter}')          # 三分类合并标签，对齐 benchmark
    fields.append('agree')
    rows = []
    for sentence in sentences:
        row = {k: sentence[k] for k in ('id','text_en','text_zh')}
        labels = []
        for user, letter in zip(USERS, letters):
            record = index.get((sentence['id'], user), {})
            for source, target in [('subjectivity','subj'),('polarity','polarity'),('aspect','aspect'),('note','note')]:
                row[f'{target}_{letter}'] = record.get(source, '')
            row[f'label_{letter}'] = merged_label(record)
            if row[f'label_{letter}']:
                labels.append(row[f'label_{letter}'])
        # 判一致要看完整三元组，不能只看合并标签 —— 三人都是 subjective/negative
        # 但方面一个 seat 一个 food，只看 label 会误判成「一致」。
        triples = {triple(index.get((sentence['id'], u), {})) for u in USERS}
        row['agree'] = ('' if len(labels) < len(USERS)
                        else '一致' if len(triples) == 1
                        else '分歧')
        rows.append(row)
    return raw, csv_bytes(fields, rows)


def merged_label(record):
    """把两步标注合成 benchmark 惯例的单一标签。

    eval.xls 的第 2 列要的就是这个：objective / positive / negative。
    unclear 单独保留——仲裁时 ≥2 人标 unclear 的句子要从金标准剔除，
    不能混进那三类里。
    """
    subj = record.get('subjectivity', '')
    if not subj:
        return ''
    if subj in ('objective', 'unclear'):
        return subj
    return record.get('polarity', '')


def triple(record):
    """完整答案三元组。判一致/分歧必须用它，不能只看 merged_label。

    merged_label 只返回 objective / positive / negative / unclear，**看不见 aspect**。
    三人都标 subjective/negative，方面一个 seat 一个 food，用 merged_label
    会判成「一致」——而方面分类器正是要被评测的三个分类器之一，这种分歧
    漏掉就等于给了它一份自相矛盾的金标准。
    """
    return (record.get('subjectivity', ''),
            record.get('polarity', ''),
            record.get('aspect', ''))


#: sentence_status() 的取值。总览页和仲裁页都从这里读，不各判一套。
UNSTARTED, PARTIAL, AGREE, DIFFER, DROP = (
    'unstarted', 'partial', 'agree', 'differ', 'drop')


def _answered(records):
    """滤掉「待定」记录（三字段全空、只有备注）—— 它不是答案，是便签。"""
    return {u: r for u, r in records.items() if r.get('subjectivity')}


def sentence_status(records):
    """一句话现在是什么状态。records 是 {username: 标注记录}。

    drop 排在 differ 前面：≥2 人标 unclear 时，不管三元组一不一样，
    该做的动作都是按 R11 剔除，不是去调解分歧。
    """
    records = _answered(records)
    n = sum(1 for u in USERS if u in records)
    if n == 0:
        return UNSTARTED
    if n < len(USERS):
        return PARTIAL
    if sum(merged_label(records[u]) == 'unclear' for u in USERS) >= 2:
        return DROP
    return DIFFER if len({triple(records[u]) for u in USERS}) > 1 else AGREE


def needs_arbitration(records):
    """这句要不要上仲裁页。

    两类：
      1. 三人答案不完全相同（含只有方面不同的情况）
      2. ≥2 人标了 unclear —— 三人都标 unclear 算「一致」，但这种一致恰恰
         说明该句不可标注，按 R11 要从金标准剔除。不单独捞出来它永远不会
         出现在仲裁页。
    尚未标满三人的句子不算——还没到能判的时候。
    """
    return sentence_status(records) in (DIFFER, DROP)


def deviation_alerts(summary):
    """某人某个标签的占比明显偏离另两人时报警。

    labels 用 .get(label, 0) 取——不同来源的 summary 可能缺 key
    （比如某人一条 unclear 都没标，Postgres 的聚合就不会给这个键），
    直接下标会 KeyError 把整个看板打挂。
    """
    alerts = []
    for user, data in summary.items():
        others = [v for k, v in summary.items() if k != user and v['count']]
        if data['count'] < 100 or len(others) != 2:
            continue
        for label, count in data['labels'].items():
            own = count / data['count']
            average = sum(d['labels'].get(label, 0) / d['count'] for d in others) / 2
            if abs(own - average) > .15:
                alerts.append((user, label, own, average))
    return alerts
