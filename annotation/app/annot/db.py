"""Same interface for shared Postgres and atomic, locked local CSV storage."""
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from filelock import FileLock
from .config import ASPECTS, USERS

FIELDS = ['username', 'sentence_id', 'subjectivity', 'polarity', 'aspect', 'note', 'updated_at']
ARB_FIELDS = ['sentence_id', 'action', 'subjectivity', 'polarity', 'aspect',
              'note', 'decided_by', 'decided_at']
SENTENCE_FIELDS = ['id', 'text_en', 'text_zh', 'risk_flag', 'seq']
SCHEMA = """
CREATE TABLE IF NOT EXISTS annotators (
 username TEXT PRIMARY KEY, display_name TEXT NOT NULL, pw_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sentences (
 id TEXT PRIMARY KEY, text_en TEXT NOT NULL, text_zh TEXT NOT NULL DEFAULT '',
 risk_flag TEXT, seq INT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS annotations (
 username TEXT REFERENCES annotators(username), sentence_id TEXT REFERENCES sentences(id),
 subjectivity TEXT NOT NULL CHECK(subjectivity IN ('objective','subjective','unclear','')),
 polarity TEXT NOT NULL CHECK(polarity IN ('positive','negative','NA','')),
 aspect TEXT NOT NULL CHECK(aspect IN ('seat','staff','food','entertainment','ground','value','overall','other','unclear','')),
 note TEXT NOT NULL DEFAULT '', updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 PRIMARY KEY(username,sentence_id),
 -- unclear = 脱离上下文无法判断（规范 R11），三个字段必须一起是 unclear，
 -- 避免出现「标不了但选了方面」这种自相矛盾的记录。
 -- 三字段全空 + 有备注 = 「待定」：标注员卡住了，先记下原因往后走。
 -- 这类记录在进度、一致率、仲裁里一律**不算已标注**，只是个占位的便签。
 CHECK((subjectivity='' AND polarity='' AND aspect='' AND note<>'') OR
       (subjectivity='unclear' AND polarity='NA' AND aspect='unclear') OR
       (subjectivity='objective' AND polarity='NA' AND aspect<>'unclear') OR
       (subjectivity='subjective' AND polarity IN ('positive','negative') AND aspect<>'unclear'))
);
-- 仲裁结果。三人分歧的句子由全组过一遍定夺，结论存这里。
-- action: adopt=采纳某个答案 | manual=手工裁定 | drop=剔除出金标准
CREATE TABLE IF NOT EXISTS arbitration (
 sentence_id TEXT PRIMARY KEY REFERENCES sentences(id),
 action TEXT NOT NULL CHECK(action IN ('adopt','manual','drop')),
 subjectivity TEXT, polarity TEXT, aspect TEXT, note TEXT NOT NULL DEFAULT '',
 decided_by TEXT NOT NULL, decided_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE arbitration ENABLE ROW LEVEL SECURITY;
ALTER TABLE annotators ENABLE ROW LEVEL SECURITY;
ALTER TABLE sentences ENABLE ROW LEVEL SECURITY;
ALTER TABLE annotations ENABLE ROW LEVEL SECURITY;
"""


def check_append_only(existing, incoming):
    """已导入的句子只能原样保留，新句子只能接在末尾。

    为什么是前缀校验，不是全等
      仲裁会把「≥2 人标 unclear」的句子剔出金标准，之后要从预留池补足回
      1,000 句 —— 那是一次**追加**。原来的全等比对连追加一起拒掉，
      补足流程根本走不了。

    为什么不放宽成「已有的都还在就行」
      标注页按 seq 顺序发题。seq 一变，三个人的进度就不再指向同一批句子，
      一致率（作业硬指标 ≥80%）失去计算基础。所以已有部分必须连内容带顺序
      原样不动。

    这道闸挡的是正规导入路径。手写 `UPDATE sentences SET text_en=...` 绕得过去，
    而且改了不会有任何报错 —— 标注表只存 sentence_id，句子换了内容，
    已有的判断就悄悄变成了对另一句话的判断。别那么做。
    """
    old = [(r['id'], r['text_en'], int(r['seq'])) for r in existing]
    new = [(r['id'], r['text_en'], int(r['seq'])) for r in incoming]
    if old and new[:len(old)] != old:
        raise ValueError(
            'Existing corpus differs; sentences may only be appended, never changed '
            f'(already imported {len(old)}, given {len(new)})')


def validate(record):
    if record.get('username') not in USERS:
        raise ValueError('Unknown annotator')
    subj, pol, asp = (record.get('subjectivity'), record.get('polarity'),
                      record.get('aspect'))
    # 「待定」：三个字段全空 + 有备注。卡住时先记下原因往后走，
    # 免得被逼着在「硬猜」和「误用 unclear」之间二选一 —— 后者更糟，
    # ≥2 人标 unclear 会把句子从金标准剔除。
    if not subj and not pol and not asp:
        if not (record.get('note') or '').strip():
            raise ValueError('Pending record must carry a note')
        if len(record.get('note', '')) > 10000:
            raise ValueError('Note too long')
        if not datetime.fromisoformat(record['updated_at']).tzinfo:
            raise ValueError('Timestamp must include timezone')
        return
    # 「标不了」是一个整体状态：三个字段要么一起 unclear，要么都不是。
    if subj == 'unclear':
        if pol != 'NA' or asp != 'unclear':
            raise ValueError('unclear must carry polarity=NA and aspect=unclear')
        if len(record.get('note', '')) > 10000:
            raise ValueError('Note too long')
        if not datetime.fromisoformat(record['updated_at']).tzinfo:
            raise ValueError('Timestamp must include timezone')
        return
    if asp not in ASPECTS:
        raise ValueError('Choose one of the eight aspects')
    if not ((subj == 'objective' and pol == 'NA') or
            (subj == 'subjective' and pol in ('positive', 'negative'))):
        raise ValueError('Invalid subjectivity/polarity combination')
    if len(record.get('note', '')) > 10000:
        raise ValueError('Note too long')
    if not datetime.fromisoformat(record['updated_at']).tzinfo:
        raise ValueError('Timestamp must include timezone')


def atomic_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def speed(timestamps):
    times = sorted(datetime.fromisoformat(str(t)) for t in timestamps)
    gaps = [(b-a).total_seconds() for a, b in zip(times, times[1:])]
    gaps = [g for g in gaps if 0 < g <= 300]
    return median(gaps) if gaps else None


def is_answered(record):
    """这条记录算不算「已标注」。

    「待定」（三字段全空、只有备注）不算 —— 它是标注员卡住时留的便签，
    不是答案。进度、一致率、仲裁全都要把它当成「还没标」，
    否则进度会虚高，而一致率会拿空答案去比。
    """
    return bool(record.get('subjectivity'))


def summarize(rows):
    result = {}
    for user in USERS:
        own = [r for r in rows if r['username'] == user and is_answered(r)]
        labels = {'objective': 0, 'positive': 0, 'negative': 0, 'unclear': 0}
        for row in own:
            if row['subjectivity'] in ('objective', 'unclear'):
                labels[row['subjectivity']] += 1
            else:
                labels[row['polarity']] += 1
        result[user] = {'count': len(own), 'labels': labels,
                        'seconds': speed([r['updated_at'] for r in own])}
    return result


class LocalStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.directory / '.store.lock'), timeout=10)

    def initialize(self, sentences, annotators):
        with self.lock:
            check_append_only(self.sentences(), sentences)
            atomic_csv(self.directory / 'sentences.csv', SENTENCE_FIELDS, sentences)
            atomic_csv(self.directory / 'annotators.csv', ['username', 'display_name', 'pw_hash'], annotators)

    def sentences(self):
        return sorted(read_csv(self.directory / 'sentences.csv'), key=lambda s: int(s['seq']))

    def own(self, username):
        if username not in USERS:
            raise ValueError('Unknown annotator')
        with self.lock:
            return [r for r in read_csv(self.directory / 'annotations.csv') if r['username'] == username]

    def upsert(self, record):
        validate(record)
        with self.lock:
            if record['sentence_id'] not in {s['id'] for s in self.sentences()}:
                raise ValueError('Unknown sentence')
            rows = read_csv(self.directory / 'annotations.csv')
            keyed = {(r['username'], r['sentence_id']): r for r in rows}
            key = (record['username'], record['sentence_id'])
            if key not in keyed or datetime.fromisoformat(record['updated_at']) >= datetime.fromisoformat(keyed[key]['updated_at']):
                keyed[key] = {field: record[field] for field in FIELDS}
                atomic_csv(self.directory / 'annotations.csv', FIELDS, keyed.values())
                self._backup(keyed.values())

    def _backup(self, rows):
        """每天留一份快照。人工标注丢了没法重建，主文件写坏时靠这个回滚。

        只按天覆盖，不会堆出上千个文件；9 小时的人工成本值这点磁盘。
        """
        try:
            day = datetime.now().strftime('%Y%m%d')
            backups = self.directory / 'backups'
            backups.mkdir(exist_ok=True)
            atomic_csv(backups / f'annotations-{day}.csv', FIELDS, rows)
        except Exception:
            pass          # 备份失败不能影响正常保存

    def stats(self):
        """给界面显示：数据到底存在哪、存了多少条。"""
        path = self.directory / 'annotations.csv'
        return {'kind': '本机 CSV', 'where': str(path),
                'rows': len(read_csv(path)), 'shared': False}

    def delete(self, username, sentence_id):
        """清掉某人某句的作答（界面上按 C）。只能删自己的。"""
        if username not in USERS:
            raise ValueError('Unknown annotator')
        with self.lock:
            rows = read_csv(self.directory / 'annotations.csv')
            kept = [r for r in rows
                    if not (r['username'] == username and r['sentence_id'] == sentence_id)]
            if len(kept) != len(rows):
                atomic_csv(self.directory / 'annotations.csv', FIELDS, kept)
            return len(rows) - len(kept)

    def save_final(self, record):
        """存一条仲裁结论。同一句重复裁定就覆盖。"""
        if record.get('action') not in ('adopt', 'manual', 'drop'):
            raise ValueError('action must be adopt/manual/drop')
        with self.lock:
            rows = read_csv(self.directory / 'arbitration.csv')
            keyed = {r['sentence_id']: r for r in rows}
            # decided_at 由存储层盖章，跟 Postgres 那边的 now() 对齐 ——
            # 两个存储写出来的 arbitration.csv 必须能互换，否则合并时对不上。
            row = {f: record.get(f, '') for f in ARB_FIELDS}
            row['decided_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
            keyed[record['sentence_id']] = row
            atomic_csv(self.directory / 'arbitration.csv', ARB_FIELDS, keyed.values())

    def delete_final(self, sentence_id):
        """撤销一条仲裁结论，让这句重新回到待裁定清单。

        剔除是个破坏性动作——点错了，那句就从金标准里消失了，而看板上
        只显示「已裁定」的计数，不会提示少了一句。所以撤销必须有入口。
        """
        with self.lock:
            rows = read_csv(self.directory / 'arbitration.csv')
            kept = [r for r in rows if r['sentence_id'] != sentence_id]
            if len(kept) != len(rows):
                atomic_csv(self.directory / 'arbitration.csv', ARB_FIELDS, kept)
            return len(rows) - len(kept)

    def finals(self):
        with self.lock:
            return {r['sentence_id']: r
                    for r in read_csv(self.directory / 'arbitration.csv')}

    def count_sentences(self):
        with self.lock:
            return len(read_csv(self.directory / 'sentences.csv'))

    def all_annotations(self):
        with self.lock:
            return read_csv(self.directory / 'annotations.csv')

    def summary(self):
        return summarize(self.all_annotations())


class PostgresStore:
    def __init__(self, dsn):
        if not dsn:
            raise ValueError('Set database_url in Streamlit secrets, or STORAGE=local')
        from psycopg.conninfo import conninfo_to_dict
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        # Supabase 的 6543 是 transaction 模式的 pgbouncer：连接在每个事务后
        # 就还给别人，服务端 prepared statement 活不过一个事务。psycopg3 默认
        # 执行 5 次后自动 PREPARE，在这个模式下会报 "prepared statement
        # already exists"。所以 6543 上必须关掉。
        #
        # 注意这个开关只能在代码里设：prepare_threshold 是 psycopg 的连接对象
        # 属性，不是 libpq 参数。写进连接串会在解析阶段就抛
        # "invalid URI query parameter"，连都连不上，而外层只看得到 PoolTimeout。
        port = conninfo_to_dict(dsn).get('port')
        transaction_pooler = str(port) == '6543'

        def configure(conn):
            if transaction_pooler:
                conn.prepare_threshold = None      # None = 永不 PREPARE

        self.pool = ConnectionPool(
            dsn, min_size=0, max_size=4, timeout=15, configure=configure,
            kwargs={'row_factory': dict_row, 'connect_timeout': 10})

    def initialize(self, sentences, annotators):
        with self.pool.connection() as conn:
            conn.execute(SCHEMA)
            existing = conn.execute('SELECT id,text_en,seq FROM sentences ORDER BY seq').fetchall()
            check_append_only(existing, sentences)
            for user in annotators:
                conn.execute('INSERT INTO annotators VALUES (%(username)s,%(display_name)s,%(pw_hash)s) '
                             'ON CONFLICT(username) DO UPDATE SET display_name=EXCLUDED.display_name,pw_hash=EXCLUDED.pw_hash', user)
            for row in sentences:
                conn.execute('INSERT INTO sentences VALUES (%(id)s,%(text_en)s,%(text_zh)s,%(risk_flag)s,%(seq)s) '
                             'ON CONFLICT(id) DO UPDATE SET text_zh=EXCLUDED.text_zh,risk_flag=EXCLUDED.risk_flag', row)

    def sentences(self):
        with self.pool.connection() as conn:
            return conn.execute('SELECT id,text_en,text_zh,risk_flag,seq FROM sentences ORDER BY seq').fetchall()

    def count_sentences(self):
        """只要个数就别把全表拉回来。

        看板每 15 秒刷新一次，原来为了 len() 把 1000 句的原文和译文
        （约 173 KB）整个搬一遍 —— 一个标签页开一天就是几 MB 的无谓流量，
        每次刷新还要等这趟往返。
        """
        with self.pool.connection() as conn:
            return conn.execute('SELECT count(*) AS n FROM sentences').fetchone()['n']

    def own(self, username):
        if username not in USERS:
            raise ValueError('Unknown annotator')
        with self.pool.connection() as conn:
            return conn.execute('SELECT * FROM annotations WHERE username=%s', (username,)).fetchall()

    def upsert(self, record):
        validate(record)
        with self.pool.connection() as conn:
            conn.execute('''INSERT INTO annotations VALUES
                (%(username)s,%(sentence_id)s,%(subjectivity)s,%(polarity)s,%(aspect)s,%(note)s,%(updated_at)s)
                ON CONFLICT(username,sentence_id) DO UPDATE SET
                subjectivity=EXCLUDED.subjectivity,polarity=EXCLUDED.polarity,
                aspect=EXCLUDED.aspect,note=EXCLUDED.note,updated_at=EXCLUDED.updated_at
                WHERE annotations.updated_at <= EXCLUDED.updated_at''', record)

    def delete(self, username, sentence_id):
        """清掉某人某句的作答（界面上按 C）。WHERE 带 username，删不到别人的。"""
        if username not in USERS:
            raise ValueError('Unknown annotator')
        with self.pool.connection() as conn:
            cur = conn.execute('DELETE FROM annotations WHERE username=%s AND sentence_id=%s',
                               (username, sentence_id))
            return cur.rowcount

    def save_final(self, record):
        if record.get('action') not in ('adopt', 'manual', 'drop'):
            raise ValueError('action must be adopt/manual/drop')
        with self.pool.connection() as conn:
            conn.execute('''INSERT INTO arbitration
                (sentence_id,action,subjectivity,polarity,aspect,note,decided_by,decided_at)
                VALUES (%(sentence_id)s,%(action)s,%(subjectivity)s,%(polarity)s,
                        %(aspect)s,%(note)s,%(decided_by)s,now())
                ON CONFLICT(sentence_id) DO UPDATE SET
                action=EXCLUDED.action, subjectivity=EXCLUDED.subjectivity,
                polarity=EXCLUDED.polarity, aspect=EXCLUDED.aspect,
                note=EXCLUDED.note, decided_by=EXCLUDED.decided_by,
                decided_at=now()''', record)

    def delete_final(self, sentence_id):
        with self.pool.connection() as conn:
            cur = conn.execute('DELETE FROM arbitration WHERE sentence_id=%s',
                               (sentence_id,))
            return cur.rowcount

    def finals(self):
        with self.pool.connection() as conn:
            return {r['sentence_id']: r
                    for r in conn.execute('SELECT * FROM arbitration').fetchall()}

    def stats(self):
        with self.pool.connection() as conn:
            n = conn.execute('SELECT count(*) AS n FROM annotations').fetchone()['n']
            host = conn.execute('SELECT inet_server_addr() AS h').fetchone()
        return {'kind': '共享数据库 (Postgres)',
                'where': str(host.get('h') or 'Supabase'), 'rows': n, 'shared': True}

    def all_annotations(self):
        with self.pool.connection() as conn:
            return conn.execute('SELECT * FROM annotations ORDER BY sentence_id,username').fetchall()

    def summary(self):
        # Only aggregates leave this method, never another annotator's sentence answers.
        with self.pool.connection() as conn:
            rows = conn.execute('''SELECT username,count(*) AS count,
                count(*) FILTER(WHERE subjectivity='objective') AS objective,
                count(*) FILTER(WHERE subjectivity='unclear') AS unclear,
                count(*) FILTER(WHERE polarity='positive') AS positive,
                count(*) FILTER(WHERE polarity='negative') AS negative,
                array_agg(updated_at ORDER BY updated_at) AS times
                FROM annotations GROUP BY username''').fetchall()
        result = summarize([])
        for r in rows:
            result[r['username']] = {'count': r['count'], 'seconds': speed(r['times']),
                'labels': {k: r[k] for k in ('objective','positive','negative','unclear')}}
        return result


def make_store(config):
    if config['storage'] == 'local':
        return LocalStore(config['local_dir'])
    if config['storage'] == 'supabase':
        return PostgresStore(config['dsn'])
    raise ValueError('STORAGE must be local or supabase')
