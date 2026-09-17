import csv
import io
import sys
import tempfile
import pytest
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import bcrypt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from annot.auth import issue_token, password_ok, verify_token
from annot.cache import Cache
from annot.db import LocalStore, speed, summarize
from annot.config import USERS
from annot.export import (AGREE, DIFFER, DROP, PARTIAL, UNSTARTED,
                          exports, deviation_alerts, merged_label,
                          needs_arbitration, sentence_status, triple)
from annot.config import DEFAULT_ROSTER, roster, settings
from annot.ui import backoff_seconds, md_safe


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = LocalStore(self.root/'db')
        self.sentences = [{'id':f'E{i:04}', 'text_en':f'Example {i}', 'text_zh':f'示例{i}', 'risk_flag':'', 'seq':i} for i in range(1,4)]
        self.db.initialize(self.sentences, [])

    def tearDown(self):
        self.temp.cleanup()

    def row(self, user='zhang', sid='E0001', **changes):
        result = {'username':user,'sentence_id':sid,'subjectivity':'subjective', 'polarity':'positive',
                  'aspect':'seat','note':'','updated_at':'2026-09-12T12:00:00+00:00'}
        result.update(changes)
        return result

    def test_identity_and_upsert(self):
        for user in ('zhang','guo','sun'):
            self.db.upsert(self.row(user))
        self.db.upsert(self.row(polarity='negative'))
        self.db.upsert(self.row(polarity='negative'))
        self.assertEqual(len(self.db.all_annotations()),3)
        self.assertEqual(self.db.own('zhang')[0]['polarity'],'negative')
        self.assertEqual(self.db.own('guo')[0]['polarity'],'positive')
        self.assertEqual({r['username'] for r in self.db.own('sun')},{'sun'})

    def test_invalid_combinations_never_saved(self):
        for changes in ({'subjectivity':'objective'}, {'aspect':'wifi'}, {'username':'attacker'}, {'sentence_id':'missing'}):
            with self.assertRaises(ValueError):
                self.db.upsert(self.row(**changes))
        self.assertEqual(self.db.all_annotations(),[])
        self.db.upsert(self.row(subjectivity='objective',polarity='NA'))

    def test_cache_outage_replay_and_stale_suppression(self):
        class Offline:
            def upsert(self, row):
                raise ConnectionError('offline')
        cache = Cache(self.root/'cache','zhang')
        cache.enqueue(self.row())
        self.assertEqual(cache.flush(Offline()),1)
        self.assertEqual(len(Cache(self.root/'cache','zhang').pending()),1)
        self.db.upsert(self.row(polarity='negative',updated_at='2026-09-12T13:00:00+00:00'))
        self.assertEqual(cache.flush(self.db),0)
        self.assertEqual(self.db.own('zhang')[0]['polarity'],'negative')
        self.assertEqual(cache.pending(),[])
        with self.assertRaises(ValueError):
            cache.enqueue(self.row(user='guo'))

    def test_partial_cache_flush_preserves_remaining(self):
        cache=Cache(self.root/'cache','zhang')
        cache.enqueue(self.row())
        cache.enqueue(self.row(sid='E0002'))
        target=self.db
        class Partial:
            def upsert(self,row):
                if row['sentence_id']=='E0002': raise ConnectionError()
                target.upsert(row)
        self.assertEqual(cache.flush(Partial()),1)
        self.assertEqual(cache.pending()[0]['sentence_id'],'E0002')
        self.assertEqual(cache.flush(self.db),0)
        self.assertEqual(len(self.db.own('zhang')),2)

    def test_concurrent_csv_writes_no_lost_users(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda u:self.db.upsert(self.row(user=u)), ['zhang','guo','sun']))
        self.assertEqual(len(self.db.all_annotations()),3)

    def test_export_alignment_and_na(self):
        self.db.upsert(self.row(subjectivity='objective',polarity='NA',note='comma, quote"\n新行'))
        self.db.upsert(self.row(user='sun',polarity='negative'))
        raw,wide=exports(self.sentences,self.db.all_annotations())
        rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        self.assertEqual(rows[0]['polarity'],'NA')
        self.assertEqual(rows[0]['note'],'comma, quote"\n新行')
        rows=list(csv.DictReader(io.StringIO(wide.decode('utf-8-sig'))))
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[0]['polarity_a'],'NA')
        self.assertEqual(rows[0]['polarity_b'],'')
        self.assertEqual(rows[0]['polarity_c'],'negative')

    def test_no_corpus_replacement(self):
        changed=[dict(s) for s in self.sentences]
        changed[0]['text_en']='Different sentence'
        with self.assertRaises(ValueError): self.db.initialize(changed,[])

    def test_corpus_append_is_allowed(self):
        # 仲裁剔除句子之后要从预留池补足，那是一次追加，必须放行。
        extra=self.sentences+[{'id':'E1001','text_en':'Reserve sentence','text_zh':'预留句',
                               'risk_flag':'','seq':4}]
        self.db.initialize(extra,[])
        self.assertEqual([s['id'] for s in self.db.sentences()],
                         ['E0001','E0002','E0003','E1001'])

    def test_append_cannot_disturb_existing(self):
        # 追加的同时改动已有句子 —— 已标的答案会悄悄变成对另一句话的判断。
        moved=[dict(s) for s in self.sentences]
        moved[1]['text_en']='Silently swapped'
        moved.append({'id':'E1001','text_en':'Reserve','text_zh':'','risk_flag':'','seq':4})
        with self.assertRaises(ValueError): self.db.initialize(moved,[])

    def test_corpus_cannot_shrink(self):
        with self.assertRaises(ValueError): self.db.initialize(self.sentences[:2],[])

    def test_speed_and_alert_thresholds(self):
        self.assertEqual(speed(['2026-09-12T12:00:00+00:00','2026-09-12T12:00:14+00:00','2026-09-12T13:00:00+00:00']),14)
        summary={u:{'count':100,'labels':{'objective':20,'positive':40,'negative':40}} for u in ('zhang','guo','sun')}
        summary['zhang']['labels']={'objective':50,'positive':25,'negative':25}
        self.assertTrue(any(a[0]=='zhang' and a[1]=='objective' for a in deviation_alerts(summary)))
        summary['zhang']['count']=99
        self.assertFalse(any(a[0]=='zhang' for a in deviation_alerts(summary)))

    def test_password_and_refresh_token(self):
        users={'zhang':{'pw_hash':bcrypt.hashpw(b'valid-password',bcrypt.gensalt(rounds=4)).decode()}}
        self.assertTrue(password_ok('zhang','valid-password',users))
        self.assertFalse(password_ok('zhang','wrong',users))
        self.assertFalse(password_ok('zhang','a'*100,users))
        token=issue_token('zhang',users)
        self.assertEqual(verify_token(token,users),'zhang')
        self.assertIsNone(verify_token(token+'x',users))
        self.assertIsNone(verify_token('garbage',users))


if __name__=='__main__':
    unittest.main()


# ── 「标不了」(unclear, 规范 R11) ──────────────────────────────────
# 加这一档是因为规范里有 R11「无法判断 → 标 unclear」，但原来的
# CHECK 约束和 validate() 都不允许，标注员碰到 "I'm still wearing it
# at home." 只能硬猜 —— 这正是拉低一致率的典型来源。

def _rec(**kw):
    base = dict(username='zhang', sentence_id='E0001', subjectivity='unclear',
                polarity='NA', aspect='unclear', note='',
                updated_at='2026-09-14T10:00:00+00:00')
    base.update(kw)
    return base


def test_unclear_accepted():
    from app.annot.db import validate
    validate(_rec())


def test_unclear_must_carry_na_and_unclear_aspect():
    """三个字段必须一起 unclear，不能出现「标不了但选了方面」。"""
    import pytest
    from app.annot.db import validate
    for bad in (_rec(polarity='negative'), _rec(aspect='seat'),
                _rec(polarity='positive', aspect='food')):
        with pytest.raises(ValueError):
            validate(bad)


def test_unclear_aspect_rejected_for_normal_labels():
    """反过来：正常标注不能把 aspect 写成 unclear。"""
    import pytest
    from app.annot.db import validate
    with pytest.raises(ValueError):
        validate(_rec(subjectivity='objective', polarity='NA', aspect='unclear'))


def test_summary_counts_unclear():
    from app.annot.db import summarize
    rows = [_rec(), _rec(sentence_id='E0002', subjectivity='subjective',
                         polarity='negative', aspect='seat')]
    assert summarize(rows)['zhang']['labels']['unclear'] == 1


# ── C 键清除作答 ───────────────────────────────────────────────
def test_delete_removes_only_own_row(tmp_path):
    from app.annot.db import LocalStore
    store = LocalStore(tmp_path)
    store.initialize([{'id': 'E0001', 'text_en': 'a', 'text_zh': '', 'risk_flag': '', 'seq': 1}],
                     [{'username': u, 'display_name': u, 'pw_hash': 'x'}
                      for u in ('zhang', 'guo', 'sun')])
    for user in ('zhang', 'guo'):
        store.upsert(_rec(username=user, subjectivity='subjective',
                          polarity='negative', aspect='seat'))
    assert len(store.all_annotations()) == 2
    assert store.delete('zhang', 'E0001') == 1
    rest = store.all_annotations()
    assert len(rest) == 1 and rest[0]['username'] == 'guo'   # 没删到别人的


def test_delete_missing_row_is_noop(tmp_path):
    from app.annot.db import LocalStore
    store = LocalStore(tmp_path)
    store.initialize([{'id': 'E0001', 'text_en': 'a', 'text_zh': '', 'risk_flag': '', 'seq': 1}],
                     [{'username': 'zhang', 'display_name': 'z', 'pw_hash': 'x'}])
    assert store.delete('zhang', 'E0001') == 0


def test_cache_drop_prevents_resurrection(tmp_path):
    """清除后队列里若还留着同一句，下次 flush 会把它写回去。"""
    from app.annot.cache import Cache
    cache = Cache(tmp_path, 'zhang')
    cache.enqueue(_rec(subjectivity='subjective', polarity='negative', aspect='seat'))
    cache.enqueue(_rec(sentence_id='E0002', subjectivity='objective',
                       polarity='NA', aspect='other'))
    assert cache.drop('E0001') == 1
    assert [r['sentence_id'] for r in cache.pending()] == ['E0002']


# ── 看板 / 导出对 unclear 的兼容 ────────────────────────────────
def test_deviation_alerts_tolerates_missing_label_key():
    """某人一条 unclear 都没标时，Postgres 聚合不会给这个键。
    直接下标会 KeyError 把整个看板打挂。"""
    from app.annot.export import deviation_alerts
    summary = {
        'zhang': {'count': 200, 'labels': {'objective': 40, 'positive': 80,
                                           'negative': 80, 'unclear': 0}},
        'guo':   {'count': 200, 'labels': {'objective': 30, 'positive': 90,
                                           'negative': 80, 'unclear': 0}},
        'sun':   {'count': 200, 'labels': {'objective': 100, 'positive': 50,
                                           'negative': 50}},          # 少 unclear
    }
    deviation_alerts(summary)          # 不抛异常即可


def test_merged_label():
    from app.annot.export import merged_label
    assert merged_label({'subjectivity': 'objective', 'polarity': 'NA'}) == 'objective'
    assert merged_label({'subjectivity': 'subjective', 'polarity': 'negative'}) == 'negative'
    assert merged_label({'subjectivity': 'unclear', 'polarity': 'NA'}) == 'unclear'
    assert merged_label({}) == ''


def test_wide_export_carries_unclear_and_agreement():
    from app.annot.export import exports
    sents = [{'id': 'E0001', 'text_en': 'a', 'text_zh': '甲', 'risk_flag': '', 'seq': 1}]
    ann = [dict(username=u, sentence_id='E0001', subjectivity='unclear',
                polarity='NA', aspect='unclear', note='',
                updated_at='2026-09-14T10:00:00+00:00')
           for u in ('zhang', 'guo', 'sun')]
    _, wide = exports(sents, ann)
    line = wide.decode('utf-8-sig').strip().splitlines()[-1]
    assert line.count('unclear') >= 3      # 三人的 label 列都落上了
    assert line.endswith('一致')            # 三人相同 → 标记为一致


# ── 仲裁分歧判定 ────────────────────────────────────────────
# 这一组锁的是一个真实踩过的坑：原先仲裁页只比 merged_label，
# 而 merged_label 看不见 aspect —— 造了 10 条分歧测试数据，页面只认出 7 条。

def _ans(subj, pol, asp):
    return {'subjectivity': subj, 'polarity': pol, 'aspect': asp}


def _trio(a, b, c):
    return dict(zip(USERS, (a, b, c)))


NEG_SEAT = _ans('subjective', 'negative', 'seat')
NEG_FOOD = _ans('subjective', 'negative', 'food')
POS_STAFF = _ans('subjective', 'positive', 'staff')
OBJ = _ans('objective', 'NA', 'other')
UNC = _ans('unclear', 'NA', 'unclear')


def test_needs_arbitration_catches_aspect_only_disagreement():
    """三人合并标签都是 negative，只有方面不同 —— 必须算分歧。"""
    assert needs_arbitration(_trio(NEG_SEAT, NEG_SEAT, NEG_FOOD))


def test_needs_arbitration_catches_unanimous_unclear():
    """三人全标 unclear 是「一致」，但按 R11 要剔除，所以仍须进仲裁页。"""
    assert needs_arbitration(_trio(UNC, UNC, UNC))


def test_needs_arbitration_catches_two_of_three_unclear():
    assert needs_arbitration(_trio(UNC, UNC, NEG_SEAT))


def test_needs_arbitration_ignores_true_agreement():
    assert not needs_arbitration(_trio(NEG_SEAT, NEG_SEAT, NEG_SEAT))


def test_needs_arbitration_ignores_single_unclear_with_majority():
    """只有 1 人标 unclear，走正常多数票，不触发剔除规则。"""
    assert needs_arbitration(_trio(NEG_SEAT, NEG_SEAT, UNC))   # 仍是分歧
    labels = [merged_label(r) for r in (NEG_SEAT, NEG_SEAT, UNC)]
    assert labels.count('unclear') < 2                          # 但不该剔除


def test_needs_arbitration_skips_incomplete():
    """没标满三人就下判断，会把「还没标」当成分歧。"""
    partial = dict(list(_trio(NEG_SEAT, POS_STAFF, OBJ).items())[:2])
    assert not needs_arbitration(partial)
    assert not needs_arbitration({})


def test_export_agree_column_sees_aspect():
    """导出的 agree 列跟仲裁页用同一套判据，不能只看合并标签。"""
    sentences = [{'id': 'E1', 'text_en': 'x', 'text_zh': 'x'},
                 {'id': 'E2', 'text_en': 'y', 'text_zh': 'y'}]
    ann = []
    for sid, trio in (('E1', (NEG_SEAT, NEG_SEAT, NEG_FOOD)),
                      ('E2', (NEG_SEAT, NEG_SEAT, NEG_SEAT))):
        for user, r in zip(USERS, trio):
            ann.append({**r, 'sentence_id': sid, 'username': user,
                        'note': '', 'updated_at': ''})
    _, wide = exports(sentences, ann)
    body = wide.decode('utf-8-sig').strip().splitlines()[1:]
    assert body[0].rstrip().endswith('分歧')     # 只有方面不同
    assert body[1].rstrip().endswith('一致')


# ── Markdown 转义 ───────────────────────────────────────────

def test_md_safe_protects_dollar_amounts():
    """成对的 $ 会被 Streamlit 当 LaTeX 公式，金额直接消失。

    航司评论里 18/1000 句含 $，而 value（性价比）的判断依据就是金额。
    """
    out = md_safe('extra costs ($3 for a seat, $25 for a bag)')
    assert out == r'extra costs (\$3 for a seat, \$25 for a bag)'
    assert md_safe('50% *off* _now_') == r'50% \*off\* \_now\_'
    assert md_safe('') == '' and md_safe(None) == ''


# ── 仲裁落库 ────────────────────────────────────────────────

def test_save_final_stamps_decided_at(tmp_path):
    """本地存储要跟 Postgres 的 now() 对齐，否则两边导出的 CSV 合不到一起。"""
    store = LocalStore(tmp_path)
    store.save_final(dict(sentence_id='E1', action='adopt', decided_by='group',
                          subjectivity='subjective', polarity='negative',
                          aspect='seat', note=''))
    row = store.finals()['E1']
    assert row['decided_at'], 'decided_at 不能为空'
    datetime.fromisoformat(row['decided_at'])          # 可解析
    assert row['action'] == 'adopt'


def test_save_final_overwrites_not_appends(tmp_path):
    store = LocalStore(tmp_path)
    base = dict(sentence_id='E1', decided_by='group', subjectivity='',
                polarity='', aspect='', note='')
    store.save_final({**base, 'action': 'adopt'})
    store.save_final({**base, 'action': 'drop', 'note': '改主意'})
    finals = store.finals()
    assert len(finals) == 1 and finals['E1']['action'] == 'drop'


# ── 状态判定与仲裁清单的一致性 ──────────────────────────────
# 总览页和仲裁页是两个入口，但必须对同一句给出相容的结论。
# 让 needs_arbitration 从 sentence_status 派生就是为了这个，这几条锁住它。

def test_sentence_status_covers_every_case():
    assert sentence_status({}) == UNSTARTED
    two = dict(list(_trio(NEG_SEAT, NEG_SEAT, NEG_SEAT).items())[:2])
    assert sentence_status(two) == PARTIAL
    assert sentence_status(_trio(NEG_SEAT, NEG_SEAT, NEG_SEAT)) == AGREE
    assert sentence_status(_trio(NEG_SEAT, NEG_SEAT, NEG_FOOD)) == DIFFER
    assert sentence_status(_trio(UNC, UNC, UNC)) == DROP
    assert sentence_status(_trio(UNC, UNC, NEG_SEAT)) == DROP


def test_drop_wins_over_differ():
    """≥2 人 unclear 时三元组也不同，但该做的动作是剔除，不是调解分歧。"""
    got = _trio(UNC, UNC, NEG_SEAT)
    assert len({triple(got[u]) for u in USERS}) > 1     # 确实是分歧
    assert sentence_status(got) == DROP                  # 但状态是待剔除


def test_needs_arbitration_matches_status():
    """两个入口不能对同一句给出相反结论。"""
    cases = [_trio(NEG_SEAT, NEG_SEAT, NEG_SEAT), _trio(NEG_SEAT, NEG_SEAT, NEG_FOOD),
             _trio(UNC, UNC, UNC), _trio(UNC, UNC, NEG_SEAT),
             _trio(NEG_SEAT, POS_STAFF, OBJ), {},
             dict(list(_trio(NEG_SEAT, NEG_SEAT, NEG_SEAT).items())[:2])]
    for got in cases:
        assert needs_arbitration(got) == (sentence_status(got) in (DIFFER, DROP))


# ── 撤销裁定 ────────────────────────────────────────────────

def test_delete_final_restores_to_pending(tmp_path):
    """剔除点错了要能撤销——否则那句从金标准消失，看板也不会提示。"""
    store = LocalStore(tmp_path)
    store.save_final(dict(sentence_id='E1', action='drop', decided_by='group',
                          subjectivity='', polarity='', aspect='', note=''))
    assert 'E1' in store.finals()
    assert store.delete_final('E1') == 1
    assert store.finals() == {}


def test_delete_final_is_idempotent_and_targeted(tmp_path):
    store = LocalStore(tmp_path)
    for sid in ('E1', 'E2'):
        store.save_final(dict(sentence_id=sid, action='drop', decided_by='group',
                              subjectivity='', polarity='', aspect='', note=''))
    assert store.delete_final('E1') == 1
    assert store.delete_final('E1') == 0          # 再删一次不报错
    assert list(store.finals()) == ['E2']         # 没误伤别的句子


def test_delete_final_does_not_touch_annotations(tmp_path):
    """撤销裁定只回滚结论，三个人的原始标注一条都不能少。"""
    store = LocalStore(tmp_path)
    store.initialize([{'id': 'E0001', 'text_en': 'x', 'text_zh': 'x',
                       'risk_flag': '', 'seq': 1}], [])
    for user in USERS:
        store.upsert(dict(username=user, sentence_id='E0001', subjectivity='subjective',
                          polarity='negative', aspect='seat', note='',
                          updated_at=datetime.now(timezone.utc).isoformat()))
    store.save_final(dict(sentence_id='E0001', action='drop', decided_by='group',
                          subjectivity='', polarity='', aspect='', note=''))
    store.delete_final('E0001')
    assert len(store.all_annotations()) == len(USERS)


# ── 仲裁 / 总览的口令闸 ─────────────────────────────────────

def test_backoff_escalates_and_caps():
    """固定 2 秒对 6 位数字没意义；递增之后「随手多试几下」这条路才堵得住。"""
    assert backoff_seconds(0) == 0
    assert [backoff_seconds(i) for i in range(1, 6)] == [2, 4, 8, 16, 32]
    assert backoff_seconds(20) == 60          # 封顶，不会等到天荒地老
    assert backoff_seconds(-1) == 0           # 不会因为脏数据炸掉


def test_reveal_password_read_from_secrets():
    assert settings({'reveal_password': 'dummy-123'})['reveal_password'] == 'dummy-123'


def test_reveal_password_absent_means_locked():
    """没配口令要锁死，不能放行。

    口令存在 gitignored 的 secrets.toml 里，不随代码分发。组员克隆下来的副本
    没有这个键——这时如果 fallback 成放行，这道闸等于从来没存在过。
    """
    assert settings({})['reveal_password'] == ''


def test_reveal_password_never_hardcoded_in_source():
    """口令不能出现在任何会进仓库的文件里。"""
    root = Path(__file__).resolve().parents[1]
    # tests/ 也要扫 —— 测试文件同样会进仓库，用真口令当测试夹具一样是泄露。
    tracked = [p for p in (list((root / 'app').rglob('*.py'))
                           + list((root / 'scripts').rglob('*.py'))
                           + list((root / 'tests').rglob('*.py'))
                           + list(root.glob('*.md'))
                           + list((root / 'docs').rglob('*.md'))
                           + [root / '.streamlit/secrets.example.toml'])
               if p.exists()]
    secret = settings()['reveal_password']
    if secret:
        for path in tracked:
            assert secret not in path.read_text(encoding='utf-8'), \
                f'口令硬编码在 {path.relative_to(root)} 里了'


# ── 名册从 secrets 读 ────────────────────────────────────────
# 「谁来标」到临门一脚还在变（现在只定了一个人）。名册写死在代码里的话，
# 每定一个人就要改代码重新部署一次。

def test_roster_reads_from_secrets():
    got = roster({'annotators': {
        'sunsijie': {'display_name': '孙思洁', 'pw_hash': 'x'},
        'zhu': {'display_name': '朱津豆', 'pw_hash': 'y'}}})
    assert got == {'sunsijie': '孙思洁', 'zhu': '朱津豆'}


def test_roster_preserves_order():
    """顺序决定导出宽表的 a/b/c 列顺序，不能被 dict 重排。"""
    people = {k: {'display_name': k.upper(), 'pw_hash': 'x'} for k in 'cab'}
    assert list(roster({'annotators': people})) == ['c', 'a', 'b']


def test_roster_falls_back_when_unset():
    assert roster({}) == DEFAULT_ROSTER
    assert roster({'annotators': {}}) == DEFAULT_ROSTER


def test_roster_tolerates_missing_display_name():
    """漏填 display_name 不能让整个应用起不来，退回用登录名显示。"""
    assert roster({'annotators': {'zhu': {'pw_hash': 'y'}}}) == {'zhu': 'zhu'}


def test_export_columns_track_roster_size():
    """列后缀按人数生成。写死 abc 的话，第 4 个人的标注会被 zip 静默吃掉。"""
    import annot.config as cfgmod
    original = dict(cfgmod.USERS)
    try:
        for n, expected in ((1, ['a']), (2, ['a', 'b']), (4, ['a', 'b', 'c', 'd'])):
            cfgmod.USERS.clear()
            cfgmod.USERS.update({f'u{i}': f'人{i}' for i in range(n)})
            _, wide = exports([{'id': 'E1', 'text_en': 'x', 'text_zh': 'x'}], [])
            header = wide.decode('utf-8-sig').splitlines()[0]
            assert [c for c in expected if f'label_{c}' in header] == expected
            assert f'label_{chr(ord("a") + n)}' not in header    # 不多不少
    finally:
        cfgmod.USERS.clear()
        cfgmod.USERS.update(original)


def test_count_sentences_matches_full_fetch(tmp_path):
    """看板每 15 秒要一次句子总数，走 count 而不是拉全表 —— 两者必须一致。"""
    store = LocalStore(tmp_path)
    corpus = [{'id': f'E{i:04}', 'text_en': f'x{i}', 'text_zh': f'中{i}',
               'risk_flag': '', 'seq': i} for i in range(1, 21)]
    store.initialize(corpus, [])
    assert store.count_sentences() == len(store.sentences()) == 20


# ── 「待定」：只提交备注，不选答案 ──────────────────────────
# 卡住时的出口。没有它，标注员只能硬猜或误用 unclear —— 后者更糟，
# ≥2 人标 unclear 会把句子从金标准剔除，删掉的正是最难的那批题。

def _pending(sid='E0001', user='zhang', note='不确定归哪类'):
    return dict(username=user, sentence_id=sid, subjectivity='', polarity='',
                aspect='', note=note,
                updated_at=datetime.now(timezone.utc).isoformat())


def _corpus(store, n=3):
    store.initialize([{'id': f'E{i:04}', 'text_en': f'x{i}', 'text_zh': f'中{i}',
                       'risk_flag': '', 'seq': i} for i in range(1, n + 1)], [])


def test_pending_record_accepted(tmp_path):
    store = LocalStore(tmp_path); _corpus(store)
    store.upsert(_pending())
    assert len(store.all_annotations()) == 1


def test_pending_requires_a_note(tmp_path):
    """空答案 + 空备注 = 什么信息都没有，不该落库。"""
    store = LocalStore(tmp_path); _corpus(store)
    for blank in ('', '   '):
        with pytest.raises(ValueError):
            store.upsert(_pending(note=blank))


def test_pending_does_not_count_as_answered(tmp_path):
    """进度必须把待定当成还没标，否则 1,000 句的完成度会虚高。"""
    store = LocalStore(tmp_path); _corpus(store)
    store.upsert(_pending())
    assert summarize(store.all_annotations())['zhang']['count'] == 0


def test_pending_ignored_by_sentence_status():
    """一致率和仲裁都不能拿空答案去比。"""
    got = {'zhang': {'subjectivity': '', 'polarity': '', 'aspect': '', 'note': 'x'}}
    assert sentence_status(got) == UNSTARTED
    two = _trio(NEG_SEAT, NEG_SEAT, NEG_SEAT)
    two['sun'] = {'subjectivity': '', 'polarity': '', 'aspect': '', 'note': 'x'}
    assert sentence_status(two) == PARTIAL          # 只剩两人有答案
    assert not needs_arbitration(two)


def test_pending_can_be_upgraded_to_a_real_answer(tmp_path):
    """想清楚之后正常标，应该覆盖掉待定记录而不是并存。"""
    store = LocalStore(tmp_path); _corpus(store)
    store.upsert(_pending())
    store.upsert(dict(username='zhang', sentence_id='E0001', subjectivity='subjective',
                      polarity='negative', aspect='seat', note='',
                      updated_at=datetime.now(timezone.utc).isoformat()))
    rows = store.all_annotations()
    assert len(rows) == 1 and rows[0]['subjectivity'] == 'subjective'
    assert summarize(rows)['zhang']['count'] == 1


def test_pending_leaves_export_label_blank(tmp_path):
    """导出时待定那一列应该是空的，且不能被算成「一致」。"""
    sentences = [{'id': 'E1', 'text_en': 'x', 'text_zh': 'x'}]
    ann = [{'username': u, 'sentence_id': 'E1', 'note': '', 'updated_at': '',
            **(NEG_SEAT if u != 'sun' else
               {'subjectivity': '', 'polarity': '', 'aspect': ''})} for u in USERS]
    _, wide = exports(sentences, ann)
    body = wide.decode('utf-8-sig').strip().splitlines()[1]
    assert not body.rstrip().endswith('一致')
    assert not body.rstrip().endswith('分歧')
