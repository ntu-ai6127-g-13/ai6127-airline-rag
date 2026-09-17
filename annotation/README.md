> ## ⚠️ 这是快照，不是开发目录
>
> 标注系统的**正式仓库**在 <https://github.com/ntu-ai6127-g-13/airlabel>，
> 部署在 Streamlit Cloud，三位标注员直接用网页版。
>
> 这份副本放在主仓库里**只为存档完整性**（作业交付时能看到标注工具的全貌），
> **不要在这里改代码**——改了不会同步到线上，还会和正式仓库分叉。
>
> 更新方式：在 airlabel 仓库改完推送后，从那边重新生成这份快照。
>
> 也**不要指望照着下面的说明在这里跑起来**：`.streamlit/secrets.toml`
> （数据库连接串、口令、账号哈希）按设计不入库，这里没有。

---

# AirLabel · 双语标注系统

AI6127 组 13 的评测集标注工具。每位标注员各标 1,000 句，产出 `eval.xls` 的金标准。

部署在 Streamlit Community Cloud，组员打开网址就能标，不用装任何东西。

---

## 页面

| 页面 | 谁用 | 说明 |
|---|---|---|
| 标注页 | 标注员 | 两步问答 + 快捷键，答完自动保存下一句 |
| 进度看板 | 所有人 | 三人进度、速度、预计完成日，每 15 秒刷新 |
| 登录 | 标注员 | 三个账号，bcrypt 口令 |
| 仲裁 | 统筹 | 分歧句逐条定夺，直接产出金标准 |
| 总览 | 统筹 | 1,000 句全景表，三人答案并排 |

**仲裁和总览会显示每位标注员各自的答案**，所以另有一道口令闸（`reveal_password`）。
标注期间被人看到别人的答案，一致率就失去意义了——那是作业的硬指标（≥80%）。

## 快捷键

`1/2/3` 回答第一问 · `1–8` 选方面 · `A` / `D` 上下句 · `C` 清除本题
`T` 中文 · `N` 备注 · `Esc` 退出输入 · `?` 规范速查

---

## 部署（统筹做一次）

### 1. 建 Supabase 库

<https://supabase.com> 新建项目 → 顶部 **Connect** → **Transaction pooler**（6543）。
形如：

```
postgresql://postgres.<项目ref>:<PASSWORD>@aws-0-<region>.pooler.supabase.com:6543/postgres?sslmode=require
```

> **两个端口都能用，代码会自动适配**（`db.py` 检测到 6543 就把
> `conn.prepare_threshold` 设成 `None`）。psycopg3 默认执行 5 次后自动
> PREPARE，而 transaction 模式下连接每个事务后就还给别人，服务端的
> prepared statement 活不过一个事务 —— 不关掉会报
> `prepared statement already exists`。
>
> ⚠️ **不要把 `prepare_threshold` 写进连接串**。它不是 libpq 参数，
> psycopg 在解析阶段就抛 `invalid URI query parameter`，连都连不上，
> 而 ConnectionPool 会把这个异常吞掉，外层只看得到 `PoolTimeout`
> —— 看起来像网络不通，实际是参数非法。这个坑踩过一次。
>
> ⚠️ 密码里的特殊字符要 URL 编码：`@` 写成 `%40`、`#` 写成 `%23`。
> psycopg 从**第一个** `@` 切开连接串，不转义会把主机名截断。

### 2. 建表 + 导语料 + 建账号

```bash
pip install -r requirements.txt
export DATABASE_URL='上一步那串'
python scripts/init_annot_db.py --storage supabase --generate-passwords \
    --annotators "sunsijie=孙思洁"
```

会打印初始密码和对应的 bcrypt 哈希。**密码单独私发给本人，不要发群里**。

`--annotators` 可以一次写多个：`"a=甲,b=乙,c=丙"`。人没定齐也能先上线，
定一个加一个（见下面「加标注员」）。

### 3. 授权私有仓库，再建 app

仓库是 private，**默认授权范围不含私有仓库**，不先做这步的话建 app 时
在仓库下拉框里根本看不到它（很容易误以为是仓库名填错了）：

> Streamlit Cloud → Settings → **Linked accounts** → Source control →
> **Connect here** → 再点一次 *Authorize streamlit*

另外 Streamlit 要求你对该仓库有 **admin** 权限（组织 owner 满足）。

然后 Create app：

- Repository：`ntu-ai6127-g-13/airlabel`
- Branch：`main`
- **Main file path：`app/annotate.py`**
- Python version：3.11 或更高

> 仓库**不要**改成 public。作业规定与任何现有项目重合超过 30% 直接零分，
> 公开等于把 1,000 句评测语料和整套规范挂到互联网上。私有仓库本来就能部署。

### 4. 填 Secrets

App → Settings → Secrets，粘贴（照着 `.streamlit/secrets.example.toml` 填）：

```toml
storage = "supabase"
database_url = "postgresql://...:6543/postgres?sslmode=require"
deadline = "2026-10-25"
reveal_password = "仲裁/总览的口令"

[annotators.sunsijie]
display_name = "孙思洁"
pw_hash = "第 2 步打印的哈希"
```

**名册就是这里的 `[annotators.*]`**，不在代码里。人数、显示名、页面文案、
进度总量、导出的列数全都跟着它走。

### 5. 限制访问

App → Settings → **Sharing** → *Who can view this app* →
**Only specific people can view this app**，把三位标注员的邮箱加进去。

> ⚠️ 这跟仓库可见性是**两个独立的开关**。仓库 private 不代表部署出来的网页
> 私有——"Who can view" 默认是 *public and searchable*，网页照样会被搜索引擎
> 收录。两个都要设对。
>
> 两个已知限制：Community Cloud **同一时间只允许一个私有 app**（从私有仓库
> 部署就占掉这个名额）；被邀请的人**自己还能再邀请别人**，所以这不是严格的
> 访问控制，只适合团队内部。

---

## 加标注员（另外两位定下来之后）

```bash
python scripts/add_annotator.py zhujindou 朱津豆
```

打印出密码和一段 TOML。把那段贴进 **Streamlit Cloud → Settings → Secrets**，
保存后应用会自动重启，名册立刻生效——**不用改代码、不用重新部署**。

然后把账号写进数据库，否则登录查不到：

```bash
DATABASE_URL='...' python scripts/init_annot_db.py --storage supabase
```

> 已经标过的数据不受影响：记录以 `(username, sentence_id)` 为主键，
> 加人是纯追加。反过来**不要从名册里删人**——删了他的标注就读不出来了。

---

## 本地跑（开发用）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.example.toml .streamlit/secrets.toml   # 改成 storage = "local"
STORAGE=local python scripts/init_annot_db.py --storage local \
    --generate-passwords --annotators "sunsijie=孙思洁"
STORAGE=local streamlit run app/annotate.py
```

造测试数据（只在 `STORAGE=local` 下允许，防止污染共享库）：

```bash
STORAGE=local python scripts/seed_test_data.py --n 50    # 50 题 × 全员，含 10 种分歧
STORAGE=local python scripts/seed_test_data.py --clear
```

跑测试：`python -m pytest tests/ -q`

## 备份（每天一次）

**Supabase 免费版不含自动备份，也没有时间点恢复。** 数据没了只能重标，
而那是 3 人 × 3–4 小时的人工。

```bash
export DATABASE_URL='...'
python scripts/backup.py
```

产出 `backups/<日期-时间>/`，三个文件：

| 文件 | 内容 |
|---|---|
| `annotations.csv` | 三人原始标注，一行一条 |
| `arbitration.csv` | 仲裁结论 |
| `annotations_wide.csv` | 一句一行、三人并排，带 `agree` 列 |

> 看板上那个「下载 annotations_raw.csv」**只有标注，不含仲裁结论**。
> 仲裁是全组碰头讨论的产物，过两周没人说得清 —— 所以备份要用这个脚本，
> 不要只导看板那一份。
>
> 语料和账号不用备份：语料在 `data/eval/`，账号在 secrets 里。

`backups/` 已在 `.gitignore` 里 —— 备份是给「云端库出事」兜底的，
跟着仓库走就失去意义了。存本机或网盘。

## 收尾

标完之后在进度看板导出，或者：

```bash
python scripts/merge_annotations.py --dir 收上来的/
```

产出一致率报告（含分维度一致率和 Cohen's Kappa）、宽表、分歧清单。
**报告里要写的是仲裁前的原始一致率**——仲裁后必然 100%，那个数字没有意义。

## 存储

`storage = "local"` 走本机 CSV（`outputs/annotation-local/`），
`storage = "supabase"` 走共享 Postgres。两者接口相同，随时可切。
云端必须用 supabase——Streamlit Cloud 的容器会重启，本地文件不保留。