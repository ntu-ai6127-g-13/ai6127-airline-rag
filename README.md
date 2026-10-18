# AI6127 小组作业 — 航空服务评价检索系统

一个基于 RAG（检索增强生成）的问答系统：用 6.1 万条真实航司评论，
回答「新加坡飞伦敦经济舱，哪家航司腿部空间评价最好？」这类**带约束的对比型问题**，
并给出**可追溯到原文的结论**。

已有知识库和检索索引后，可运行 `.venv/bin/streamlit run app/streamlit_app.py`
打开本地证据查询界面，按航司、舱位、方面、航线与日期筛选，并对照 2–3 家航司的原文。
界面也接受中文或英文自然语言问题，并用本地 Qwen3 4B 根据检索原文生成带编号引用的回答。
模型在独立 MLX 环境中常驻；若模型服务不可用或回答未通过引用校验，系统会自动返回程序生成的有据摘要。
查询质量目标、可选本地分类器、无 gold 选参监控及评测命令见
[查询质量实验说明](experiments/query_quality/README.md)。实验分类器尚需独立查询验收；本机存在完整标签索引时，
标签辅助排序默认开启且只调整顺序，不按预测标签删除评论。

### 启动自然语言问答

Apple Silicon 本机已经按下列流程准备好 Qwen 权重时，分别在两个终端运行：

```bash
# 终端 1：模型只加载一次，默认仅监听本机 127.0.0.1:8765
outputs/query_quality_runtime/venv311/bin/python scripts/local_qa_server.py

# 终端 2：检索与问答页面；异常退出时会自动重启
.venv/bin/python scripts/run_query_app.py --port 8517
```

首次准备模型环境与固定版本权重时运行：

```bash
.venv/bin/python scripts/setup_query_model.py
```

页面支持“新航经济舱的座位怎么样？”和“比较新航与卡塔尔航空商务舱的餐饮评价”等问题。
自然语言只会被解析成语料中已有的航司、舱位、航线和评价方面；检索计数由程序计算，
生成模型不能修改筛选条件、证据编号或全量匹配数。评论数据截至 2019 年，回答不代表当前服务。
配对结果的人工评审界面可用 `.venv/bin/streamlit run app/query_review.py --server.port 8518` 打开；未评分的结果不会通过质量验收。

2026-10-19 定稿的 1,000 句 gold 已接入复评。当前代码可重建 C 组弱监督权重；
正式标签复评、训练监控、批量导出联调和后续验收要求见
[定稿 gold 后续推进](docs/定稿gold后续推进.md)。分类质量尚未全部达标；529,031 条基础标签索引可由固定命令重建，
本地生成产物不进入 Git。

> 📘 **新加入的同学从这里开始**：先读 [`docs/项目执行手册.md`](docs/项目执行手册.md)。
> 里面每个技术名词都有解释，每一步都有可直接运行的代码。

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/ntu-ai6127-g-13/ai6127-airline-rag.git
cd ai6127-airline-rag

# 2. ⚠️ 配置 git 身份（第一次用 git 的话必做）
#    邮箱必须是你 GitHub 账号里已验证的那个，否则提交不会算在你名下
git config --global user.name "你的GitHub用户名"
git config --global user.email "绑定GitHub账号的邮箱"

# 3. 建独立的 Python 环境（需要 Python ≥ 3.10）
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 4. 装依赖（约 5–10 分钟）
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 5. 验证环境
python scripts/test_env.py
```

看到 `全部就绪 ✅` 就说明装好了。自检会验证四件事：依赖版本、spaCy 切句、
BM25 关键词检索、FAISS 向量检索。

> ✅ **这套流程 2026-09-08 在 macOS (Apple Silicon) + Python 3.11 上完整跑通过**，
> 依赖无冲突。实测的版本组合记在 `requirements.txt` 末尾。

### 先把数据"看一眼"

跑完 `python -m src.build_kb` 生成 `chunks.parquet` 之后，用这个界面直观地看表：

```bash
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
python -m src.build_kb                     # 先建全量（约 14 分钟）
python scripts/make_sample.py              # 可选：抽个小样本给开发用（几秒）
streamlit run app/data_explorer.py
```

> ⚠️ **必须先激活 venv**。直接敲 `streamlit` 很可能用到系统里另一个 streamlit
> （macOS 上常见的是 Homebrew 装的 `/opt/homebrew/bin/streamlit`），
> 那个环境里没有项目依赖，会报 `ModuleNotFoundError: No module named 'duckdb'`。
> 嫌麻烦就不激活、直接指定 venv 里的那个：
>
> ```bash
> .venv/bin/streamlit run app/data_explorer.py     # Windows: .venv\Scripts\streamlit run ...
> ```
>
> 想确认自己用对了没有：`which streamlit` 的输出里应该带 `.venv`。

浏览器会自动打开。左边选筛选条件（航司 / 舱位 / 出行类型 / 日期 / 评分 / 关键词），
右边四个标签页：

| 标签页 | 用来干什么 |
|---|---|
| 📋 表格 | 翻页看原始行，可以排序、导出当前页 CSV |
| 📖 逐条精读 | 随机抽几句，**把句子放回它所在的整条评论里**看（就是 parent-document retrieval 的效果） |
| 📊 图表 | 六个维度的平均分、航司排名、评分分布、每年评论量、舱位×维度热力图 |
| 🛠 自己写 SQL | 直接写 DuckDB SQL，附常用模板（含 C10 参考排名那条） |

> 这个界面**只读**，不参与评分，纯粹是给大家熟悉数据用的。
> 它和 `app/streamlit_app.py`（检索组负责的 RAG 查询界面）是两个东西，别改错文件。

**看数据前先展开「各列的空值情况」**：六个维度的评分很稀疏（value 覆盖最好、ground 最差），
而且约 **21.6%** 的评论没有日期。不知道这两件事很容易把统计数字算错。

### 数据

原始数据**不在 git 里**（24 MB，没必要进版本控制）。

### 下载数据（一条命令）

```bash
pip install kagglehub          # 只需装一次
python scripts/download_data.py
```

自动从 Kaggle 下载、复制到 `data/raw/`、并跑自检。
**公开数据集，不需要 Kaggle 账号或 API token。**

> 想手动下也行：https://www.kaggle.com/datasets/efehandanisman/skytrax-airline-reviews
> 下载后把 `capstone_airline_reviews3.xlsx` 放进 `data/raw/`，再跑
> `python scripts/profile_data.py` 自检。

**自检应该看到这些数字：**

| 项 | 应为 |
|---|---|
| 原始行数 | 131,895 |
| 其中空行 | 65,948（**数据集自带的，正常**） |
| 有正文的评论 | **65,947** |
| 去重后 | 61,172 |
| **最终进知识库** | **61,167 条评论 / 546,139 句** |
| **最终总词数** | **8,276,082** |
| 航司数 / 时间范围 | **81** / **2002-08 ~ 2019-06** |
| 评价维度 | **6 个**（座位、客舱服务、餐饮、娱乐、地勤、性价比——**没有 wifi**） |

> ⚠️ 网上同名的航空评论数据集有好几个，规模从 8 千到 1 亿条都有。
> **以自检数字为准**，对不上就是下错了。

开发时先用真实数据的一小块（几秒生成，全量建索引要几十分钟）：

```bash
python scripts/make_sample.py        # 从真实数据切一小块供开发用
```

---

## 项目结构

```
├── docs/                   📘 文档（先读这里）
│   ├── 分工与任务清单.md    ⚠️ 先看这个：你负责什么、交什么、什么时候交
│   ├── 协作规范.md          ⚠️ 再看这个：分支、提交、什么进 git
│   ├── 项目执行手册.md      做的时候查：每步都有可运行代码
│   ├── 实验发现记录.md      ⭐ 边做边记，报告从这里搬
│   ├── 选题与标注方案.md    为什么选这个题 + 标注规范
│   └── Assignment.pdf      作业原文
│
├── data/
│   ├── raw/                原始数据（不进 git）
│   ├── processed/          清洗、切句后的数据（不进 git）
│   └── eval/               ✅ 人工标注结果（进 git）
│
├── index/                  检索索引（不进 git，可重建）
├── models/                 训练好的模型（不进 git）
├── outputs/                中间结果（不进 git）
│
├── src/                    源码
│   ├── config.py           所有路径和参数集中在这
│   ├── build_kb.py         知识库组
│   ├── retriever.py        检索组
│   ├── rerank.py           检索组
│   ├── weak_labels.py      下游组
│   ├── classify.py         下游组
│   ├── generate.py         下游组
│   └── evaluate.py         评测
│
├── app/streamlit_app.py    RAG 查询界面（检索组）
├── app/data_explorer.py    数据浏览界面（公共工具，谁都能改）
├── scripts/                辅助脚本
├── experiments/log.md      ⭐ 每做一次实验记一行
└── report/                 报告草稿与图表
```

---

## 分工

> 📘 **要用数据先读 [`docs/数据使用指南.md`](docs/数据使用指南.md)** ——
> 怎么拿到数据、31 列是什么、六个会让你算错数字的坑、各组的常用配方。
> 五分钟看完，所有代码片段都在完整数据上实测过。

> **完整分工见 [`docs/分工与任务清单.md`](docs/分工与任务清单.md)** ——
> 每个人的任务清单、交付物、截止日、验收标准都在里面。

> 🔴 **B / C 组开工前先读 [`docs/BC组推进计划.md`](docs/BC组推进计划.md)** ——
> 开工前必做的三件事（切分支、锁版本、重建语料）、逐周排期、
> 以及三条会让评测作废的硬规矩。**不做 §0 后面全白干。**

| 小组 | 人数 | 分值 | 负责 | 主要文件 | 自己写的报告 |
|---|---|---|---|---|---|
| **A 知识库** | 1 | 20 | 清洗、切句、建库、抽样 | `src/build_kb.py`、`src/clean_text.py` | **Q1** |
| **B 检索** | **3** | 40 | 两种检索 + 融合 + 重排 + 界面 + 评测 | `src/retriever.py`、`app/` | **Q2 + Q3** |
| **C 下游** | 2 | 40 | 弱标签、分类、生成、RAG 评测 | `src/classify.py`、`src/generate.py` | **Q4 + Q5** |

**没有单独的统筹岗位**——标注协调、周会、合稿、格式检查由 B 组某人兼任（约 8–10 h）。
**也没有报告组和 PPT 组**：五个 Question 天然对应三个小组，**谁做的谁写**，
统筹只负责把三份拼成一份 PDF，不代写章节。详见
[分工文档第 9 节](docs/分工与任务清单.md)。

标注那 1,000 句由其中 3 人共同完成（各 3–4 小时），规范见
[`docs/选题与标注方案.md`](docs/选题与标注方案.md)。

---

## 关键日期

| 日期 | 事项 |
|---|---|
| **10/25** | 🔴 1,000 句标注必须完成（压缩不了的工作量） |
| **11/01** | 所有实验跑完，指标和图表齐了 |
| **11/07 23:59** | 🔴 报告交 Blackboard（**迟交每天扣 5%**） |
| **11/10** | 现场展示 |

---

## 协作约定

> **完整规则见 [`docs/协作规范.md`](docs/协作规范.md)，开工前必读。** 要点：

- **每人开自己的分支**，通过 Pull Request 合并，不要直接推 main
- **卡住超过 3 天必须在群里说**，不要自己憋着
- **每做一次实验就往 `experiments/log.md` 加一行**——报告里的表格全靠它
- 大文件（数据、索引、模型）**走网盘，不要提交进 git**
- **不要提交任何密钥**（`.env` 已在 `.gitignore` 里）
- **确认自己的提交被正确关联**：推送后去仓库的 Contributors 页面看有没有自己。
  没有的话说明 git 邮箱和 GitHub 账号对不上，按上面第 2 步重新配置
  （不确定用哪个邮箱？去 https://github.com/settings/emails 查）
