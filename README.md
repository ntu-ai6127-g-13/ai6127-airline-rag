# AI6127 小组作业 — 航空服务评价检索系统

一个基于 RAG（检索增强生成）的问答系统：用 6.1 万条真实航司评论，
回答「新加坡飞伦敦经济舱，哪家航司腿部空间评价最好？」这类**带约束的对比型问题**，
并给出**可追溯到原文的结论**。

> 📘 **新加入的同学从这里开始**：先读 [`docs/项目执行手册.md`](docs/项目执行手册.md)。
> 里面每个技术名词都有解释，每一步都有可直接运行的代码。

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/RenXinghan/ai6127-airline-rag.git
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

看到 `全部就绪 ✅` 就说明装好了。

### 数据

原始数据是 `data/raw/capstone_airline_reviews3.xlsx`（24 MB）。
**它不在 git 里**（超过 GitHub 建议的单文件大小，且没必要进版本控制）——
**问组长要网盘链接，下载后放到 `data/raw/` 目录下。**

**实测规模**（跑 `python scripts/profile_data.py` 可复现）：

| 项 | 数值 |
|---|---|
| 原始行数 | 131,895（**其中 65,948 行全空，必须先滤掉**） |
| 去重后可用评论 | **61,172** |
| 总词数 | **8,402,547** |
| 航司 / 时间范围 | 81 家 / **2002-08 ~ 2019-06** |
| 评价维度 | **6 个**（座位、客舱服务、餐饮、娱乐、地勤、性价比——**没有 wifi**） |

还没同步到数据时，先用假数据开工：

```bash
python scripts/make_fake_data.py     # 生成 100 行假数据供开发用
```

---

## 项目结构

```
├── docs/                   📘 文档（先读这里）
│   ├── 项目执行手册.md      怎么做，每步都有代码
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
├── app/streamlit_app.py    网页界面
├── scripts/                辅助脚本
├── experiments/log.md      ⭐ 每做一次实验记一行
└── report/                 报告草稿与图表
```

---

## 分工

| 小组 | 分值 | 负责 | 主要文件 |
|---|---|---|---|
| **知识库** | 20 | 下数据、清洗、切句、抽样 | `src/build_kb.py` |
| **检索** | 40 | 两种检索 + 合并 + 重排 + 界面 | `src/retriever.py`、`app/` |
| **下游任务** | 40 | 弱标签、分类、生成、评测 | `src/classify.py`、`src/generate.py` |

标注那 1,000 句由三人共同完成，规范见 `docs/选题与标注方案.md`。

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

- **每人开自己的分支**，合并前简单看一眼
- **卡住超过 3 天必须在群里说**，不要自己憋着
- **每做一次实验就往 `experiments/log.md` 加一行**——报告里的表格全靠它
- 大文件（数据、索引、模型）**走网盘，不要提交进 git**
- **不要提交任何密钥**（`.env` 已在 `.gitignore` 里）
- **确认自己的提交被正确关联**：推送后去仓库的 Contributors 页面看有没有自己。
  没有的话说明 git 邮箱和 GitHub 账号对不上，按上面第 2 步重新配置
  （不确定用哪个邮箱？去 https://github.com/settings/emails 查）
