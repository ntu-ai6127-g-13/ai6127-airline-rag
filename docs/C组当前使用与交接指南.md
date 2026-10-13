# C 组当前使用与交接指南

更新：2026-10-13。本页是本次精简提交的入口；数据、模型和旧实验继续保留在本地。

**当前可以交付的是分类程序及候选标签包，尚未完成正式最终模型冻结、人工金标准评测或 B 组实际联调。** 已有模型和结果不需要重新训练、建库或预测。

## 1. 当前版本和 C1–C5 的关系

| 部分 | 程序负责什么 | 当前状态 |
| --- | --- | --- |
| C1 弱标签 | 用句子线索和父评论评分生成训练参考标签 | 已实现六方面弱标签；默认规则为 `context_v4`，保留历史规则复现入口 |
| C2 简单基线 | 用 TF-IDF 和逻辑回归跑通分类与弱标签开发验证 | 六方面及正负情感基线已实现，指标不是金标准成绩 |
| C3 主客观分类 | 判断是在陈述事实还是表达观点 | 已实现规则与模型组合；客观训练样本仅 175 句，仍是局限 |
| C4 情感分类 | 判断主观句为正面还是负面 | 客观句的情感为字面值 `NA`，不强判正负 |
| C5 方面分类 | 判断句子谈什么 | 联合流程支持六具体方面及 `overall/other`，共八类；已有开发验证、规则回归及两策略对照 |

统一入口是 `src/classify.py`；八方面联合流程在 `src/classification_pipeline.py`。
旧 `train` 命令是六方面基线；三个分类器共用评论切分的入口是 `train-all`。
`src/weak_labels.py` 的六方面文件不是当前八方面联合训练材料，不能混用名称或结果。

当前交接使用 `eight-aspect-subjectivity-polarity-v4.1`。
它继承 v3 的 TF-IDF 词表、三个分类器权重及父评论训练/开发名单，修正了推理规则，**没有重新训练**。
历史语言、方面和主客观规则快照保留，是为了旧模型行为不随新规则变化。
中间 v4 候选已经撤回；预测入口拒绝使用它。
新 C1 `context_v4` 标签尚未重新生成并用于训练，不能声称当前模型学过它。

## 2. 训练保护和三条硬规矩

1. **训练与开发材料排除金标准。** 不仅排除 `is_eval=True` 的 1,100 句，还排除其 1,091 条父评论及规范化后重复的评测文本。
2. **不重新抽取冻结评测集。** 人工标注按固定句子编号连接，重新抽样会破坏对应关系。
3. **不重建受保护的评测文件。** 评测目录中的四份冻结文件保持原样；本次精简也不重建知识库。

`src/training_guards.py` 统一检查训练/开发的句子编号、父评论、重复文本、角色和标签来源。
有 `split` 的标签只选择 `split == "train"`；开发、未分配、屏蔽记录不能整文件送去训练。
生成和重新切分入口拒绝把已分角色的材料当原始输入；沿用名单时使用原始句子及单独的 `split-reference`。
模型预测的 `hybrid_prediction` 来源不能冒充弱标签或人工监督答案。

新增 11,535 条兄弟句来自金标准父评论，但自身不是金标准题。
它们只供推理，标为 `heldout_parent_prediction_only`，**不能用于训练、开发验证、调参或模型选择**。
这次补预测没有修改训练隔离名单，也没有预测那 1,100 条金标准句。

## 3. 两套弱标签对照保留什么

`scripts/compare_weak_label_strategies.py` 保留完整可复现流程：

- 宽松版：把六个方面的可用极端评分下放到句子，允许同句有多个标签。
- 严格版：关键词只命中一个方面，且该方面评分为 1–2 或 4–5，取宽松版的子集。
- 两边用相同父评论分配、算法和参数，在同一份 26,779 句旧弱标签开发题上比较；词表只在各自训练侧拟合。
- 同句跨方面、正负冲突和训练内部重复在该对照中按原方案保留并统计；“标签记录数”不是“独立句子数”。

| 已有对照结果 | 宽松版 | 严格版 |
| --- | ---: | ---: |
| 实际训练标签记录 | 1,528,146 | 114,869 |
| 实际训练独立句子 | 347,953 | 114,869 |
| 六方面宏 F1，目标为旧弱标签 | 0.3581 | 0.8428 |
| 正负情感宏 F1，目标为旧弱标签 | 0.8128 | 0.8255 |

结果在 `outputs/weak_strategy_comparison_v1/`，不重复运行或覆盖。
这两个诊断模型只比较六方面和情感，不能替代八方面联合候选。
旧开发题来自严格关键词弱标签且参与过规则设计，存在偏差；**0.8428 不是人工准确率**。
该对照明确使用历史 `context_v3`，没有因为 C1 默认规则变化就自动改成 `context_v4`。

## 4. 环境版本的范围

`requirements-c.lock.txt` 记录本机 C 流程的 60 个包，使用 Python 3.11.16、macOS ARM64。
关键版本包括 spaCy 3.8.16、英文切句模型 3.8.0、pandas 3.0.5、scikit-learn 1.9.0、pyarrow 25.0.1。
切句器和知识库版本会影响句子编号；B 应核对实际索引的编号，不能只比总行数。

这个 lock 覆盖分类、清洗、切句和表格读写，**不是整个 B 组检索环境的锁定文件，也不保证跨平台安装复现**。
普通包使用精确版本及公开 PyPI 目标；英文模型使用官方 wheel 地址及已安装元数据提供的 SHA-256。
普通包没有完整轮子哈希，未做重新安装验证；当前环境记录不能补造旧实验未记录的历史版本。
本机现有环境无需重新安装。本次暂不提交环境快照辅助脚本，不影响 lock 和这些边界说明。

## 5. 已有结果怎样检查和使用

以下命令在项目根目录运行；使用现有本地数据和模型，不重新建库。
两条检查不带 `--output`，只读核验、不写报告、不训练、不推理；首次克隆但没有本地数据包时不能直接完成这些检查。

```sh
.venv/bin/python -B scripts/check_c_training.py
.venv/bin/python -B scripts/check_c_b_extended.py \
  --package outputs/c_b_sync_v4_1_pr_ready
```

确需试一条自己的句子时，下面只进行单句推理，不训练，不改模型：

```sh
.venv/bin/python -B -m src.classify predict \
  --model outputs/classification_eight_rules_v4_1/model.joblib \
  --text "The food was terrible."
```

精简清单保留 15 个测试模块，另带 `tests/__init__.py`，避免其他环境的同名测试包抢占共享测试辅助代码。已在基础提交 `8ecd8d2` 加本次 40 个文件的干净副本中复测，**170 项测试全部通过**，包括历史来源与重打包保护测试。
此前 42 文件精简方案的干净导出副本有 155 项测试，已全部通过；当时完整本地版本有 157 项，额外 2 项属于暂不提交的历史上下文对照。
本地仍保留全部旧文件，因此本地全量发现测试和精简交付副本的数量不同，不是丢失了安全测试。
训练隔离、版本兼容、两策略、包读取、补预测、编号及空值边界测试都保留。
小型合成数据的测试会在临时目录拟合模型验证接口，与重训项目全量模型不同。

## 6. 给 B 组的当前数据包

只发送 `outputs/c_b_sync_v4_1_pr_ready.zip`；数据、模型、虚拟环境和 `outputs/` 不加入 Git 提交。
接口为 `c-b-sync-v2`，状态为 `candidate`。B 读取包不需要分类模型或完整 C 仓库。
本次从已核验的旧包沿用全部 527,912 条结果，只合并说明并保留来源核验记录；没有新增推理或训练。
标签、覆盖表、JSONL、读取器及 B 源码快照这 8 个文件的字节不变，改变的是 README 和清单，另带内部历史校验档案。

### 包里有什么

| 文件 | 用途 |
| --- | --- |
| `labels.parquet` | 527,912 条有预测的标签 |
| `supplemental_predictions.parquet` | 原补预测的 11,535 条兄弟句 |
| `coverage.parquet` | 546,139 条全知识库覆盖记录 |
| `b_coverage.parquet` | 529,031 条 B 候选池记录，包含跳过行 |
| `non_eval_coverage.parquet` | 545,039 条非金标准覆盖记录 |
| `predictions.jsonl` | 标签表镜像，额外保留全为空的 `confidence` |
| `read_labels.py` | 独立轻量读取器 |
| `b_retriever_source.py` | 重建本地 B 候选范围使用的源码快照，不是 B 当前索引的证明 |
| `README.md` | 本指南，包内唯一用户说明 |
| `manifest.json` | 来源、版本、数量、文件校验值和候选限制 |
| `historical_packaging_sources.zip` | 原包来源校验档案，B 读取标签无需打开 |

发送者把新 ZIP 的实际大小和 SHA-256 随文件单独交给 B，不能套用旧 `extended.zip` 的值。
B 对下载文件运行 `shasum -a 256`，与发送者提供的完整校验值核对。
包内读取器同时检查覆盖文件的校验值；压缩包下载完整和实际索引对齐是两个独立检查。

### 数量、字段和使用边界

| 范围 | 总句数 | 有预测 | 无预测 |
| --- | ---: | ---: | ---: |
| 全部非金标准句 | 545,039 | 527,912 | 17,127 |
| B 候选池 | 529,031 | 527,912 | 1,119 |

已有预测 516,377 条保持原字段不变，补 11,535 条后共 527,912 条。
B 的 1,119 条无预测记录为 1,100 金标准句及 19 碎片句。
全部非金标准句都有覆盖记录，17,127 条跳过包括 16,772 纯航线、336 明显非英语、19 碎片。
**覆盖记录完整不等于每句都有分类标签**，该质量跳过策略需要 B 明确接受。
纯航线在语义上是客观信息；当前依据检索质量规则跳过分类，空标签不能解释成“已判客观”。

八方面是 `seat/staff/food/entertainment/ground/value/overall/other`。
主客观为 `subjective/objective`；主观情感为 `positive/negative`，客观情感为字符串 `NA`。
真正空值表示没有预测，不能补成 `other`、负面或客观。
Parquet 不提供置信度；JSONL 的 `confidence` 全为 `null`，不能补成概率。

B 解压后使用包内 `read_labels.py`，环境只需 `pandas/pyarrow`。
先读 `b_coverage.parquet`，再传入 B **真实索引**的 `chunk_id/review_id` 清单核对集合和父评论映射。
以 `chunk_id` 连接，不能按行序拼接；只对 `prediction_status == "predicted"` 的行使用标签筛选。

### B 可直接使用的读取代码

解压后把 `package` 改为实际文件夹路径；它不能直接指向 ZIP。
第一段只检查包本身，不证明 B 实际索引已经接通。

```python
from pathlib import Path
import importlib.util
import pandas as pd

package = Path("c_b_sync_v4_1_pr_ready")
spec = importlib.util.spec_from_file_location(
    "c_label_reader", package / "read_labels.py"
)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

table = reader.load_b_labels(package)
print(len(table))  # 529031
print(table["prediction_status"].value_counts())
# predicted: 527912; skipped: 1119
```

下一段必须改为 B 从**真实索引或元数据**取得的编号清单；不能用本包覆盖表冒充 B 清单。
核对缺少、多出、重复编号或父评论不一致时停止接入，排查数据与索引版本，不能删行凑通过。

```python
b_pool = pd.read_parquet(
    "填入B真实索引编号清单.parquet",
    columns=["chunk_id", "review_id"],
)
table = reader.load_b_labels(package, b_pool=b_pool)

usable = table.loc[table["prediction_status"].eq("predicted")]
food_negative = usable.loc[
    usable["aspect"].eq("food") & usable["polarity"].eq("negative")
]
```

只需要 JSONL 时可读取下面的镜像；它不含 1,119 条 B 池跳过行，不能代替完整池编号核验。

```python
json_labels = pd.read_json(package / "predictions.jsonl", lines=True)
assert len(json_labels) == 527912
assert json_labels["confidence"].isna().all()
objective = json_labels["subjectivity"].eq("objective")
assert json_labels.loc[objective, "polarity"].eq("NA").all()
```

`confidence: null` 仅表示未提供数值，不代表零概率或经过校准的概率。
新接口增加 `is_eval_parent`、`prediction_origin` 和新的父评论角色；B 的旧六类读取流程须确认兼容。
八方面、质量跳过、空置信度和兄弟句推理边界仍需 B 确认，并回传真实索引核对和查询样例。

## 7. 这次精简与后续验收

本次提交方案为 40 个文件：13 个源码、8 个脚本、15 个测试模块、1 个测试包标识、本指南、实验日志和 lock。
12 份旧专题及技术交接文档的必要内容归入本页；四个辅助脚本、人工抽查 HTML、含 2 项测试的旧上下文实验暂不提交。
它们全部保留本地；精简改变的是提交范围，不删已有证据，不改原预测数据、旧包、模型或金标准。
旧来源归档用于校验原包，不是让新人按旧版重跑；当前分类、规则和轻量读取器仍需与既有来源校验值一致。
旧源码开头和历史日志可能仍提到旧示例或专题文件，以本页的当前入口、新包及候选状态为准。
本机提交范围使用 `outputs/c_pr_ready_files_v1.txt`，替代旧的 56、42 及 39 文件清单；清单本身不进 Git。不要用全目录添加代替精简清单，以免把本地历史材料一并提交。

后续按以下顺序完成：

1. 保存已做过的人工抽查结论及导出记录，不要求重复标注；不同抽查材料不能互相冒充。
2. B 确认八方面、跳过策略、空置信度和新增父评论角色，并回传真实编号核对及检索样例。
3. W5 根据训练侧抽查和弱标签开发结果，正式记录并冻结模型、规则、超参及对照方案。
4. A 组金标准定稿、仲裁后，W7 对预先冻结方案做正式评测并写 Q4；不能用金标准选模型或反向调规则。

代码审查或包校验通过不等于模型质量验收；本包没有正式人工金标准准确率，C6 正式评测与报告仍待完成。
