# 冻结评测、真实生成消融与 C10

本实验的运行产物保存在被 Git 忽略的 `outputs/`。代码和协议定义可以审阅，人工判断保持空白。
生成器与自动评委均为本机 Qwen3 4B，因此代理分数与人工分数分别记录，不称为独立模型评测或 RAGAS 库分数。

目录由原任务编号名称调整为 `downstream_evaluation`，对应回答质量、生成策略消融和排名评测。
历史协议与结果保留原路径和内容；其源码哈希绑定旧包路径，不能直接用于运行改名后的代码。
原始源码已保存在各历史协议目录的 `frozen_source_tree/`。新运行应按下列命令冻结新协议，使用新的输出目录。

## 协议与数据

`freeze.py` 在新题计分前冻结 12 道题、评分规则、模型 revision、输入和源码 SHA-256、消融配置及 C10 纳入规则。
原 20 题只作历史开发复核。新题由本轮编写，使用相同历史语料，不能称为第三方盲测。
题目涵盖单航司、比较、细分项、事实、多方面、小样本和无答案；4 个合成引用检查单独计数，不混入真实问答准确率。
小样本切片按元数据评论量选择，未来年份无答案切片使用显式结构化条件；这些条件会写入运行记录。

```bash
.venv/bin/python -m experiments.downstream_evaluation.freeze \
  --output-dir outputs/downstream_protocol_replay
```

已存在的协议目录拒绝覆盖。原题解析发现的措辞与语料实体差异在冻结前检查；新题侧重生成与证据支持，不能证明解析器对任意自然语言都有效。

## 真实生成与消融

启动本机模型服务后执行：

```bash
.venv/bin/python scripts/local_qa_server.py --backend transformers --device cuda

# 在另一终端执行
.venv/bin/python -m experiments.downstream_evaluation.qa \
  --protocol outputs/downstream_protocol_replay/protocol.json \
  --output-dir outputs/downstream_answers_replay
```

逐题保存真实提示、初稿、校验原因、修正版、证据、程序汇总与代理评委的原始输出。
三个策略共享同一次真实初稿和同一份检索证据：

| 策略 | 行为 |
|---|---|
| full | 现有生产链路：校验、一次修正、必要时程序回退 |
| no_repair | 初稿有效则采用，否则直接使用同一程序回退 |
| no_guard | 保留真实初稿，关闭校验与修正，仅作离线分析 |

这是策略消融的配对回放，初稿由真实生产生成器产生，不使用模板冒充模型回答，也不把共享初稿说成三次独立生成。
程序直接回答的题目标为 `ablation_applicable=false`，不进入生成策略配对差值。
当前分类标签未接入，三组都关闭标签辅助；本轮不评估分类标签、重排或正反平衡组件的独立收益。
每航司/方面使用相同 top-6 预算，初稿和修正版上限均 480 token、temperature=0；中英文模式保留原页面的翻译逻辑。
相同答案只调用一次自动评委。代理指标缺失时保留缺失，并报告配对样本数；区间是问题级配对 bootstrap。

任务中断后，可加 `--resume` 沿用完整逐题 checkpoint。源码或输入变化会被冻结哈希检查拒绝，需要重新冻结新版本并说明原因。

v1 首题的自动评委把父评论 ID、统计字段名当成引用编号，导致两次评分均无效；失败记录原样保存在
`outputs/c_q5_answers_20261024_v1/`。v2 只修正评委输入序列化和紧凑计分输出，不调整问题、生成策略或 C10 参数。
v1 原始实验源码在对应协议目录的 `source_snapshot/` 保存，C10 源码与预测哈希已另行核验。

本轮 v2 的 12 题已完成：7 题实际调用生成器，完整系统采用模型初稿 1 题，6 题修正后仍使用程序回退；
另有 2 题直接使用评分比较、1 题使用菜品证据汇总、2 题为空切片程序回答。
完整流程与关闭修正的最终回答完全相同，本轮未观察到修正收益。
完整系统的同模型代理 Faithfulness 为 0.7800（10 个有可核查论断的题目），Answer Relevance 为 0.9583（12 题）。
与关闭校验相比，7 个生成题的配对 Faithfulness 差值为 +0.0430，95% 区间 [-0.1713, 0.2370]，不能宣称显著改善。
Context Precision 的总体代理均值为 0.7583（10 个有展示证据的题目）；各策略使用相同证据，
答案条件下评委对上下文打分的波动不代表检索收益，不用该消融证明检索改善。
4 个引用错误探针符合预期；逐题生成/校验跟踪与无答案空证据检查保存在 `artifact_audit.json`。

## 人工复核

运行目录包含历史 20 题人工包和新题 A/B/C 隐去配置名的人工包。评委分数没有预填进人评材料。
启动评审页面，例如：

```bash
.venv/bin/python -m streamlit run experiments/downstream_evaluation/review_app.py \
  --server.address 127.0.0.1 --server.port 8519 -- \
  outputs/c_q5_answers_20261024_v2/new_cases_human_review.json
```

原 20 题也可使用 `outputs/c_q5_human_20261024_v1/historical_20_human_review.json`。
新题 36 条策略记录中有重复答案，已另外生成 `new_cases_human_review_unique.json`，只需评 18 条不同答案；
`human_review_duplicate_map.json` 保留完整记录到去重记录的对应关系，后续计算分策略人评分数时需要展开该映射。
页面要求填写评审人、证据相关性、回答相关性、逐条论断及支持理由；无答案题还需判断拒答。
保存到同目录的 `*_annotations.json`，不改原空白包。要检验标注一致性，需要另一个评审人使用独立文件评分后再比较；本版不自动声称有人际一致率。

```bash
.venv/bin/python -m experiments.downstream_evaluation.review \
  outputs/c_q5_answers_20261024_v2/new_cases_human_review_annotations.json \
  --output outputs/c_q5_answers_20261024_v2/human_summary.json
```

未填写的记录不会计为成功；已填子集的分数不会自动当成整套验收。历史机器评委与人评偏差的比较需要先有人评，当前仍待完成。

## E6 回答端 2×2×2 析因消融（`factorial.py`）

报告 E6 的三个因素：X 全量统计（完整直接匹配池的标签计数与由此得出的总体方向，同时用于生成、校验和程序回退）、
Y 正反平衡且标签与措辞一致的示例、Z 校验 + 一次修复 + 程序回退。八种配置共享每题同一个 30 条候选池
（排序不使用标签）、同一提示词和 4 条引用预算；X 关闭时提示词、校验和回退都看不到全量统计；
Y 关闭时取候选池前 4 条；Z 关闭时直接返回初稿。每个 X/Y 组合一份初稿，由 Z 开/关共用。
20 道题来自 `data/eval/query_smoke_40_20261019.json` 的 Q01–Q20，改写为固定英文句式并检查解析结果与原题的航司、舱位、细分项一致。
生成与评委都是本机 MLX 版 Qwen3-4B（4-bit），评委沿用 `experiments/final_rag_eval/run.py` 的 `judge()`，
每种配置给评委相同的程序统计。所有模型调用和评分写入缓存，中断后用同一命令续跑。

```bash
outputs/query_quality_runtime/venv311/bin/python scripts/local_qa_server.py   # 127.0.0.1:8765
.venv/bin/python -u experiments/downstream_evaluation/factorial.py \
  --output-dir outputs/q5_factorial_replay
```

输出：`protocol.json`（题目与配置）、`pools.json`（候选池与全量统计）、`model_calls.jsonl`、`judgements.jsonl`、
`answers.jsonl`（初稿、校验原因、修复稿、最终回答、评分）、`per_answer.csv` 和 `results.json`
（各配置均值、相对基线的配对差值、主效应与交互效应，均为以题目为单位的 2,000 次 bootstrap，种子 6127）。
报告数字来自 `outputs/q5_factorial_20261104/`。读 `per_answer.csv` 时用 `dtype={'config': str}`，以保留 `000` 这类编码的前导零。

## C10 评分隔离排名

```bash
.venv/bin/python -m experiments.downstream_evaluation.ranking \
  --protocol outputs/downstream_protocol_replay/protocol.json \
  --output-dir outputs/downstream_ranking_replay
```

父评论按固定 hash 划出约 20% 留出集；原 1,100 句 gold 的父评论与规范化文本另行排除。
训练/开发材料还排除所有留出文本的规范化重复，然后复用 C·甲的弱标签、切分和三个模型拟合 API。
这是专用诊断模型，不替换已有正式分类器。开发分数参照弱标签，不是人工 gold 质量。

预测函数只接受 `chunk_id/review_id/text/airline/cabin` 五列；读取模型与这个白名单 parquet，不打开原始评分。
预测写盘并冻结哈希之后，计分阶段才读取留出父评论的 `seat_comfort`。训练侧评分用于弱监督，评测侧评分不参与预测。
按预测为 seat 且主观的句子先计算每条评论正向比例，再对航司取评论均值，避免长评论权重过高。
经济舱航司纳入条件预先设为至少 30 条参照评分、15 条可预测评论；固定纳入集合后，进行 2,000 次航司内父评论配对重采样。
并列分数使用平均名次；覆盖数、未纳入航司及区间有效次数都会输出。

报告 E7 使用 2026-11-04 在 Apple M1 Pro 上的重跑：协议 `outputs/q5_protocol_20261104/`，结果
`outputs/q5_ranking_20261104/report.json`：58 家航司，Spearman 0.8056，95% 区间 [0.6749, 0.8329]；
8,365 条留出评分评论中，3,154 条有可用座位情感预测。（2026-10-24 在 Linux 上的首次运行为 0.8095 [0.6820, 0.8379]、3,152 条。）
`freeze.py` 读取 `outputs/minimal_rag_models.json` 记录模型版本；Mac 上该文件记录 MLX 4-bit 生成模型，需手动创建或由 `scripts/setup_rag_models.py` 生成。
区间反映固定航司集合内的评论采样不确定性，不包含重新训练的模型不确定性。
它评估的是隔离的文本分类—评论聚合—航司排名分支，不代表当前 UI 使用原始评分的排名，也不检验细分腿部空间的绝对真值。

## 验证

`tests/test_c_q5_protocol.py` 覆盖父评论与文本隔离、评分列拒绝、评论去重聚合、真实初稿策略回放、非模型题排除、空白人评与引用错误探针。
新增 10 项测试及原有问答/边界 52 项测试通过；其中新增评委检查拒绝不可能的论断计数和统计字段引用 ID。
评审页初始加载无异常。完整生成实验和 C10 均另外保留实际运行记录。
