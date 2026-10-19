# 回答级 RAG 评测、性能与统一消融

本实验使用 20 个冻结问题运行完整中文问答链路，保存检索证据、翻译、最终回答、程序校验状态和机器 judge 结果。
问题覆盖 12 个单航司细分项、5 个双航司比较、2 个多细分项和 1 个多方面查询。

## 评测口径

- Faithfulness：judge 识别的、受到引用原文或程序汇总支持的可核查论断数，除以全部可核查论断数。
- Answer Relevance：0–4 rubric 归一化到 0–1，检查回答是否覆盖问题中的航司、舱位、方面、细分项和比较要求。
- Context Precision：judge 判定直接有助于回答的展示证据数，除以全部展示证据数。
- judge 可读取程序计算的原始评分和全量评论汇总，避免把这些数字因不在单条展示引文中而判为无依据。
- 生成器与 judge 都是本机 `mlx-community/Qwen3-4B-Instruct-2507-4bit`，因此结果标记为
  `same_qwen4b_automatic_proxy_not_human_independent`，不能写成人工或独立模型评测。

统一指标消融在方向明确的单航司题上比较：展示 top-10 投票、完整评论池方向、正反平衡证据、标签—文本一致性过滤。
方向守卫另使用 18 个反方向对抗回答评测，结果保存在相邻的 `final_auto_eval` 实验中。

## 复现

先启动本地模型服务：

```bash
outputs/query_quality_runtime/venv311/bin/python scripts/local_qa_server.py
```

再运行完整评测：

```bash
.venv/bin/python -u experiments/final_rag_eval/run.py \
  --output-dir outputs/final_rag_eval_replay
```

完整运行逐题写入 `full_checkpoint.jsonl`，生成阶段结束后立即写入 `answers.jsonl`、
`per_question_metrics.csv` 和 `full_summary.json`。如果只需在已完成的回答目录上重跑消融：

```bash
.venv/bin/python -u experiments/final_rag_eval/run.py \
  --phase ablation --output-dir outputs/final_rag_eval_replay
```

如果需要让 judge 重新读取程序汇总后评分：

```bash
.venv/bin/python -u experiments/final_rag_eval/rejudge_with_aggregates.py \
  --output-dir outputs/final_rag_eval_replay
```

标注一致率和提交版 `eval.xls` 使用：

```bash
.venv/bin/python scripts/finalize_frozen_eval.py
```

完整汇总见 [`results_20261019/README.md`](results_20261019/README.md)，逐题精简证据和回答见
[`results_20261019/answers_slim.jsonl`](results_20261019/answers_slim.jsonl)。完整的 1.3 MB 运行快照保存在本机
`outputs/final_rag_eval_20261019/answers.jsonl`。
