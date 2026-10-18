# 定稿 gold 后续推进（2026-10-19）

本轮已接入人工审核定稿的 1,000 句，完成冻结候选复评，并从当前仓库代码重新生成弱标签和权重。
gold 保持评测用途，没有进入本轮训练；Qwen 权重没有微调，原生产/保留模型指针没有替换。

## 数据与指标口径

数据版本为 `gold_frozen_1000_20261019_v1`，CSV 与相对路径发布登记来自标注分支提交 `820acd8`。
CSV SHA-256：`8a4926c0bbfd22c79ed06cf2b8ef783e09902a5d5743a2994d64a6679a57b223`。
本轮复评代码核验了哈希、审核状态、唯一 ID、原始句子/父评论映射和标签合法性。

本版包含 919 条主观句、81 条客观句。相对之前评测的 1,073 句，保留其中 1,000 句；
保留句中 9 句标签变化，原文没有变化。下面四项指标在 **919 条主观 gold** 上计算，
误判为客观的主观句仍计为错误，不从分母中删掉。

| 模型 | 方面 macro-F1 | 正负 macro-F1 | 方面与正负同时正确 | 主观召回 | 全部过线 |
|---|---:|---:|---:|---:|---|
| 目标 | 85.0% | 90.0% | 80.0% | 97.0% | — |
| 之前保留 r6 | 71.3% | 84.1% | 61.0% | 94.3% | 否 |
| 非 gold 预选 r34 | 74.1% | 87.8% | 68.1% | 88.6% | 否 |
| 事后对照 r36 | 74.1% | 91.8% | 70.8% | 96.8% | 否 |

全部 13 个模型都复用了先前冻结的预测，按稳定 ID、原文及可用元数据核验后重新计分。
这轮分数变化来自样本组成与最终标签修订，**不是新训练带来的涨分**。
`audit.json` 同时记录同样 1,000 句在旧标签下的分数，可区分标签修订与子集组成的影响。
r36 在之前评测前已经冻结，但不是非 gold monitor 预先选中的方案，只作为事后诊断候选。

本版有历史开发曝光，而且是事后按原始方面一致性选择的子集，不能称为全新独立测试。
人工审核完成率 100% 不等于标注者一致率。数据来源、原始一致率、分布与修订政策见
[`data/eval/README.md`](../data/eval/README.md) 和发布登记。

## 当前仓库可以重建的训练流程

C 组训练实现来自 Git 提交 `0538fd9`，保留其弱标签规则、三个任务、父评论切分和拟合算法。
为兼容已有查询接口，原训练入口改名为 `src/train_classifiers.py`；
`src/classify.py` 继续承担查询用本地模型的推理桥接。训练报告中的入口文件哈希也随之调整。

训练仍排除原 1,100 条评测句及其 1,091 条父评论和规范化重复文本，没有因最终只交付 1,000 句而缩小排除范围。
CLI 训练进程另加读取拦截，禁止打开 gold 答案、历史 gold 审计及按 gold 调过的权重；
只允许读取原始无答案句子及元数据用于排除。实际训练、开发集与定稿 gold 的
chunk ID、父评论 ID、规范化文本三种交集均为零。

运行前需已有 `data/processed/chunks.parquet`、原始 `to_label.csv` / `to_label_meta.csv` 和仓库依赖。
从仓库根目录执行，输出和控制目录必须是新路径：

```bash
.venv/bin/python scripts/run_training_monitor.py \
  --output-dir outputs/weak_rebuild_new \
  --control-dir outputs/weak_rebuild_monitor_new
```

monitor 每 5 秒写心跳，默认 90 分钟截止；在控制目录创建 `STOP` 文件可停止该任务。
它监控一次可复现训练，不表示无限期自动调参。直接训练也可执行：

```bash
.venv/bin/python -m src.train_classifiers train-all \
  --output-dir outputs/weak_rebuild_direct --seed 42
```

本轮实际产物在 `outputs/weak_rebuild_20261019/`，监控状态为 `completed`。
生成 289,999 条弱标签（训练 231,779、开发 58,220）。与上一轮从 Git 快照重建的基线相比：

- `weak_labels.parquet` 和 `split.parquet` 数据及文件哈希完全相同。
- 58,220 条开发集的原始/最终预测、推理来源等 17 列完全相同，文件哈希也相同。
- 权重保存后重新加载，全部开发预测保持一致。详细对照见该目录的 `verification.json`。
- TF-IDF 词表、IDF、三个模型的类别顺序、系数和截距也逐元素完全相同，见 `full_reproduction_check.json`。

已有参考产物时可用仓库脚本复核：

```bash
.venv/bin/python scripts/verify_training_reproduction.py \
  --reference-run outputs/c_gold_smoke_20261018/reproduced_current_git_seed42 \
  --rebuilt-run outputs/weak_rebuild_20261019 \
  --output outputs/reproduction_check_new.json
```

这是复现基线和接通工程流程，不是质量突破。训练日志里很高的弱标签验证分数衡量规则一致性，不能替代人工 gold 分数。

## 批量导出联调

`export` 支持加载刚训练的权重，按批次生成八方面、主客观和正负标签；保留原文、ID、模型版本、预测来源及父评论所在的训练/开发角色。
它拒绝知识库或评测排除清单版本不匹配的模型，不覆盖输入知识库或已有输出。

```bash
.venv/bin/python -m src.train_classifiers export \
  --model outputs/weak_rebuild_20261019/model.joblib \
  --input outputs/weak_rebuild_20261019/export_smoke/input.parquet \
  --output outputs/labels_smoke_new.parquet
```

本轮随机抽取 2,048 条知识库句子（seed=42），排除评测父评论、重复评测文本及不合格句后，
成功导出 1,916 条本地联调标签。ID、标签组合、评测隔离、输入哈希和 parquet 回读检查全部通过。
这是接口冒烟产物，不是正式标签索引；全量标签索引仍待模型与查询验收后再建。

## 冻结模型复评与查询验收

已有上一轮冻结候选、模型文件及预测缓存时，可复评定稿 gold：

```bash
.venv/bin/python scripts/audit_frozen_gold.py \
  --output-dir outputs/gold_release_audit_new
```

脚本在读取/计分前验证冻结的代码、输入和权重哈希，并拒绝覆盖旧报告。
本轮结果位于 `outputs/gold_release_audit_20261019/`，包括全部模型指标、9 句标签修订、父评论 bootstrap 区间及发布门槛结果。
旧实验产物是本机输出，不随 Git 提交；全新克隆要先按
[`experiments/query_quality/README.md`](../experiments/query_quality/README.md) 重建相应实验，不能仅复制路径。

查询实验将 r36 标记为事后诊断候选，单独保存 manifest，不改原预选记录。
与原基线使用同样的 120 个查询和每题同样的 top-10 候选池，比较标签辅助排序。
这只能评估排序，不能证明候选召回改善；顺序改变也不等于相关性改善。

本轮 120 个查询已完成，复用 523 条不同证据，批处理约 309 秒（不能当成单题交互延迟）。
相对基线有 61 题的 top-5 顺序改变；与上轮 r34 相比只有 4 题不同。
原文、引用、元数据、候选池一致性和去重/父评论限制均通过检查；10/10 条明确无语料的条件返回证据不足。
240 条两种模式的待评审记录仍为空白，没有填入模型猜测的相关性。

```bash
.venv/bin/python scripts/label_query_results.py \
  --baseline-dir outputs/query_quality_benchmark_20261018_full \
  --manifest outputs/query_r36_experiment_20261019/manifest.json \
  --output-dir outputs/query_r36_paired_new
```

查询指标仍需人工判定：P@5 ≥90%、Success@10 ≥95%、论断支持率 ≥95%、回答成功率 ≥90%、无答案正确拒答率 ≥95%。
查询分类标签和结果顺序不能代填人工相关性；开发查询也不会被标成独立测试。
可在配对结果完成后启动专用评审界面：

```bash
QUERY_REVIEW_DIR=outputs/query_r36_paired_20261019 \
  .venv/bin/streamlit run app/query_review.py --server.port 8519
```

本机本轮已在 `http://127.0.0.1:8519` 启动 r36 专用评审页面；应用测试确认可加载，初始进度 0/120。
重启电脑或结束服务后，可用上面的命令重新启动。

用户随后在对话中确认“小批查询没有问题，可以通过”。已记录为**小批查询整体体验的定性评审通过**，
可继续推进开发。反馈存于该配对结果目录的 `user_feedback.json`，页面也显示这一状态。
本次反馈未指定题号、数量或 A/B 偏好，页面尚无保存的逐题评分；P@5、答案支持率等定量指标仍待统计，
不将这项反馈扩大为全部 120 题或独立测试通过。

## 验证与接下来的交付

默认 `pytest` 只收集当前应用与标注工具测试，历史冻结实验单独运行，避免其旧版 `src` 导入及进程级读取拦截污染应用测试进程。

```bash
.venv/bin/python -m pytest -q
PYTHONPATH=experiments/no_gold_optimization \
  .venv/bin/python experiments/no_gold_optimization/test_boundaries.py
```

本轮应用与标注测试 166 项、100 个子测试通过；历史实验边界测试另有 4 项通过。

下一步的实质工作是提高方面识别和联合正确率，并完成查询结果人工评审。
质量达标后再冻结模型版本、生成全量标签索引和验收查询系统。
目前界面可以筛选、比较和展示可追溯评论；完整比较型自然语言答案、独立查询验收、
作业 `eval.xls`、完整报告和展示材料仍未因 gold 冻结而自动完成。
