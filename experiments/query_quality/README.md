# 查询质量目标、无 gold 选参监控与本地模型

本轮围绕航空服务的方面查询优化。分类器在人工主观句上的目标：方面 macro-F1 ≥ 0.85、正负极性 macro-F1 ≥ 0.90、方面和正负同时正确率 ≥ 0.80、主观召回率 ≥ 0.97。客观 precision/recall 保留为诊断，不能为了多筛出客观句而丢失有效主观评价。

查询层目标：P@5 ≥ 0.90、Success@10 ≥ 0.95、回答论断证据支持率 ≥ 0.95、人工完整问答成功率 ≥ 0.90、无答案正确说明不足率 ≥ 0.95。上述是本项目建议门槛，不是通用行业保证。

## 本轮方法和隔离边界

- 固定本地 `mlx-community/Qwen3-4B-Instruct-2507-4bit`，revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`，比较两套遵循项目分类规范的提示。
- 对照重新按新目标排名的旧弱监督组件，比较弱模型方面 + Qwen 情感、Qwen 方面 + NLI 情感、两套提示对主观性的共识策略，以及 R16 规则辅助。
- 在非 gold 诊断中发现自由输出格式错误后，追加两步固定选项分类：先从八个方面选一个，再针对该方面判断正向/负向/客观；输出 token 限定为有效选项。
- 增加 80 条与原训练、评测及旧验证样本父评论隔离的真实开发句，由助手在查看候选预测前提供 AI 辅助 silver 标签；不确定字段留空。这不是人工 gold，仍只能作为代理诊断。选择分数为新 silver monitor 的 80% 与旧弱标签/合成 monitor 的 20%。
- 最后在 silver calibration 部分分别选择方面与情感组件，再比较直接组合和保留主观评价的共识策略；monitor 部分只用于候选比较。
- 本轮是冻结模型推理和组合优化，**没有微调 Qwen 权重**。旧分类头来自前轮无 gold 的弱监督训练。
- 训练/选择进程和 MLX 子进程均拦截 gold 答案及历史 gold 调参产物读取。子进程只接受 `id` 和 `text`，不接受标签字段。
- 非 gold calibration/monitor 保留父评论隔离。弱标签与合成诊断分数均是代理，不作为真实质量验收。
- 全部候选、选择指针、源码和模型权重先冻结，随后独立进程评估现有 1,073 条 gold。已参与历史开发的 gold 不会因仲裁少量修改而重新成为独立测试集。
- monitor 每 10 秒写心跳，最长 90 分钟；在输出目录创建 `STOP` 可停止。中断/完成目录均不覆盖。

## 在当前工作区复跑这一轮

```sh
.venv/bin/python experiments/query_quality/run_monitor.py --output-dir outputs/query_quality_replay
.venv/bin/python experiments/query_quality/extend_choices.py --run-dir outputs/query_quality_replay
.venv/bin/python experiments/query_quality/extend_silver.py --run-dir outputs/query_quality_replay
.venv/bin/python experiments/query_quality/extend_specialists.py --run-dir outputs/query_quality_replay
.venv/bin/python experiments/query_quality/evaluate_gold.py --run-dir outputs/query_quality_replay
```

需要上一轮的 `outputs/c_no_gold_optimization_20261018/` 非 gold 选参记录和权重，以及 `outputs/c_gold_smoke_20261018/` 的冻结 C 源码和评估输入。silver 扩展会自动复制本目录固定的 80 条 AI 开发标签，并生成只含 ID、原文、父评论和分区的文本清单。大文件不进 Git；这些是现有工作区的复跑命令，不代表仅 clone 就拥有全部历史数据。

本地 MLX 运行环境单独位于 `outputs/query_quality_runtime/venv311/`，不会更改项目 `.venv` 的依赖。固定 `mlx-lm==0.31.2`；实际完整依赖已保存到本目录的 `requirements-mlx.lock.txt`，重建脚本按这个锁定文件安装。

## 从 Git 代码独立重建零样本分类器

Apple Silicon 上，用项目 Python（≥ 3.11）运行：

```sh
.venv/bin/python scripts/setup_query_model.py
```

它创建独立运行环境、下载固定 revision 的模型和聊天模板，默认生成本轮选中的 `closed_choice` 两步固定选项 manifest；也可用 `--variant taxonomy` 或 `--variant evidence_first` 重建自由输出的对照版本。**这一入口不依赖历史 C 权重或 gold，也不宣称自动达到质量门槛。** 这是重新下载同一预训练权重并复现推理，不是从头训练该大模型。

预测输入 JSONL 每行只能有 `id`、`text`：

```sh
.venv/bin/python experiments/query_quality/predict_cli.py \
  --manifest outputs/query_quality_standalone/frozen_candidates.json \
  --input new_texts.jsonl --output predictions.jsonl
```

实际优化候选改用 `outputs/query_quality_20261018/frozen_candidates.json`。无效或多个标签的输出记为 `invalid` 并计错，不偷偷当成正确答案。

## 查询界面和命令行

```sh
.venv/bin/streamlit run app/streamlit_app.py
.venv/bin/python -m src.query_service --aspect seat --airline 'Singapore Airlines' \
  --output outputs/seat_query.json
.venv/bin/python -m src.query_service --aspect seat --airline 'Singapore Airlines' \
  --classifier-manifest outputs/query_quality_20261018/frozen_candidates.json \
  --output outputs/seat_query_with_labels.json
```

界面支持结构化方面、航司、舱位、出发地、目的地和评论日期查询，以及 2–3 家航司在相同条件下的证据对照。补充自由文本目前按英文检索词使用；尚未加入任意中文问题的翻译和条件解析，也不自动生成总体航司排名。

查询使用现有 BM25 + MiniLM + cross-encoder，先形成最多 10 条去重后的证据池，再调用实验分类器辅助排序。客观/不确定标签不直接删除句子；同一父评论最多展示两句。标签阶段只能改变池内排序，不能提升这个池的召回率；配对评测使用同一条执行路径。每条结果显示句子 ID 和完整父评论。当前回答是带原文引用的证据摘要及检索样本计数，不是自由生成的长篇评价，也不把检索样本当成全库好评率。少于三条独立评论时说明证据不足；该数量检查本身不代表语义证据充分。

分类器辅助排序默认关闭，打开时明确标为实验候选。全量标签索引和上轮模型未被替换。
界面优先使用本轮实验 manifest；新工作区没有历史实验目录时，自动使用 `setup_query_model.py` 生成的 standalone manifest。

## 查询评测与人工判定

```sh
.venv/bin/python scripts/query_benchmark.py \
  --output-dir outputs/query_benchmark_replay --run --limit 12 \
  --classifier-manifest outputs/query_quality_20261018/frozen_candidates.json
.venv/bin/python -m src.evaluate outputs/query_benchmark_replay/judgments_to_review.jsonl \
  --output outputs/query_benchmark_replay/evaluation.json
```

已有完整的基线结果时，可以复用同一个证据池批量加标签，避免重复检索：

```sh
.venv/bin/python scripts/label_query_results.py \
  --baseline-dir outputs/query_quality_benchmark_20261018_full \
  --manifest outputs/query_quality_20261018/frozen_candidates.json \
  --output-dir outputs/query_quality_benchmark_20261018_paired
```

先冻结 120 个查询，包含单方面、正负评价、舱位条件、成对航司比较组件、超出语料日期和不存在的航司。`--limit 120` 执行全部；默认 pilot 跨类型抽 12 个。每个已执行查询导出原文结果及基线/标签两种排序，人工判定表的字段保持 `null`，不伪造相关性或回答质量分数。

填写 `expected_answerable`、前十条 `relevance`、`answer_successful`、`total_claims`/`supported_claims`，无答案题填写 `correct_abstention`。`judgment_provenance` 记录判定来源。只有真正独立保留且人工完成的评估，才可设置 `independent_test=true` 和 `human_reviewed=true`。缺判定不会通过；返回不足五条仍以 5 为 P@5 分母；点估计通过与独立验收通过分别记录。

含两种模式的判定文件会输出 `by_mode.baseline` 与 `by_mode.labels`，不把两个系统平均成一个分数。只有同一组查询、两边判定齐全且无答案标记一致时，才计算 `label_minus_baseline` 的配对分差。

本地人工评审界面可直接查看这 120 个查询的两种结果，A/B 展示顺序按查询固定打散，隐藏分类器标签，并支持展开完整评论：

```sh
.venv/bin/streamlit run app/query_review.py --server.address 127.0.0.1 --server.port 8518
```

默认读取本轮配对目录；其他运行可设置 `QUERY_REVIEW_DIR`。只有点击保存且完成必填判定后，才写入新的 `judgments_reviewed.jsonl` 与分模式 `evaluation_reviewed.json`。原始空白表不会覆盖，未评审内容仍为空。保存记录固定为 `independent_test=false`，不把开发评审变成独立验收。比较组件只评该航司的结果，完整比较回答还需另外验收。

相关性判定看原文是否能支持目标方面、是否满足航司/舱位/航线/日期条件，以及显式好评或差评要求；不要照抄分类器标签。多方面句即使主标签是另一方面，只要确实支持查询所需的评价，也可能是相关证据。判断无答案应依据语料是否有充分证据，不能因为当前 top-10 没检索到就标为无答案。比较题要核对同组双方；有一方缺证据就不能给出确定胜负。当前 120 条检索用例包含 10 个比较组件，对应 5 组双航司比较。

## 验证

```sh
.venv/bin/python -m pytest -q tests
```

覆盖主观误过滤、缺失弱标签、重复证据、父评论数量、引用原文、缺失人工判定、P@5 短结果分母、gold 文件边界和不合法模型输出。

模型及评测定义来源：

- [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [固定 MLX 量化模型](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit)
- [MLX LM](https://github.com/ml-explore/mlx-lm)
- [检索指标定义](https://ir-measur.es/en/latest/measures.html)
- [Faithfulness 定义](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
