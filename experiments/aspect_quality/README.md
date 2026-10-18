# 方面分类优化：固定规则、非 gold 示例和选项顺序共识

这轮直接优化 r36 的 Qwen 方面判断；沿用相同的正负/客观判断提示与既有 NLI 主观共识。
固定本地 Qwen3 4B 权重，没有微调权重，也没有把 gold 当训练或选参材料。
目标仍为方面 F1 85%、极性 F1 90%、联合正确率 80%、主观召回 97%。

## 固定的五个方案

1. `baseline`：重跑原固定选项 Qwen，并应用既有 NLI 主观共识。
2. `hierarchy`：根据项目规范明确“评价对象—转折—多方面—泛泛整体评价”的决策顺序。
3. `fewshot`：在上述规则上加 15 条既有非 gold calibration 示例，覆盖八个方面；每类取最短和最长例句，seat 只有一例。
4. `rotated`：相同规则与示例，固定轮换选项字母的对应顺序，检查选项位置偏差。
5. `ensemble`：四种方面预测投票；最多票并列时依次偏好 fewshot、hierarchy、baseline、rotated。按最终方面重新判断情感。

所有方案都保留同一 NLI 共识策略：只有 Qwen 判断客观、而 NLI 判断主观时，才采用 NLI 的主观及极性输出。
这样可以直接比较方面策略，同时监测情感和主观召回退化。

## 数据边界与选择规则

- 新增 96 条真实开发句：按已有规则方面各取 8 条，加 32 条规则未覆盖的句子，seed=20261019，每条来自不同开发父评论。
- 新样本排除训练父评论、原始全部评测父评论、旧代理及 silver 父评论，并排除重复评测文本；运行前另检查与弱训练文本的重叠。
- 助手在查看新样本的候选预测前完成 silver 标注。89 条方面可判断、90 条三分类标签可判断；不确定字段保持缺失。
- `development_silver.jsonl` 是 **AI 辅助开发标签**，不是人工 gold。抽样也不是总体随机样本，不能以它估计全库准确率。
- 另加入旧 silver monitor 40 条、旧弱标签/合成 monitor 294 条，共 430 条。15 条示例只来自旧 calibration，与 monitor 的父评论及规范化文本均不重叠。
- 固定选择分数：新 silver 70%、旧 silver 20%、旧代理 10%；各组内为方面 F1 65%、联合正确率 25%、主观召回 10%。
- 只有合并 silver 的极性 F1 和主观召回均不比基线下降超过 2 个百分点，候选才可入选；总选择分数至少提高 0.005，否则保留基线。相同分数优先预先列出的方案。
- 最长 90 分钟、每约 2 秒刷新心跳；创建输出目录下的 `STOP` 可停止该实验进程组。完成或失败的目录不覆盖。

`plan.json` 在任何本轮模型预测前锁定选择规则、源代码和数据哈希。全部候选运行结束后写入 `frozen_candidates.json`。
之后单独启动 gold 审计进程，模型子进程只接收 ID 和原文。审计不改变开发阶段的选择，不因看到 gold 分数而追加挑选。
定稿 gold 有历史曝光及事后子集筛选，因此本轮 gold 结果仍是开发审计，不能作为独立泛化保证。

## 本地运行与复现

工作区需已有知识库、原始评测排除清单、前两轮冻结组件、MLX 运行环境及本地模型、`weak_rebuild_20261019` 产物。
这些大文件和历史缓存不随 Git 提交。默认运行结果在 `outputs/aspect_quality_20261019/`。

第一次原始抽样由 `prepare_development.py` 生成；版本化的 silver 已固定，重跑时使用：

```bash
.venv/bin/python experiments/aspect_quality/prepare_replay.py --output-dir outputs/aspect_quality_replay
.venv/bin/python experiments/aspect_quality/run_monitor.py --output-dir outputs/aspect_quality_replay
.venv/bin/python experiments/aspect_quality/evaluate_frozen.py --run-dir outputs/aspect_quality_replay
```

复用开发集重跑不等于新的独立实验。重跑前不要修改 silver 标签来迎合模型输出。

gold 审计还需 `outputs/gold_release_audit_20261019/` 的先前缓存登记。审计先校验旧缓存哈希，
复用原 r34 原始预测和 r35 的 NLI 情感输出，并核验合成结果与原 r36 完全一致。
开发阶段复跑的 334 条旧样本方面预测全部相同，但有 1 条极性不同；批次组成发生了变化，具体原因尚未确定。
因此 gold 审计在相同 1,000 句输入与顺序上重新运行原基线，并记录相对旧缓存的差异，确保候选比较口径一致。
新候选只对预选方案重新推理；若预选共识方案，则运行其所需的全部方面组件。
审计最长 60 分钟，创建 `STOP_AUDIT` 可停止。报告包含四项目标、与 r36 的差值、父评论分组 bootstrap 区间及发布门槛。

```bash
.venv/bin/python -m pytest -q tests/test_aspect_quality.py tests/test_gold_release.py tests/test_query_quality.py
```

这轮实验通过开发代理，不代表分类验收通过；小批查询的用户体验认可也不替代完整定量验收。
正式模型与全量标签索引保持原版本，直到质量评估完成。

## 对新句子使用预选候选

输入 JSONL 每行仅包含字符串 `id` 和非空 `text`。独立预测入口核验冻结代码、权重和示例，
按预选方案运行本地 Qwen 与 NLI 共识，保持实验状态，不修改原查询模型指针：

```bash
.venv/bin/python experiments/aspect_quality/predict_cli.py \
  --manifest outputs/aspect_quality_20261019/frozen_candidates.json \
  --input new_texts.jsonl --output outputs/aspect_predictions_new.jsonl
```

运行目录及输出文件必须使用未占用路径。共识方案需要运行多个 Qwen 组件，耗时高于单个提示方案；
全量索引的推理吞吐与缓存仍需单独验证。

## 2026-10-19 五方案运行结果：未晋升

5 个预先规定的方案已全部跑完（约 14 分 44 秒），按非 gold 开发分数冻结 `rotated`，随后进行独立进程 gold 审计（约 15 分 47 秒）。
新 silver 上方面 F1 从 83.67% 提高至约 94.14%，但未迁移到人工 gold，不能作为质量改进交付。

| 指标 | 目标 | 同批重跑 r36 | 开发预选 rotated | 变化（百分点） |
|---|---:|---:|---:|---:|
| 方面 macro-F1 | 85.00% | 74.13% | 69.91% | -4.22 |
| 正负 macro-F1 | 90.00% | 91.76% | 91.62% | -0.14 |
| 方面与正负联合正确率 | 80.00% | 70.84% | 66.81% | -4.03 |
| 主观召回 | 97.00% | 96.84% | 96.95% | +0.11 |

1,000 条 gold 中有 919 条主观句；方面、正负和联合指标在这些句子上统计，误判为客观仍计错。
主观召回 96.95% 仍低于 97%，不按四舍五入算过线。
同批 r36 与旧缓存 1,000 条预测完全一致。联合正确率差值的父评论 bootstrap 95% 区间为 -6.38 至 -1.94 个百分点。
本轮候选在 gold 上回退，未替换旧模型、查询配置或全量索引，也没有改选其他候选来迎合 gold。

本轮固定 Qwen3 4B 权重，优化的是推理规则、非 gold 示例、选项顺序和投票；没有权重微调。
gold 只在冻结之后计分，没有进入训练或选参；已有 gold 的历史开发曝光与事后子集选择仍限制独立性。
结果表明自动 silver 的优化收益在本轮没有可靠迁移；不能仅凭本次实验断言是模型容量不足或指定唯一原因。
下一轮应优先建立与 gold 父评论和文本隔离的人工开发集，覆盖 other、overall、staff 等薄弱类别，
再比较弱监督噪声处理与方面分类器训练。开发选择依据改善前，不继续用同类少量 silver 追逐分数。

完整本机结果：`outputs/aspect_quality_20261019/audit.json`、`release_gate.json`、`audit_predictions.jsonl`。
开发冻结清单 SHA-256：`76088cd14a306fa800a57153eff7a596524f9e6dddc94868ee260724f2b03d52`。

验证：当前测试 170 项及 100 个子测试通过；独立 CLI 的 8 条非 gold 冒烟预测与已冻结批次逐项一致。
冻结源码与权重、最终 gold、旧模型及旧候选清单的哈希核验通过。冒烟复现不等于分类准确率达标。
