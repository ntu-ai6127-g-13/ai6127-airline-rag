# 检索示例辅助分类（独立实验，不替换原系统）

本目录是 `main@00bad033164807a44ecc42a406672f082f0fccba` 之上的**纯新增**实验。
原 `src/`、`app/`、顶层 README、依赖和默认启动方式均不修改。切换到本分支后，
原系统仍按原方式运行；只有显式执行本目录命令才会使用新分类器。

该方案尚未通过独立人工验收，不是正式替代版本。它也不是新训练的权重。

## 方法与边界

1. 从 63 条已冻结、非评测的 AI 银标签校准句中检索参考示例。
2. 排序固定为 `0.7 × MiniLM 单位向量内积 + 0.3 × TF-IDF 余弦`；取 4 条，同一方面最多 2 条。
3. 固定 Qwen3-4B 依据项目规则，仅判断 TARGET，输出原文片段、方面、情感。
4. 严格检查 JSON、类别和逐字引用；无效输出交给固定单词分类提示回退。
5. 保留原始输出、示例 ID、无效原因、回退路线、数据/代码/权重哈希。

示例不是目标上下文，AI 标注不是人工金标准。逐字引用只核对来源，不证明语义正确。
本轮没有改 BM25/FAISS、RRF、Cross-encoder、问答或全库标签索引。
详细结果及限制见 [RESULTS.md](RESULTS.md)。

## 1. 环境与数据

主环境 Python 3.11；在仓库根目录运行：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r experiments/example_assisted_classifier/requirements.txt
```

已有主环境可直接复用；后一个文件记录本轮实测核心版本，**没有修改顶层依赖**。
如需保持原环境完全不动，也可另建虚拟环境运行这些命令。

按 [数据使用指南](../../docs/数据使用指南.md) 第 1 节重建知识库。
必须使用 `spacy==3.8.16` 和 `en_core_web_sm==3.8.0`，并确认：

```bash
.venv/bin/python scripts/check_data.py
# 1,100 句 is_eval 与清单逐行对应
```

本实验的示例库重建需要完整 `chunks.parquet`，因为默认 sample 不含全部冻结示例。
**这不意味着先做全量推理**：先跑不依赖真实数据的单测，再对下方 3 条合成句试跑，
最后才处理自己的小批量句子。不会训练模型或重打 546,139 句标签。

## 2. 固定模型与独立 MLX 环境

模型推理目前只支持 **macOS Apple Silicon**；Windows/Linux 可以运行不涉及模型的检查，
不能运行 MLX 推理。它不在查询主环境里导入 MLX。

```bash
python3.11 -m venv outputs/query_quality_runtime/venv311
outputs/query_quality_runtime/venv311/bin/python -m pip install \
  -r experiments/query_quality/requirements-mlx.lock.txt

.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3-4B-Instruct-2507-4bit', revision='50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b', allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41', allow_patterns=['*.json','*.safetensors','*.txt'])"
```

已有相同 revision 的缓存/MLX 环境无需重建。下载是明确的准备步骤；分类运行离线。
默认使用标准 Hugging Face 缓存目录，可通过 `--model`、`--encoder`、`--mlx-python` 指定其他位置。
`model_fingerprints.json` 检查权重、分词器与配置哈希，不接受静默换模型。

## 3. 先测试，再重建示例库和试跑

```bash
.venv/bin/python -m pytest experiments/example_assisted_classifier/tests -q

.venv/bin/python -m experiments.example_assisted_classifier.build_bank \
  --out outputs/example_assisted_bank

.venv/bin/python -m experiments.example_assisted_classifier.predict \
  --input experiments/example_assisted_classifier/demo_inputs.jsonl \
  --bank outputs/example_assisted_bank/bank.jsonl \
  --out outputs/example_assisted_demo \
  --experimental
```

所有输出必须使用**新的** `outputs/` 子目录；已有目录会拒绝覆盖。
`bank_spec.json` 只版本化不可自动重建的 AI 标注配置：ID、方面、情感、文本哈希，
**不含原始评论文本或人工答案**。原文从自己的知识库按 ID 恢复。
重建时逐条核对文本哈希、父评论和 1,100 句评测映射；示例与评测句、评测父评论、
规范化重复文本的交集必须为零。任何后续训练仍必须排除这些评测边界。

输入 JSONL 每行仅含非空字符串 `id`、`text`，ID 唯一，例如：

```json
{"id":"my-001","text":"Seats were great but food was awful."}
```

主要输出：

- `predictions.jsonl`：方面、主客观、情感、原文片段、原始输出、示例 ID、推理路线。
- `report.json`：成功状态、耗时、回退数、数据边界、输入/输出/代码/模型指纹。
- `examples/`、`words/`（有回退时）、`*_lock.json`、日志：供逐步检查和追溯。

`valid=False` 在回退后仍保留：它表示**原始引用/结构不合格**，不是最终标签缺失。
不要把回退标签当作有已验证片段支持的标签。异常时保留日志和 `run_started.json`，
不会生成成功的最终报告。每个工作进程默认 600 秒超时，可用 `--timeout` 调整。

**已知失败**：本机三句 demo 中，`demo-contrast` 的原始输出无效，回退误判为
`seat/positive`，而按转折规则应为 `food/negative`。运行完成不表示 demo 全部分类正确；
这也是不能接管默认系统的原因之一。详见 RESULTS。

不同工作目录可用 `--kb`、`--eval-meta`、`--eval-text` 指定已有本地数据，
无需复制或覆盖旧文件。不要把新结果写回原知识库或原标签索引。

## 4. 接入策略

当前只交付独立 CLI，不接入 Streamlit、不替换 `src/classify.py`，没有修改默认标签排序。
将来如通过独立验收，建议另开集成 PR，增加**默认关闭**的后端开关，再做分类、检索和回答的配对验证。
本分支不包含模型、语料、全量预测、历史输出或虚拟环境；这些始终本地生成。
