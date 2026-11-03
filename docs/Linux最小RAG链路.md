# Linux 上运行最小 RAG 链路

本入口复用现有知识库、检索和回答校验代码。生成服务新增 Transformers 后端，可使用 PyTorch 能识别的 GPU；AMD ROCm 在 PyTorch 中也使用 `cuda` 设备名。

## 这条链路做什么

```text
问题 → BM25 词语检索 + 向量语义检索 → 融合、重排与筛选
     → 原文证据及引用编号 → Qwen 生成 → 校验引用和数字 → 回答
```

知识库的句子已经存在 `data/processed/chunks.parquet`，无需为启动问答重新切句。
索引是查找这些句子的工具。生成器只接收检索出来的证据，不会把全库评论装进上下文。
本入口不要求已有分类标签，缺少标签时可先验证检索和生成。分类标签属于后续接入项。

## 第一次准备

使用项目已有 `.venv`。需要 `huggingface_hub`、`sentence-transformers`、`transformers`、PyTorch 和项目检索依赖；
Qwen3 要求 Transformers 至少支持 `qwen3` 架构，参考官方模型卡：
<https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507>。

```bash
.venv/bin/python scripts/setup_rag_models.py
.venv/bin/python scripts/linux_rag.py build-indexes
```

第一条下载 embedding、reranker 和 Qwen3 4B 模型，把确切 revision 记录到
`outputs/minimal_rag_models.json`。重复运行沿用已记录的 revision。
第二条保留已有 BM25 索引，并在缺少完整向量索引时建立向量索引；发现残缺索引会停止，交由使用者检查。
下载不训练模型；建立索引只对原文编码。权重位于 `models/huggingface/`，索引位于 `index/`，均被 Git 忽略。

## 启动与体验

在仓库根目录打开两个终端。

终端一保持模型服务运行：

```bash
.venv/bin/python scripts/local_qa_server.py --backend transformers --device cuda
```

终端二启动查询页面：

```bash
.venv/bin/python -m streamlit run scripts/linux_query_app.py \
  --server.port 8517 --server.address 127.0.0.1 \
  --server.headless true --server.fileWatcherType none \
  --browser.gatherUsageStats false
```

打开 <http://127.0.0.1:8517>，先问“新航经济舱的座位怎么样？”。检查结构化条件是否正确，
展开回答引用，核对对应原文。中文模式还会调用模型翻译证据，首次查询通常比 English 模式多一道生成步骤。

生成服务默认地址为 `127.0.0.1:8765`。停止服务使用所在终端的 Ctrl+C；重启后模型需重新加载。
本 Linux 入口在进程中选择本地固定版本的检索模型，不修改 `src/config.py`，不要求向默认全局 Hugging Face 缓存重复下载。

## 检查服务与完整链路

```bash
# 同步与流式生成应给出相同的简短结果
.venv/bin/python scripts/linux_rag.py probe

# 真实检索 → 英文证据 → 生成、校验及引用；记录模型是否被采用
.venv/bin/python scripts/linux_rag.py smoke
```

`probe` 的记录写到 `outputs/minimal_rag_server_probe.json`。
`smoke` 的问题、原文、回答、流式事件及模型版本写入带时间戳的 `outputs/minimal_rag_smoke_*.json`。
检查 `answer.mode`：`local_qwen` 或 `local_qwen_repaired` 表示模型答案被采用；
`validation_fallback` 表示模型尝试后未通过校验，最终使用程序回答。带引用只说明引用编号有效，答案是否被原文支持仍需人工核对。

## 与历史实验的关系

这里使用官方 `Qwen/Qwen3-4B-Instruct-2507` 的 PyTorch 权重，GPU 加载为 BF16。
历史结果使用 MLX 4bit 版本；后端与数值格式发生变化，后续实验应使用本轮实际版本重新记录结果。
最小链路通过说明接口可用，不代表分类、回答质量或独立验收已经达标。
