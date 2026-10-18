# 不用项目 gold 训练和选参的优化 monitor

本实验只使用 C 组已有弱标签训练分类头。新的规则来源重加权、平衡采样 MLP、本地 NLI 零样本模型和融合方法，均先在非 gold 的数据上选参并冻结；gold 在独立的事后评估进程中读取。

## 边界

- 训练进程禁止打开项目 gold 答案目录和此前使用 gold 调参的权重目录。启动时进行一次拒绝读取自检。
- 从原 weak dev 抽取固定样本，按父评论拆成 calibration / monitor；不从 gold 取训练样本。
- 按项目标注规范生成合成诊断句，calibration / monitor 使用不同句式。它们不进入权重拟合，也不能证明真实泛化质量。
- 选择分数为 weak proxy 和合成诊断分数等权平均。每个候选的阈值只由 calibration 部分选择，候选间比较使用 monitor。
- MiniLM 和 DeBERTa 均冻结；MLP 最多 20 轮，连续 4 轮无有效改善停止。整个训练进程最长 45 分钟，每 10 秒更新心跳；输出目录创建 `STOP` 可在下一检查点停止。
- 预定方法与融合全部完成后写入 `frozen_candidates.json`。事后 gold 审计只报告，不改选中指针或阈值。原候选和全量标签索引不会被覆盖。

## 方法

1. 原始 TF-IDF 分类器作为非 gold 校准基线。
2. 固定 MiniLM 向量上的 LogisticRegression，按类别及弱规则来源重新加权，C=0.1 / 1。
3. 平衡抽样的 64 / 128 单隐层 MLP，按非 gold 校准数据早停。
4. 两组预先写定类别描述的本地零样本分类。
5. 在非 gold 校准部分选出各方法来源，比较 NLI 权重 0.25 / 0.5 / 0.75 的概率融合。
6. 可选教师扩展：从原 weak train 的主客观弃权句中固定抽取 2,048 条，仅在两个零样本描述模板一致且达到预设分数门槛时加入主客观／极性弱标签；方面标签保持原样。新增标签及两组分数逐条保存。再次比较 C=0.1 / 1，全部在读取 gold 前完成。

零样本模型：[MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33)，固定 revision `613e8c52c33e2bc0677ada4ad760f693e5e0f581`。模型用自然语言类别描述进行分类；本项目句子仅在本机做推理。MLP 行为参见 [scikit-learn 文档](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)。

## 依赖与运行

沿用项目 `.venv` 中的 numpy、pandas、scikit-learn、scipy、joblib、faiss、torch、transformers、sentence-transformers。输入是已生成的 `outputs/c_gold_smoke_20261018/` 冻结 C 代码、weak labels 和分组切分，以及现有 `index/` MiniLM 向量。不会自动下载或修改标注答案。

首次使用时缓存公开模型（不上传项目文本）：

```sh
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33', revision='613e8c52c33e2bc0677ada4ad760f693e5e0f581', allow_patterns=['*.json','*.safetensors','*.model','README.md'])"
```

在仓库根目录执行。使用新的输出目录，已有 run 不覆盖：

```sh
.venv/bin/python experiments/no_gold_optimization/run_monitor.py --output-dir outputs/c_no_gold_replay
.venv/bin/python experiments/no_gold_optimization/teacher_augment.py --run-dir outputs/c_no_gold_replay
.venv/bin/python experiments/no_gold_optimization/evaluate_gold.py --run-dir outputs/c_no_gold_replay
```

教师扩展要求第一阶段已完成，且 gold 审计尚未开始；保留第一阶段 freeze 记录，重新冻结完整候选后才允许评估。最后一条命令读取现有冻结 gold 仅做评估。历史实验已经使用过同一份 gold，因此这些成绩仍属于开发审计，不能声称独立测试通过。

教师扩展有单独的 `teacher_control/monitor.json` 心跳及最长 45 分钟限制；在 `teacher_control/STOP` 创建文件可停止该阶段。实际推理和选参仍在本机进行。

预测需使用本实验入口及保存的完整 bundle：

```sh
.venv/bin/python experiments/no_gold_optimization/inference.py --model outputs/c_no_gold_replay/<candidate>/model.joblib --text 'The cabin attendant was courteous.'
```

状态文件 `monitor.json` 与 `audit_monitor.json` 可直接读取；实际结果在输出目录 README、`gold_audit.json`、逐候选 metrics 和 predictions 文件中。

边界检查：`.venv/bin/python experiments/no_gold_optimization/test_boundaries.py`，覆盖缺失弱标签处理、误判为客观仍计极性错误、重跑保护、冻结源码完整性以及 gold 文件读取拦截。
