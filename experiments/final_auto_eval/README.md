# 自动检索评测与下游消融

> 本报告使用自动代理 qrels 和程序化一致性检查，不是人工独立评测，也不应写成人工 RAGAS 分数。

运行时间：2026-10-19T02:15:39.264833+00:00；总耗时 198.3 秒。

## 检索质量

| 方法 | P@5 | Recall@50 | MRR | NDCG@10 | Success@10 | 平均(s) | 中位(s) | P95(s) | 最大(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.8800 | 0.4033 | 0.9500 | 0.7896 | 1.0000 | 0.547 | 0.534 | 0.640 | 1.278 |
| dense | 0.7533 | 0.3067 | 0.8667 | 0.6864 | 1.0000 | 0.464 | 0.057 | 0.096 | 12.141 |
| hybrid_rrf | 0.8800 | 0.3808 | 0.9611 | 0.8095 | 1.0000 | 1.011 | 0.592 | 0.701 | 13.419 |
| hybrid_rerank | 0.8267 | 0.3865 | 0.9325 | 0.7439 | 1.0000 | 1.452 | 0.714 | 0.837 | 22.980 |

qrels 口径：在相同航司、舱位等条件下直接命中细分项的句子记为相关；同时被离线标签判为所问方面且为主观评价时记为二级相关。该口径可复现，但会偏向显式关键词，不能替代人工语义判断。

### 配对差值与 95% bootstrap 区间

```json
{
  "hybrid_rrf_minus_bm25": {
    "precision_at_5": {
      "mean_delta": 0.0,
      "ci95_low": -0.0733,
      "ci95_high": 0.0667
    },
    "recall_at_50": {
      "mean_delta": -0.0225,
      "ci95_low": -0.0456,
      "ci95_high": 0.0005
    },
    "mrr": {
      "mean_delta": 0.0111,
      "ci95_low": -0.05,
      "ci95_high": 0.0667
    },
    "ndcg_at_10": {
      "mean_delta": 0.0199,
      "ci95_low": -0.0357,
      "ci95_high": 0.0726
    }
  },
  "hybrid_rerank_minus_hybrid_rrf": {
    "precision_at_5": {
      "mean_delta": -0.0533,
      "ci95_low": -0.12,
      "ci95_high": 0.0133
    },
    "recall_at_50": {
      "mean_delta": 0.0057,
      "ci95_low": -0.0108,
      "ci95_high": 0.0222
    },
    "mrr": {
      "mean_delta": -0.0286,
      "ci95_low": -0.0794,
      "ci95_high": 0.0
    },
    "ndcg_at_10": {
      "mean_delta": -0.0656,
      "ci95_low": -0.1141,
      "ci95_high": -0.0193
    }
  }
}
```

## 下游组件消融

```json
{
  "evaluated_airline_subtopics": 18,
  "direction_accuracy": {
    "displayed_top10_vote": 0.6667,
    "full_direct_aggregate": 1.0,
    "displayed_top10_vote_count": 12,
    "full_direct_aggregate_count": 18
  },
  "two_sided_example_coverage_when_available": {
    "naive_rank_order": 0.6471,
    "balanced_selection": 1.0,
    "eligible_count": 17,
    "naive_count": 11,
    "balanced_count": 17
  },
  "obvious_label_text_conflicts_in_selected_examples": {
    "naive": 13,
    "consistency_filter": 0
  },
  "adversarial_wrong_direction_rejection_rate": {
    "citation_only_validation": 0.0,
    "full_aggregate_direction_guard": 1.0,
    "citation_only_rejected_count": 0,
    "direction_guard_rejected_count": 18
  }
}
```

方向准确率以完整直接匹配池为自动参考，因此 full_direct_aggregate 是实现一致性检查，不是独立真实性证明。消融用于量化过去的 top-k 投票错误、正反例截断、明显标签文本冲突和错误方向回答拦截。

## 复现

```bash
.venv/bin/python -u experiments/final_auto_eval/run.py \
  --output-dir outputs/final_auto_eval_replay
```

输入固定为 `data/eval/query_smoke_40_20261019.json`、当前知识库与
`outputs/label_index_20261019/labels.parquet`。脚本拒绝覆盖已有输出目录，使用固定随机种子
`6127` 计算 10,000 次查询级配对 bootstrap 区间。检索评测取前 20 个单航司细分问题和
5 个双航司比较问题，拆成 30 个航司×细分项单元；下游消融用同一批前 25 个问题，只有
18 个航司×细分项具有足够的全量正负标签以形成明确方向，其余保持“正负接近/证据不足”，
不强行纳入方向准确率。

四个检索配置共享同一元数据过滤范围和自动 qrels。Dense 首次加载模型造成最大延迟
12.14 秒，因此均值高于 P95；热态中位数更能表示后续查询。Cross-encoder 在这套显式词
qrels 上令 NDCG@10 相对 Hybrid+RRF 下降 0.0656，查询级 bootstrap 95% 区间为
[-0.1141, -0.0193]。这项结果应如实报告：它说明当前重排器与显式细分相关性口径不完全
一致，不能写成“重排已证明提升”；真实语义收益仍需要独立的人或外部 judge qrels。
