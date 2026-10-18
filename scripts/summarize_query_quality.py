"""Generate the local experiment report from finished audits, without changing models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path,required=True)
    parser.add_argument('--paired-dir',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir
    report=json.loads((out/'audit.json').read_text())
    checks=json.loads((args.baseline_dir/'structural_checks.json').read_text())
    paired=json.loads((args.paired_dir/'progress.json').read_text())
    if paired['status']!='completed_pending_judgments':raise RuntimeError('Wait for paired query scoring to finish')
    selected=report['selected_without_gold'];old=report['previous_retained_metrics'];new=report['selected_metrics']
    names={'subjective_aspect_f1':'主观句上的方面 macro-F1','subjective_polarity_f1':'主观句上的正负极性 macro-F1',
           'subjective_joint_accuracy':'方面和正负同时正确率','subjective_recall':'主观评价保留率'}
    lines=['# 查询质量优化与端到端试跑','',
        f'本轮共 {len(report["candidates"])} 个冻结配置；非 gold 选择锁定 `{selected}`。',
        'gold 标签未进入本轮权重拟合、提示、参数或模型选择。候选全部冻结后，独立进程才读取 gold 进行对照。',
        '使用了 80 条新真实开发句的 AI 辅助 silver 标签，不是人工 gold；模糊字段留空。现有项目 gold 有历史开发曝光，仍不是独立测试。','',
        '## 分类质量','',f'统一在现有 {report["gold_n"]:,} 条 gold 中的 {new["subjective_n"]:,} 条主观句上统计；误判为客观仍计入错误。','',
        '| 指标 | 目标 | 上轮保留模型 | 本轮预先选中模型 |','|---|---:|---:|---:|']
    for key,name in names.items():lines.append(f'| {name} | {report["targets"][key]:.1%} | {old[key]:.1%} | {new[key]:.1%} |')
    lines.extend(['',f'选中候选通过全部分类点估计目标：**{new["point_targets_passed"]}**。',
        f'联合正确率按父评论分组的 bootstrap 95% 区间：{report["selected_joint_accuracy_95"][0]:.1%}–{report["selected_joint_accuracy_95"][1]:.1%}。',
        f'相对旧模型的联合正确率差值区间：{report["selected_joint_delta_vs_old_95"][0]*100:+.1f} 至 {report["selected_joint_delta_vs_old_95"][1]*100:+.1f} 个百分点。',
        '这些区间未计入历史选参、标签不确定性及未来分布变化。','',
        '## 错误方向与使用边界','',
        f'主观句被判成客观的数量：旧模型 {old["subjective_false_objective_n"]} 条，预选候选 {new["subjective_false_objective_n"]} 条。',
        f'预选候选的客观 precision 为 {new["objective_precision"]:.1%}，客观 recall 为 {new["objective_recall"]:.1%}。不能用客观召回的提高掩盖主观评价被误删。',
        '查询链路保留所有已检索证据，只允许标签调整顺序；这能避免标签硬过滤造成的丢失，但不代表标签语义已正确。','',
        '| 方面 | 主观 gold 数 | 旧模型 F1 | 预选候选 F1 |','|---|---:|---:|---:|'])
    for aspect, part in new['subjective_aspect_per_class'].items():
        before=old['subjective_aspect_per_class'][aspect]['f1']
        lines.append(f'| {aspect} | {part["support"]} | {before:.1%} | {part["f1"]:.1%} |')
    lines.extend(['',
        '## 全部冻结候选的事后对照','',
        '| 候选 | 方面 F1 | 极性 F1 | 联合正确率 | 主观召回 | 四项目标全过 |',
        '|---|---:|---:|---:|---:|---|'])
    for name,m in report['candidates'].items():
        lines.append(f'| {name} | {m["subjective_aspect_f1"]:.1%} | {m["subjective_polarity_f1"]:.1%} | {m["subjective_joint_accuracy"]:.1%} | {m["subjective_recall"]:.1%} | {m["point_targets_passed"]} |')
    lines.extend(['','## 查询系统','',
        '- 支持方面、航司、舱位、航线和评论日期条件；支持 2–3 家航司的同条件证据对照。',
        '- 先检索、精排和去重，形成最多 10 条证据池；可选分类器只调整池内排序，不因客观标签直接删除证据。',
        '- 引用保留原文句子 ID，可展开完整父评论。回答是证据摘要与检索样本统计，不推断总体航司好评率。',
        '- 当前补充自由文本按英文关键词使用，尚未实现任意中文问题解析和自由生成的长篇结论。','',
        f'已执行 {checks["queries"]} 个检索用例，含 5 组双航司比较的独立组件。来源引用、元数据筛选、重复句与父评论数量检查错误数：{len(checks["metadata_citation_duplicate_errors"])}。',
        f'明确无语料的元数据条件：{checks["correct_empty_evidence_responses"]}/{checks["known_metadata_no_answer_cases"]} 正确返回证据不足。',
        f'非空基线查询耗时中位数 {checks["nonempty_query_latency_median_seconds"]:.2f} 秒，P95 {checks["nonempty_query_latency_p95_seconds"]:.2f} 秒；不包括本地分类器耗时。',
        f'标签配对试跑复用 {paired["unique_sentences"]} 条不同证据，{paired["queries_with_changed_top5"]}/{paired["queries"]} 个查询的前五条排序改变。排序改变不等于质量提高。',
        '前五条相关性、回答成功率和论断证据支持率仍待人工判定；缺少判定时评测器不会通过验收。',
        '同一 top-10 池内排序不会提高 Success@10；比较关注 P@5 和回答质量。','',
        '## 文件','',
        '- `frozen_candidates.json`：选择、源码和权重的冻结记录。',
        '- `audit.json`：全部分类结果、分组区间与核验。',
        '- `monitor.json` 与各 `*_control/monitor.json`：监控和阶段状态。',
        f'- `{args.paired_dir / "judgments_to_review.jsonl"}`：基线/标签两种模式的待评审记录。',
        '- `experiments/query_quality/README.md`：代码、依赖、重建与复跑说明。','',
        '原候选和全量索引保留。新模型仅用于实验预览，未因本次 gold 对照自动晋升。'])
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(str(out/'README.md'))


if __name__=='__main__':main()
