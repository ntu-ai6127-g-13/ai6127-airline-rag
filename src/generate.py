"""Evidence-first extractive answers. Counts describe this retrieved sample only."""
from __future__ import annotations


def evidence_answer(evidence: list[dict], *, aspect: str | None = None, min_reviews: int = 3) -> dict:
    review_ids = {item.get("review_id", item["chunk_id"]) for item in evidence}
    citations = [{"number": i+1, "chunk_id": item["chunk_id"], "quote": item["text"]}
                 for i,item in enumerate(evidence)]
    if len(review_ids) < min_reviews:
        text = f"当前检索仅找到 {len(review_ids)} 条不同评论的证据，材料不足以给出可靠的好坏判断。"
        return {"status": "insufficient_evidence", "text": text, "citations": citations,
                "review_count": len(review_ids), "sentiment_counts": None}
    labeled = [item for item in evidence if item.get("pred_subjectivity") == "subjective"
               and (aspect is None or item.get("pred_aspect") == aspect)]
    counts = {label: sum(item.get("pred_polarity") == label for item in labeled) for label in ["positive","negative"]}
    text = f"本次检索返回 {len(evidence)} 条原文片段，来自 {len(review_ids)} 条不同评论。"
    if labeled:
        text += f"模型将其中与所选方面匹配的 {len(labeled)} 条评价标为：正向 {counts['positive']} 条、负向 {counts['negative']} 条。"
    text += "以下原文可逐条核对；检索样本的数量不代表全航司的好评率或排名。"
    return {"status": "evidence_available", "text": text, "citations": citations,
            "review_count": len(review_ids), "sentiment_counts": counts if labeled else None}
