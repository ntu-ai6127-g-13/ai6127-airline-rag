"""各环节耗时实测，用于估算工时和排期。

用法：python scripts/benchmark.py --n 500
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_RAW, COL_TEXT, EMBED_MODEL
from src.clean_text import clean_review

FULL_REVIEWS = 61_172          # 去重后的真实语料规模


class Timer:
    def __init__(self, name): self.name = name
    def __enter__(self):
        self.t = time.perf_counter(); return self
    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    n = args.n
    scale = FULL_REVIEWS / n

    print(f"基准测试：{n} 条评论，换算系数 ×{scale:.1f}\n")
    results = []

    # ---- 1. 读取 ----
    with Timer("读取") as t:
        path = next(f for f in DATA_RAW.iterdir() if f.suffix == ".xlsx")
        df = pd.read_excel(path)
    print(f"读取整个 xlsx        {t.elapsed:6.1f}s  (一次性，不随规模变)")
    read_time = t.elapsed

    df = df[df[COL_TEXT].notna()].drop_duplicates(subset=[COL_TEXT])
    sample = df.sample(n, random_state=42)

    # ---- 2. 清洗 ----
    with Timer("清洗") as t:
        texts = sample[COL_TEXT].astype(str).map(clean_review).tolist()
    results.append(("文本清洗", t.elapsed, t.elapsed * scale))

    # ---- 3. 切句 ----
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])
    with Timer("切句") as t:
        sents = []
        for doc in nlp.pipe(texts, batch_size=100):
            sents += [s.text.strip() for s in doc.sents
                      if len(s.text.split()) >= 3]
    results.append(("spaCy 切句", t.elapsed, t.elapsed * scale))
    sents_per_review = len(sents) / n
    full_sents = int(FULL_REVIEWS * sents_per_review)
    print(f"切出 {len(sents):,} 句（每条 {sents_per_review:.1f} 句）"
          f" → 全量约 {full_sents:,} 句\n")

    # ---- 4. BM25 建索引 ----
    from rank_bm25 import BM25Okapi
    with Timer("BM25") as t:
        bm25 = BM25Okapi([s.lower().split() for s in sents])
    results.append(("BM25 建索引", t.elapsed, t.elapsed * scale))

    with Timer("BM25查询") as t:
        for _ in range(20):
            bm25.get_scores(["legroom", "seat"])
    q_bm25 = t.elapsed / 20

    # ---- 5. 向量编码（最耗时的一步）----
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    with Timer("编码") as t:
        emb = model.encode(sents, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False)
    enc_per_sent = t.elapsed / len(sents)
    results.append(("向量编码", t.elapsed, enc_per_sent * full_sents))

    # ---- 6. FAISS ----
    import faiss, numpy as np
    with Timer("FAISS") as t:
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb.astype("float32"))
    results.append(("FAISS 建索引", t.elapsed, t.elapsed / len(sents) * full_sents))

    qv = model.encode(["legroom in economy"], normalize_embeddings=True)
    with Timer("向量查询") as t:
        for _ in range(20):
            index.search(qv.astype("float32"), 50)
    q_dense = t.elapsed / 20

    # ---- 汇总 ----
    print("=" * 62)
    print(f"{'环节':<16}{'本次':>10}{'换算到全量':>16}")
    print("-" * 62)
    for name, actual, full in results:
        fs = f"{full:.0f}s" if full < 90 else f"{full/60:.0f} 分钟"
        print(f"{name:<16}{actual:>9.1f}s{fs:>16}")
    total_full = read_time + sum(f for _, _, f in results)
    print("-" * 62)
    print(f"{'离线建库合计':<16}{'':>10}{total_full/60:>13.0f} 分钟")
    print("=" * 62)
    print(f"\n单次查询延迟：BM25 {q_bm25*1000:.0f}ms  ·  向量 {q_dense*1000:.1f}ms")
    print(f"内存估算：全量向量 {full_sents * emb.shape[1] * 4 / 1024**3:.2f} GB "
          f"({full_sents:,} × {emb.shape[1]} 维 × 4 字节)")


if __name__ == "__main__":
    main()
