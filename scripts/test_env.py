"""环境自检。装完依赖后跑一次，确认全部就绪。

用法：python scripts/test_env.py
"""
import sys


def main() -> None:
    print("=== 依赖版本 ===")
    import pandas, torch, faiss, spacy, sklearn, transformers, streamlit
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer   # noqa: F401

    print(f"  pandas        {pandas.__version__}")
    print(f"  torch         {torch.__version__}")
    print(f"  scikit-learn  {sklearn.__version__}")
    print(f"  transformers  {transformers.__version__}")
    print(f"  streamlit     {streamlit.__version__}")
    print(f"  faiss         OK (v{faiss.__version__})")

    print("\n=== 切句（spaCy）===")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("The seat was comfortable. But the food was terrible.")
    sents = [s.text for s in doc.sents]
    print(f"  {sents}")
    assert len(sents) == 2, "❌ 切句结果不对，检查 spacy 模型是否装好"

    print("\n=== 关键词检索（BM25）===")
    # 注意：文档数太少时 IDF 会算成 0，所以这里用 5 篇文档才看得出效果
    corpus = [
        ["the", "seat", "had", "plenty", "of", "legroom"],
        ["the", "food", "was", "cold", "and", "tasteless"],
        ["cabin", "crew", "were", "friendly"],
        ["boarding", "was", "slow"],
        ["the", "screen", "was", "broken"],
    ]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(["legroom"])
    top = int(scores.argmax())
    print(f"  查询 'legroom' -> 命中第 {top} 篇，得分 {scores[top]:.3f}")
    assert top == 0 and scores[0] > 0, "❌ BM25 没有正确命中含 legroom 的文档"

    print("\n=== 向量检索（FAISS）===")
    import numpy as np
    vecs = np.random.rand(10, 8).astype("float32")
    # ⚠️ 用内积(IndexFlatIP)检索必须先归一化，否则模长大的向量会排前面，
    #    而不是真正最相似的。retriever.py 里同样要注意这点。
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(8)
    index.add(vecs)
    sims, idx = index.search(vecs[:1], 3)
    print(f"  索引 10 条向量，查最相似的 3 条 -> {idx[0].tolist()}")
    print(f"  自己和自己的相似度 = {sims[0][0]:.4f}（应接近 1.0）")
    assert idx[0][0] == 0, "❌ FAISS 检索结果异常"

    print("\n全部就绪 ✅")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"\n❌ 缺少依赖：{e}")
        print("   请先跑：pip install -r requirements.txt")
        sys.exit(1)
    except AssertionError as e:
        print(f"\n{e}")
        sys.exit(1)
