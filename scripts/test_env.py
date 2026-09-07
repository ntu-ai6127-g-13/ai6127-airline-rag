"""环境自检。装完依赖后跑一次，确认全部就绪。"""

def main():
    import pandas, torch, faiss, spacy
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer   # noqa: F401

    print(f"pandas  {pandas.__version__}")
    print(f"torch   {torch.__version__}")
    print(f"faiss   OK")

    nlp = spacy.load("en_core_web_sm")
    doc = nlp("The seat was comfortable. But the food was terrible.")
    sents = [s.text for s in doc.sents]
    print(f"切句测试: {sents}")
    assert len(sents) == 2, "切句结果不对，检查 spacy 模型"

    bm25 = BM25Okapi([["seat", "legroom"], ["food", "meal"]])
    scores = bm25.get_scores(["legroom"])
    print(f"BM25 测试: {scores.round(3).tolist()}")

    print("\n全部就绪 ✅")


if __name__ == "__main__":
    main()
