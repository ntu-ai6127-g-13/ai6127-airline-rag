"""Build local indexes or run one real retrieval-to-answer check on Linux."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def configure_models():
    """Select downloaded, revision-pinned models before importing retrieval."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    manifest = json.loads((ROOT / "outputs/minimal_rag_models.json").read_text())
    from src import config
    config.EMBED_MODEL = manifest["embedding"]["path"]
    config.RERANK_MODEL = manifest["reranker"]["path"]
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build-indexes", "probe", "smoke"])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    manifest = configure_models()
    if args.action == "build-indexes":
        from src.config import BM25_INDEX, FAISS_INDEX, CHUNK_IDS, CHUNK_TEXTS
        from src.retriever import build_bm25, build_faiss
        if not BM25_INDEX.exists():
            print("BM25:", build_bm25(), flush=True)
        if all(path.exists() for path in (FAISS_INDEX, CHUNK_IDS, CHUNK_TEXTS)):
            print("Dense index already exists; preserving it", flush=True)
        elif any(path.exists() for path in (FAISS_INDEX, CHUNK_IDS, CHUNK_TEXTS)):
            raise RuntimeError("Partial dense index exists; inspect it before rebuilding")
        else:
            print("Dense:", build_faiss(batch_size=args.batch_size), flush=True)
        return

    from src.query_service import query_reviews
    from src.grounded_qa import answer_question, model_health
    if model_health().get("status") != "ready":
        raise RuntimeError("Start local_qa_server.py --backend transformers first")
    if args.action == "probe":
        from src.grounded_qa import _generate, DEFAULT_ENDPOINT
        messages = [{"role": "user", "content": "Reply with exactly: ready"}]
        events = []
        direct = _generate(messages, endpoint=DEFAULT_ENDPOINT, timeout=90, max_tokens=32)
        streamed = _generate(messages, endpoint=DEFAULT_ENDPOINT, timeout=90, max_tokens=32,
                             on_stream=lambda text, event: events.append(event))
        if direct["text"] != streamed["text"] or not events or not direct["text"]:
            raise RuntimeError("Synchronous and streaming generation did not agree")
        report = {"health": model_health(), "direct": direct, "streamed": streamed,
                  "delta_events": len(events)}
        (ROOT / "outputs/minimal_rag_server_probe.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    question = "How do passengers describe Singapore Airlines economy class seats?"
    print("Retrieving real reviews...", flush=True)
    result = query_reviews("seat comfort legroom", aspect="seat", top_k=5,
                           filters={"airline": "Singapore Airlines", "cabin": "Economy Class"})
    result["language"] = "en"
    print(f"Retrieved {len(result['evidence'])} evidence passages; generating...", flush=True)
    stream_events = []
    answer = answer_question(question, result,
                             on_stream=lambda text, event: stream_events.append(event))
    # Check the observed model mode as well as the program's citation validation.
    if not result["evidence"] or not answer["valid"] or not answer["citations"]:
        raise RuntimeError("The smoke query did not return valid cited evidence")
    report = {"question": question, "models": manifest, "health": model_health(),
              "classifier_used": result["classifier_used"],
              "retrieval_seconds": result["retrieval_seconds"],
              "evidence": result["evidence"], "answer": answer,
              "stream_events": stream_events}
    target = ROOT / "outputs" / ("minimal_rag_smoke_" +
             datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps(answer, ensure_ascii=False, indent=2), flush=True)
    print("Saved:", target, flush=True)


if __name__ == "__main__":
    main()
