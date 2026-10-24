"""Freeze questions, rubrics, model versions and ranking rules before scoring."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.downstream_evaluation.common import SEED, new_dir, now, read, sha, write


def cases():
    entries = [
        ("single", "en", "What do passengers think of SAS Scandinavian economy class seats?"),
        ("single", "zh", "韩亚航空商务舱的空乘服务怎么样？"),
        ("comparison", "en", "Compare Qatar Airways and Emirates business class seats."),
        ("comparison", "zh", "比较国泰航空和汉莎航空经济舱的餐饮评价。"),
        ("subtopic", "en", "How is meal portion size in Lufthansa economy class?"),
        ("subtopic", "zh", "卡塔尔航空商务舱座椅的隐私怎么样？"),
        ("fact", "en", "KLM Royal Dutch Airlines economy class: what meals were served?"),
        ("fact", "en", "Is seat power available in ANA All Nippon Airways economy class?"),
        ("multi_aspect", "en", "Describe food and cabin crew service in Air France economy class."),
    ]
    result = [{"id": f"A{i:02d}", "category": kind, "language": lang, "question": q,
               "expected_answerable": None} for i, (kind, lang, q) in enumerate(entries, 1)]
    # Select a small slice using metadata counts only, without looking at answers or scores.
    import pandas as pd
    frame = pd.read_parquet(ROOT / "data/processed/chunks.parquet",
                            columns=["review_id", "airline", "cabin", "route_from", "route_to"])
    counts = frame.drop_duplicates("review_id").groupby(
        ["airline", "cabin", "route_from", "route_to"]).size().rename("n").reset_index()
    small = counts.loc[counts.n.between(3, 8)].sort_values(
        ["airline", "cabin", "route_from", "route_to"]).iloc[0]
    filters = {key: str(small[key]) for key in ["airline", "cabin", "route_from", "route_to"]}
    result.append({"id": "A10", "category": "small_slice", "language": "en",
                   "question": f"How were the seats on {small.airline} {small.cabin} flights from "
                               f"{small.route_from} to {small.route_to}?",
                   "structured_filters": filters, "metadata_review_count": int(small.n),
                   "expected_answerable": None})
    for ident, lang, airline, question in [
        ("A11", "en", "Singapore Airlines", "How was Singapore Airlines economy class food in 2030?"),
        ("A12", "zh", "United Airlines", "2030年联合航空经济舱的座位怎么样？"),
    ]:
        result.append({"id": ident, "category": "no_answer", "language": lang, "question": question,
                       "structured_filters": {"airline": airline, "cabin": "Economy Class",
                                              "date": ["2030-01-01", "2030-12-31"]},
                       "expected_answerable": False})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dest = new_dir(args.output_dir)
    source_files = [*ROOT.joinpath("experiments/downstream_evaluation").glob("*.py"),
                    ROOT / "scripts/linux_rag.py", ROOT / "scripts/local_qa_server.py",
                    ROOT / "scripts/transformers_qa_runtime.py", ROOT / "experiments/final_rag_eval/run.py",
                    *ROOT.joinpath("src").glob("*.py")]
    inputs = ["data/processed/chunks.parquet", "data/processed/reviews.parquet",
              "data/eval/to_label.csv", "data/eval/to_label_meta.csv",
              "data/eval/query_smoke_40_20261019.json", "outputs/minimal_rag_models.json",
              "index/faiss.index", "index/bm25.pkl", "index/chunk_ids.npy", "index/chunk_texts.npy",
              "experiments/final_rag_eval/results_20261019/answers_slim.jsonl"]
    protocol = {
        "version": "c-yi-q5-v1", "created_at": now(), "seed": SEED,
        "evaluation_status": "frozen_before_new_scores; human_review_pending",
        "question_independence": "newly_authored_frozen_cases; same_corpus; not_blind_independent_test",
        "development_cases": {"source": "experiments/final_rag_eval/results_20261019/answers_slim.jsonl",
                              "count": 20, "use": "human calibration and development regression only"},
        "cases": cases(), "models": read(ROOT / "outputs/minimal_rag_models.json"),
        "generation": {"temperature": 0, "max_tokens": 480, "timeout": 90,
                       "top_k_per_airline_or_aspect": 6, "classifier_labels": None},
        "variants": {
            "full": "production answer_question, validation, repair, program fallback",
            "no_repair": "same actual initial generation; use program fallback if initial validation fails",
            "no_guard": "same actual initial generation; keep it without validation or repair; offline only",
        },
        "ablation_scope": "paired policy replay of shared real model generations; no template model substitutes; "
                          "same retrieved evidence and initial model tokens; retrieval/label components held fixed",
        "rubric": {
            "answerable": "human judges whether supplied evidence can answer the exact question and constraints",
            "claims": "split answer into independently checkable claims; supported only if cited original text "
                      "or separately supplied verified aggregates entail them; citations alone are insufficient",
            "faithfulness": "supported_claims / total_checkable_claims; null if no checkable claims",
            "answer_relevance": "0 off-topic, 1 barely related, 2 partial, 3 mostly complete, 4 complete; divide by 4",
            "context_precision": "human relevant displayed evidence / all displayed evidence",
            "abstention": "when corpus slice empty, explicitly say no historical evidence; no fabricated facts",
            "reviewer": "human ID, evidence judgments and claim annotations required; all blanks stay missing",
            "automatic_judge": "same_qwen4b_proxy; scored separately; never independent/human/RAGAS-library score",
        },
        "ranking": {"split": "SHA256(seed:review_id) first 8 hex / 2**32 < .20 => heldout",
                    "test_fraction": .20, "dev_fraction_within_remaining": .20,
                    "exclude_gold": "all original 1100 sentences, parents and normalized texts",
                    "duplicate_policy": "remove every test-normalized-text match from train/dev before labels",
                    "classifier": "reuse C甲 build_labels, split_labels and fit_models; separate diagnostic model",
                    "max_features": 20000, "max_iter": 1000,
                    "prediction_input": ["chunk_id", "review_id", "text", "airline", "cabin"],
                    "sentence_selection": "quality-filtered predicted seat + subjective sentences",
                    "review_score": "mean(binary positive) among selected sentences per review",
                    "airline_score": "mean(review_score); one parent review one vote",
                    "reference": "heldout economy parents mean seat_comfort; scorer reads after predictions frozen",
                    "min_reference_reviews": 30, "min_predicted_reviews": 15,
                    "ties": "average ranks", "bootstrap": "2000 within-airline paired parent-review resamples",
                    "scope": "heldout text-to-label-to-airline ranking branch; not current UI rating-based ranking",
                    "no_hyperparameter_selection": True},
        "bootstrap_replicates": 2000,
        "input_hashes": {p: sha(ROOT / p) for p in inputs},
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in source_files},
    }
    write(dest / "protocol.json", protocol)
    write(dest / "protocol.sha256.json", {"sha256": sha(dest / "protocol.json")})
    print(dest / "protocol.json", flush=True)


if __name__ == "__main__":
    main()
