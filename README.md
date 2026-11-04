# Ask the Reviews — grounded question answering over airline reviews

AI6127 Deep Neural Networks for NLP, Group 13 (NTU, 2026).

A retrieval-augmented system that answers constrained, attribute-level questions about airline service, such as
*"Flying Singapore to London in economy, which airline has the best legroom?"*, from 61,167 Skytrax passenger
reviews (546,139 sentences). It turns route, cabin, traveller type, date, aspect and one of 46 sub-attributes into
hard filters; retrieves evidence with BM25 and MiniLM dense search fused by reciprocal rank fusion; counts positive
and negative opinions over every matching review with an offline sentence classifier; and lets a local Qwen3-4B
model write a cited answer that program checks accept only if its citations, numbers and direction match the
evidence. Questions can be asked in English or Chinese.

- **Report:** [`report/final/13.pdf`](report/final/13.pdf) (Chinese reading copy: `report/final/13_zh.pdf`)
- **Data archive:** `ai6127-g13-data.zip` on the [v1.0-submission release](https://github.com/ntu-ai6127-g-13/ai6127-airline-rag/releases/tag/v1.0-submission)
- Team-internal Chinese documentation: [`README_zh.md`](README_zh.md) and [`docs/`](docs/)

## 1. Requirements

- Python 3.11 (3.10+ works for the core system).
- macOS or Linux; about 3 GB of free disk space.
- Optional, for generated answers: Apple Silicon (MLX backend) **or** an NVIDIA/AMD GPU (Transformers backend).
  Without a local model the interface still works and shows program-written, evidence-cited answers.

## 2. Install

```bash
git clone https://github.com/ntu-ai6127-g-13/ai6127-airline-rag.git
cd ai6127-airline-rag
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/python scripts/test_env.py          # checks dependencies, spaCy, BM25 and FAISS
```

## 3. Get the data

**Fast path (recommended).** Download `ai6127-g13-data.zip` from the release and unzip it in the repository root.
It contains the knowledge base, the evaluation set, the label indexes and every result file used in the report
(see `DATA_README.md` inside the archive). Then build the two retrieval indexes from the knowledge base:

```bash
unzip -n ai6127-g13-data.zip                    # in the repository root; -n keeps the tracked copies of data/eval/
.venv/bin/python -m src.retriever build-bm25    # ~6 s
.venv/bin/python -m src.retriever build-dense --batch-size 64   # ~5 min on Apple M1 Pro (encodes 529k sentences)
```

**Full rebuild from the raw data.** Downloads the public Kaggle dataset (no account needed), cleans it and splits
it into sentences (about 14 minutes), then builds the indexes and the v4.1 label index:

```bash
.venv/bin/python scripts/download_data.py
.venv/bin/python -m src.build_kb
.venv/bin/python -m src.retriever build-bm25
.venv/bin/python -m src.retriever build-dense --batch-size 64
.venv/bin/python -m src.label_index --model outputs/weak_rebuild_20261019/model.joblib \
    --output-dir outputs/label_index_replay
```

The label index needs the classifier in `outputs/weak_rebuild_20261019/model.joblib` and the interface needs
`data/processed/reviews.parquet` for its rating summary; both are in the data archive. The interface reads the label
index from `outputs/label_index_20261019`, also in the archive.

## 4. Run the interface

```bash
.venv/bin/python scripts/run_query_app.py --port 8517
```

Open http://localhost:8517. Type a question (English or Chinese) or set airline, comparison airlines, cabin, aspect,
route and date in the sidebar. The page shows the answer with numbered citations, the positive/negative overview over
all matching reviews, the airline's own ratings with a significance check, and the evidence per airline.
Example questions:

- `How is the legroom in Singapore Airlines economy class?`
- `Flying Singapore to London in economy, which airline has the best legroom?`
- `新加坡航空经济舱的餐食、性价比和地面服务分别怎么样？`
- `Which has better legroom in economy, Cathay Pacific or Singapore Airlines?` or
  `国泰航空和新加坡航空经济舱腿部空间哪家评价更好？` (the sidebar comparison also works)

### Optional: local answer generation

**Apple Silicon (MLX).** Once, create the pinned model runtime and download Qwen3-4B-Instruct-2507 (4-bit); this
setup script is for Apple Silicon only:

```bash
.venv/bin/python scripts/setup_query_model.py
```

Then keep the model server running in a second terminal while using the app:

```bash
outputs/query_quality_runtime/venv311/bin/python scripts/local_qa_server.py      # 127.0.0.1:8765
```

**Linux GPU (Transformers).** See [`docs/Linux最小RAG链路.md`](docs/Linux最小RAG链路.md):

```bash
.venv/bin/python scripts/setup_rag_models.py
.venv/bin/python scripts/local_qa_server.py --backend transformers --device cuda
```

The app detects the server automatically; if it is not running, answers fall back to the program-written mode.

## 5. Tests

```bash
.venv/bin/pip install pytest -r annotation/requirements.txt   # the annotation tool's tests need bcrypt etc.
.venv/bin/python -m pytest -q                                   # needs the data archive unzipped
```

## 6. Reproducing the report

Experiment scripts never overwrite earlier results: most refuse an existing output directory and the long-running
ones resume from checkpoints, so replay into a new directory name. Commands are in Appendix A of the report.

| Report | What | Entry point | Results (data archive) |
|---|---|---|---|
| Q1, §3 | Cleaning, chunking, corpus statistics | `src/clean_text.py`, `src/build_kb.py` | `data/processed/chunks.parquet` |
| Q2/Q3, E1 | Retrieval benchmark (P@5, Recall@50, MRR, NDCG@10) | `experiments/final_auto_eval/run.py` | `outputs/final_auto_eval_20261019_v2/` |
| Q2/Q4, E2, E5 | Query latency, RAG metrics on 20 questions | `experiments/final_rag_eval/run.py` | `outputs/final_rag_eval_20261019/` |
| Q2, E2 | Retrieval components timed alone | `experiments/final_auto_eval/component_latency.py` | `outputs/component_latency_20261104/` |
| Q3, E1b | Retrieval innovations added one at a time (LLM-judged relevance; needs the model server) | `experiments/final_auto_eval/retrieval_ablation.py` | `outputs/retrieval_ablation_20261104/` |
| Q4, E3 | Classifier quality on the gold set | `experiments/distilled_label_index/evaluate_gold.py` | `outputs/distill_gold_audit_20261029/` |
| Q5, E4 | LLM-to-student distillation | `experiments/distilled_label_index/` | `outputs/distill_*_20261029/`, `outputs/label_index_20261029_distilled/` |
| Q5, E6 | Factorial ablation of X/Y/Z, targeted checks | `experiments/downstream_evaluation/`, `experiments/final_auto_eval/run.py` | see release notes |
| Q5, E7 | Airline ranking vs. held-out ratings | `python -m experiments.downstream_evaluation.ranking` | see release notes |

## 7. Repository layout

```
app/streamlit_app.py        query interface (Q2)
src/                        cleaning, knowledge base, retrieval, filters, classification, grounded QA
scripts/                    data download, app launcher, local model server, setup helpers
experiments/                evaluation, distillation and ablation scripts
data/eval/                  eval.xls and the frozen 1,000-sentence gold set
tests/                      unit tests
report/final/               report source (LaTeX) and PDFs
docs/                       team documentation (Chinese)
```

Data: Skytrax reviews from the public Kaggle dataset
[`efehandanisman/skytrax-airline-reviews`](https://www.kaggle.com/datasets/efehandanisman/skytrax-airline-reviews).
Reviews end in mid-2019; answers describe historical experiences, not current service.
