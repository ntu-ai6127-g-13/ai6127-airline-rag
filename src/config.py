"""全项目共用的路径和参数。改配置只改这里，不要在各处硬编码。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------- 路径 ----------------
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_EVAL = ROOT / "data" / "eval"
INDEX_DIR = ROOT / "index"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

RAW_CSV = DATA_RAW / "skytrax.csv"
REVIEWS = DATA_PROC / "reviews.parquet"
CHUNKS = DATA_PROC / "chunks.parquet"
CHUNKS_SAMPLE = DATA_PROC / "chunks_sample.parquet"   # 开发用假数据
WEAK_LABELS = DATA_PROC / "weak_labels.parquet"
EVAL_XLS = DATA_EVAL / "eval.xls"                     # 作业指定的文件名

BM25_INDEX = INDEX_DIR / "bm25.pkl"
FAISS_INDEX = INDEX_DIR / "faiss.index"
CHUNK_IDS = INDEX_DIR / "chunk_ids.npy"

for _d in (DATA_RAW, DATA_PROC, DATA_EVAL, INDEX_DIR, MODEL_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------- 模型 ----------------
EMBED_MODEL = "all-MiniLM-L6-v2"                      # 小而快，CPU 可跑
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CLS_MODEL = "distilbert-base-uncased"

# ---------------- 检索参数 ----------------
TOP_K_CANDIDATES = 100    # 粗筛保留多少条
TOP_K_FINAL = 20          # 最终返回多少条
RRF_K = 60                # RRF 公式里的常数，论文经验值

# ---------------- 七个评价维度 ----------------
# 方面 -> (对应的评分列, 判断句子是否在讲该方面的关键词)
ASPECTS = {
    "seat": ("r_seat", ["seat", "legroom", "leg room", "pitch",
                        "recline", "cramped", "comfortable"]),
    "staff": ("r_staff", ["crew", "staff", "attendant", "service",
                          "stewardess", "friendly", "rude"]),
    "food": ("r_food", ["food", "meal", "drink", "breakfast",
                        "dinner", "snack", "catering"]),
    "entertainment": ("r_entertainment", ["entertainment", "movie", "screen",
                                          "ife", "film", "tv"]),
    "ground": ("r_ground", ["check-in", "checkin", "boarding", "gate",
                            "luggage", "baggage", "ground"]),
    "wifi": ("r_wifi", ["wifi", "wi-fi", "internet", "connectivity"]),
    "value": ("r_value", ["value", "price", "expensive", "cheap",
                          "worth", "money"]),
}

RATING_COLS = [cfg[0] for cfg in ASPECTS.values()]
