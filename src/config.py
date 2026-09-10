"""全项目共用的路径和参数。改配置只改这里，不要在各处硬编码。

数据事实（从 data/raw/capstone_airline_reviews3.xlsx 实测，2026-09-07）：
  原始行数        131,895   ← 其中 65,948 行是全空的填充行
  有正文的评论     65,947
  去重后           61,172   ← 这是真正的语料规模
  总词数        8,402,547   （平均每条 137 词）
  航司数               81
  时间范围    2002-08 ~ 2019-06   ← 注意：数据到 2019 年为止
  预计切句后    约 566,000 句
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------- 路径 ----------------
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_EVAL = ROOT / "data" / "eval"
INDEX_DIR = ROOT / "index"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

RAW_XLSX = DATA_RAW / "capstone_airline_reviews3.xlsx"
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

# ---------------- 原始文件的真实列名 ----------------
# ⚠️ 这些是实测的列名，不要凭印象改
COL_AIRLINE = "airline"
COL_OVERALL = "overall"            # 1–10，不叫 overall_rating
COL_AUTHOR = "author"
COL_DATE = "review_date"           # 不叫 date
COL_TEXT = "customer_review"       # 不叫 review_text
COL_AIRCRAFT = "aircraft"          # 有值 30.2%（不是之前写的 15%），2,088 种机型
COL_TRAVELLER = "traveller_type"   # Solo/Couple/Family Leisure、Business
COL_CABIN = "cabin"                # Economy Class / Business Class / ...
COL_ROUTE = "route"
COL_DATE_FLOWN = "date_flown"      # 有值 61.2%，**只有月精度**（原始值都是每月 1 号）
COL_RECOMMENDED = "recommended"

# ---------------- 模型 ----------------
EMBED_MODEL = "all-MiniLM-L6-v2"                      # 小而快，CPU 可跑
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CLS_MODEL = "distilbert-base-uncased"

# ---------------- 检索参数 ----------------
TOP_K_CANDIDATES = 100    # 粗筛保留多少条
TOP_K_FINAL = 20          # 最终返回多少条
RRF_K = 60                # RRF 公式里的常数，论文经验值

# ---------------- 六个评价维度 ----------------
# ⚠️ 实际数据只有 6 个维度评分，没有 wifi。不要按 7 个写。
# 方面 -> (原始文件里的评分列名, 判断句子是否在讲该方面的关键词)
ASPECTS = {
    "seat": ("seat_comfort", [
        "seat", "legroom", "leg room", "pitch", "recline",
        "cramped", "comfortable", "uncomfortable", "space"]),
    "staff": ("cabin_service", [
        "crew", "staff", "attendant", "service", "stewardess",
        "friendly", "rude", "helpful", "cabin service"]),
    "food": ("food_bev", [
        "food", "meal", "drink", "breakfast", "dinner",
        "snack", "catering", "beverage", "lunch"]),
    "entertainment": ("entertainment", [
        "entertainment", "movie", "screen", "ife", "film",
        "tv", "headphone", "audio", "wifi"]),
    "ground": ("ground_service", [
        "check-in", "checkin", "check in", "boarding", "gate",
        "luggage", "baggage", "ground", "counter", "queue"]),
    "value": ("value_for_money", [
        "value", "price", "expensive", "cheap", "worth",
        "money", "fare", "cost"]),
}

RATING_COLS = [cfg[0] for cfg in ASPECTS.values()]

# 各维度有评分的评论数（实测，决定弱标签规模）
# seat 56,202 ｜ staff 56,231 ｜ food 48,332
# entertainment 40,222 ｜ ground 37,165 ｜ value 59,317
