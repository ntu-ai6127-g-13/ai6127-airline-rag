"""Review-level rating aggregates aligned with the query filters."""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path

import pandas as pd

from src.config import CHUNKS, REVIEWS


RATING_FIELDS = {
    "seat": ("seat_comfort", 5),
    "staff": ("cabin_service", 5),
    "food": ("food_bev", 5),
    "entertainment": ("entertainment", 5),
    "ground": ("ground_service", 5),
    "value": ("value_for_money", 5),
    "overall": ("overall", 10),
}
_FILTER_FIELDS = {"airline", "cabin", "traveller_type", "route_from", "route_to", "date"}


@lru_cache(maxsize=2)
def _load_review_ratings_cached(review_path: str, review_mtime: int, review_size: int,
                                chunks_path: str, chunks_mtime: int, chunks_size: int) -> pd.DataFrame:
    del review_mtime, review_size, chunks_mtime, chunks_size
    rating_columns = [field for field, _ in RATING_FIELDS.values()]
    reviews = pd.read_parquet(review_path, columns=["review_id", "airline", "cabin", "traveller_type",
        "review_date", *rating_columns]).rename(columns={"review_date": "date"})
    routes = pd.read_parquet(chunks_path, columns=["review_id", "route_from", "route_to"])
    routes = routes.drop_duplicates("review_id", keep="first")
    return reviews.merge(routes, on="review_id", how="left", validate="one_to_one")


def _load_review_ratings(reviews_path: str | Path, chunks_path: str | Path) -> pd.DataFrame:
    reviews_path, chunks_path = Path(reviews_path), Path(chunks_path)
    review_stat, chunks_stat = reviews_path.stat(), chunks_path.stat()
    return _load_review_ratings_cached(str(reviews_path.resolve()), review_stat.st_mtime_ns, review_stat.st_size,
        str(chunks_path.resolve()), chunks_stat.st_mtime_ns, chunks_stat.st_size)


def _apply_filters(frame: pd.DataFrame, filters: dict) -> pd.DataFrame:
    unsupported = set(filters) - _FILTER_FIELDS
    if unsupported:
        raise ValueError(f"Unsupported rating filters: {sorted(unsupported)}")
    mask = pd.Series(True, index=frame.index)
    for field, value in filters.items():
        if field == "date":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("Rating date filter must be a start/end pair")
            start, end = value
            if start is not None: mask &= frame[field] >= pd.Timestamp(start)
            if end is not None: mask &= frame[field] <= pd.Timestamp(end)
        elif isinstance(value, (list, tuple, set)):
            mask &= frame[field].isin(value)
        else:
            mask &= frame[field] == value
    return frame.loc[mask]


def _one_summary(values: pd.Series, scale: int) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "ci95_low": None, "ci95_high": None, "high_share": None, "low_share": None}
    mean, std = float(values.mean()), float(values.std(ddof=1)) if n > 1 else 0.0
    margin = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    high_threshold, low_threshold = (8, 4) if scale == 10 else (4, 2)
    return {"n": n, "mean": round(mean, 3), "median": round(float(values.median()), 3),
            "std": round(std, 3), "ci95_low": round(mean - margin, 3),
            "ci95_high": round(mean + margin, 3),
            "high_share": round(float((values >= high_threshold).mean()), 4),
            "low_share": round(float((values <= low_threshold).mean()), 4)}


def summarize_ratings(aspect: str, *, filters: dict | None = None, airlines: list[str] | None = None,
                      reviews_path: str | Path = REVIEWS, chunks_path: str | Path = CHUNKS,
                      min_samples: int = 30) -> dict:
    """Aggregate original review ratings and produce a conservative comparison."""
    if aspect not in RATING_FIELDS:
        return {"available": False, "reason": "no_rating_field", "aspect": aspect,
                "source": "review_level_ratings"}
    field, scale = RATING_FIELDS[aspect]
    query_filters = dict(filters or {})
    if airlines:
        query_filters.pop("airline", None)
        query_filters["airline"] = airlines
    frame = _apply_filters(_load_review_ratings(reviews_path, chunks_path), query_filters)
    if airlines:
        names = list(airlines)
        summaries = {name: _one_summary(frame.loc[frame["airline"] == name, field], scale) for name in names}
    elif isinstance(query_filters.get("airline"), str):
        names = [query_filters["airline"]]
        summaries = {names[0]: _one_summary(frame[field], scale)}
    else:
        names = ["全部航司"]
        summaries = {names[0]: _one_summary(frame[field], scale)}
    result = {"available": True, "source": "original_review_ratings", "aspect": aspect,
              "rating_field": field, "scale": scale, "filters": query_filters,
              "airlines": summaries, "minimum_reliable_n": min_samples}
    eligible = [(name, data) for name, data in summaries.items()
                if data["mean"] is not None and data["n"] >= min_samples]
    comparison = {"status": "not_applicable"}
    if len(eligible) >= 2:
        ranked = sorted(eligible, key=lambda item: (-item[1]["mean"], item[0]))
        winner, winner_data = ranked[0]
        runner, runner_data = ranked[1]
        delta = winner_data["mean"] - runner_data["mean"]
        se = math.sqrt((winner_data["std"] ** 2 / winner_data["n"]) +
                       (runner_data["std"] ** 2 / runner_data["n"]))
        low, high = delta - 1.96 * se, delta + 1.96 * se
        threshold = 0.5 if scale == 10 else 0.25
        decisive = delta >= threshold and low > 0
        comparison = {"status": "clear_difference" if decisive else "no_clear_difference",
                      "winner": winner if decisive else None, "leader": winner,
                      "runner_up": runner, "difference": round(delta, 3),
                      "difference_ci95_low": round(low, 3), "difference_ci95_high": round(high, 3),
                      "practical_threshold": threshold}
    elif len(names) >= 2:
        comparison = {"status": "insufficient_ratings", "winner": None}
    result["comparison"] = comparison
    return result
