"""生成 100 行假数据，让检索组和下游组不用等真数据就能开工。

用法：python scripts/make_fake_data.py
产出：data/processed/chunks_sample.parquet
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(0)
np.random.seed(0)

AIRLINES = ["Airline X", "Airline Y", "Airline Z"]
CABINS = ["Economy", "Business", "Premium Economy"]

# (句子, 讲的是哪个方面, 该方面的评分)
SAMPLES = [
    ("The seat was comfortable with plenty of legroom.", "seat", 5),
    ("Seat pitch was far too tight for a 12-hour flight.", "seat", 1),
    ("Cabin crew were attentive and friendly throughout.", "staff", 5),
    ("The flight attendants ignored us for most of the flight.", "staff", 1),
    ("The food was cold and tasteless.", "food", 1),
    ("Meals were surprisingly good for economy class.", "food", 4),
    ("Entertainment system had a good selection of movies.", "entertainment", 4),
    ("The screen was broken and could not be fixed.", "entertainment", 1),
    ("Check-in staff were slow and unhelpful.", "ground", 2),
    ("Boarding was efficient and well organised.", "ground", 5),
    ("Wifi did not work for the entire flight.", "wifi", 1),
    ("Good value for the price paid.", "value", 4),
    ("Overpriced for what you actually get.", "value", 2),
    ("I flew from Singapore to London in March.", "other", 3),
]

RATING_COLS = ["r_seat", "r_staff", "r_food", "r_entertainment",
               "r_ground", "r_wifi", "r_value"]
ASPECT_TO_COL = {
    "seat": "r_seat", "staff": "r_staff", "food": "r_food",
    "entertainment": "r_entertainment", "ground": "r_ground",
    "wifi": "r_wifi", "value": "r_value",
}


def main(n: int = 100) -> None:
    rows = []
    for i in range(n):
        text, aspect, score = random.choice(SAMPLES)
        review_id = f"R{10000 + i // 3}"

        row = {
            "chunk_id": f"{review_id}_s{i % 3:02d}",
            "review_id": review_id,
            "text": text,
            "review_text": text + " Overall an average experience.",
            "airline": random.choice(AIRLINES),
            "cabin": random.choice(CABINS),
            "route": "Singapore to London",
            "date": pd.Timestamp("2023-01-01")
                    + pd.Timedelta(days=random.randint(0, 1000)),
            "overall_rating": random.randint(1, 10),
            "recommended": random.choice([True, False]),
        }
        # 七个维度评分随机填，但把这句话对应的那个方面填成它该有的分数
        for col in RATING_COLS:
            row[col] = float(random.randint(1, 5))
        if aspect in ASPECT_TO_COL:
            row[ASPECT_TO_COL[aspect]] = float(score)

        rows.append(row)

    out = Path("data/processed/chunks_sample.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    print(f"已生成 {n} 行假数据 -> {out}")
    print("检索组和下游组现在可以开工了 ✅")


if __name__ == "__main__":
    main()
