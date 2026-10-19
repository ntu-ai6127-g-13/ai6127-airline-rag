#!/usr/bin/env python3
"""Measure local Qwen service behavior at concurrency 1 and 2."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.grounded_qa import DEFAULT_ENDPOINT, _generate  # noqa: E402

MESSAGES = [
    {"role": "system", "content": "Answer concisely in Chinese."},
    {"role": "user", "content": "用两句话说明航空评论检索系统为何需要引用原文，并说明历史评论不能代表当前服务。"},
]


def request() -> dict:
    started = time.monotonic()
    result = _generate(MESSAGES, endpoint=DEFAULT_ENDPOINT, timeout=120, max_tokens=120)
    return {"wall_seconds": time.monotonic() - started,
            "server_seconds": float(result.get("seconds", 0.0)),
            "characters": len(str(result.get("text", "")))}


def trial(concurrency: int) -> dict:
    started = time.monotonic()
    if concurrency == 1:
        calls = [request(), request()]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            calls = list(pool.map(lambda _: request(), range(2)))
    elapsed = time.monotonic() - started
    return {"concurrency": concurrency, "requests": 2, "wall_seconds": elapsed,
            "requests_per_second": 2 / elapsed, "calls": calls}


def main() -> int:
    output = Path(__file__).with_name("concurrency_results.json")
    result = {
        "scope": "local Qwen generation service only; fixed 120-token cap",
        "service_design": "single-threaded HTTPServer with serialized MLX access",
        "concurrency_1": trial(1),
        "concurrency_2": trial(2),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
