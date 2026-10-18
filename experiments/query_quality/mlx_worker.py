"""Isolated local inference worker. Input schema contains IDs/text only, never answers."""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

from prompts import messages, parse_output, MODEL_ID, MODEL_REVISION
from runtime import guard_gold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["taxonomy", "evidence_first"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    denied = guard_gold()
    if args.output.exists():
        raise RuntimeError("Preserve existing inference output")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    for row in rows:
        if set(row) != {"id", "text"}:
            raise ValueError("Inference receives only id and text; labels are forbidden")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from mlx_lm import load, batch_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.cache import make_prompt_cache
    import mlx.core as mx
    model, tokenizer = load(str(args.model))
    started = time.monotonic()
    prompts = [tokenizer.apply_chat_template(messages(r["text"], args.variant), add_generation_prompt=True) for r in rows]
    # All rows share the same taxonomy. Cache its exact token prefix once.
    prefix_length = min(map(len,prompts)) - 1
    first = prompts[0]
    for prompt in prompts[1:]:
        for j in range(prefix_length):
            if first[j] != prompt[j]: prefix_length = j; break
    prefix_cache = make_prompt_cache(model)
    if prefix_length:
        model(mx.array([first[:prefix_length]]), cache=prefix_cache)
        mx.eval([c.state for c in prefix_cache])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            batch_prompts = [p[prefix_length:] for p in prompts[offset:offset + len(batch)]]
            result = batch_generate(model, tokenizer, batch_prompts, max_tokens=16, verbose=False,
                                    prompt_caches=[copy.deepcopy(prefix_cache) for _ in batch],
                                    sampler=make_sampler(temp=0.), prefill_batch_size=8,
                                    completion_batch_size=args.batch_size)
            for row, output in zip(batch, result.texts):
                handle.write(json.dumps({"id": row["id"], **parse_output(output)}, ensure_ascii=False) + "\n")
            handle.flush()
            mx.clear_cache()
            print(json.dumps({"completed": offset + len(batch), "total": len(rows),
                              "seconds": round(time.monotonic() - started, 2)}, ensure_ascii=False), flush=True)
    assert len(denied) == 1
    print(json.dumps({"status": "completed", "model": MODEL_ID, "revision": MODEL_REVISION,
                      "gold_guard_only_self_test": True}), flush=True)


if __name__ == "__main__":
    main()
