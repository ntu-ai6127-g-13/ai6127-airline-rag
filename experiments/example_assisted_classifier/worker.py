"""MLX-only subprocess. It reads prepared target text, never evaluation answers."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

from .core import (ROOT, WEIGHT_SHA256, check_source_hashes, label_prediction,
                   read_jsonl, sha256, validate_inputs, write_json, write_jsonl)
from .evidence import ASPECTS, LABELS, DIRECT_PROMPT, parse_evidence
from .guard import install_guard
from .prompts import ASPECT_PROMPT, SENTIMENT_PROMPT, EXAMPLE_INSTRUCTION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=['examples', 'words'], required=True)
    args = parser.parse_args()
    denied = install_guard(ROOT)
    lock = json.loads(args.lock.read_text(encoding='utf-8'))
    check_source_hashes(lock['source_hashes'])
    if sha256(args.input) != lock['input_sha256']:
        raise ValueError('Prepared input changed')
    if sha256(args.model / 'model.safetensors') != WEIGHT_SHA256:
        raise ValueError('Unexpected model weights; use the documented fixed revision')
    for name, digest in lock['model_files'].items():
        if sha256(args.model / name) != digest:
            raise ValueError('Model/tokenizer configuration changed: ' + name)
    rows = read_jsonl(args.input)
    validate_inputs([{key: row[key] for key in ['id', 'text']} for row in rows])
    if args.mode == 'examples':
        for row in rows:
            if set(row) != {'id', 'text', 'examples'} or len(row['examples']) != 4:
                raise ValueError('Expected four examples per target')
            for example in row['examples']:
                if example['aspect'] not in ASPECTS or example['label'] not in LABELS:
                    raise ValueError('Illegal example annotation')
    else:
        validate_inputs(rows)
    if args.out.exists():
        raise ValueError('Preserve existing worker output')
    args.out.mkdir(parents=True)
    os.environ['HF_HUB_OFFLINE'] = '1'
    import mlx.core as mx
    from mlx_lm import load, batch_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(str(args.model))
    started = time.monotonic()

    def cached_prompts(system, contents):
        prompts = [tokenizer.apply_chat_template([
            {'role': 'system', 'content': system}, {'role': 'user', 'content': text}],
            add_generation_prompt=True) for text in contents]
        prefix = min(map(len, prompts)) - 1
        for tokens in prompts[1:]:
            for i in range(prefix):
                if tokens[i] != prompts[0][i]:
                    prefix = i
                    break
        cache = make_prompt_cache(model)
        model(mx.array([prompts[0][:prefix]]), cache=cache)
        mx.eval([item.state for item in cache])
        return prompts, prefix, cache

    if args.mode == 'examples':
        contents = [json.dumps({'AI_silver_examples': [
            {key: example[key] for key in ['text', 'aspect', 'label']}
            for example in row['examples']], 'TARGET': row['text']}, ensure_ascii=False)
            for row in rows]
        prompts, prefix, cache = cached_prompts(DIRECT_PROMPT + EXAMPLE_INSTRUCTION, contents)
        output = []
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8]
            response = batch_generate(model, tokenizer,
                [tokens[prefix:] for tokens in prompts[start:start + len(batch)]],
                prompt_caches=[copy.deepcopy(cache) for _ in batch], max_tokens=320,
                sampler=make_sampler(temp=0.0), prefill_batch_size=8,
                completion_batch_size=8, verbose=False)
            if len(response.texts) != len(batch):
                raise ValueError('Model returned wrong number of responses')
            for row, raw in zip(batch, response.texts):
                output.append({'id': row['id'], 'text': row['text'],
                    **parse_evidence(row['text'], raw, variant='direct'),
                    'example_ids': [item['id'] for item in row['examples']],
                    'raw_output': raw, 'human_reviewed': False,
                    'source': 'local_retrieved_silver_examples'})
            mx.clear_cache()
            print(json.dumps({'completed': len(output), 'total': len(rows)}), flush=True)
    else:
        def choose(system, contents, words):
            prompts, prefix, cache = cached_prompts(system, contents)
            token_ids = [tokenizer.encode(word, add_special_tokens=False) for word in words]
            if any(len(ids) != 1 for ids in token_ids):
                raise ValueError('Fallback choices must be single tokens')
            allowed = mx.array([ids[0] for ids in token_ids])

            def restrict(tokens, logits):
                return mx.where(mx.any(mx.arange(logits.shape[-1])[:, None] == allowed[None, :], axis=1),
                                logits, -float('inf'))

            answers = []
            for start in range(0, len(rows), 16):
                batch = prompts[start:start + 16]
                response = batch_generate(model, tokenizer, [tokens[prefix:] for tokens in batch],
                    prompt_caches=[copy.deepcopy(cache) for _ in batch], max_tokens=1,
                    sampler=make_sampler(temp=0.0), logits_processors=[restrict],
                    prefill_batch_size=8, completion_batch_size=16, verbose=False)
                values = [text.strip() for text in response.texts]
                if len(values) != len(batch) or any(value not in words for value in values):
                    raise ValueError('Invalid word fallback output')
                answers.extend(values)
                mx.clear_cache()
            return answers

        aspects = choose(ASPECT_PROMPT, [
            'Sentence: ' + json.dumps(row['text']) + '\nCategory word:' for row in rows],
            ['seat', 'staff', 'food', 'media', 'ground', 'value', 'other', 'overall'])
        aspects = ['entertainment' if aspect == 'media' else aspect for aspect in aspects]
        labels = choose(SENTIMENT_PROMPT, [
            'Sentence: ' + json.dumps(row['text']) + '\nSpecified aspect: ' + aspect + '\nSentiment word:'
            for row, aspect in zip(rows, aspects)], ['positive', 'negative', 'objective'])
        output = [{'id': row['id'], **label_prediction(aspect, label)}
                  for row, aspect, label in zip(rows, aspects, labels)]
    if len(denied) != 1:
        raise RuntimeError('Unexpected protected answer read')
    target = args.out / 'predictions.jsonl'
    write_jsonl(target, output)
    write_json(args.out / 'manifest.json', {'status': 'completed', 'rows': len(output),
        'mode': args.mode, 'input_sha256': sha256(args.input), 'lock_sha256': sha256(args.lock),
        'output_sha256': sha256(target), 'gold_answers_read': False, 'human_reviewed': False,
        'seconds': round(time.monotonic() - started, 2)})


if __name__ == '__main__':
    main()
