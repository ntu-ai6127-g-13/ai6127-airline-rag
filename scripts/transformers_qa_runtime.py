"""Local PyTorch adapter for the existing QA server's MLX-shaped calls."""
from threading import Event, Thread
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, TextIteratorStreamer


def load(path, device="auto"):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    return model, tokenizer


def make_sampler(temp):
    return temp


def _inputs(model, tokenizer, prompt, max_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if inputs.input_ids.shape[1] + max_tokens > 8192:
        raise ValueError("Local QA requests are limited to 8192 input plus output tokens")
    kwargs = dict(**inputs, max_new_tokens=max_tokens, do_sample=temperature > 0,
                  pad_token_id=tokenizer.eos_token_id)
    if temperature > 0:
        kwargs["temperature"] = temperature
    return kwargs


def generate(model, tokenizer, *, prompt, max_tokens, sampler, verbose=False):
    kwargs = _inputs(model, tokenizer, prompt, max_tokens, sampler)
    with torch.inference_mode():
        output = model.generate(**kwargs)
    return tokenizer.decode(output[0, kwargs["input_ids"].shape[1]:], skip_special_tokens=True)


def stream_generate(model, tokenizer, *, prompt, max_tokens, sampler):
    kwargs = _inputs(model, tokenizer, prompt, max_tokens, sampler)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60)
    stop = Event()
    errors = []

    class Cancelled(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return stop.is_set()

    def produce():
        try:
            with torch.inference_mode():
                model.generate(**kwargs, streamer=streamer, stopping_criteria=[Cancelled()])
        except Exception as error:
            errors.append(error)
        finally:
            streamer.end()

    worker = Thread(target=produce, daemon=True)
    worker.start()
    try:
        for text in streamer:
            yield SimpleNamespace(text=text)
        if errors:
            raise errors[0]
    finally:
        stop.set()
        worker.join()
