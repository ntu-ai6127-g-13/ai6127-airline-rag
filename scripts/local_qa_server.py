"""Persistent, localhost-only MLX generation service for grounded QA."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import threading
import time

DEFAULT_MODEL = (Path.home() / ".cache/huggingface/hub/"
                 "models--mlx-community--Qwen3-4B-Instruct-2507-4bit/"
                 "snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The QA model service is intentionally localhost-only")
    if not args.model.exists():
        raise FileNotFoundError(f"Local model not found: {args.model}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from mlx_lm import generate, load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    started = time.monotonic()
    model, tokenizer = load(str(args.model))
    loaded_seconds = round(time.monotonic() - started, 3)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "AirlineGroundedQA/1.0"

        def log_message(self, format: str, *values) -> None:
            print(f"{self.address_string()} {format % values}", flush=True)

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                self._json(404, {"error": "not_found"}); return
            self._json(200, {"status": "ready", "model": args.model.name,
                             "loaded_seconds": loaded_seconds})

        def do_POST(self) -> None:
            if self.path not in {"/generate", "/generate_stream"}:
                self._json(404, {"error": "not_found"}); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 2 or length > 200_000: raise ValueError("invalid_request_size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                messages = payload["messages"]
                if not isinstance(messages, list) or not 1 <= len(messages) <= 8:
                    raise ValueError("invalid_messages")
                max_tokens = min(max(int(payload.get("max_tokens", 420)), 32), 600)
                temperature = float(payload.get("temperature", 0.0))
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                request_started = time.monotonic()
                if self.path == "/generate_stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    try:
                        with lock:
                            for response in stream_generate(
                                    model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                                    sampler=make_sampler(temp=temperature)):
                                if response.text:
                                    event = json.dumps({"type": "delta", "text": response.text},
                                                       ensure_ascii=False).encode("utf-8") + b"\n"
                                    self.wfile.write(event)
                                    self.wfile.flush()
                        done_payload = {"type": "done", "seconds": round(
                            time.monotonic() - request_started, 3)}
                    except Exception as error:
                        done_payload = {"type": "error", "error": type(error).__name__,
                                        "detail": str(error)[:500]}
                    self.wfile.write(json.dumps(done_payload, ensure_ascii=False).encode("utf-8") + b"\n")
                    self.wfile.flush()
                    return
                with lock:
                    output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                                      sampler=make_sampler(temp=temperature), verbose=False)
                self._json(200, {"text": output.strip(),
                                 "seconds": round(time.monotonic() - request_started, 3)})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as error:
                self._json(500, {"error": type(error).__name__, "detail": str(error)[:500]})

    # MLX Metal streams are bound to the thread which loaded the model.  A
    # single-threaded HTTPServer keeps generation on that same thread; the lock
    # still documents that the model is intentionally one-request-at-a-time.
    server = HTTPServer((args.host, args.port), Handler)
    print(json.dumps({"status": "ready", "host": args.host, "port": args.port,
                      "model": str(args.model), "loaded_seconds": loaded_seconds}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
