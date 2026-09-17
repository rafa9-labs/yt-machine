#!/usr/bin/env python3
"""
Ollama API bridge for the local MLX LM server.
================================================

WHY THIS EXISTS
───────────────
yt-machine talks to Ollama over its native REST API:
    POST /api/generate   (prompt completion, streaming NDJSON)
    POST /api/chat       (chat completion, streaming NDJSON)
    GET  /api/tags       (installed models)
    GET  /api/ps         (loaded models)

On this machine the LLM runs on mlx_lm.server (Qwen3.8-27B OptiQ 4-bit),
which only exposes an OpenAI-compatible API (/v1/chat/completions).

This bridge translates Ollama API calls → OpenAI chat completions and
streams the results back in Ollama's NDJSON format, so the pipeline works
unmodified.

USAGE
─────
    # 1. Start the MLX model server (loads ~18GB into unified memory)
    mlx_lm.server --model mlx-community/Qwen3.8-27B-OptiQ-4bit \
        --host 127.0.0.1 --port 8080

    # 2. Start this bridge
    python3 tools/ollama_mlx_bridge.py            # listens on :11434

    # Or just use tools/start_llm.sh which does both.

ENV
───
    MLX_BASE_URL   default http://127.0.0.1:8080
    MLX_MODEL      default mlx-community/Qwen3.8-27B-OptiQ-4bit
    BRIDGE_PORT    default 11434
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_MLX_BASE_URL = os.environ.get("MLX_BASE_URL", "http://127.0.0.1:8080")
DEFAULT_MODEL = os.environ.get("MLX_MODEL", "mlx-community/Qwen3.8-27B-OptiQ-4bit")
DEFAULT_PORT = int(os.environ.get("BRIDGE_PORT", "11434"))

BACKEND_TIMEOUT = float(os.environ.get("MLX_BACKEND_TIMEOUT", "1800"))


def log(msg: str) -> None:
    print(f"[bridge] {msg}", file=sys.stderr, flush=True)


class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    mlx_base_url = DEFAULT_MLX_BASE_URL
    served_model = DEFAULT_MODEL

    # ── helpers ──────────────────────────────────────────────────────

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _stream_chunk(self, data: bytes) -> None:
        payload = data + b"\n"
        self.wfile.write(f"{len(payload):X}\r\n".encode("ascii"))
        self.wfile.write(payload)
        self.wfile.write(b"\r\n")

    def _stream_end(self) -> None:
        self.wfile.write(b"0\r\n\r\n")

    # ── backend call ─────────────────────────────────────────────────

    def _backend_chat_stream(self, messages, temperature, max_tokens):
        payload = {
            "model": self.served_model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        req = urllib.request.Request(
            f"{self.mlx_base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(req, timeout=BACKEND_TIMEOUT)

    @staticmethod
    def _iter_backend_deltas(response):
        """Yield content deltas from an OpenAI-compatible SSE stream."""
        for raw_line in response:
            line = raw_line.strip()
            if not line or not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                return
            try:
                chunk = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                yield content

    def _run_completion(self, body: dict, chat: bool) -> None:
        stream = bool(body.get("stream", True))
        options = body.get("options") or {}
        temperature = options.get("temperature", body.get("temperature", 0.7))
        max_tokens = options.get("num_predict", body.get("max_tokens")) or 0

        if chat:
            messages = body.get("messages") or []
        else:
            messages = []
            system_prompt = body.get("system")
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": body.get("prompt", "")})

        if not messages:
            self._send_json(400, {"error": "missing prompt/messages"})
            return

        model_name = body.get("model") or self.served_model

        try:
            backend = self._backend_chat_stream(messages, temperature, max_tokens)
        except urllib.error.URLError as exc:
            log(f"backend unreachable: {exc}")
            self._send_json(
                503,
                {
                    "error": (
                        f"MLX backend not reachable at {self.mlx_base_url}. "
                        "Start it with tools/start_llm.sh"
                    )
                },
            )
            return

        try:
            if stream:
                self._stream_start()
                for content in self._iter_backend_deltas(backend):
                    if chat:
                        chunk = {
                            "model": model_name,
                            "message": {"role": "assistant", "content": content},
                            "response": content,
                            "done": False,
                        }
                    else:
                        chunk = {"model": model_name, "response": content, "done": False}
                    self._stream_chunk(json.dumps(chunk).encode("utf-8"))
                final = {
                    "model": model_name,
                    "done": True,
                    "done_reason": "stop",
                    "response": "",
                }
                if chat:
                    final["message"] = {"role": "assistant", "content": ""}
                self._stream_chunk(json.dumps(final).encode("utf-8"))
                self._stream_end()
            else:
                text = "".join(self._iter_backend_deltas(backend))
                if chat:
                    payload = {
                        "model": model_name,
                        "message": {"role": "assistant", "content": text},
                        "response": text,
                        "done": True,
                    }
                else:
                    payload = {"model": model_name, "response": text, "done": True}
                self._send_json(200, payload)
        except (BrokenPipeError, ConnectionResetError):
            log("client disconnected mid-stream")
        finally:
            backend.close()

    # ── HTTP methods ─────────────────────────────────────────────────

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/api/tags", "/api/ps"):
            model = {
                "name": self.served_model,
                "model": self.served_model,
                "size": 0,
                "digest": "mlx",
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            key = "models" if path == "/api/tags" else "models"
            self._send_json(200, {key: [model]})
        elif path in ("/health", "/"):
            self._send_json(200, {"status": "ok"})
        elif path == "/api/embeddings":
            self._send_json(
                501,
                {"error": "MLX bridge does not provide embeddings; use Ollama nomic-embed-text"},
            )
        else:
            self._send_json(404, {"error": f"unsupported path {path}"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        body = self._read_body()

        # Ollama unload signal (keep_alive=0) — no-op for MLX
        if "keep_alive" in body and not body.get("prompt") and not body.get("messages"):
            self._send_json(200, {"done": True})
            return

        if path == "/api/generate":
            self._run_completion(body, chat=False)
        elif path == "/api/chat":
            self._run_completion(body, chat=True)
        elif path == "/api/pull":
            self._send_json(
                200,
                {"status": "success", "note": "MLX models are managed via Hugging Face"},
            )
        else:
            self._send_json(404, {"error": f"unsupported path {path}"})

    def log_message(self, fmt: str, *args) -> None:
        log(fmt % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama API bridge for mlx_lm.server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bridge port")
    parser.add_argument("--mlx-url", default=DEFAULT_MLX_BASE_URL, help="mlx_lm.server base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name exposed to clients")
    args = parser.parse_args()

    BridgeHandler.mlx_base_url = args.mlx_url.rstrip("/")
    BridgeHandler.served_model = args.model

    log(f"listening on http://{args.host}:{args.port}")
    log(f"proxying to {BridgeHandler.mlx_base_url} (model={BridgeHandler.served_model})")

    server = ThreadingHTTPServer((args.host, args.port), BridgeHandler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
