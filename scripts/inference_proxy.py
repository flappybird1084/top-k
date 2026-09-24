"""Local W&B Inference relay: keep the spend key out of worker environment.

The relay is bound to loopback, accepts one configured model, caps request size
and cumulative observed usage, and forwards only chat completions. Repository
code still needs an isolated OS identity/container for a strong trust boundary.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config

ENDPOINT = "https://api.inference.wandb.ai/v1/chat/completions"
MAX_REQUEST_BYTES = 2_000_000
MAX_COMPLETION_TOKENS = 8192


class InferenceRelay:
    def __init__(self, model: str, spend_cap: float):
        self.model = model
        self.spend_cap = spend_cap
        self.spent_usd = 0.0
        self.calls = 0
        self.credits_exhausted = False
        self.key = os.getenv("WANDB_INFERENCE_API_KEY") or os.environ["WANDB_API_KEY"]
        self.project = os.getenv("WANDB_INFERENCE_PROJECT") or (
            f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}")
        self.lock = threading.Lock()
        self.server = None
        self.thread = None

    @contextmanager
    def serving(self):
        relay = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    return self._reply(404, {"error": "unsupported path"})
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    return self._reply(400, {"error": "invalid content length"})
                if not 0 < size <= MAX_REQUEST_BYTES:
                    return self._reply(413, {"error": "request too large"})
                try:
                    body = json.loads(self.rfile.read(size))
                except (ValueError, UnicodeDecodeError):
                    return self._reply(400, {"error": "invalid JSON"})
                if body.get("model") != relay.model or body.get("stream"):
                    return self._reply(400, {"error": "unsupported model or stream"})
                max_tokens = body.get("max_tokens", 0)
                if not isinstance(max_tokens, int) or not 0 < max_tokens <= MAX_COMPLETION_TOKENS:
                    return self._reply(400, {"error": "invalid max_tokens"})
                with relay.lock:
                    return self._forward(body)

            def _forward(self, body):
                # Serialize all inference calls, including spend checks and
                # accounting, so concurrent search roles cannot race the cap.
                if relay.spent_usd >= relay.spend_cap:
                    return self._reply(429, {"error": "benchmark spend cap reached"})
                relay.calls += 1
                payload = json.dumps(body).encode("utf-8")
                request = urllib.request.Request(
                    ENDPOINT, payload, method="POST",
                    headers={"Authorization": f"Bearer {relay.key}",
                             "OpenAI-Project": relay.project,
                             "Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(request, timeout=180) as upstream:
                        response = upstream.read()
                        status = upstream.status
                except urllib.error.HTTPError as error:
                    response, status = error.read(), error.code
                except (urllib.error.URLError, TimeoutError) as error:
                    return self._reply(502, {"error": f"inference relay: {type(error).__name__}"})
                try:
                    data = json.loads(response)
                except ValueError:
                    data = {}
                error_text = json.dumps(data).lower()
                if status == 402 or (status in (403, 429) and any(
                        word in error_text for word in
                        ("credit", "quota", "payment", "balance"))):
                    relay.credits_exhausted = True
                usage = data.get("usage") or {}
                input_price, output_price = config.price_for(relay.model)
                relay.spent_usd += (
                    usage.get("prompt_tokens", len(payload)) * input_price +
                    usage.get("completion_tokens", body["max_tokens"]) *
                    output_price) / 1_000_000
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def _reply(self, status, data):
                payload = json.dumps(data).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        try:
            yield f"http://127.0.0.1:{self.server.server_port}/v1"
        finally:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=5)
