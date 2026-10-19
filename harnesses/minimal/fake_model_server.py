#!/usr/bin/env python3
"""A stand-in for the model server (no GPU): answers the minimal harness the way vLLM answers the recording proxy
(message, output token ids, chosen-token logprobs, prompt token ids). The answer depends only on the request's seed and
last message, so runs are reproducible. --weights is whatever the controller serves this round; it is only echoed."""
import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--weights", default="none")
a = ap.parse_args()


class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": a.weights}]} if self.path.startswith("/v1/models") else {"ok": True})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        msgs = body.get("messages") or []
        last = msgs[-1].get("content") if msgs else ""
        n = 1 + int(hashlib.sha256(f"{body.get('seed')}|{last}".encode()).hexdigest(), 16) % 2
        prompt_ids = [1] * (5 * len(msgs))
        self._send({"id": "x", "object": "chat.completion", "model": body.get("model"), "prompt_token_ids": prompt_ids,
                    "choices": [{"index": 0, "finish_reason": "stop", "stop_reason": None, "token_ids": [100, 10 + n],
                                 "message": {"role": "assistant", "content": f"MOVE {n}"},
                                 "logprobs": {"content": [{"token": "MOVE", "logprob": -0.1}, {"token": str(n), "logprob": -0.69}]}}],
                    "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": 2, "total_tokens": len(prompt_ids) + 2}})


ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
