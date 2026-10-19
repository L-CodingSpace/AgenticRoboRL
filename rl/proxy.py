"""Recording proxy between the RPent planner and its vLLM server (RL rollouts).

One proxy per GPU. The planner is pointed at
    http://127.0.0.1:<port>/ep/<episode key>/v1
where <episode key> is the cell's path under the run's output root
(<suite>/task_XX/seed_XXX/attempt_KK). For every chat request the proxy

  * overrides the sampling fields with the RL sampling (no top-k / top-p /
    penalties), so the logprobs vLLM returns (raw_logprobs, its default) are
    the behaviour policy's own probabilities;
  * asks vLLM for the prompt / output token ids and the chosen-token logprobs;
  * writes one gzip JSON per request to <out root>/<key>/rl_turns/turn_NNN.json.gz
    (images replaced by sha1 references, bytes stored once in <key>/rl_images/);
  * returns the response WITHOUT the added fields, so the planner sees the
    same shape of answer it gets without the proxy.

A request that times out, cannot connect, or gets 408 / 429 / 5xx from vLLM is
sent again after 1, 2 and 4 s; the fourth failure returns 502 to the planner.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_sampling import RL_SAMPLING  # noqa: E402

SAMPLING_KEYS = tuple(RL_SAMPLING)
ADDED_KEYS = {"logprobs": True, "return_token_ids": True}

_RETRY_WAITS_S = (1.0, 2.0, 4.0)
_DATA_URL = re.compile(r"^data:(?P<mime>[\w/+.-]+);base64,(?P<b64>.*)$", re.S)
_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp"}
_KEY = re.compile(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$")


class Recorder:
    def __init__(self, out_root: Path, policy_version: str) -> None:
        self.out_root = out_root
        self.policy_version = policy_version
        self._locks: dict[str, asyncio.Lock] = {}

    def episode_dir(self, key: str) -> Path:
        if not _KEY.match(key) or ".." in key.split("/"):
            raise HTTPException(status_code=400, detail=f"bad episode key {key!r}")
        return self.out_root / key

    def lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())

    @staticmethod
    def next_index(turn_dir: Path) -> int:
        # Restart-safe: the index is the number of records already on disk.
        return sum(1 for p in turn_dir.glob("turn_*.json.gz"))

    @staticmethod
    def strip_images(messages: list[dict[str, Any]], image_dir: Path) -> list[dict[str, Any]]:
        """Replace base64 data URLs by {"type": "image_ref", ...}; store each image once."""
        out = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                out.append(msg)
                continue
            parts = []
            for part in content:
                url = (part.get("image_url") or {}).get("url") if part.get("type") == "image_url" else None
                m = _DATA_URL.match(url) if isinstance(url, str) else None
                if m is None:
                    parts.append(part)
                    continue
                raw = base64.b64decode(m["b64"])
                sha1 = hashlib.sha1(raw).hexdigest()
                ext = _EXT.get(m["mime"], "bin")
                path = image_dir / f"{sha1}.{ext}"
                if not path.exists():
                    image_dir.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
                    tmp.write_bytes(raw)
                    tmp.rename(path)
                ref = {"type": "image_ref", "sha1": sha1, "mime": m["mime"], "bytes": len(raw),
                       "path": f"rl_images/{path.name}"}
                detail = (part.get("image_url") or {}).get("detail")
                if detail is not None:
                    ref["detail"] = detail
                parts.append(ref)
            out.append({**msg, "content": parts})
        return out

    async def write(self, key: str, record: dict[str, Any], messages: list[dict[str, Any]]) -> int:
        ep = self.episode_dir(key)
        turn_dir = ep / "rl_turns"
        async with self.lock(key):
            turn_dir.mkdir(parents=True, exist_ok=True)
            idx = self.next_index(turn_dir)
            record["turn_index"] = idx
            record["messages"] = await asyncio.to_thread(self.strip_images, messages, ep / "rl_images")
            data = gzip.compress(json.dumps(record, separators=(",", ":")).encode(), compresslevel=1)
            path = turn_dir / f"turn_{idx:03d}.json.gz"
            tmp = turn_dir / f".turn_{idx:03d}.tmp{os.getpid()}"
            await asyncio.to_thread(tmp.write_bytes, data)
            tmp.rename(path)
        return idx


def build_app(upstream: str, recorder: Recorder, timeout_s: float) -> FastAPI:
    app = FastAPI()
    client = httpx.AsyncClient(base_url=upstream, timeout=httpx.Timeout(timeout_s, connect=30.0))

    async def send(path: str, body: dict[str, Any]) -> httpx.Response:
        for wait in (*_RETRY_WAITS_S, None):  # None: last try, no further retry
            try:
                resp = await client.post(path, json=body)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if wait is None:
                    raise HTTPException(status_code=502, detail=f"upstream failed: {exc!r}") from exc
                await asyncio.sleep(wait)
                continue
            if wait is not None and (resp.status_code in (408, 429) or resp.status_code >= 500):
                await asyncio.sleep(wait)
                continue
            return resp

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "policy_version": recorder.policy_version}

    @app.get("/ep/{key:path}/v1/models")
    async def models(key: str) -> Response:
        resp = await client.get("/v1/models")
        return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

    @app.post("/ep/{key:path}/v1/chat/completions")
    async def chat(key: str, request: Request) -> Response:
        recorder.episode_dir(key)  # validate before doing any work
        body = await request.json()
        if body.get("stream"):
            raise HTTPException(status_code=400, detail="streaming is not supported by the RL proxy")
        client_sampling = {k: body.get(k) for k in SAMPLING_KEYS + ("seed", "logprobs", "top_logprobs")}
        client_asked_logprobs = bool(body.get("logprobs"))
        sent = {**body, **RL_SAMPLING, **ADDED_KEYS}
        sent.pop("top_logprobs", None)
        t0 = time.time()
        resp = await send("/v1/chat/completions", sent)
        t1 = time.time()
        try:
            payload = resp.json()
        except ValueError:
            return Response(content=resp.content, status_code=resp.status_code)
        if resp.status_code != 200:
            return JSONResponse(content=payload, status_code=resp.status_code)

        choice = payload["choices"][0]
        content = ((choice.get("logprobs") or {}).get("content")) or []
        output_ids = choice.get("token_ids")
        record = {
            "key": key,
            "policy_version": recorder.policy_version,
            "t_start": t0,
            "t_end": t1,
            "latency_s": round(t1 - t0, 4),
            "client_sampling": client_sampling,
            "sent_sampling": {k: sent.get(k) for k in SAMPLING_KEYS + ("seed",)},
            "request": {k: v for k, v in body.items() if k not in ("messages",)},
            "prompt_token_ids": payload.get("prompt_token_ids"),
            "output_token_ids": output_ids,
            "output_logprobs": [c.get("logprob") for c in content],
            "finish_reason": choice.get("finish_reason"),
            "stop_reason": choice.get("stop_reason"),
            "message": choice.get("message"),
            "usage": payload.get("usage"),
        }
        await recorder.write(key, record, body.get("messages") or [])

        # Give the planner the answer it would have got without the added fields.
        payload.pop("prompt_token_ids", None)
        for c in payload.get("choices", []):
            c.pop("token_ids", None)
            if not client_asked_logprobs:
                c["logprobs"] = None
        return JSONResponse(content=payload)

    @app.on_event("shutdown")
    async def _close() -> None:
        await client.aclose()

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--upstream", required=True, help="e.g. http://127.0.0.1:18100")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--policy-version", required=True)
    ap.add_argument("--timeout-s", type=float, default=7200.0)
    a = ap.parse_args()
    rec = Recorder(Path(a.out_root), a.policy_version)
    uvicorn.run(build_app(a.upstream, rec, a.timeout_s), host="127.0.0.1", port=a.port,
                log_level="warning", timeout_keep_alive=7200)


if __name__ == "__main__":
    main()
