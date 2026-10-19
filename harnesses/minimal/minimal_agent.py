#!/usr/bin/env python3
"""A harness in 60 lines: the smallest program the RL core can train through (harnesses/minimal/README.md).

A 1-D "gripper" starts at 0 and must stop exactly on the target. Each turn the agent shows the model where the gripper
and the target are, the model answers "MOVE <n>", the agent moves. The episode ends on the target (success), past it
(failure) or after --max-turns. Nothing here knows about the RL code: the agent only talks to the OpenAI-compatible URL
it was given and leaves a result.json.
"""
import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--init", required=True)
ap.add_argument("--attempt", type=int, default=0)
ap.add_argument("--model-url", required=True, help="OpenAI base URL, e.g. http://127.0.0.1:18200/ep/<key>/v1")
ap.add_argument("--model", default="policy")
ap.add_argument("--out", required=True)
ap.add_argument("--max-turns", type=int, default=4)
a = ap.parse_args()

target = 3 + int(hashlib.sha256(f"{a.task}|{a.init}".encode()).hexdigest(), 16) % 3      # the "initial state": 3, 4 or 5
messages = [{"role": "system", "content": "You move a gripper along a line. Answer with: MOVE <n>, n = 1 or 2."}]
pos, calls, end_turn = 0, 0, None
for turn in range(a.max_turns):
    messages.append({"role": "user", "content": f"task {a.task}: gripper at {pos}, target at {target}."})
    body = {"model": a.model, "messages": messages, "max_tokens": 64, "seed": 1000 * a.attempt + turn}
    req = urllib.request.Request(a.model_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        reply = json.loads(r.read())["choices"][0]["message"]["content"] or ""
    calls += 1
    messages.append({"role": "assistant", "content": reply})
    m = re.search(r"MOVE\s+(-?\d+)", reply)
    pos += int(m.group(1)) if m else 0
    if pos >= target:
        end_turn = turn
        break

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / "result.json").write_text(json.dumps({"reward": 1.0 if pos == target else 0.0, "valid": True, "requests": calls,
                                              "end_turn": end_turn if pos == target else None, "report": ["final_pos", "target"],
                                              "final_pos": pos, "target": target}))
