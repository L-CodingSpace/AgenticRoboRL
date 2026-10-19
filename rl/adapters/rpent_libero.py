"""Adapter for RPent on LIBERO / LIBERO-PRO. See rl/adapters/__init__.py.

Episode dirs: <root>/<run id>/<suite>/task_XX/seed_XXX/attempt_KK/ with the RPent transcript_*.json;
a group is (suite, task, seed).
"""
from __future__ import annotations

import collections
import gzip
import json
import re
from pathlib import Path
from typing import Any

# ---- verdict of one RPent transcript ----
# A fault outside the planner: the episode never got a fair chance and is not a task failure.
_INFRASTRUCTURE = (
    "connection refused", "connection error", "connection reset", "apiconnectionerror",
    "server disconnected", "remote end closed", "read timeout", "readtimeout",
    "temporarily unavailable", "service unavailable", "bad gateway",
    "internal server error", "cuda", "out of memory", "no such device", "env server",
    "environment server", "failed to start", "address already in use", "broken pipe",
    "killed", "mujoco", "egl",
)
_INFRASTRUCTURE_STATUS = re.compile(r"(?<!\d)(?:502|503|504)(?!\d)")
_CONTEXT_OVERFLOW = (
    "maximum context length", "context length is", "context window",
    "context_length_exceeded", "prompt is too long", "too many tokens",
)
_ATTEMPT = re.compile(
    r"/(?P<suite>libero_[a-z0-9_]+)/task_(?P<task>\d+)/seed_(?P<seed>\d+)/attempt_(?P<attempt>\d+)"
    r"/transcript_[^/]+\.json$"
)


def classify_failure(error: str | None) -> str | None:
    """None (finished), context_overflow, infrastructure, or agent."""
    if not error:
        return None
    text = str(error).lower()
    if any(pattern in text for pattern in _CONTEXT_OVERFLOW):
        return "context_overflow"
    if any(pattern in text for pattern in _INFRASTRUCTURE) or _INFRASTRUCTURE_STATUS.search(text):
        return "infrastructure"
    return "agent"


def read_verdict(path: Path) -> dict[str, Any]:
    m = _ATTEMPT.search(path.as_posix())
    assert m, path
    row: dict[str, Any] = {"suite": m["suite"], "task": int(m["task"]), "seed": int(m["seed"]),
                           "attempt": int(m["attempt"]), "transcript": str(path)}
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rec, dict):
            raise ValueError("transcript root must be a JSON object")
    except (OSError, ValueError) as exc:
        return {**row, "failure_kind": "infrastructure", "solved": None, "error": f"unreadable: {exc}"}
    error = rec.get("agent_error")
    kind = "infrastructure" if "solved" not in rec else classify_failure(error)
    stats = rec.get("stats") or {}
    return {**row, "failure_kind": kind, "solved": None if kind == "infrastructure" else bool(rec.get("solved")),
            "elapsed_s": rec.get("elapsed_s"), "error": str(error) if error else None,
            "turns_used": stats.get("turns_used"), "tool_calls": stats.get("tool_calls"),
            "input_tokens": stats.get("total_input_tokens"), "output_tokens": stats.get("total_output_tokens")}


NAME = "rpent_libero"

EPISODE_KEYS = ("suite", "task", "seed", "attempt", "status", "reward", "failure_kind",
                "turns_used", "requests", "n_records", "elapsed_s", "advantage",
                "n_calls", "n_back_project", "n_move_to", "longest_same_call_run")
OUTCOME_METRICS = ("turns_used", "elapsed_s", "n_calls", "n_back_project", "n_move_to", "longest_same_call_run")


def find_episodes(root: Path):
    for tr in sorted(Path(root).rglob("transcript_*.json")):
        if "/attempt_" not in tr.as_posix() or ".stale_" in tr.as_posix():
            continue
        v = read_verdict(tr)
        try:
            stats = (json.loads(tr.read_text()).get("stats") or {})
        except (OSError, ValueError):
            stats = {}
        v["requests"] = stats.get("requests")
        v["valid"] = v["failure_kind"] != "infrastructure"
        v["reward"] = (1.0 if v["solved"] else 0.0) if v["valid"] else None
        yield tr.parent, v


def group_key(row) -> tuple:
    return (row["suite"], row["task"], row["seed"])


def group_name(gk) -> str:
    return f"{gk[0]}/t{gk[1]}/s{gk[2]}"


def outcome_kind(row) -> str:
    return ("solved" if row["reward_terminal"] == 1 else
            "budget_100" if (row.get("turns_used") or 0) >= 100 else
            f"failed_{row['failure_kind'] or 'finished'}")


_TERMINATED = re.compile(r'"terminated"\s*:\s*true')


def post_success_turns(ep_dir: Path, n_turns: int):
    """Solved episode: the turns AFTER the first one whose tool results report success ("terminated": true), except turns
    that call `finish` (those are kept: stopping after success is what we want). None when the transcript's assistant turns
    do not line up one-to-one with the rl_turns records (then nothing is filtered)."""
    trs = sorted(Path(ep_dir).glob("transcript_*.json"))
    if not trs:
        return None
    msgs = json.loads(trs[0].read_text()).get("messages") or []
    text = lambda c: c if isinstance(c, str) else " ".join(  # noqa: E731
        (x.get("text") or "") if isinstance(x, dict) else str(x) for x in (c or []))
    ai = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    if len(ai) != n_turns:
        return None
    success = None
    for k, i in enumerate(ai):
        j = i + 1
        while j < len(msgs) and msgs[j].get("role") == "tool":
            if _TERMINATED.search(text(msgs[j].get("content"))):
                success = k
                break
            j += 1
        if success is not None:
            break
    if success is None:
        return {"success_turn": None, "excluded": [], "finish_kept": []}
    calls = lambda m: [x.get("name") for x in (m.get("content") if isinstance(m.get("content"), list) else [])  # noqa: E731
                       if isinstance(x, dict) and x.get("type") == "tool_use"]
    after = range(success + 1, n_turns)
    finish = [t for t in after if "finish" in calls(msgs[ai[t]])]
    return {"success_turn": success, "excluded": [t for t in after if t not in finish], "finish_kept": finish}


def episode_extras(turns) -> dict:
    """Tool-use repetition of one episode (to see whether RL rewards repeating actions)."""
    calls = []
    for tp in turns:
        msg = json.loads(gzip.decompress(tp.read_bytes())).get("message") or {}
        for tc in msg.get("tool_calls") or []:
            calls.append(tc["function"]["name"])
    best = run = 1 if calls else 0
    for x, y in zip(calls, calls[1:]):
        run = run + 1 if x == y else 1
        best = max(best, run)
    c = collections.Counter(calls)
    return {"n_calls": len(calls), "n_back_project": c.get("back_project", 0), "n_move_to": c.get("move_to", 0),
            "longest_same_call_run": best}
