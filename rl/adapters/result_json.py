"""Generic adapter: any harness that leaves a result.json in each episode dir. See rl/adapters/__init__.py.

An episode dir is any directory under the round's root that holds result.json (beside the proxy's rl_turns/):

  {"task": "<task id>",        # any string; with "init" it names the group
   "init": 17,                 # initial state / seed the group shares (int or string)
   "attempt": 0,               # which repeat of the group this episode is
   "reward": 1.0,              # the episode's reward (ignored when not valid)
   "valid": true,              # false = infrastructure failure: the episode says nothing about the policy
   "requests": 12,             # optional: model calls the harness made (checked against the proxy records)
   "end_turn": 7}              # optional: index of the model call after which the task was already done;
                               #           later calls are not trained on under --drop-post-success

Anything else in the file is carried into round_stats.json "episode_table" if listed in "report" (a list of keys).
"""
from __future__ import annotations

import json
from pathlib import Path

NAME = "result_json"
RESULT = "result.json"

EPISODE_KEYS = ("task", "init", "attempt", "status", "reward", "requests", "n_records", "end_turn", "advantage", "n_output_tokens")
OUTCOME_METRICS = ("n_records", "n_output_tokens")


def find_episodes(root: Path):
    for rf in sorted(Path(root).rglob(RESULT)):
        base = {"task": None, "init": None, "attempt": None, "requests": None, "end_turn": None}
        try:
            rec = json.loads(rf.read_text())
            if not isinstance(rec, dict):
                raise ValueError("result.json root must be an object")
            task, attempt = rec["task"], int(rec["attempt"])
            init = rec["init"]
            if not isinstance(task, str) or not isinstance(init, (int, str)) or isinstance(init, bool):
                raise ValueError("task must be a string and init an int or a string")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # unreadable result: an infrastructure failure of an episode we cannot even place in a group
            yield rf.parent, {**base, "task": "_unreadable", "init": str(rf.parent), "attempt": 0, "valid": False,
                              "reward": None, "error": f"unreadable result.json: {exc}"}
            continue
        valid = rec.get("valid") is True and isinstance(rec.get("reward"), (int, float)) and not isinstance(rec.get("reward"), bool)
        yield rf.parent, {**base, **{k: rec.get(k) for k in rec.get("report") or [] if isinstance(k, str)},
                          "task": task, "init": init, "attempt": attempt, "valid": valid,
                          "reward": float(rec["reward"]) if valid else None,
                          "requests": rec.get("requests") if isinstance(rec.get("requests"), int) else None,
                          "end_turn": rec.get("end_turn") if isinstance(rec.get("end_turn"), int) else None}


def group_key(row) -> tuple:
    return (row["task"], row["init"])


def group_name(gk) -> str:
    return f"{gk[0]}/{gk[1]}"


def outcome_kind(row) -> str:
    return "solved" if row["reward_terminal"] == 1 else "failed"


def post_success_turns(ep_dir: Path, n_turns: int):
    """result.json "end_turn": the calls after it are excluded; None when the harness did not say."""
    try:
        end = json.loads((Path(ep_dir) / RESULT).read_text()).get("end_turn")
    except (OSError, ValueError):
        return None
    if not isinstance(end, int) or isinstance(end, bool) or not 0 <= end < n_turns:
        return None
    return {"success_turn": end, "excluded": list(range(end + 1, n_turns)), "finish_kept": []}


def episode_extras(turns) -> dict:
    import gzip
    n = 0
    for tp in turns:
        n += len(json.loads(gzip.decompress(tp.read_bytes())).get("output_token_ids") or [])
    return {"n_output_tokens": n}
