#!/usr/bin/env python3
"""Run ONE CaP-X LIBERO episode for the RL controller and leave a result.json (harnesses/README.md).

CaP-X (https://github.com/capgym/cap-x, MIT) is used as it is: this script only writes a per-episode copy of one of its
YAML configs, calls its own headless launcher for exactly one trial, and reads the outcome CaP-X saved.

  $CAPX_PYTHON run_episode.py --capx-dir <cap-x checkout> --base-config env_configs/libero/franka_libero_spatial_0.yaml
      --suite libero_object_swap --task 3 --trial 17 --model-url http://127.0.0.1:18210/ep/<key>/v1 --model <served name>
      --out <episode dir>

What is changed in the copy of the config, and nothing else:
  env.cfg.low_level.suite_name / task_id   the task
  resume_idx = trials = <trial>            CaP-X runs trials resume_idx..trials; a trial number selects the initial state
                                           (state index (trial - 1) % number of init states) and seeds the environment
  num_workers 1, record_video false, output_dir <out>/capx/run
  use_parallel_ensemble / use_multimodel / use_img_differencing / use_video_differencing: false
                                           (they add model calls to other models; one policy is trained here)
  use_visual_feedback                      as asked (--visual-feedback): raw images to the model each turn
The config's api_servers stay: CaP-X skips a server whose port already answers, so servers started once by the controller
(harness.yaml services) are reused, and a stand-alone run still starts its own.

CaP-X retries a trial that hit its wall-clock limit up to 3 times; the abandoned tries' model calls would land in the same
proxy records as the final one, so the retry count is set to --retries (default 1 = no retry) before launching.

CaP-X gives up on a model request after 200 s and on a trial after 1000 s (constants in its code). A model that reasons
at length before answering needs more: --request-timeout-s and --trial-timeout-s set both for this run, in the running
process only (CaP-X's files are not edited). With the defaults nothing is changed.

Outcome: CaP-X saves <output>/.../trial_NN_sandboxrc_R_reward_X_taskcompleted_T/. reward 1.0 iff taskcompleted = 1
(LIBERO's own check_success()). No such directory = no result.json = the controller counts an infrastructure failure --
with one exception: CaP-X crashes (TypeError on a None) when the model's reply has no answer at all (everything went into
its reasoning). That is the policy failing, not the machinery: when the last recorded model reply of the episode has an
empty content, the episode is a valid failure (reward 0, "empty_answer": true).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
from pathlib import Path

import yaml

TRIAL_DIR = re.compile(r"^trial_(\d+)_sandboxrc_(-?\d+)_reward_(-?[0-9.]+)_taskcompleted_([01])$")


def make_config(base: dict, suite: str, task: int, trial: int, out: Path, visual_feedback: bool) -> dict:
    cfg = json.loads(json.dumps(base))   # deep copy of plain YAML data
    try:
        low = cfg["env"]["cfg"]["low_level"]
    except (KeyError, TypeError):
        raise SystemExit("base config has no env.cfg.low_level (not a CaP-X LIBERO config?)") from None
    if "suite_name" not in low or "task_id" not in low:
        raise SystemExit("base config's env.cfg.low_level has no suite_name / task_id")
    low["suite_name"], low["task_id"] = suite, int(task)
    cfg.update({"resume_idx": int(trial), "trials": int(trial), "num_workers": 1, "record_video": False,
                "output_dir": str(out / "capx" / "run"), "use_parallel_ensemble": False, "use_multimodel": False,
                "use_img_differencing": False, "use_video_differencing": False, "use_visual_feedback": bool(visual_feedback)})
    return cfg


def find_outcome(out: Path, trial: int) -> dict | None:
    """The saved outcome of `trial` under <out>/capx, or None. Several dirs for one trial (should not happen): the newest."""
    found = []
    for d in (out / "capx").rglob("trial_*"):
        m = TRIAL_DIR.match(d.name)
        if d.is_dir() and m and int(m.group(1)) == int(trial):
            found.append((d.stat().st_mtime, d, m))
    if not found:
        return None
    _, d, m = max(found, key=lambda x: x[0])
    summary = (d / "summary.txt").read_text(errors="replace") if (d / "summary.txt").is_file() else ""
    done = m.group(4) == "1"
    return {"reward": 1.0 if done else 0.0, "valid": True, "capx_reward": float(m.group(3)), "sandbox_rc": int(m.group(2)),
            "timed_out": "timed out after" in summary, "capx_trial_dir": str(d), "empty_answer": False,
            "report": ["empty_answer", "capx_reward", "sandbox_rc", "timed_out"]}


def alarm_with(real_alarm, trial_timeout_s: int):
    """signal.alarm, with CaP-X's literal 1000 s replaced by our trial limit; every other value passes through."""
    def alarm(seconds):
        return real_alarm(trial_timeout_s if seconds == 1000 else seconds)
    return alarm


class with_timeout:
    """`requests`, with every post() given our timeout instead of the caller's (CaP-X hard-codes 200 s)."""

    def __init__(self, real, timeout_s: float):
        self._real, self._timeout_s = real, timeout_s

    def __getattr__(self, name):
        return getattr(self._real, name)

    def post(self, *args, **kwargs):
        kwargs["timeout"] = self._timeout_s
        return self._real.post(*args, **kwargs)


def last_reply_is_empty(out: Path) -> bool:
    """True when the proxy recorded at least one model call for this episode and the last reply carried no content."""
    import gzip
    turns = sorted((out / "rl_turns").glob("turn_*.json.gz"))
    if not turns:
        return False
    try:
        msg = json.loads(gzip.decompress(turns[-1].read_bytes())).get("message") or {}
    except (OSError, ValueError):
        return False
    c = msg.get("content")
    return c is None or (isinstance(c, str) and not c.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capx-dir", required=True)
    ap.add_argument("--base-config", required=True, help="a CaP-X LIBERO YAML, relative to --capx-dir or absolute")
    ap.add_argument("--suite", required=True); ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--trial", type=int, required=True, help="CaP-X trial number (>= 1) = initial state and environment seed")
    ap.add_argument("--model-url", required=True, help="OpenAI base URL of this episode (…/v1)")
    ap.add_argument("--model", required=True, help="the served model name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--visual-feedback", action="store_true")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--request-timeout-s", type=float, default=0, help="seconds CaP-X waits for one model reply (0 = its own 200)")
    ap.add_argument("--trial-timeout-s", type=int, default=0, help="CaP-X's wall-clock limit of the trial (0 = its own 1000)")
    a = ap.parse_args()
    if a.trial < 1:
        raise SystemExit("--trial must be >= 1 (CaP-X numbers trials from 1)")
    capx = Path(a.capx_dir).resolve(); out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base_path = Path(a.base_config) if os.path.isabs(a.base_config) else capx / a.base_config
    cfg = make_config(yaml.safe_load(base_path.read_text()), a.suite, a.task, a.trial, out, a.visual_feedback)
    cfg_path = out / "capx_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    os.chdir(capx)                      # CaP-X resolves its assets and relative paths from its checkout
    sys.path.insert(0, str(capx))
    from capx.envs import launch, runner   # noqa: E402  (needs CaP-X's own environment)
    runner.MAX_TRIAL_RETRIES = max(1, a.retries)
    if a.trial_timeout_s > 0:
        runner.TRIAL_TIMEOUT_SECONDS = a.trial_timeout_s
        # CaP-X re-arms the trial timer after the scene is loaded with a literal signal.alarm(1000) (capx/envs/trial.py):
        # without this, a longer limit only lasted until the environment reset (episodes were cut at
        # ~1020 s although the limit was 3600)
        signal.alarm = alarm_with(signal.alarm, a.trial_timeout_s)
    if a.request_timeout_s > 0:
        from capx.llm import client as _client
        _client.requests = with_timeout(_client.requests, a.request_timeout_s)
    args = launch.LaunchArgs(config_path=str(cfg_path), server_url=a.model_url.rstrip("/") + "/chat/completions", model=a.model,
                             temperature=a.temperature, max_tokens=a.max_tokens, total_trials=a.trial, num_workers=1)
    rc = 0
    try:
        launch.main(args)
    except SystemExit as e:
        rc = int(e.code or 0) if isinstance(e.code, int) else 1
    except Exception as e:  # noqa: BLE001 - the outcome on disk decides, the exception is kept for the log
        print(f"run_episode: CaP-X raised {type(e).__name__}: {e}", file=sys.stderr)
        rc = 1
    res = find_outcome(out, a.trial)
    if res is None and last_reply_is_empty(out):
        print("run_episode: CaP-X saved no trial directory and the model's last reply was empty: a valid failure", file=sys.stderr)
        res = {"reward": 0.0, "valid": True, "empty_answer": True, "capx_reward": None, "sandbox_rc": None, "timed_out": False,
               "report": ["empty_answer", "capx_reward", "sandbox_rc", "timed_out"]}
    if res is None:
        print("run_episode: CaP-X saved no trial directory; no result.json", file=sys.stderr)
        return rc or 1
    (out / "result.json").write_text(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
