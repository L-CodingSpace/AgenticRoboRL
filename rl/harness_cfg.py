"""harness.yaml: everything the single-machine controller (rl/local_ctl.py) needs to run ONE harness.

  name: my_harness
  adapter: result_json            # rl/adapters/<adapter>.py: how build_round reads the episodes
  tasks: tasks.jsonl              # one JSON object per line: {"task_id": "...", "init_ids": [...], "eval_init_ids": [...],
                                  #   "val_init_ids": [...]}: training, evaluation and (optional) validation inits, no init
                                  #   in two of them; `local_ctl.py eval --init-key val_init_ids` runs the validation inits
                                  # (any other field is available to the templates below)
  group_size: 6                   # episodes per (task, init) group
  episodes_parallel: 4
  proxy_port: 18200               # the recording proxy (rl/proxy.py) the harness talks to
  model:
    url: http://127.0.0.1:18100   # the model server the proxy forwards to (OpenAI-compatible, vLLM)
    serve_cmd: "vllm serve {weights} --port 18100 ..."   # optional: started for every round with that round's weights
    ready_url: http://127.0.0.1:18100/v1/models          # default: <url>/v1/models
    ready_timeout_s: 1800
  services:                       # optional: started before the episodes of a round, stopped after them
    - {name: sam3, cmd: "python -m ... --port 18114", ready_url: "http://127.0.0.1:18114/health", ready_timeout_s: 600}
  episode:
    cmd: "python my_agent.py --task {task_id} --init {init_id} --model-url {model_url} --out {episode_dir}"
    dir: "{task_path}/init_{init_id}/attempt_{attempt:02d}"   # under the round's episode root (this is the default)
    timeout_s: 7200
    on_timeout: failure           # failure = a valid episode with reward 0 (default) | invalid = not counted
    result_cmd: "python to_result.py {episode_dir}"           # optional: writes result.json after the episode
    cwd: .                        # default: the directory of harness.yaml
    env: {MUJOCO_GL: egl}
  build:
    turns_per_episode: 16
    args: ["--drop-post-success"]                             # extra rl/build_round.py arguments
  eval:                           # optional: how `local_ctl.py eval` differs from a training rollout
    direct: false                 # true = episodes talk to the model server directly (no recording proxy, so the
                                  #   harness's own sampling settings stay as they are); model_url = <model.url>/v1
    cmd: "..."                    # optional: the episode command for evaluation (default: episode.cmd)
  train:                          # optional; without it a round stops after the samples are built
    cmd: "torchrun --nproc_per_node 8 rl/train_rl.py --samples {samples} --model_path {weights0} --ref_files {weights0} {resume_arg} --output_dir {new_weights}"

Templates take: task_id, task_path (task_id made path-safe), init_id, attempt, episode_dir, episode_key, model_url
(= http://127.0.0.1:<proxy_port>/ep/<episode_key>/v1, the OpenAI base URL of THIS episode), weights, weights0, round,
run_dir, config_dir, python (this interpreter), every field of the task's line in tasks.jsonl, and every --var NAME=VALUE
given to rl/local_ctl.py. train.cmd also takes samples, build_dir, new_weights and resume_arg ("--resume <weights>" when
the round's weights come from an earlier round of this run and hold the trainer's state in dcp/, else empty). proxy_port and model.url / model.ready_url may use the --var names too, so several controllers
can share one harness.yaml on one machine (one per GPU: --var gpu=3 --var model_port=18113 --var proxy_port=18213).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

_SAFE = re.compile(r"[^A-Za-z0-9_.\-/]")
_KEY = re.compile(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$")   # what rl/proxy.py accepts as an episode key
DEFAULT_DIR = "{task_path}/init_{init_id}/attempt_{attempt:02d}"
INIT_KEYS = ("init_ids", "eval_init_ids", "val_init_ids")    # training / evaluation / validation inits of a task


class ConfigError(ValueError):
    pass


def safe_path(s: str) -> str:
    """task ids become directory names: anything outside [A-Za-z0-9_.-/] -> '_'; no '..', no leading or doubled '/'"""
    p = _SAFE.sub("_", str(s)).strip("/")
    p = "/".join(x for x in p.split("/") if x and x != "." and x != "..")
    if not p:
        raise ConfigError(f"task id {s!r} gives an empty path")
    return p


def fmt(template: str, where: str, **vars) -> str:
    try:
        return template.format(**vars)
    except KeyError as e:
        raise ConfigError(f"{where}: unknown placeholder {{{e.args[0]}}} (available: {sorted(vars)})") from None
    except (IndexError, ValueError) as e:
        raise ConfigError(f"{where}: bad template {template!r}: {e}") from None


def load_tasks(path: Path) -> list[dict]:
    tasks, seen = [], set()
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            t = json.loads(line)
        except ValueError as e:
            raise ConfigError(f"{path}:{n}: {e}") from None
        if not isinstance(t, dict) or not isinstance(t.get("task_id"), str) or not isinstance(t.get("init_ids"), list) or not t["init_ids"]:
            raise ConfigError(f"{path}:{n}: need {{\"task_id\": str, \"init_ids\": [non-empty]}}")
        lists = {k: t.get(k) or [] for k in INIT_KEYS}
        if any(not isinstance(v, list) for v in lists.values()):
            raise ConfigError(f"{path}:{n}: {', '.join(INIT_KEYS)} must be lists")
        if any(isinstance(i, bool) or not isinstance(i, (int, str)) for v in lists.values() for i in v):
            raise ConfigError(f"{path}:{n}: init ids must be ints or strings")
        for a, b in ((a, b) for k, a in enumerate(INIT_KEYS) for b in INIT_KEYS[k + 1:]):
            if set(lists[a]) & set(lists[b]):
                raise ConfigError(f"{path}:{n}: {a} and {b} share {sorted(set(lists[a]) & set(lists[b]), key=str)}")
        if t["task_id"] in seen:
            raise ConfigError(f"{path}:{n}: duplicate task_id {t['task_id']!r}")
        seen.add(t["task_id"])
        t["task_path"] = safe_path(t["task_id"])
        tasks.append(t)
    if not tasks:
        raise ConfigError(f"{path}: no tasks")
    if len({t["task_path"] for t in tasks}) != len(tasks):
        raise ConfigError(f"{path}: two task ids map to the same directory name")
    return tasks


def load(path: str | Path) -> dict:
    path = Path(path).resolve()
    try:
        cfg = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"{path}: {e}") from None
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: the top level must be a mapping")
    for k in ("name", "adapter", "tasks", "proxy_port"):
        if k not in cfg:
            raise ConfigError(f"{path}: missing '{k}'")
    ep = cfg.get("episode")
    if not isinstance(ep, dict) or not isinstance(ep.get("cmd"), str):
        raise ConfigError(f"{path}: missing 'episode.cmd'")
    model = cfg.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("url"), str):
        raise ConfigError(f"{path}: missing 'model.url'")
    cfg["config_dir"] = str(path.parent)
    cfg["tasks_list"] = load_tasks((path.parent / cfg["tasks"]).resolve())
    cfg.setdefault("group_size", 6)
    cfg.setdefault("episodes_parallel", 1)
    cfg.setdefault("services", [])
    cfg.setdefault("build", {})
    ev = cfg.setdefault("eval", {})
    if not isinstance(ev, dict):
        raise ConfigError(f"{path}: 'eval' must be a mapping")
    ev.setdefault("direct", False)
    if not isinstance(ev["direct"], bool) or ("cmd" in ev and not isinstance(ev["cmd"], str)):
        raise ConfigError(f"{path}: eval.direct must be true/false and eval.cmd a string")
    ep.setdefault("dir", DEFAULT_DIR)
    ep.setdefault("timeout_s", 7200)
    ep.setdefault("on_timeout", "failure")
    ep.setdefault("cwd", ".")
    ep.setdefault("env", {})
    if ep["on_timeout"] not in ("failure", "invalid"):
        raise ConfigError(f"{path}: episode.on_timeout must be 'failure' or 'invalid'")
    if not isinstance(cfg["group_size"], int) or cfg["group_size"] < 2:
        raise ConfigError(f"{path}: group_size must be an integer >= 2 (a group is compared within itself)")
    model.setdefault("ready_url", model["url"].rstrip("/") + "/v1/models")
    model.setdefault("ready_timeout_s", 1800)
    for s in cfg["services"]:
        if not isinstance(s, dict) or not isinstance(s.get("cmd"), str) or not isinstance(s.get("name"), str):
            raise ConfigError(f"{path}: every service needs 'name' and 'cmd'")
    return cfg


def apply_vars(cfg: dict, user_vars: dict) -> dict:
    """--var NAME=VALUE: kept in cfg["vars"] for every template, and resolved now in proxy_port / model.url / ready_url."""
    cfg["vars"] = dict(user_vars)
    v = {**cfg["vars"], "config_dir": cfg["config_dir"]}
    m = cfg["model"]
    m["url"] = fmt(str(m["url"]), "model.url", **v)
    m["ready_url"] = fmt(str(m["ready_url"]), "model.ready_url", **v)
    try:
        cfg["proxy_port"] = int(fmt(str(cfg["proxy_port"]), "proxy_port", **v))
    except ValueError:
        raise ConfigError(f"proxy_port must be an integer (got {cfg['proxy_port']!r})") from None
    return cfg


def parse_vars(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        k, sep, val = it.partition("=")
        if not sep or not k.isidentifier():
            raise ConfigError(f"--var wants NAME=VALUE (got {it!r})")
        out[k] = val
    return out


def episode_vars(cfg: dict, task: dict, init_id, attempt: int, eps_root: Path, weights: str, extra: dict | None = None,
                 direct: bool = False) -> dict:
    """The template variables of one episode; checks that its key is one the proxy accepts. direct: the episode talks to
    the model server itself (eval.direct), not to its proxy URL."""
    base = {**cfg.get("vars", {}), **{k: v for k, v in task.items() if k not in INIT_KEYS},
            "init_id": init_id, "attempt": attempt, "weights": weights, "config_dir": cfg["config_dir"], "python": sys.executable,
            **(extra or {})}
    key = fmt(cfg["episode"]["dir"], "episode.dir", **base)
    if not _KEY.match(key) or ".." in key.split("/"):
        raise ConfigError(f"episode.dir gives {key!r}: only [A-Za-z0-9_.-] and '/' are allowed (the proxy's episode key)")
    url = cfg["model"]["url"].rstrip("/") + "/v1" if direct else f"http://127.0.0.1:{cfg['proxy_port']}/ep/{key}/v1"
    return {**base, "episode_key": key, "episode_dir": str(eps_root / key), "model_url": url}
