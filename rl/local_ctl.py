#!/usr/bin/env python3
"""Single-machine RL controller for any harness described by a harness.yaml (rl/harness_cfg.py).

  python rl/local_ctl.py train --config harness.yaml --run-dir runs/x --weights <start weights> --rounds 3 --groups 16
  python rl/local_ctl.py eval  --config harness.yaml --run-dir runs/x --weights <weights> --tag v000 --init-key val_init_ids

One training round K (policy version vK):

  rollout   pick --groups (task, init) groups (never one used before in this run); start the model server on vK's
            weights, the recording proxy (rl/proxy.py) and the harness's services; run group_size episodes per group,
            each talking to its own proxy URL; stop everything started here
  build     rl/build_round.py --adapter <adapter>  ->  rounds/rK/build/{samples.jsonl, round_stats.json}
  train     the harness.yaml train.cmd  ->  weights/v(K+1)   (skipped when no group has a non-zero advantage)

eval runs every task once per eval init (no training) and writes eval/<tag>/report.json; its per-task success rates
can restrict the training pool (--pool-report, --rate-lo, --rate-hi: only tasks that sometimes succeed and sometimes fail
carry a signal).

The run is resumable: state.json holds the round and phase, finished episodes (episode_done.json) are not rerun.
Stop it by creating <run dir>/STOP (checked between episodes and between phases). Only processes started here are
ever signalled, by their own process group.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapters as _adapters  # noqa: E402
import harness_cfg as HC  # noqa: E402

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def version(k: int) -> str:
    return f"v{k:03d}"


def resolve_cwd(cfg: dict, rel: str) -> str:
    """a cwd from harness.yaml: environment variables expanded ($RPENT_DIR), relative to the directory of harness.yaml"""
    return str((Path(cfg["config_dir"]) / os.path.expandvars(rel)).resolve())


# ---------------------------------------------------------------------------
# processes we start (and are the only ones we stop)
# ---------------------------------------------------------------------------


class Proc:
    def __init__(self, name: str, cmd: str, log_path: Path, cwd: str, env: dict | None = None):
        self.name = name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path
        self._log = open(log_path, "ab")
        self.p = subprocess.Popen(cmd, shell=True, cwd=cwd, env={**os.environ, **{k: str(v) for k, v in (env or {}).items()}},
                                  stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True)

    def alive(self) -> bool:
        return self.p.poll() is None

    def stop(self, grace_s: float = 20.0) -> None:
        """TERM to the process group, KILL to whatever of the group is still there after grace_s. The group is what counts,
        not the shell we started: `sh -c "... vllm serve ..."` dies on TERM at once while the server it started can take
        long or hang (a model server can survive a stop that only waits for the shell)."""
        def group_alive() -> bool:
            try:
                os.killpg(self.p.pid, 0)      # its own session = its own process group: nothing else is hit
                return True
            except (ProcessLookupError, PermissionError):
                return False
        if group_alive():
            for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 10.0)):
                try:
                    os.killpg(self.p.pid, sig)
                except ProcessLookupError:
                    break
                deadline = time.time() + wait
                while time.time() < deadline:
                    self.p.poll()             # reap the shell so that it does not count as a live group member
                    if not group_alive():
                        break
                    time.sleep(0.2)
                if not group_alive():
                    break
        self.p.poll()
        self._log.close()


def wait_ready(proc: Proc, url: str | None, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while True:
        if not proc.alive():
            raise RuntimeError(f"{proc.name} exited with {proc.p.returncode} while starting (log: {proc.log_path})")
        if url is None:
            return
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status < 500:
                    return
        except Exception:  # noqa: BLE001 - not up yet
            pass
        if time.time() > deadline:
            raise RuntimeError(f"{proc.name} not ready after {timeout_s:.0f} s at {url} (log: {proc.log_path})")
        time.sleep(1.0)


class Stack:
    """model server (optional) + recording proxy + the harness's services, for one set of weights"""

    def __init__(self, cfg: dict, eps_root: Path, logs: Path, weights: str, policy_version: str, run_dir: Path,
                 proxy: bool = True):
        self.procs: list[Proc] = []
        v = {**cfg.get("vars", {}), "weights": weights, "config_dir": cfg["config_dir"], "run_dir": str(run_dir), "python": sys.executable}
        cwd = resolve_cwd(cfg, cfg["episode"]["cwd"])
        try:
            m = cfg["model"]
            if m.get("serve_cmd"):
                p = Proc("model", HC.fmt(m["serve_cmd"], "model.serve_cmd", **v), logs / "model.log", cwd)
                self.procs.append(p)
                wait_ready(p, m["ready_url"], m["ready_timeout_s"])
            eps_root.mkdir(parents=True, exist_ok=True)
            if proxy:
                cmd = (f"{shlex.quote(sys.executable)} {shlex.quote(str(HERE / 'proxy.py'))} --port {int(cfg['proxy_port'])} "
                       f"--upstream {shlex.quote(m['url'].rstrip('/'))} --out-root {shlex.quote(str(eps_root))} "
                       f"--policy-version {policy_version} --timeout-s {float(cfg['episode']['timeout_s'])}")
                p = Proc("proxy", cmd, logs / "proxy.log", str(HERE), env={"PYTHONPATH": ""})
                self.procs.append(p)
                wait_ready(p, f"http://127.0.0.1:{int(cfg['proxy_port'])}/healthz", 120)
            for s in cfg["services"]:
                p = Proc(s["name"], HC.fmt(s["cmd"], f"services.{s['name']}.cmd", **v), logs / f"service_{s['name']}.log", cwd)
                self.procs.append(p)
                wait_ready(p, s.get("ready_url"), s.get("ready_timeout_s", 600))
        except Exception:
            self.stop()
            raise

    def check(self) -> None:
        for p in self.procs:
            if not p.alive():
                raise RuntimeError(f"{p.name} died during the round (log: {p.log_path})")

    def stop(self) -> None:
        for p in reversed(self.procs):
            p.stop()
        self.procs = []


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------


def run_episode(cfg: dict, v: dict, stop_file: Path) -> dict:
    """One episode. Returns its episode_done.json content (also when it had been finished before)."""
    ep = Path(v["episode_dir"])
    done = ep / "episode_done.json"
    if done.is_file():
        try:
            return {**json.loads(done.read_text()), "resumed": True}
        except ValueError:
            pass
    if stop_file.exists():
        return {"skipped": "STOP"}
    if ep.exists():
        # a half-run episode: its records are not trusted (the harness may append to its own files)
        ep.rename(ep.with_name(f"{ep.name}.stale_{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"))
    ep.mkdir(parents=True)
    e = cfg["episode"]
    cwd = resolve_cwd(cfg, e["cwd"])
    t0 = time.time()
    proc = Proc("episode", HC.fmt(e["cmd"], "episode.cmd", **v), ep / "episode.log", cwd, env=e["env"])
    timed_out = False
    try:
        proc.p.wait(timeout=float(e["timeout_s"]))
    except subprocess.TimeoutExpired:
        timed_out = True
    proc.stop(grace_s=10.0)
    rc = proc.p.returncode
    if e.get("result_cmd"):
        r = subprocess.run(HC.fmt(e["result_cmd"], "episode.result_cmd", **v), shell=True, cwd=cwd, capture_output=True, text=True,
                           env={**os.environ, **{k: str(x) for k, x in e["env"].items()}})
        if r.returncode != 0:
            (ep / "result_cmd.log").write_text(r.stdout + r.stderr)
    info = {"task_id": v["task_id"], "init_id": v["init_id"], "attempt": v["attempt"], "episode_key": v["episode_key"],
            "rc": rc, "timed_out": timed_out, "t_start": t0, "t_end": time.time()}
    if cfg["adapter"] == "result_json":
        info["result"] = finish_result(cfg, ep, v, rc, timed_out)
    write_json(done, info)
    return info


def finish_result(cfg: dict, ep: Path, v: dict, rc: int, timed_out: bool) -> str:
    """result.json for the generic adapter: which group the episode belongs to is OURS to say (we launched it); how it
    ended is the harness's. No result from the harness: a timeout is what episode.on_timeout says, anything else is an
    infrastructure failure."""
    rf = ep / "result.json"
    rec, how = None, "harness"
    try:
        rec = json.loads(rf.read_text())
        if not isinstance(rec, dict):
            rec = None
    except (OSError, ValueError):
        rec = None
    if rec is None:
        if timed_out and cfg["episode"]["on_timeout"] == "failure":
            rec, how = {"reward": 0.0, "valid": True, "timed_out": True}, "timeout_failure"
        else:
            rec, how = {"reward": None, "valid": False, "error": "timed out" if timed_out else f"no result.json (exit code {rc})"}, "missing"
    rec.update({"task": v["task_id"], "init": v["init_id"], "attempt": v["attempt"]})
    write_json(rf, rec)
    return how


def run_episodes(cfg: dict, jobs: list[dict], stack: Stack, stop_file: Path) -> list[dict]:
    out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(cfg["episodes_parallel"])) as pool:
        futs = {pool.submit(run_episode, cfg, v, stop_file): v for v in jobs}
        for n, f in enumerate(concurrent.futures.as_completed(futs), 1):
            v = futs[f]
            info = f.result()
            out.append(info)
            if "skipped" not in info:
                log(f"episode {n}/{len(jobs)} {v['episode_key']} rc={info.get('rc')} timed_out={info.get('timed_out')}"
                    + (" (already done)" if info.get("resumed") else ""))
            stack.check()
    return out


# ---------------------------------------------------------------------------
# which groups a round runs
# ---------------------------------------------------------------------------


def pool_tasks(tasks: list[dict], report: dict | None, lo: float, hi: float) -> list[dict]:
    """Tasks eligible for training: all of them, or those whose success rate in an eval report lies in [lo, hi]."""
    if report is None:
        return list(tasks)
    rates = {t: r["success_rate"] for t, r in report["tasks"].items() if r.get("success_rate") is not None}
    return [t for t in tasks if t["task_id"] in rates and lo <= rates[t["task_id"]] <= hi]


def choose_groups(pool: list[dict], k: int, n_groups: int, used: set, seed: int) -> list[list]:
    """n_groups (task_id, init_id) pairs spread evenly over the pool's tasks (each task floor or ceil of n_groups / tasks;
    which get the extra one: seeded shuffle), never a pair in `used`. Fewer when the pool runs out of unused inits."""
    rng = random.Random(seed * 100003 + 1000 + k)
    tasks = sorted(pool, key=lambda t: t["task_id"])
    rng.shuffle(tasks)
    used = set(used)
    groups: list[list] = []
    while len(groups) < n_groups:
        added = False
        for t in tasks:
            if len(groups) == n_groups:
                break
            free = [i for i in t["init_ids"] if (t["task_id"], i) not in used]
            if not free:
                continue
            i = rng.choice(free)
            used.add((t["task_id"], i)); groups.append([t["task_id"], i]); added = True
        if not added:
            break
    return groups


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def weights_arg(w: str) -> str:
    """a local directory -> its absolute path; anything else (a model id on a hub) stays as given"""
    return str(Path(w).resolve()) if Path(w).exists() else w


def build_samples(cfg: dict, eps: Path, build: Path, rd: Path, V: str, k: int) -> dict:
    """rl/build_round.py over a round's episodes -> build/{samples.jsonl, round_stats.json}; returns round_stats"""
    b = cfg["build"]
    cmd = [sys.executable, str(HERE / "build_round.py"), "--root", str(eps), "--policy-version", V, "--round-seed", str(k + 1),
           "--mode", "rloo", "--turns-per-episode", str(b.get("turns_per_episode", 16)), "--group-size", str(cfg["group_size"]),
           "--out", str(build), "--adapter", cfg["adapter"], *[str(x) for x in b.get("args", [])]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    (rd / "build.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"round {k}: build_round failed, see {rd / 'build.log'}")
    return json.loads((build / "round_stats.json").read_text())


def train_command(cfg: dict, build: Path, weights: str, weights0: str, new: Path, k: int, run_dir: Path,
                  python: str = sys.executable) -> tuple[str, str]:
    """(train.cmd filled in, its cwd). After round 0 the trainer continues from the previous round's state
    (weights + optimizer in <weights>/dcp): resume_arg."""
    resume_arg = f"--resume {shlex.quote(weights)}" if (Path(weights) / "dcp").is_dir() else ""
    cmd = HC.fmt(cfg["train"]["cmd"], "train.cmd", **cfg.get("vars", {}), samples=str(build / "samples.jsonl"), build_dir=str(build),
                 weights=weights, weights0=weights0, new_weights=str(new), round=k, run_dir=str(run_dir),
                 config_dir=cfg["config_dir"], python=python, resume_arg=resume_arg)
    return cmd, resolve_cwd(cfg, cfg["train"].get("cwd", "."))


def eval_report(cfg: dict, eps: Path, jobs: list[dict], tag: str, weights: str, shard: str) -> dict:
    """per-task success rates of an evaluation: how each episode ended comes from the adapter, like in training;
    which task it was is what we launched"""
    AD = _adapters.load(cfg["adapter"])
    task_of = {str(Path(v["episode_dir"])): v["task_id"] for v in jobs}
    per: dict[str, dict] = {t["task_id"]: {"episodes": 0, "valid": 0, "reward_sum": 0.0} for t in cfg["tasks_list"]}
    for ep, verdict in AD.find_episodes(eps):
        tid = task_of.get(str(ep))
        if tid is None:
            continue
        per[tid]["episodes"] += 1
        if verdict["valid"]:
            per[tid]["valid"] += 1; per[tid]["reward_sum"] += float(verdict["reward"])
    for r in per.values():
        r["success_rate"] = (r["reward_sum"] / r["valid"]) if r["valid"] else None
    tot_v = sum(r["valid"] for r in per.values())
    return {"tag": tag, "weights": weights, "shard": shard or "0/1", "launched": len(jobs), "valid": tot_v,
            "success_rate": (sum(r["reward_sum"] for r in per.values()) / tot_v) if tot_v else None, "tasks": per}


def eval_plan(cfg: dict, init_key: str, repeats: int) -> list[tuple]:
    """(task, init, attempt) of every evaluation episode, in tasks.jsonl order"""
    plan = []
    for t in cfg["tasks_list"]:
        inits = t.get(init_key) or []
        if not inits:
            raise SystemExit(f"task {t['task_id']!r} has no {init_key} in {cfg['tasks']}")
        plan += [(t, i, att) for i in inits for att in range(int(t.get("repeats", repeats)))]
    return plan


def load_state(run_dir: Path, a) -> dict:
    sp = run_dir / "state.json"
    if sp.is_file():
        st = json.loads(sp.read_text())
        if a.weights and st["weights"]["v000"] != weights_arg(a.weights):
            raise SystemExit(f"{sp} was started from {st['weights']['v000']}, not {a.weights}: use another --run-dir")
        return st
    if not a.weights:
        raise SystemExit("--weights is needed to start a run")
    return {"round": 0, "phase": "rollout", "used": [], "weights": {"v000": weights_arg(a.weights)}, "history": [],
            "seed": a.seed, "config": str(Path(a.config).resolve())}


def cmd_train(a) -> int:
    cfg = HC.apply_vars(HC.load(a.config), HC.parse_vars(a.var))
    _adapters.load(cfg["adapter"])
    run_dir = Path(a.run_dir).resolve(); run_dir.mkdir(parents=True, exist_ok=True)
    stop_file = run_dir / "STOP"
    st = load_state(run_dir, a)
    report = json.loads(Path(a.pool_report).read_text()) if a.pool_report else None
    pool = pool_tasks(cfg["tasks_list"], report, a.rate_lo, a.rate_hi)
    if not pool:
        raise SystemExit("the training pool is empty (no task with a success rate in [--rate-lo, --rate-hi])")
    by_id = {t["task_id"]: t for t in cfg["tasks_list"]}
    G = int(cfg["group_size"])
    while st["round"] < a.rounds:
        k = st["round"]; V = version(k)
        rd = run_dir / "rounds" / f"r{k}"; eps = rd / "eps"; build = rd / "build"
        rd.mkdir(parents=True, exist_ok=True)
        weights = st["weights"][V]
        if stop_file.exists():
            log(f"STOP found: leaving at round {k} phase {st['phase']}"); return 3
        if st["phase"] == "rollout":
            gp = rd / "groups.json"
            if gp.is_file():
                groups = json.loads(gp.read_text())
            else:
                groups = choose_groups(pool, k, a.groups, {tuple(x) for x in st["used"]}, st["seed"])
                if not groups:
                    raise SystemExit(f"round {k}: no unused (task, init) pair left in the pool")
                write_json(gp, groups)
                st["used"] += groups; write_json(run_dir / "state.json", st)
            log(f"round {k} ({V}): {len(groups)} groups x {G} episodes, weights {weights}")
            jobs = [HC.episode_vars(cfg, by_id[t], i, att, eps, weights, {"round": k, "run_dir": str(run_dir), "weights0": st["weights"]["v000"]})
                    for t, i in groups for att in range(G)]
            stack = Stack(cfg, eps, rd / "logs", weights, V, run_dir)
            try:
                infos = run_episodes(cfg, jobs, stack, stop_file)
            finally:
                stack.stop()
            if any("skipped" in i for i in infos):
                log(f"STOP found: round {k} rollouts incomplete"); return 3
            st["phase"] = "build"; write_json(run_dir / "state.json", st)
        if st["phase"] == "build":
            stats = build_samples(cfg, eps, build, rd, V, k)
            st["history"].append({"round": k, "version": V, "episodes": stats["episodes"], "status": stats["status"],
                                  "success_rate_valid": stats["success_rate_valid"], "groups_used": stats["groups_used"],
                                  "samples": stats["samples"]})
            log(f"round {k}: success {stats['success_rate_valid']:.3f}, groups {stats['groups_used']}, samples {stats['samples']}")
            st["phase"] = "train"; write_json(run_dir / "state.json", st)
        if st["phase"] == "train":
            VN = version(k + 1); new = run_dir / "weights" / VN
            n_samples = st["history"][-1]["samples"]
            if not cfg.get("train") or a.no_train:
                log(f"round {k}: no training (no train.cmd or --no-train); the run ends here"); st["weights"][VN] = weights
                st["round"] = k + 1; st["phase"] = "rollout"; write_json(run_dir / "state.json", st)
                break
            if n_samples == 0:
                log(f"round {k}: no group with a non-zero advantage -> nothing to train on, {VN} = {V}")
                st["weights"][VN] = weights
            else:
                new.parent.mkdir(parents=True, exist_ok=True)
                if new.exists():
                    # left by a training that did not finish (state.json still says "train"): never train into it
                    new.rename(new.with_name(f"{new.name}.stale_{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"))
                cmd, cwd = train_command(cfg, build, weights, st["weights"]["v000"], new, k, run_dir)
                with open(rd / "train.log", "ab") as lf:
                    r = subprocess.run(cmd, shell=True, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
                if r.returncode != 0 or not new.is_dir():
                    raise SystemExit(f"round {k}: training failed (exit {r.returncode}, {new} {'exists' if new.is_dir() else 'missing'}), see {rd / 'train.log'}")
                st["weights"][VN] = str(new)
                log(f"round {k}: trained {VN} on {n_samples} samples -> {new}")
            st["round"] = k + 1; st["phase"] = "rollout"; write_json(run_dir / "state.json", st)
    log(f"done: {st['round']} round(s); latest weights {st['weights'][version(st['round'])]}")
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def parse_shard(spec: str | None) -> tuple[int, int]:
    """'i/n' -> (i, n): this controller takes the tasks whose index in tasks.jsonl is i modulo n ('' = all of them)"""
    if not spec:
        return 0, 1
    try:
        i, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise HC.ConfigError(f"--shard wants I/N (got {spec!r})") from None
    if not 0 <= i < n:
        raise HC.ConfigError(f"--shard {spec}: need 0 <= I < N")
    return i, n


def cmd_eval(a) -> int:
    cfg = HC.apply_vars(HC.load(a.config), HC.parse_vars(a.var))
    si, sn = parse_shard(a.shard)
    if a.shard_by == "task":
        cfg["tasks_list"] = [t for k, t in enumerate(cfg["tasks_list"]) if k % sn == si]
    # (task, init, attempt) of every episode, in tasks.jsonl order; --shard-by episode deals THEM out instead of the tasks
    # (few tasks with many episodes each, e.g. collecting data on 26 tasks with 64 GPUs)
    plan = eval_plan(cfg, a.init_key, a.repeats)
    if a.shard_by == "episode":
        plan = [x for k, x in enumerate(plan) if k % sn == si]
        mine = {x[0]["task_id"] for x in plan}
        cfg["tasks_list"] = [t for t in cfg["tasks_list"] if t["task_id"] in mine]
    if not plan:
        # more shards than tasks (e.g. 64 GPUs, 60 tasks): nothing to run here, and that is not an error -- an empty
        # report keeps "one report per shard" true for whoever merges them
        out = Path(a.run_dir).resolve() / "eval" / a.tag; out.mkdir(parents=True, exist_ok=True)
        write_json(out / "report.json", {"tag": a.tag, "weights": weights_arg(a.weights), "shard": a.shard, "launched": 0, "valid": 0,
                                         "success_rate": None, "tasks": {}})
        log(f"eval {a.tag}: shard {a.shard} has no task; empty report written")
        return 0
    run_dir = Path(a.run_dir).resolve()
    out = run_dir / "eval" / a.tag; eps = out / "eps"
    out.mkdir(parents=True, exist_ok=True)
    weights = weights_arg(a.weights)
    direct = cfg["eval"]["direct"]
    if cfg["eval"].get("cmd"):
        cfg["episode"]["cmd"] = cfg["eval"]["cmd"]
    jobs = [HC.episode_vars(cfg, t, i, att, eps, weights, {"round": 0, "run_dir": str(run_dir), "weights0": weights}, direct=direct)
            for t, i, att in plan]
    log(f"eval {a.tag}: {len(jobs)} episodes, weights {weights}" + (" (direct to the model server, no proxy)" if direct else ""))
    stack = Stack(cfg, eps, out / "logs", weights, a.tag, run_dir, proxy=not direct)
    try:
        infos = run_episodes(cfg, jobs, stack, run_dir / "STOP")
    finally:
        stack.stop()
    if any("skipped" in i for i in infos):
        log("STOP found: eval incomplete"); return 3
    report = eval_report(cfg, eps, jobs, a.tag, weights, a.shard)
    write_json(out / "report.json", report)
    log(f"eval {a.tag}: {report['success_rate']} over {report['valid']} valid of {len(jobs)} episodes -> {out / 'report.json'}")
    return 0


def cmd_merge(a) -> int:
    """Sum the per-task counts of several eval reports (one per shard) into one report.json."""
    per: dict[str, dict] = {}
    weights, launched = set(), 0
    for f in a.reports:
        r = json.loads(Path(f).read_text())
        weights.add(r["weights"]); launched += r["launched"]
        for tid, t in r["tasks"].items():
            if tid in per:
                if not a.sum_tasks:
                    raise SystemExit(f"task {tid!r} is in two of the reports: shards must not overlap")
                for k in ("episodes", "valid", "reward_sum"):
                    per[tid][k] += t[k]
                per[tid]["success_rate"] = (per[tid]["reward_sum"] / per[tid]["valid"]) if per[tid]["valid"] else None
                continue
            per[tid] = {k: t[k] for k in ("episodes", "valid", "reward_sum", "success_rate")}
    if len(weights) != 1:
        raise SystemExit(f"the reports are of different weights: {sorted(weights)}")
    tot_v = sum(t["valid"] for t in per.values())
    out = {"tag": a.tag, "weights": weights.pop(), "merged_from": [str(Path(f).resolve()) for f in a.reports], "launched": launched,
           "valid": tot_v, "success_rate": (sum(t["reward_sum"] for t in per.values()) / tot_v) if tot_v else None,
           "tasks": dict(sorted(per.items()))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(a.out), out)
    in_pool = sum(1 for t in per.values() if t["success_rate"] is not None and a.rate_lo <= t["success_rate"] <= a.rate_hi)
    log(f"merged {len(a.reports)} reports: {len(per)} tasks, {tot_v} valid of {launched} episodes, success {out['success_rate']}; "
        f"{in_pool} tasks with a success rate in [{a.rate_lo}, {a.rate_hi}] -> {a.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True); t.add_argument("--run-dir", required=True)
    t.add_argument("--weights", default=None, help="start weights (v000); not needed when resuming")
    t.add_argument("--rounds", type=int, required=True, help="total number of rounds of the run")
    t.add_argument("--groups", type=int, required=True, help="(task, init) groups per round")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--pool-report", default=None, help="eval report.json: train only on tasks with a success rate in [--rate-lo, --rate-hi]")
    t.add_argument("--rate-lo", type=float, default=0.1); t.add_argument("--rate-hi", type=float, default=0.9)
    t.add_argument("--no-train", action="store_true", help="rollouts and samples only")
    t.add_argument("--var", action="append", default=[], metavar="NAME=VALUE", help="extra template variable (repeatable)")
    e = sub.add_parser("eval")
    e.add_argument("--config", required=True); e.add_argument("--run-dir", required=True)
    e.add_argument("--weights", required=True); e.add_argument("--tag", required=True)
    e.add_argument("--repeats", type=int, default=1, help="episodes per (task, init); a task's own \"repeats\" in tasks.jsonl wins")
    e.add_argument("--var", action="append", default=[], metavar="NAME=VALUE", help="extra template variable (repeatable)")
    e.add_argument("--shard", default="", metavar="I/N", help="only the tasks whose index in tasks.jsonl is I modulo N (one controller per GPU)")
    e.add_argument("--shard-by", choices=("task", "episode"), default="task",
                   help="episode: deal out the episodes, not the tasks (a task is then in several reports: merge with --sum-tasks)")
    e.add_argument("--init-key", required=True, choices=HC.INIT_KEYS,
                   help="which inits of tasks.jsonl to run, always named: val_init_ids (choosing checkpoints / the training pool), "
                        "eval_init_ids (reported numbers), init_ids (training inits, e.g. collecting data)")
    g = sub.add_parser("merge-eval")
    g.add_argument("--sum-tasks", action="store_true", help="a task in several reports: add its counts up (reports of --shard-by episode)")
    g.add_argument("--out", required=True); g.add_argument("--tag", required=True)
    g.add_argument("--rate-lo", type=float, default=0.1); g.add_argument("--rate-hi", type=float, default=0.9)
    g.add_argument("reports", nargs="+")
    a = ap.parse_args()
    try:
        return {"train": cmd_train, "eval": cmd_eval, "merge-eval": cmd_merge}[a.what](a)
    except HC.ConfigError as exc:
        raise SystemExit(f"config error: {exc}") from None


if __name__ == "__main__":
    sys.exit(main())
