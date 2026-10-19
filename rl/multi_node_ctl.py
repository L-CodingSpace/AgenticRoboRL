#!/usr/bin/env python3
"""Multi-machine RL controller: the episodes of a round (or of an evaluation) are spread over several machines; the
samples are built and the policy is trained as in rl/local_ctl.py (same functions). Nothing in it is specific to one
cluster: machines are reached through a command template (ssh by default) and all of them share one directory (NFS or
similar) that holds the run. The run directory, the weights and the harness's data must have the same absolute paths
on every machine.

  python rl/multi_node_ctl.py train --cluster cluster.yaml --config harnesses/x/harness.yaml --run-dir /shared/runs/x \\
         --weights /shared/ckpt/start --rounds 3 --groups 64 [--pool-report /shared/runs/x/eval/v000/report.json]
  python rl/multi_node_ctl.py eval  --cluster cluster.yaml --config harnesses/x/harness.yaml --run-dir /shared/runs/x \\
         --weights /shared/ckpt/start --tag v000 --init-key val_init_ids

cluster.yaml (an annotated example: rl/cluster.example.yaml):
  hosts: [gpu-node-01, gpu-node-02]      machines that run episodes; an entry may also be
                                         {name: gpu-node-03, gpus: [0, 1], exec: "ssh -p 2222 {host}"} (own GPUs / own exec)
  gpus: [0, 1, 2, 3, 4, 5, 6, 7]         one episode worker per listed GPU of every host ({gpu} in the templates)
  exec: "ssh -o BatchMode=yes {host}"    runs a command on a host; the command follows as ONE argument
                                         ("" = on this machine: tests, or one machine with several workers)
  code_dir: /shared/AgenticRoboRL        this repository as the hosts see it
  python: /shared/envs/rl/bin/python     python on the hosts (pyyaml, fastapi, uvicorn, httpx)
  setup_cmd: ""                          optional: run on a host before its first worker of this controller
  port_vars: {model_port: 18100, proxy_port: 18200}   template variables that are <base> + <gpu> on each worker
  vars: {}                               more template variables for every worker and for train.cmd
  train_host: gpu-node-01                where harness.yaml's train.cmd runs
  lease_s: 1200                          an episode whose worker has been silent this long is run again elsewhere
  max_leases: 3                          ... at most this many times in all (then it counts as invalid)
  poll_s: 60
  worker_restarts: 3                     per host and phase: dead workers are started again at most this often
  train_restarts: 2                      a failed training is started again (from the same state) at most this often
  train_silence_s: 3600                  a training that has stopped beating for this long counts as failed

How the work is shared (files in the phase directory only; the controller never talks to a worker directly):
  jobs.json                 the phase's episodes, written once by the controller
  claims/<j>.claim          created with O_EXCL by the worker that takes episode j (exactly one wins); holds its id
  claims/<j>.done           episode j finished (written by its worker)
  claims/<j>.claim.stale.N  the controller -- the ONLY process that takes a claim away -- renames the claim of a worker
                            that has stopped beating for lease_s; episode j is free again and runs under eps/lease_<N+1>/,
                            so a late writer of the old lease never touches the new episode directory; after max_leases
                            leases it is marked claims/<j>.invalid
  workers/<label>~<id>.beat.<n>  a worker's heartbeat: every 30 s it creates the next number and deletes the previous one
A worker starts the model server, the recording proxy and the harness's services on its GPU (rl/local_ctl.Stack) at its
first episode, takes free episodes until every episode is done or invalid, and stops everything it started. The
controller starts workers (after setup_cmd) on the hosts, starts them again when they die, builds the samples, runs
train.cmd on train_host and goes on to the next round. state.json makes the controller resumable; <run dir>/STOP stops
the controller and the workers between episodes.

Liveness never relies on file times or on clocks agreeing: a heartbeat is a NEW file name, and the controller notes on
its own clock when a worker's number last changed (some shared file systems, e.g. HDFS through FUSE, keep a file's
modification time when it is touched, and show data a remote process is still writing only after it closes the file).
The training job on train_host beats the same way (rounds/r<k>/train_<try>.beat.<n>).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import harness_cfg as HC  # noqa: E402
import local_ctl as LC  # noqa: E402
from local_ctl import log, write_json, version  # noqa: E402

import yaml  # noqa: E402

CLUSTER_DEFAULTS = {"exec": "ssh -o BatchMode=yes {host}", "setup_cmd": "", "port_vars": {}, "vars": {}, "lease_s": 1200,
                    "max_leases": 3, "poll_s": 60, "worker_restarts": 3, "train_restarts": 2, "train_silence_s": 3600}
BEAT_S = float(os.environ.get("MULTI_NODE_BEAT_S", "30"))   # heartbeat period (tests use a short one)


# ---------------------------------------------------------------------------
# cluster.yaml and running commands on hosts
# ---------------------------------------------------------------------------


def load_cluster(path: str) -> dict:
    cl = yaml.safe_load(Path(path).read_text())
    if not isinstance(cl, dict):
        raise SystemExit(f"{path}: the top level must be a mapping")
    for k in ("hosts", "gpus", "code_dir", "python", "train_host"):
        if k not in cl:
            raise SystemExit(f"{path}: missing '{k}'")
    if not cl["hosts"] or not cl["gpus"]:
        raise SystemExit(f"{path}: 'hosts' and 'gpus' must not be empty")
    for k, v in CLUSTER_DEFAULTS.items():
        cl.setdefault(k, v)
    hosts = []
    for h in cl["hosts"]:
        if isinstance(h, str):
            hosts.append({"name": h, "gpus": list(cl["gpus"]), "exec": cl["exec"]})
        elif isinstance(h, dict) and isinstance(h.get("name"), str):
            hosts.append({"name": h["name"], "gpus": list(h.get("gpus") or cl["gpus"]), "exec": h.get("exec", cl["exec"])})
        else:
            raise SystemExit(f"{path}: a host is a name or {{name: ..., gpus: [...]}} (got {h!r})")
    cl["hosts"] = hosts
    return cl


def run_on(cl: dict, host: str, cmd: str, timeout_s: float = 600) -> tuple[int, str]:
    """run a shell command on a host (its 'exec', else the cluster's); (exit code, output); 255 when it could not be
    reached in time"""
    ex = next((h["exec"] for h in cl["hosts"] if h["name"] == host), cl["exec"])
    argv = ["bash", "-c", cmd] if not ex.strip() else shlex.split(ex.format(host=host)) + [cmd]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)
        return r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return 255, f"timed out after {timeout_s:.0f} s"


def controller_cfg(cl: dict, a) -> dict:
    """harness.yaml for the controller itself: the per-worker variables take the values of the first GPU (the controller
    only needs them to read the file; every worker gets its own)"""
    g = cl["gpus"][0]
    v = {"gpu": str(g), **{n: str(int(b) + int(g)) for n, b in cl["port_vars"].items()}, **{n: str(x) for n, x in cl["vars"].items()}}
    return HC.apply_vars(HC.load(a.config), {**v, **HC.parse_vars(a.var)})


def host_path(cl: dict, p: str | Path) -> str:
    """a path of this repository -> the same file in the hosts' code_dir; anything else is the same on every machine"""
    p = Path(p).resolve()
    try:
        return str(Path(cl["code_dir"]) / p.relative_to(REPO))
    except ValueError:
        return str(p)


class Beats:
    """Heartbeats seen by the controller: for every beating id (file '<id>.beat.<n>' in a directory), the time on the
    controller's own clock at which its number last changed."""

    def __init__(self, d: Path):
        self.d, self.seen = d, {}            # id -> (number, time it last changed)

    def scan(self, now: float) -> None:
        cur: dict[str, int] = {}
        try:
            names = os.listdir(self.d)
        except FileNotFoundError:
            names = []
        for f in names:
            ident, sep, num = f.rpartition(".beat.")
            if sep and num.isdigit():
                cur[ident] = max(cur.get(ident, -1), int(num))
        for ident, num in cur.items():
            if ident not in self.seen or self.seen[ident][0] != num:
                self.seen[ident] = (num, now)

    def silent_for(self, ident: str, now: float) -> float:
        """seconds since ident's number last changed; an id never seen counts from now (it gets the full grace)"""
        if ident not in self.seen:
            self.seen[ident] = (None, now)
        return now - self.seen[ident][1]

    def label_silent_for(self, label: str, now: float, launched: float | None) -> float:
        """seconds since ANY worker of this label beat (several ids when it was restarted); since launch when none did"""
        ts = [t for ident, (num, t) in self.seen.items() if ident.split("~", 1)[0] == label and num is not None]
        if ts:
            return now - max(ts)
        return now - launched if launched is not None else float("inf")


def beat_loop(d: Path, ident: str, period: float, stop) -> None:
    """create <ident>.beat.<n> every period seconds (n = 1, 2, ...), deleting the previous one, until stop()"""
    n, prev = 0, None
    while not stop():
        n += 1
        f = d / f"{ident}.beat.{n}"
        try:
            f.touch()
            if prev is not None:
                prev.unlink(missing_ok=True)
            prev = f
        except OSError:
            pass
        time.sleep(period)


# ---------------------------------------------------------------------------
# worker (runs on a host, one per GPU)
# ---------------------------------------------------------------------------


def _key(j: int) -> str:
    return f"{j:05d}"


def finished(claims: Path, n: int) -> bool:
    return all((claims / f"{_key(j)}.done").exists() or (claims / f"{_key(j)}.invalid").exists() for j in range(n))


def claim(claims: Path, n: int, owner: str) -> int | None:
    """the first free episode, claimed with O_EXCL (exactly one worker wins it); None when none is free right now"""
    for j in range(n):
        k = _key(j)
        if (claims / f"{k}.done").exists() or (claims / f"{k}.invalid").exists() or (claims / f"{k}.claim").exists():
            continue
        try:
            fd = os.open(claims / f"{k}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"owner": owner, "pid": os.getpid()}) + "\n")
        return j
    return None


def cmd_worker(a) -> int:
    D = Path(a.phase_dir)
    spec = json.loads((D / "jobs.json").read_text())
    cfg = HC.apply_vars(HC.load(a.config), HC.parse_vars(a.var))
    if spec.get("eval_cmd") and cfg["eval"].get("cmd"):
        cfg["episode"]["cmd"] = cfg["eval"]["cmd"]
    cfg["episode"]["dir"] = "lease_{lease}/" + cfg["episode"]["dir"]
    by_id = {t["task_id"]: t for t in cfg["tasks_list"]}
    claims, workers, eps = D / "claims", D / "workers", D / "eps"
    for d in (claims, workers, eps):
        d.mkdir(parents=True, exist_ok=True)
    stop_file = Path(spec["run_dir"]) / "STOP"
    n = len(spec["jobs"])
    ident = f"{a.label}~{os.uname().nodename}-{os.getpid()}-{int(time.time())}"   # this worker instance
    cur = {"stop": False}
    threading.Thread(target=beat_loop, args=(workers, ident, BEAT_S, lambda: cur["stop"]), daemon=True).start()
    P = int(cfg["episodes_parallel"])
    log(f"{a.label}: worker {ident} started, {n} episodes in {D}, {P} at a time")
    lock = threading.Lock()
    shared = {"stack": None, "err": None}

    def get_stack():
        with lock:
            if shared["stack"] is None:
                shared["stack"] = LC.Stack(cfg, eps, workers / f"{a.label}_logs", spec["weights"], spec["policy_version"],
                                           Path(spec["run_dir"]), proxy=not spec["direct"])
            return shared["stack"]

    def slot():
        """one of P episode slots sharing this worker's model server, proxy and services"""
        while not stop_file.exists() and shared["err"] is None:
            j = claim(claims, n, ident)
            if j is None:
                if finished(claims, n):
                    return
                time.sleep(BEAT_S)        # episodes still running elsewhere may come back (expired lease)
                continue
            k = _key(j)
            try:
                stack = get_stack()
            except Exception as e:  # noqa: BLE001
                # nothing of episode j has run yet: give it back at once instead of after the lease
                (claims / f"{k}.claim").unlink(missing_ok=True)
                shared["err"] = e
                return
            job = spec["jobs"][j]
            lease = len(list(claims.glob(f"{k}.claim.stale.*")))
            v = HC.episode_vars(cfg, by_id[job["task_id"]], job["init_id"], job["attempt"], eps, spec["weights"],
                                {**job.get("extra", {}), "lease": lease}, direct=spec["direct"])
            info = LC.run_episode(cfg, v, stop_file)
            if "skipped" in info:
                return                    # STOP: the claim stays; the controller hands it out again if it goes on
            write_json(claims / f"{k}.done", {"episode_dir": v["episode_dir"], "lease": lease, "worker": ident, **info})
            log(f"{a.label}: episode {j} {v['episode_key']} rc={info.get('rc')} timed_out={info.get('timed_out')}")
            try:
                stack.check()
            except Exception as e:  # noqa: BLE001
                shared["err"] = e
                return

    try:
        slots = [threading.Thread(target=slot) for _ in range(P)]
        for t in slots:
            t.start()
            time.sleep(1.0)               # spread the first claims
        for t in slots:
            t.join()
    finally:
        cur["stop"] = True
        if shared["stack"] is not None:
            shared["stack"].stop()
    if shared["err"] is not None:
        raise shared["err"]
    return 0


# ---------------------------------------------------------------------------
# controller
# ---------------------------------------------------------------------------


def write_jobs(D: Path, jobs: list[dict], weights: str, policy_version: str, run_dir: Path, direct: bool, eval_cmd: bool) -> None:
    D.mkdir(parents=True, exist_ok=True)
    if not (D / "jobs.json").is_file():
        write_json(D / "jobs.json", {"weights": weights, "policy_version": policy_version, "run_dir": str(run_dir),
                                     "direct": direct, "eval_cmd": eval_cmd, "jobs": jobs})
    for d in ("claims", "workers", "eps"):
        (D / d).mkdir(exist_ok=True)


def expire_leases(cl: dict, D: Path, n: int, now: float, beats: Beats, owners: dict) -> None:
    """hand out again the episodes whose worker has stopped beating for lease_s (owners: cache claim file -> owner)"""
    claims = D / "claims"
    for j in range(n):
        k = _key(j)
        c = claims / f"{k}.claim"
        if (claims / f"{k}.done").exists() or (claims / f"{k}.invalid").exists() or not c.exists():
            continue
        if k not in owners:
            try:
                owners[k] = json.loads(c.read_text()).get("owner") or "?"
            except (OSError, ValueError):
                continue                          # being written right now
        silent = beats.silent_for(owners[k], now)
        if silent < cl["lease_s"]:
            continue
        stale = len(list(claims.glob(f"{k}.claim.stale.*")))
        try:
            os.rename(c, claims / f"{k}.claim.stale.{stale}")
        except FileNotFoundError:
            continue
        owners.pop(k, None)
        if (claims / f"{k}.done").exists():
            continue                              # it finished just now: nothing to hand out again
        if stale + 1 >= cl["max_leases"]:
            write_json(claims / f"{k}.invalid", {"leases": stale + 1})
            log(f"episode {j}: {stale + 1} leases without a result -> invalid")
        else:
            log(f"episode {j}: its worker has not beaten for {silent:.0f} s -> handed out again (lease {stale + 1})")


def worker_cmd(cl: dict, a, D: Path, host: str, gpu, label: str) -> str:
    v = [f"gpu={gpu}"] + [f"{name}={int(base) + int(gpu)}" for name, base in cl["port_vars"].items()]
    v += [f"{name}={val}" for name, val in cl["vars"].items()]
    args = ["worker", "--config", host_path(cl, a.config), "--phase-dir", str(D), "--label", label]
    for x in v:
        args += ["--var", x]
    logf = D / "workers" / f"{label}.log"
    # only the nohup command goes to the background, with every stream redirected: `cd X && cmd &` would background a
    # subshell that keeps the caller's output open, and the launching ssh (or bash) would wait for the worker to end
    return (f"cd {shlex.quote(cl['code_dir'])} && {{ nohup {shlex.quote(cl['python'])} rl/multi_node_ctl.py "
            f"{' '.join(shlex.quote(x) for x in args)} >> {shlex.quote(str(logf))} 2>&1 < /dev/null & }}")


def run_phase(cl: dict, a, st: dict, D: Path, run_dir: Path, save) -> bool:
    """drive one phase (rollout or eval) until every episode is done or invalid; False when STOP was found.
    save(): writes the controller state."""
    n = len(json.loads((D / "jobs.json").read_text())["jobs"])
    ps = st.setdefault("phases", {}).setdefault(str(D), {"launched": {}, "restarts": {}})
    setup_done = st.setdefault("setup_done", [])
    stop_file = run_dir / "STOP"
    beats, owners = Beats(D / "workers"), {}
    while True:
        if stop_file.exists():
            log("STOP found"); return False
        now = time.time()
        beats.scan(now)
        expire_leases(cl, D, n, now, beats, owners)
        claims = D / "claims"
        done = sum(1 for j in range(n) if (claims / f"{_key(j)}.done").exists())
        invalid = sum(1 for j in range(n) if (claims / f"{_key(j)}.invalid").exists())
        if done + invalid >= n:
            log(f"{D.name}: {done} done, {invalid} invalid of {n}")
            return True
        for hd in cl["hosts"]:
            host = hd["name"]
            # dead: no worker of this label has beaten for 4 periods (a worker launched less than 5 min ago may still be
            # starting: it is left alone)
            dead = [g for g in hd["gpus"] if beats.label_silent_for(f"{host}_g{g}", now, ps["launched"].get(f"{host}_g{g}")) > 4 * BEAT_S
                    and now - ps["launched"].get(f"{host}_g{g}", 0) > 300]
            if not dead:
                continue
            if cl["setup_cmd"] and host not in setup_done:
                rc, out = run_on(cl, host, cl["setup_cmd"], timeout_s=7200)
                if rc != 0:
                    log(f"{host}: setup failed ({rc}): {out[-300:]}"); continue
                setup_done.append(host); save()
            for g in dead:
                label = f"{host}_g{g}"
                r = ps["restarts"].get(host, 0)
                if label in ps["launched"]:
                    if r >= cl["worker_restarts"] * len(hd["gpus"]):
                        continue                  # this host has used up its restarts for this phase
                    ps["restarts"][host] = r + 1
                rc, out = run_on(cl, host, worker_cmd(cl, a, D, host, g, label), timeout_s=120)
                if rc == 0:
                    ps["launched"][label] = now
                    log(f"{label}: worker started")
                else:
                    log(f"{label}: could not start a worker ({rc}): {out[-200:]}")
        save()
        log(f"{D.name}: {done} done, {invalid} invalid of {n}")
        time.sleep(cl["poll_s"])


def run_training(cl: dict, cfg: dict, st: dict, rd: Path, build: Path, weights: str, new: Path, k: int, run_dir: Path) -> None:
    """train.cmd on train_host, in the background there; waits for its exit code file; restarts at most train_restarts"""
    cfg_host = {**cfg, "config_dir": host_path(cl, cfg["config_dir"]), "vars": {**cl["vars"], **cfg.get("vars", {})}}
    cmd, cwd = LC.train_command(cfg_host, build, weights, st["weights"]["v000"], new, k, run_dir, python=cl["python"])
    tries = st.setdefault("train_tries", {}).setdefault(str(k), 0)
    tlog = rd / "train.log"
    beats = Beats(rd)
    while True:
        rcf = rd / f"train_{tries}.rc"
        if not rcf.exists() and not st.get("train_launched", {}).get(str(k)) == tries:
            if new.exists():
                new.rename(new.with_name(f"{new.name}.stale_{time.strftime('%Y%m%dT%H%M%S')}"))
            # a beat loop next to the training (new file names, see the module docstring), stopped when it ends
            b = shlex.quote(str(rd / f"train_{tries}"))
            beat = f"( n=0; while :; do n=$((n+1)); : > {b}.beat.$n; rm -f {b}.beat.$((n-1)); sleep {int(BEAT_S)}; done ) & BP=$!; "
            inner = (f"cd {shlex.quote(cwd)} && {{ {beat}( {cmd} ) >> {shlex.quote(str(tlog))} 2>&1; rc=$?; kill $BP; "
                     f"echo $rc > {shlex.quote(str(rcf))}; }}")
            rc, out = run_on(cl, cl["train_host"], f"nohup bash -c {shlex.quote(inner)} > /dev/null 2>&1 < /dev/null &", timeout_s=120)
            if rc != 0:
                log(f"round {k}: could not start the training on {cl['train_host']} ({rc}): {out[-200:]}"); time.sleep(cl["poll_s"]); continue
            st.setdefault("train_launched", {})[str(k)] = tries
            write_json(run_dir / "state.json", st)
            log(f"round {k}: training started on {cl['train_host']} (try {tries}): {cmd}")
        code = None
        while not rcf.exists():
            if (run_dir / "STOP").exists():
                raise SystemExit("STOP found during training (the training itself keeps running)")
            now = time.time()
            beats.scan(now)
            if beats.silent_for(f"train_{tries}", now) > cl["train_silence_s"]:
                code = f"no heartbeat for more than {cl['train_silence_s']} s (train host lost?)"
                break
            time.sleep(cl["poll_s"])
        if code is None:
            code = rcf.read_text().strip()
        if code == "0" and new.is_dir():
            return
        tries += 1
        st["train_tries"][str(k)] = tries
        write_json(run_dir / "state.json", st)
        if tries > cl["train_restarts"]:
            raise SystemExit(f"round {k}: training failed {tries} times (last exit {code}), see {rd / 'train.log'}")
        log(f"round {k}: training ended with {code}; starting it again ({tries}/{cl['train_restarts']})")


def cmd_train(a) -> int:
    cl = load_cluster(a.cluster)
    cfg = controller_cfg(cl, a)
    run_dir = Path(a.run_dir).resolve(); run_dir.mkdir(parents=True, exist_ok=True)
    st = LC.load_state(run_dir, a)
    save = lambda: write_json(run_dir / "state.json", st)  # noqa: E731
    report = json.loads(Path(a.pool_report).read_text()) if a.pool_report else None
    pool = LC.pool_tasks(cfg["tasks_list"], report, a.rate_lo, a.rate_hi)
    if not pool:
        raise SystemExit("the training pool is empty (no task with a success rate in [--rate-lo, --rate-hi])")
    G = int(cfg["group_size"])
    while st["round"] < a.rounds:
        k = st["round"]; V = version(k)
        rd = run_dir / "rounds" / f"r{k}"; build = rd / "build"
        rd.mkdir(parents=True, exist_ok=True)
        weights = st["weights"][V]
        if st["phase"] == "rollout":
            gp = rd / "groups.json"
            if gp.is_file():
                groups = json.loads(gp.read_text())
            else:
                groups = LC.choose_groups(pool, k, a.groups, {tuple(x) for x in st["used"]}, st["seed"])
                if not groups:
                    raise SystemExit(f"round {k}: no unused (task, init) pair left in the pool")
                write_json(gp, groups)
                st["used"] += groups; write_json(run_dir / "state.json", st)
            jobs = [{"task_id": t, "init_id": i, "attempt": att, "extra": {"round": k, "run_dir": str(run_dir), "weights0": st["weights"]["v000"]}}
                    for t, i in groups for att in range(G)]
            write_jobs(rd, jobs, weights, V, run_dir, direct=False, eval_cmd=False)
            log(f"round {k} ({V}): {len(groups)} groups x {G} episodes over {len(cl['hosts'])} hosts, weights {weights}")
            if not run_phase(cl, a, st, rd, run_dir, save):
                return 3
            st["phase"] = "build"; write_json(run_dir / "state.json", st)
        if st["phase"] == "build":
            stats = LC.build_samples(cfg, rd / "eps", build, rd, V, k)
            st["history"].append({"round": k, "version": V, "episodes": stats["episodes"], "status": stats["status"],
                                  "success_rate_valid": stats["success_rate_valid"], "groups_used": stats["groups_used"],
                                  "samples": stats["samples"]})
            log(f"round {k}: success {stats['success_rate_valid']:.3f}, groups {stats['groups_used']}, samples {stats['samples']}")
            st["phase"] = "train"; write_json(run_dir / "state.json", st)
        if st["phase"] == "train":
            VN = version(k + 1); new = run_dir / "weights" / VN
            if st["history"][-1]["samples"] == 0:
                log(f"round {k}: no group with a non-zero advantage -> nothing to train on, {VN} = {V}")
                st["weights"][VN] = weights
            else:
                new.parent.mkdir(parents=True, exist_ok=True)
                run_training(cl, cfg, st, rd, build, weights, new, k, run_dir)
                st["weights"][VN] = str(new)
                log(f"round {k}: trained {VN} -> {new}")
            st["round"] = k + 1; st["phase"] = "rollout"; write_json(run_dir / "state.json", st)
    log(f"done: {st['round']} round(s); latest weights {st['weights'][version(st['round'])]}")
    return 0


def cmd_eval(a) -> int:
    cl = load_cluster(a.cluster)
    cfg = controller_cfg(cl, a)
    run_dir = Path(a.run_dir).resolve()
    D = run_dir / "eval" / a.tag
    st_path = run_dir / f"eval_{a.tag}_state.json"
    st = json.loads(st_path.read_text()) if st_path.is_file() else {}
    weights = LC.weights_arg(a.weights)
    plan = LC.eval_plan(cfg, a.init_key, a.repeats)
    jobs = [{"task_id": t["task_id"], "init_id": i, "attempt": att, "extra": {"round": 0, "run_dir": str(run_dir), "weights0": weights}}
            for t, i, att in plan]
    write_jobs(D, jobs, weights, a.tag, run_dir, direct=cfg["eval"]["direct"], eval_cmd=bool(cfg["eval"].get("cmd")))
    log(f"eval {a.tag}: {len(jobs)} episodes over {len(cl['hosts'])} hosts, weights {weights}")
    if not run_phase(cl, a, st, D, run_dir, lambda: write_json(st_path, st)):
        return 3
    # one result per episode: the lease that finished it
    claims = D / "claims"
    AD = LC._adapters.load(cfg["adapter"])
    per = {t["task_id"]: {"episodes": 0, "valid": 0, "reward_sum": 0.0} for t in cfg["tasks_list"]}
    for j, job in enumerate(jobs):
        df = claims / f"{_key(j)}.done"
        per[job["task_id"]]["episodes"] += 1
        if not df.exists():
            continue
        found = list(AD.find_episodes(Path(json.loads(df.read_text())["episode_dir"])))
        if len(found) == 1 and found[0][1]["valid"]:
            per[job["task_id"]]["valid"] += 1; per[job["task_id"]]["reward_sum"] += float(found[0][1]["reward"])
    for r in per.values():
        r["success_rate"] = (r["reward_sum"] / r["valid"]) if r["valid"] else None
    tot_v = sum(r["valid"] for r in per.values())
    report = {"tag": a.tag, "weights": weights, "shard": "0/1", "launched": len(jobs), "valid": tot_v,
              "success_rate": (sum(r["reward_sum"] for r in per.values()) / tot_v) if tot_v else None, "tasks": per}
    write_json(D / "report.json", report)
    log(f"eval {a.tag}: {report['success_rate']} over {tot_v} valid of {len(jobs)} episodes -> {D / 'report.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    t = sub.add_parser("train")
    t.add_argument("--cluster", required=True); t.add_argument("--config", required=True); t.add_argument("--run-dir", required=True)
    t.add_argument("--weights", default=None, help="start weights (v000); not needed when resuming")
    t.add_argument("--rounds", type=int, required=True); t.add_argument("--groups", type=int, required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--pool-report", default=None, help="eval report.json: train only on tasks with a success rate in [--rate-lo, --rate-hi]")
    t.add_argument("--rate-lo", type=float, default=0.1); t.add_argument("--rate-hi", type=float, default=0.9)
    t.add_argument("--var", action="append", default=[], metavar="NAME=VALUE")
    e = sub.add_parser("eval")
    e.add_argument("--cluster", required=True); e.add_argument("--config", required=True); e.add_argument("--run-dir", required=True)
    e.add_argument("--weights", required=True); e.add_argument("--tag", required=True)
    e.add_argument("--repeats", type=int, default=1)
    e.add_argument("--init-key", required=True, choices=HC.INIT_KEYS,
                   help="which inits of tasks.jsonl to run, always named: val_init_ids (choosing checkpoints / the training pool), "
                        "eval_init_ids (reported numbers), init_ids (training inits, e.g. collecting data)")
    e.add_argument("--var", action="append", default=[], metavar="NAME=VALUE")
    w = sub.add_parser("worker", help="(started by the controller on a host)")
    w.add_argument("--config", required=True); w.add_argument("--phase-dir", required=True); w.add_argument("--label", required=True)
    w.add_argument("--var", action="append", default=[], metavar="NAME=VALUE")
    a = ap.parse_args()
    try:
        return {"train": cmd_train, "eval": cmd_eval, "worker": cmd_worker}[a.what](a)
    except HC.ConfigError as exc:
        raise SystemExit(f"config error: {exc}") from None


if __name__ == "__main__":
    sys.exit(main())
