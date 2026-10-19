"""The multi-machine controller (rl/multi_node_ctl.py) with the minimal harness: two "hosts" with two "GPUs" each, all
on this machine (cluster exec ""), sharing the run directory -- a whole RL round, an evaluation, and an episode whose
worker vanished being handed out again under a new lease directory."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml
from conftest import ROOT

MIN = ROOT / "harnesses" / "minimal"
ENV = {**os.environ, "MULTI_NODE_BEAT_S": "1"}


def free_base() -> int:
    """a base port such that base + 0..3 are free (proxy) and base + 100..103 (model)"""
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            base = s.getsockname()[1]
        if base + 103 >= 65535:
            continue
        ok = True
        for off in (0, 1, 2, 3, 100, 101, 102, 103):
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", base + off))
                except OSError:
                    ok = False
        if ok:
            return base


def setup(tmp_path, lease_s=600):
    base = free_base()
    d = tmp_path / "cfg"; d.mkdir()
    cfg = yaml.safe_load((MIN / "harness.yaml").read_text())
    cfg["proxy_port"] = "{proxy_port}"
    cfg["model"] = {"url": "http://127.0.0.1:{model_port}", "ready_timeout_s": 60,
                    "serve_cmd": f"{{python}} {MIN}/fake_model_server.py --port {{model_port}} --weights {{weights}}"}
    # each episode takes a second, so that the four workers really share the round
    cfg["episode"]["cmd"] = "sleep 1 && " + cfg["episode"]["cmd"].replace("{config_dir}", str(MIN))
    cfg["train"]["cmd"] = cfg["train"]["cmd"].replace("{config_dir}", str(MIN))
    cfg["episodes_parallel"] = 2                    # two episode slots per worker, sharing its servers
    (d / "harness.yaml").write_text(yaml.safe_dump(cfg))
    (d / "tasks.jsonl").write_text((MIN / "tasks.jsonl").read_text())
    # two "hosts" on this machine: different GPU numbers, so that their per-GPU ports differ
    cl = {"hosts": [{"name": "hostA", "gpus": [0, 1]}, {"name": "hostB", "gpus": [2, 3]}], "gpus": [0], "exec": "",
          "code_dir": str(ROOT), "python": sys.executable,
          "port_vars": {"proxy_port": base, "model_port": base + 100}, "train_host": "hostA",
          "lease_s": lease_s, "max_leases": 3, "poll_s": 1, "worker_restarts": 2, "train_restarts": 1}
    (d / "cluster.yaml").write_text(yaml.safe_dump(cl))
    return d


def ctl(*args, timeout=300):
    return subprocess.run([sys.executable, str(ROOT / "rl" / "multi_node_ctl.py"), *map(str, args)], capture_output=True,
                          text=True, cwd=str(ROOT), env=ENV, timeout=timeout)


def test_one_round_over_two_hosts(tmp_path):
    d = setup(tmp_path)
    run = tmp_path / "run"
    r = ctl("train", "--cluster", d / "cluster.yaml", "--config", d / "harness.yaml", "--run-dir", run,
            "--weights", "w0", "--rounds", "1", "--groups", "4")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rd = run / "rounds" / "r0"
    stats = json.loads((rd / "build" / "round_stats.json").read_text())
    assert stats["episodes"] == 24 and stats["status"] == {"valid": 24}
    done = [json.loads(f.read_text()) for f in sorted((rd / "claims").glob("*.done"))]
    assert len(done) == 24 and {x["lease"] for x in done} == {0}
    assert len({x["worker"] for x in done}) >= 2                      # the work really was shared
    st = json.loads((run / "state.json").read_text())
    assert st["round"] == 1 and Path(st["weights"]["v001"]).is_dir()
    assert (rd / "train_0.rc").read_text().strip() == "0"


def test_vanished_worker_episode_runs_again_under_a_new_lease(tmp_path):
    d = setup(tmp_path, lease_s=5)
    run = tmp_path / "run"
    # round 0's jobs as the controller will write them, and a claim of episode 0 by a worker that is gone (old heartbeat)
    groups_cmd = ctl("train", "--cluster", d / "cluster.yaml", "--config", d / "harness.yaml", "--run-dir", run,
                     "--weights", "w0", "--rounds", "1", "--groups", "2", "--var", "unused=1")
    assert groups_cmd.returncode == 0, groups_cmd.stdout[-3000:] + groups_cmd.stderr[-3000:]
    # second run directory: plant a dead claim before the controller starts
    run2 = tmp_path / "run2"
    rd = run2 / "rounds" / "r0"
    (rd / "claims").mkdir(parents=True)
    (rd / "claims" / "00000.claim").write_text('{"owner": "gone"}\n')
    old = time.time() - 3600
    os.utime(rd / "claims" / "00000.claim", (old, old))
    r = ctl("train", "--cluster", d / "cluster.yaml", "--config", d / "harness.yaml", "--run-dir", run2,
            "--weights", "w0", "--rounds", "1", "--groups", "2")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert (rd / "claims" / "00000.claim.stale.0").exists()
    d0 = json.loads((rd / "claims" / "00000.done").read_text())
    assert d0["lease"] == 1 and "/lease_1/" in d0["episode_dir"]
    stats = json.loads((rd / "build" / "round_stats.json").read_text())
    assert stats["episodes"] == 12 and stats["status"] == {"valid": 12}


def test_eval_over_two_hosts(tmp_path):
    d = setup(tmp_path)
    run = tmp_path / "run"
    r = ctl("eval", "--cluster", d / "cluster.yaml", "--config", d / "harness.yaml", "--run-dir", run, "--weights", "w0", "--tag", "v000",
            "--init-key", "eval_init_ids")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rep = json.loads((run / "eval" / "v000" / "report.json").read_text())
    n_tasks = len((MIN / "tasks.jsonl").read_text().strip().splitlines())
    assert rep["launched"] == 3 * n_tasks and rep["valid"] == 3 * n_tasks
    assert set(rep["tasks"]) == {json.loads(l)["task_id"] for l in (MIN / "tasks.jsonl").read_text().splitlines() if l.strip()}
    assert not (run / "state.json").exists()                          # the eval state is kept apart from a training run's


def test_heartbeats_need_no_file_times(tmp_path):
    """liveness comes from new beat file names seen on the controller's clock: file modification times are never read
    (some shared file systems keep them unchanged when a file is touched)"""
    sys.path.insert(0, str(ROOT / "rl"))
    import multi_node_ctl as M
    d = tmp_path / "workers"; d.mkdir()
    b = M.Beats(d)
    (d / "h_g0~a.beat.1").touch(); (d / "h_g1~b.beat.7").touch()
    old = time.time() - 10 ** 6
    for f in d.iterdir():
        os.utime(f, (old, old))                   # file times are far in the past: must not matter
    b.scan(100.0)
    assert b.silent_for("h_g0~a", 100.0) == 0 and b.label_silent_for("h_g0", 100.0, None) == 0
    (d / "h_g0~a.beat.1").unlink(); (d / "h_g0~a.beat.2").touch()   # g0 beats on, g1 does not
    b.scan(160.0)
    assert b.silent_for("h_g0~a", 160.0) == 0 and b.silent_for("h_g1~b", 160.0) == 60.0
    assert b.silent_for("unknown~x", 160.0) == 0 and b.silent_for("unknown~x", 200.0) == 40.0   # grace from first look
    assert b.label_silent_for("h_g2", 200.0, launched=150.0) == 50.0 and b.label_silent_for("h_g2", 200.0, None) == float("inf")
