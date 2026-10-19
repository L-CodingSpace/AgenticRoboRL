"""The single-machine controller (rl/local_ctl.py) with the minimal harness (harnesses/minimal/): a whole RL round --
model server, recording proxy, episodes, result.json, build_round, trainer -- without a GPU and without any RPent file."""
from __future__ import annotations

import gzip
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from conftest import ROOT

sys.path.insert(0, str(ROOT / "rl"))
import harness_cfg as HC  # noqa: E402
import local_ctl as LC  # noqa: E402

MIN = ROOT / "harnesses" / "minimal"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_cfg(tmp_path, **episode):
    """the minimal harness.yaml on free ports, scripts by absolute path; episode.* overridable"""
    cfg = yaml.safe_load((MIN / "harness.yaml").read_text())
    proxy, model = free_port(), free_port()
    cfg["proxy_port"] = proxy
    cfg["model"] = {"url": f"http://127.0.0.1:{model}", "ready_timeout_s": 60,
                    "serve_cmd": f"{{python}} {MIN}/fake_model_server.py --port {model} --weights {{weights}}"}
    cfg["episode"]["cmd"] = cfg["episode"]["cmd"].replace("{config_dir}", str(MIN))
    cfg["train"]["cmd"] = cfg["train"]["cmd"].replace("{config_dir}", str(MIN))
    cfg["episode"].update(episode)
    d = tmp_path / "cfg"; d.mkdir(exist_ok=True)
    (d / "tasks.jsonl").write_text((MIN / "tasks.jsonl").read_text())
    (d / "harness.yaml").write_text(yaml.safe_dump(cfg))
    return d / "harness.yaml"


def ctl(*args, expect=0):
    r = subprocess.run([sys.executable, str(ROOT / "rl" / "local_ctl.py"), *map(str, args)], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == expect, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout + r.stderr


def test_two_rounds_end_to_end(tmp_path):
    cfg, run = make_cfg(tmp_path), tmp_path / "run"
    ctl("train", "--config", cfg, "--run-dir", run, "--weights", "start-weights", "--rounds", 2, "--groups", 6)
    st = json.loads((run / "state.json").read_text())
    assert st["round"] == 2 and st["phase"] == "rollout" and [h["version"] for h in st["history"]] == ["v000", "v001"]
    assert len({tuple(g) for g in st["used"]}) == len(st["used"]) == 12            # no (task, init) used twice
    for k, V in ((0, "v000"), (1, "v001")):
        eps = run / "rounds" / f"r{k}" / "eps"
        done = sorted(eps.rglob("episode_done.json"))
        assert len(done) == 36
        turns = sorted(eps.rglob("turn_*.json.gz"))
        recs = [json.loads(gzip.decompress(t.read_bytes())) for t in turns]
        assert turns and {r["policy_version"] for r in recs} == {V}                # recorded by this round's proxy
        for d in done:                                                             # one record per model call
            res = json.loads((d.parent / "result.json").read_text())
            assert res["requests"] == len(list((d.parent / "rl_turns").glob("turn_*.json.gz")))
            assert res["task"] == json.loads(d.read_text())["task_id"] and res["valid"] is True
        stats = json.loads((eps.parent / "build" / "round_stats.json").read_text())
        assert stats["status"] == {"valid": 36} and stats["samples"] == st["history"][k]["samples"] > 0
    # the trainer got round 0's samples and the start weights; round 1 was served and trained from its output
    i1 = json.loads((run / "weights" / "v001" / "info.json").read_text())
    i2 = json.loads((run / "weights" / "v002" / "info.json").read_text())
    assert i1["from"] == "start-weights" and i1["samples"] == st["history"][0]["samples"]
    assert i2["from"] == str(run / "weights" / "v001") and st["weights"]["v002"] == str(run / "weights" / "v002")
    # nothing we started is left behind
    for port in (yaml.safe_load(Path(cfg).read_text())["proxy_port"],):
        with socket.socket() as s:
            assert s.connect_ex(("127.0.0.1", port)) != 0


def test_resume_continues_and_reruns_only_what_is_missing(tmp_path):
    cfg, run = make_cfg(tmp_path), tmp_path / "run"
    ctl("train", "--config", cfg, "--run-dir", run, "--weights", "w", "--rounds", 1, "--groups", 3)
    r0 = run / "rounds" / "r0"
    before = {p: p.stat().st_mtime_ns for p in r0.rglob("episode_done.json")}
    samples0 = (r0 / "build" / "samples.jsonl").read_bytes()
    out = ctl("train", "--config", cfg, "--run-dir", run, "--rounds", 2, "--groups", 3)       # no --weights: resumes
    assert "round 1 (v001)" in out and "round 0 (v000)" not in out
    assert {p: p.stat().st_mtime_ns for p in r0.rglob("episode_done.json")} == before
    # an episode that never finished is run again (its old dir is set aside); the others are kept
    st = json.loads((run / "state.json").read_text()); st.update(round=1, phase="rollout"); st["history"] = st["history"][:1]
    (run / "state.json").write_text(json.dumps(st))
    r1 = run / "rounds" / "r1"
    victim = sorted(r1.rglob("episode_done.json"))[0]; victim.unlink()
    kept = {p: p.stat().st_mtime_ns for p in r1.rglob("episode_done.json")}
    ctl("train", "--config", cfg, "--run-dir", run, "--rounds", 2, "--groups", 3)
    assert victim.exists() and {p: p.stat().st_mtime_ns for p in kept} == kept
    assert len(list(victim.parent.parent.glob(victim.parent.name + ".stale_*"))) == 1
    assert (r0 / "build" / "samples.jsonl").read_bytes() == samples0
    # a different start is refused instead of silently mixing runs
    assert "use another --run-dir" in ctl("train", "--config", cfg, "--run-dir", run, "--weights", "other", "--rounds", 2, "--groups", 3, expect=1)


@pytest.mark.parametrize("on_timeout,status", [("failure", {"valid": 4}), ("invalid", {"invalid_infrastructure": 4})])
def test_timeout_is_a_failure_or_an_infrastructure_error_as_configured(tmp_path, on_timeout, status):
    # the agent talks to the model, then hangs without a result (an episode with no model call at all has nothing to
    # train on and is "invalid_records" whatever the harness says)
    cfg = make_cfg(tmp_path, timeout_s=4, on_timeout=on_timeout)
    c = yaml.safe_load(cfg.read_text()); c["group_size"] = 2; c["train"] = None
    c["episode"]["cmd"] += " && rm {episode_dir}/result.json && sleep 60"
    cfg.write_text(yaml.safe_dump(c))
    run = tmp_path / "run"
    ctl("train", "--config", cfg, "--run-dir", run, "--weights", "w", "--rounds", 1, "--groups", 2)
    stats = json.loads((run / "rounds" / "r0" / "build" / "round_stats.json").read_text())
    assert stats["status"] == status and stats["samples"] == 0
    res = [json.loads(p.read_text()) for p in (run / "rounds" / "r0" / "eps").rglob("result.json")]
    assert len(res) == 4 and all(r["valid"] is (on_timeout == "failure") for r in res)
    assert all(json.loads(p.read_text())["timed_out"] for p in (run / "rounds" / "r0" / "eps").rglob("episode_done.json"))


def test_harness_that_leaves_no_result_is_an_infrastructure_failure(tmp_path):
    cfg = make_cfg(tmp_path, cmd="true")
    c = yaml.safe_load(cfg.read_text()); c["group_size"] = 2; cfg.write_text(yaml.safe_dump(c))
    run = tmp_path / "run"
    ctl("train", "--config", cfg, "--run-dir", run, "--weights", "w", "--rounds", 1, "--groups", 2)
    st = json.loads((run / "state.json").read_text())
    assert st["history"][0]["status"] == {"invalid_infrastructure": 4} and st["weights"]["v001"] == "w"   # nothing to train on


def test_eval_report_and_pool(tmp_path):
    cfg, run = make_cfg(tmp_path), tmp_path / "run"
    ctl("eval", "--config", cfg, "--run-dir", run, "--weights", "w", "--tag", "v000", "--repeats", 2, "--init-key", "eval_init_ids")
    rep = json.loads((run / "eval" / "v000" / "report.json").read_text())
    assert rep["launched"] == rep["valid"] == 18 and set(rep["tasks"]) == {"reach/a", "reach/b", "reach/c"}
    assert all(t["episodes"] == 6 and 0.0 <= t["success_rate"] <= 1.0 for t in rep["tasks"].values())
    tasks = HC.load(cfg)["tasks_list"]
    fake = {"tasks": {"reach/a": {"success_rate": 1.0}, "reach/b": {"success_rate": 0.5}, "reach/c": {"success_rate": None}}}
    assert [t["task_id"] for t in LC.pool_tasks(tasks, fake, 0.1, 0.9)] == ["reach/b"]
    assert len(LC.pool_tasks(tasks, None, 0.1, 0.9)) == 3
    out = ctl("train", "--config", cfg, "--run-dir", tmp_path / "run2", "--weights", "w", "--rounds", 1, "--groups", 2,
              "--pool-report", run / "eval" / "v000" / "report.json", "--rate-lo", 2, "--rate-hi", 3, expect=1)
    assert "training pool is empty" in out


def test_choose_groups_spreads_evenly_never_repeats_and_is_seeded():
    tasks = [{"task_id": f"t{i}", "init_ids": [1, 2, 3]} for i in range(4)]
    g0 = LC.choose_groups(tasks, 0, 6, set(), seed=0)
    assert g0 == LC.choose_groups(tasks, 0, 6, set(), seed=0) != LC.choose_groups(tasks, 0, 6, set(), seed=1)
    per = {t: sum(1 for x in g0 if x[0] == t) for t in ("t0", "t1", "t2", "t3")}
    assert sorted(per.values()) == [1, 1, 2, 2] and len({tuple(x) for x in g0}) == 6
    used = {tuple(x) for x in g0}
    g1 = LC.choose_groups(tasks, 1, 6, used, seed=0)
    assert len(g1) == 6 and not used & {tuple(x) for x in g1}
    assert LC.choose_groups(tasks, 2, 6, used | {tuple(x) for x in g1}, seed=0) == []      # 12 pairs exist, all used


def test_config_errors_are_explained(tmp_path):
    cfg = make_cfg(tmp_path, cmd="echo {no_such_thing}")
    assert "unknown placeholder {no_such_thing}" in ctl("train", "--config", cfg, "--run-dir", tmp_path / "r", "--weights", "w",
                                                         "--rounds", 1, "--groups", 1, expect=1)
    cfg = make_cfg(tmp_path, dir="{task_id} x/{init_id}")
    assert "proxy's episode key" in ctl("train", "--config", cfg, "--run-dir", tmp_path / "r2", "--weights", "w",
                                        "--rounds", 1, "--groups", 1, expect=1)
    assert HC.safe_path("libero goal:3/../x") == "libero_goal_3/x"


def test_vars_shards_and_merged_report(tmp_path):
    """one harness.yaml, two controllers side by side (what a multi-GPU pilot does): ports and GPU come from --var,
    each controller takes its shard of the tasks, the two reports merge into one"""
    cfg = yaml.safe_load((MIN / "harness.yaml").read_text())
    cfg["proxy_port"] = "{proxy_port}"
    cfg["model"] = {"url": "http://127.0.0.1:{model_port}", "ready_timeout_s": 60,
                    "serve_cmd": f"{{python}} {MIN}/fake_model_server.py --port {{model_port}} --weights {{weights}}-gpu{{gpu}}"}
    cfg["episode"]["cmd"] = cfg["episode"]["cmd"].replace("{config_dir}", str(MIN))
    d = tmp_path / "cfg"; d.mkdir()
    (d / "tasks.jsonl").write_text((MIN / "tasks.jsonl").read_text())
    (d / "harness.yaml").write_text(yaml.safe_dump(cfg))
    reports = []
    procs = []
    for g in (0, 1):
        run = tmp_path / f"shard{g}"
        cmd = [sys.executable, str(ROOT / "rl" / "local_ctl.py"), "eval", "--config", str(d / "harness.yaml"), "--run-dir", str(run),
               "--weights", "w", "--tag", "pilot", "--init-key", "eval_init_ids", "--shard", f"{g}/2", "--var", f"gpu={g}", "--var", f"model_port={free_port()}",
               "--var", f"proxy_port={free_port()}"]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT)))
        reports.append(run / "eval" / "pilot" / "report.json")
    for p in procs:
        out, _ = p.communicate(timeout=240)
        assert p.returncode == 0, out[-3000:]
    r0, r1 = (json.loads(r.read_text()) for r in reports)
    assert sorted(r0["tasks"]) == ["reach/a", "reach/c"] and sorted(r1["tasks"]) == ["reach/b"]      # index modulo 2
    assert r0["shard"] == "0/2" and r0["launched"] == 6 and r1["launched"] == 3
    merged = tmp_path / "merged" / "report.json"
    out = ctl("merge-eval", "--out", merged, "--tag", "pilot", *reports)
    m = json.loads(merged.read_text())
    assert sorted(m["tasks"]) == ["reach/a", "reach/b", "reach/c"] and m["launched"] == 9 == m["valid"]
    assert abs(m["success_rate"] - (r0["success_rate"] * 6 + r1["success_rate"] * 3) / 9) < 1e-12 and "3 tasks" in out
    assert len(LC.pool_tasks(HC.load(d / "harness.yaml")["tasks_list"], m, 0.0, 1.0)) == 3             # usable as --pool-report
    assert "shards must not overlap" in ctl("merge-eval", "--out", merged, "--tag", "x", reports[0], reports[0], expect=1)
    assert "unknown placeholder {model_port}" in ctl("eval", "--config", d / "harness.yaml", "--run-dir", tmp_path / "x", "--weights", "w",
                                                     "--tag", "t", "--init-key", "eval_init_ids", expect=1)
    assert "need 0 <= I < N" in ctl("eval", "--config", d / "harness.yaml", "--run-dir", tmp_path / "x", "--weights", "w", "--tag", "t",
                                    "--init-key", "eval_init_ids", "--shard", "2/2", "--var", "gpu=0", "--var", "model_port=1", "--var", "proxy_port=2", expect=1)


def test_stop_kills_a_child_that_ignores_term_after_its_shell_is_gone(tmp_path):
    """sh dies on TERM at once; the process it started ignores TERM: stop() must still end it (KILL to the group)"""
    import os, time
    pidfile = tmp_path / "child.pid"
    child = f"import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(600)"
    p = LC.Proc("stubborn", f"{sys.executable} -c \"{child}\" & wait", tmp_path / "log.txt", str(tmp_path))
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text():
            break
        time.sleep(0.1)
    pid = int(pidfile.read_text())
    t0 = time.time(); p.stop(grace_s=1.0)
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, 9); raise AssertionError("the child survived stop()")
    assert time.time() - t0 < 15


def test_a_shard_without_tasks_writes_an_empty_report_and_merges(tmp_path):
    cfg = make_cfg(tmp_path)
    out = ctl("eval", "--config", cfg, "--run-dir", tmp_path / "s5", "--weights", "w", "--tag", "t", "--init-key", "eval_init_ids", "--shard", "5/8")   # 3 tasks, 8 shards
    assert "has no task" in out
    empty = tmp_path / "s5" / "eval" / "t" / "report.json"
    assert json.loads(empty.read_text())["launched"] == 0
    ctl("eval", "--config", cfg, "--run-dir", tmp_path / "s1", "--weights", "w", "--tag", "t", "--init-key", "eval_init_ids", "--shard", "1/8")
    ctl("merge-eval", "--out", tmp_path / "m.json", "--tag", "t", empty, tmp_path / "s1" / "eval" / "t" / "report.json")
    m = json.loads((tmp_path / "m.json").read_text())
    assert list(m["tasks"]) == ["reach/b"] and m["launched"] == 3


def test_episodes_dealt_out_over_shards_on_training_inits_and_summed(tmp_path):
    """data collection: few tasks, many episodes each -> the EPISODES are dealt out, on the training inits, a task may
    ask for its own number of repeats, and the reports (the same task in several of them) are added up"""
    cfg = make_cfg(tmp_path)
    tasks = [json.loads(x) for x in (Path(cfg).parent / "tasks.jsonl").read_text().splitlines()]
    tasks[0]["repeats"] = 2
    (Path(cfg).parent / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    want = sum(len(t["init_ids"]) * t.get("repeats", 1) for t in tasks)
    reports, seen = [], set()
    for g in range(3):
        run = tmp_path / f"e{g}"
        ctl("eval", "--config", cfg, "--run-dir", run, "--weights", "w", "--tag", "c", "--shard", f"{g}/3", "--shard-by", "episode",
            "--init-key", "init_ids")
        reports.append(run / "eval" / "c" / "report.json")
        for r in (run / "eval" / "c" / "eps").rglob("result.json"):
            d = json.loads(r.read_text()); key = (d["task"], d["init"], d["attempt"])
            assert key not in seen and d["init"] in {t["task_id"]: t for t in tasks}[d["task"]]["init_ids"]
            seen.add(key)
    assert len(seen) == want
    assert "shards must not overlap" in ctl("merge-eval", "--out", tmp_path / "m.json", "--tag", "c", *reports, expect=1)
    ctl("merge-eval", "--sum-tasks", "--out", tmp_path / "m.json", "--tag", "c", *reports)
    m = json.loads((tmp_path / "m.json").read_text())
    assert m["launched"] == want and sum(t["episodes"] for t in m["tasks"].values()) == want
    assert m["tasks"][tasks[0]["task_id"]]["episodes"] == 2 * len(tasks[0]["init_ids"])


def test_validation_inits_choose_the_pool_and_never_mix_with_the_other_inits(tmp_path):
    """--init-key val_init_ids runs only the validation inits; its report is a valid --pool-report; an init listed in
    two of init_ids / eval_init_ids / val_init_ids is refused; the init lists are not template variables"""
    cfg, run = make_cfg(tmp_path), tmp_path / "val"
    ctl("eval", "--config", cfg, "--run-dir", run, "--weights", "w", "--tag", "val_v000", "--init-key", "val_init_ids")
    rep = json.loads((run / "eval" / "val_v000" / "report.json").read_text())
    assert rep["launched"] == rep["valid"] == 6 and all(t["episodes"] == 2 for t in rep["tasks"].values())
    inits = {json.loads(r.read_text())["init"] for r in (run / "eval" / "val_v000" / "eps").rglob("result.json")}
    assert inits == {4, 5}
    ctl("train", "--config", cfg, "--run-dir", tmp_path / "train", "--weights", "w", "--rounds", 1, "--groups", 2,
        "--pool-report", run / "eval" / "val_v000" / "report.json", "--rate-lo", 0, "--rate-hi", 1)
    used = {g[1] for g in json.loads((tmp_path / "train" / "state.json").read_text())["used"]}
    assert used and used <= {11, 12, 13, 14, 15, 16}
    t = HC.load(cfg)["tasks_list"][0]
    assert not set(HC.INIT_KEYS) & set(HC.episode_vars(HC.apply_vars(HC.load(cfg), {}), t, 4, 0, tmp_path, "w"))
    tasks = [json.loads(x) for x in (Path(cfg).parent / "tasks.jsonl").read_text().splitlines()]
    out = ctl("eval", "--config", cfg, "--run-dir", tmp_path / "x", "--weights", "w", "--tag", "t", expect=2)
    assert "the following arguments are required: --init-key" in out                           # no silent default
    out = ctl("eval", "--config", cfg, "--run-dir", tmp_path / "x", "--weights", "w", "--tag", "t", "--init-key", "val_ids", expect=2)
    assert "invalid choice: 'val_ids'" in out
    tasks[1]["val_init_ids"] = [3, 4]
    (Path(cfg).parent / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    with pytest.raises(HC.ConfigError, match=r"eval_init_ids and val_init_ids share \[3\]"):
        HC.load(cfg)
