"""harnesses/capx_libero: the wrapper around CaP-X, tested against a stand-in for CaP-X's launcher (same entry points and
the same on-disk outcome; the real one needs its simulator environment and is exercised by the wiring test on a GPU node)."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from conftest import ROOT

CAPX = ROOT / "harnesses" / "capx_libero"
MIN = ROOT / "harnesses" / "minimal"
sys.path.insert(0, str(CAPX))
import run_episode as RE  # noqa: E402

BASE = {"env": {"_target_": "x.Env", "cfg": {"low_level": {"_target_": "x.Low", "suite_name": "libero_spatial", "task_id": 0, "max_steps": 8000},
                                              "privileged": False, "apis": ["FrankaLiberoApi"], "multi_turn_prompt": "..."}},
        "api_servers": [{"_target_": "x.server", "port": 8116}], "record_video": True, "output_dir": "./outputs/x",
        "use_parallel_ensemble": True, "use_multimodel": True, "use_img_differencing": True, "trials": 50, "num_workers": 10}


def test_make_config_changes_only_what_it_says(tmp_path):
    cfg = RE.make_config(BASE, "libero_goal_swap", 7, 23, tmp_path, visual_feedback=True)
    assert cfg["env"]["cfg"]["low_level"] == {"_target_": "x.Low", "suite_name": "libero_goal_swap", "task_id": 7, "max_steps": 8000}
    assert cfg["resume_idx"] == cfg["trials"] == 23 and cfg["num_workers"] == 1 and cfg["record_video"] is False
    assert not (cfg["use_parallel_ensemble"] or cfg["use_multimodel"] or cfg["use_img_differencing"] or cfg["use_video_differencing"])
    assert cfg["use_visual_feedback"] is True and cfg["output_dir"] == str(tmp_path / "capx" / "run")
    assert cfg["api_servers"] == BASE["api_servers"] and cfg["env"]["cfg"]["apis"] == ["FrankaLiberoApi"]
    assert BASE["env"]["cfg"]["low_level"]["task_id"] == 0 and BASE["trials"] == 50          # the base is not modified
    with pytest.raises(SystemExit):
        RE.make_config({"env": {"cfg": {}}}, "s", 0, 1, tmp_path, False)


def test_find_outcome_reads_capx_trial_dir(tmp_path):
    assert RE.find_outcome(tmp_path, 23) is None
    d = tmp_path / "capx" / "some_model" / "run"
    (d / "trial_23_sandboxrc_0_reward_1.000_taskcompleted_1").mkdir(parents=True)
    (d / "trial_22_sandboxrc_0_reward_1.000_taskcompleted_1").mkdir()                        # another trial: ignored
    (d / "trial_23_notes").mkdir()
    r = RE.find_outcome(tmp_path, 23)
    assert r["reward"] == 1.0 and r["valid"] is True and r["sandbox_rc"] == 0 and r["timed_out"] is False
    t = tmp_path / "t2" / "capx" / "run" / "trial_05_sandboxrc_1_reward_0.000_taskcompleted_0"; t.mkdir(parents=True)
    (t / "summary.txt").write_text("Trial 5 timed out after 1000 seconds.\n")
    r = RE.find_outcome(tmp_path / "t2", 5)
    assert r["reward"] == 0.0 and r["valid"] is True and r["timed_out"] is True and r["sandbox_rc"] == 1


# ---------------------------------------------------------------------------
# through the controller, with a stand-in for CaP-X's launcher
# ---------------------------------------------------------------------------

FAKE_LAUNCH = '''
    import json, hashlib, urllib.request
    from dataclasses import dataclass
    from pathlib import Path
    import yaml
    from capx.envs import runner

    @dataclass
    class LaunchArgs:
        config_path: str
        server_url: str = "http://127.0.0.1:8110/chat/completions"
        model: str = "m"
        temperature: float = 1.0
        max_tokens: int = 2048
        total_trials: int | None = None
        num_workers: int | None = None

    def main(args):
        cfg = yaml.safe_load(Path(args.config_path).read_text())
        assert cfg["resume_idx"] == cfg["trials"] == args.total_trials and runner.MAX_TRIAL_RETRIES == 1
        assert not cfg["use_parallel_ensemble"] and not cfg["use_multimodel"] and args.server_url.endswith("/v1/chat/completions")
        low = cfg["env"]["cfg"]["low_level"]; trial = args.total_trials
        if low["task_id"] == 9:
            raise RuntimeError("simulator crashed")          # an infrastructure failure: nothing is saved
        msgs = [{"role": "user", "content": f"Goal of {low['suite_name']} task {low['task_id']} (trial {trial}). Write code."}]
        done = False
        for turn in range(3):                                  # write code -> run -> look -> REGENERATE / FINISH
            body = {"model": args.model, "messages": msgs, "temperature": args.temperature, "max_tokens": args.max_tokens}
            req = urllib.request.Request(args.server_url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            reply = json.loads(urllib.request.urlopen(req, timeout=60).read())["choices"][0]["message"]["content"]
            msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": f"stdout after block {turn}: ok"}]
            if reply.endswith("2"):
                done = True
                break
        out = Path(cfg["output_dir"]); parts = list(out.parts); parts.insert(-1, str(args.model).replace("/", "_"))
        d = Path(*parts) / f"trial_{trial:02d}_sandboxrc_0_reward_{1.0 if done else 0.0:.3f}_taskcompleted_{int(done)}"
        d.mkdir(parents=True); (d / "summary.txt").write_text("ok")
'''


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_capx_wrapper_through_the_controller(tmp_path):
    fake = tmp_path / "fake_capx"; (fake / "capx" / "envs").mkdir(parents=True); (fake / "env_configs").mkdir()
    (fake / "capx" / "__init__.py").write_text(""); (fake / "capx" / "envs" / "__init__.py").write_text("")
    (fake / "capx" / "envs" / "runner.py").write_text("MAX_TRIAL_RETRIES = 3\nTRIAL_TIMEOUT_SECONDS = 1000\n")
    (fake / "capx" / "llm").mkdir(); (fake / "capx" / "llm" / "__init__.py").write_text("")
    (fake / "capx" / "llm" / "client.py").write_text("import requests\n")   # run_episode.py wraps its requests module
    (fake / "capx" / "envs" / "launch.py").write_text(textwrap.dedent(FAKE_LAUNCH))
    (fake / "env_configs" / "base.yaml").write_text(yaml.safe_dump(BASE))
    proxy, model = free_port(), free_port()
    cfg = yaml.safe_load((CAPX / "harness.yaml").read_text())
    cfg.update({"proxy_port": proxy, "services": [], "group_size": 3, "episodes_parallel": 3, "train": None,
                "model": {"url": f"http://127.0.0.1:{model}", "ready_timeout_s": 60,
                          "serve_cmd": f"{{python}} {MIN}/fake_model_server.py --port {model} --weights {{weights}}"}})
    cfg["episode"]["cmd"] = (cfg["episode"]["cmd"].replace("{config_dir}", str(CAPX))
                             .replace("env_configs/libero/franka_libero_goal_1.yaml", "env_configs/base.yaml"))
    cfg["episode"]["env"] = {"CAPX_DIR": str(fake), "CAPX_PYTHON": sys.executable}
    d = tmp_path / "cfg"; d.mkdir()
    (d / "tasks.jsonl").write_text("\n".join(json.dumps({"task_id": f"libero_goal_swap/{t}", "suite": "libero_goal_swap", "task": t,
                                                         "init_ids": [11, 12], "eval_init_ids": [1]}) for t in (0, 1, 9)) + "\n")
    (d / "harness.yaml").write_text(yaml.safe_dump(cfg))
    run = tmp_path / "run"
    r = subprocess.run([sys.executable, str(ROOT / "rl" / "local_ctl.py"), "train", "--config", str(d / "harness.yaml"), "--run-dir", str(run),
                        "--weights", "ckpt", "--rounds", "1", "--groups", "3", "--var", "gpu=0"], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    eps = run / "rounds" / "r0" / "eps"
    stats = json.loads((eps.parent / "build" / "round_stats.json").read_text())
    assert stats["status"] == {"valid": 6, "invalid_infrastructure": 3}                       # task 9 "crashed" 3 times
    by_group = {tuple(g["group"]): g["used"] for g in stats["group_table"]}
    assert by_group[("libero_goal_swap/9", [x for x in (11, 12) if ("libero_goal_swap/9", x) in by_group][0])] == "dropped_invalid"
    for res in eps.rglob("result.json"):
        rec = json.loads(res.read_text())
        ep = res.parent
        conf = yaml.safe_load((ep / "capx_config.yaml").read_text())
        assert conf["env"]["cfg"]["low_level"]["suite_name"] == "libero_goal_swap" and conf["trials"] == rec["init"] in (11, 12)
        assert rec["task"] == f"libero_goal_swap/{conf['env']['cfg']['low_level']['task_id']}"
        if rec["valid"]:
            n = len(list((ep / "rl_turns").glob("turn_*.json.gz")))
            assert 1 <= n <= 3 and rec["reward"] in (0.0, 1.0) and Path(rec["capx_trial_dir"]).is_dir()
        else:
            assert "simulator crashed" in (ep / "episode.log").read_text()


def _turn(ep, i, content):
    import gzip
    (ep / "rl_turns").mkdir(parents=True, exist_ok=True)
    (ep / "rl_turns" / f"turn_{i:03d}.json.gz").write_bytes(gzip.compress(json.dumps({"message": {"role": "assistant", "content": content}}).encode()))


def test_empty_last_reply_is_recognised(tmp_path):
    """CaP-X crashes when the model's reply has no answer (all of it in the reasoning): that is the policy's failure"""
    a = tmp_path / "a"; _turn(a, 0, "```python\nx=1\n```"); _turn(a, 1, None)
    b = tmp_path / "b"; _turn(b, 0, None); _turn(b, 1, "FINISH")
    c = tmp_path / "c"; _turn(c, 0, "  \n ")
    assert RE.last_reply_is_empty(a) and not RE.last_reply_is_empty(b) and RE.last_reply_is_empty(c)
    assert not RE.last_reply_is_empty(tmp_path / "no_calls_at_all")            # the simulator died before any call: infrastructure


def test_capx_crash_on_an_empty_answer_is_a_valid_failure_through_the_wrapper(tmp_path):
    fake = tmp_path / "fake_capx"; (fake / "capx" / "envs").mkdir(parents=True); (fake / "env_configs").mkdir()
    (fake / "capx" / "__init__.py").write_text(""); (fake / "capx" / "envs" / "__init__.py").write_text("")
    (fake / "capx" / "envs" / "runner.py").write_text("MAX_TRIAL_RETRIES = 3\nTRIAL_TIMEOUT_SECONDS = 1000\n")
    (fake / "capx" / "llm").mkdir(); (fake / "capx" / "llm" / "__init__.py").write_text("")
    (fake / "capx" / "llm" / "client.py").write_text("import requests\n")   # run_episode.py wraps its requests module
    (fake / "capx" / "envs" / "launch.py").write_text(textwrap.dedent('''
        from dataclasses import dataclass
        @dataclass
        class LaunchArgs:
            config_path: str
            server_url: str = ""
            model: str = "m"
            temperature: float = 1.0
            max_tokens: int = 2048
            total_trials: int | None = None
            num_workers: int | None = None
        def main(args):
            raise TypeError("object of type 'NoneType' has no len()")
    '''))
    (fake / "env_configs" / "base.yaml").write_text(yaml.safe_dump(BASE))
    for name, content, want_rc, want in (("empty", None, 0, {"valid": True, "reward": 0.0, "empty_answer": True}), ("crash", "some code", 1, None)):
        out = tmp_path / name; _turn(out, 0, content)
        r = subprocess.run([sys.executable, str(CAPX / "run_episode.py"), "--capx-dir", str(fake), "--base-config", "env_configs/base.yaml",
                            "--suite", "libero_goal_swap", "--task", "3", "--trial", "13", "--model-url", "http://127.0.0.1:1/ep/x/v1",
                            "--model", "m", "--out", str(out)], capture_output=True, text=True)
        assert r.returncode == want_rc, r.stderr[-1500:]
        if want is None:
            assert not (out / "result.json").exists()
        else:
            res = json.loads((out / "result.json").read_text())
            assert {k: res[k] for k in want} == want


def test_with_timeout_replaces_only_the_post_timeout():
    calls = []
    class FakeRequests:
        exceptions = "kept"
        @staticmethod
        def post(url, **kw):
            calls.append((url, kw)); return "resp"
    r = RE.with_timeout(FakeRequests, 1500.0)
    assert r.post("http://x", headers={"a": 1}, data="d", timeout=200) == "resp" and r.exceptions == "kept"
    assert calls == [("http://x", {"headers": {"a": 1}, "data": "d", "timeout": 1500.0})]


def test_alarm_with_replaces_only_capx_literal_1000():
    seen = []
    alarm = RE.alarm_with(lambda s: seen.append(s) or 7, 3600)
    assert alarm(1000) == 7 and alarm(0) == 7 and alarm(3600) == 7 and alarm(5) == 7
    assert seen == [3600, 0, 3600, 5]
