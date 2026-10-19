"""--tail_batch (rl/plan.py, rl/train_rl.py): a last batch with fewer than global_batch samples is either a step of its
own (the original) or joins the step before it. Pure-Python checks: every sample once, padding only as padding, the same
number of passes on every rank, resume = uninterrupted, full batches unchanged. The planning lines of the trainer are
executed from its source text (the trainer itself needs GPUs and a process group)."""
import random
import re
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "rl"))
from plan import n_steps_for, step_gas, step_rows, stride_plan  # noqa: E402

B = 128
SIZES = [0, 1, 50, 127, 128, 129, 255, 256, 2584, 2756, 2817]
WORLDS = [4, 32, 64, 128]
SRC = (ROOT / "rl" / "train_rl.py").read_text()


def _legacy_stride(N, n_steps, B, world):
    """rl/plan.py stride_plan at 101e942"""
    gas = B // world
    return [[[min(s * B + r + k * world, N) for k in range(gas)] for r in range(world)] for s in range(n_steps)]


@pytest.mark.parametrize("N", SIZES)
@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("tail", ["own", "merge"])
def test_every_sample_once_and_same_passes_on_every_rank(N, world, tail):
    steps = n_steps_for(N, B, tail)
    plan = stride_plan(N, steps, B, world, tail)
    seen = []
    for s, step in enumerate(plan):
        lo, hi = step_rows(s, N, B, tail)
        g = step_gas(s, N, B, world, tail)
        assert all(len(m) == g for m in step)                       # every rank: the same number of passes
        real = sorted(i for m in step for i in m if i < N)
        assert real == list(range(lo, hi))                          # exactly the step's rows
        assert sum(i >= N for m in step for i in m) == g * world - (hi - lo)   # the rest is padding
        if s < steps - 1:
            assert g == B // world and hi - lo == B                 # only the last step can differ
        seen += real
    assert seen == list(range(N))                                   # nothing lost, nothing twice


@pytest.mark.parametrize("N", SIZES)
def test_merge_has_no_step_below_a_full_batch(N):
    steps = n_steps_for(N, B, "merge")
    rows = [step_rows(s, N, B, "merge") for s in range(steps)]
    sizes = [hi - lo for lo, hi in rows]
    assert sum(sizes) == N
    if N >= B:
        assert min(sizes) >= B and max(sizes) < 2 * B and steps == N // B
    else:
        assert sizes == ([N] if N else [])                          # one step when 0 < N < B, none when N == 0


def test_the_round_that_had_one_sample_alone():
    N, world = 2817, 32
    assert n_steps_for(N, B, "own") == 23 and step_rows(22, N, B, "own") == (2816, 2817)
    assert n_steps_for(N, B, "merge") == 22 and step_rows(21, N, B, "merge") == (2688, 2817)
    assert step_gas(21, N, B, world, "merge") == 5 and step_gas(20, N, B, world, "merge") == 4
    last = stride_plan(N, 22, B, world, "merge")[-1]
    assert sum(i >= N for m in last for i in m) == 5 * 32 - 129
    assert [r for r, m in enumerate(last) if 2816 in m] == [0]       # row 2816 = 2688 + 0 + 4 * 32: rank 0, fifth pass


@pytest.mark.parametrize("N", SIZES)
@pytest.mark.parametrize("world", WORLDS)
def test_default_is_the_old_plan(N, world):
    steps = -(-N // B)
    assert n_steps_for(N, B) == n_steps_for(N, B, "own") == steps
    assert stride_plan(N, steps, B, world) == stride_plan(N, steps, B, world, "own") == _legacy_stride(N, steps, B, world)


@pytest.mark.parametrize("N", [128, 256, 2816])
@pytest.mark.parametrize("world", WORLDS)
def test_full_batches_are_the_same_in_both_modes(N, world):
    steps = N // B
    assert n_steps_for(N, B, "merge") == n_steps_for(N, B, "own") == steps
    assert stride_plan(N, steps, B, world, "merge") == _legacy_stride(N, steps, B, world)


@pytest.mark.parametrize("max_steps", [1, 2, 21, 22, 23, 40])
def test_max_steps_does_not_move_the_merged_step(max_steps):
    N, world = 2817, 32
    steps = min(n_steps_for(N, B, "merge"), max_steps)
    plan = stride_plan(N, steps, B, world, "merge")
    for s, step in enumerate(plan):
        real = [i for m in step for i in m if i < N]
        assert len(real) == (129 if s == 21 else 128)                # a cut run never lets an earlier step swallow the rest


# ---- the trainer's own planning lines, executed from its source ----
def _trainer_plan(N, world, rank, tail, max_steps=-1, S0=0):
    def line(start):
        hits = [l for l in SRC.splitlines() if l.startswith(start)]
        assert len(hits) == 1, (start, hits)
        return hits[0]

    class A:
        global_batch = B; tail_batch = tail
    ns = {"N": N, "world": world, "rank": rank, "args": A, "gas": B // world, "BAL": None, "S0": S0,
          "step_gas": step_gas, "n_steps_for": n_steps_for}
    exec(line("n_steps = n_steps_for("), ns)
    if max_steps > 0:
        ns["n_steps"] = min(ns["n_steps"], max_steps)
    exec(line("GS = ["), ns)
    body = SRC[SRC.index("def my_indices(s):"):SRC.index("ds = RLDataset()")]
    exec(body, ns)
    exec(line("first_mine = "), ns)
    exec(line("plan = ["), ns)
    exec(line("    _rest = plan[").strip(), ns)
    return ns


@pytest.mark.parametrize("N", [50, 128, 129, 255, 256, 2817])
@pytest.mark.parametrize("world", [4, 32])
@pytest.mark.parametrize("tail", ["own", "merge"])
def test_trainer_lines_agree_with_plan_py_on_every_rank(N, world, tail):
    steps = n_steps_for(N, B, tail)
    ref = stride_plan(N, steps, B, world, tail)
    used = []
    for rank in range(world):
        ns = _trainer_plan(N, world, rank, tail)
        assert ns["n_steps"] == steps and len(ns["GS"]) == steps
        flat = [i for s in range(steps) for i in ref[s][rank]]
        assert len(ns["plan"]) == len(flat) == sum(ns["GS"])
        for (i, real), want in zip(ns["plan"], flat):
            assert real == (want < N) and (i == want if real else i == ns["first_mine"])
        used += [i for i, real in ns["plan"] if real]
    assert sorted(used) == list(range(N))


@pytest.mark.parametrize("tail", ["own", "merge"])
@pytest.mark.parametrize("N", [255, 300, 2817])
def test_resume_continues_exactly_where_the_uninterrupted_run_is(N, tail):
    world = 32
    steps = n_steps_for(N, B, tail)
    for rank in (0, 1, 31):
        full = _trainer_plan(N, world, rank, tail)
        per_step, pos = [], 0
        for s in range(steps):                                       # what the update loop consumes step by step
            per_step.append(full["plan"][pos:pos + full["GS"][s]]); pos += full["GS"][s]
        assert pos == len(full["plan"])
        for S0 in range(1, steps):                                   # mid checkpoints exist only after steps < n_steps
            rest = _trainer_plan(N, world, rank, tail, S0=S0)["_rest"]
            assert rest == [e for st in per_step[S0:] for e in st]


def _step_gradient(N, world, tail, s, w, g):
    """the step's gradient as the trainer builds it: every rank adds weight * grad / D * world for its passes
    (padding: weight 0), then the ranks are averaged"""
    step = stride_plan(N, n_steps_for(N, B, tail), B, world, tail)[s]
    D = sum(w[i] for m in step for i in m if i < N)
    total = 0.0
    for m in step:
        total += sum((w[i] if i < N else 0.0) * (g[i] if i < N else 123.0) / D * world for i in m)
    return total / world, D


def test_merged_step_is_the_weighted_mean_and_padding_adds_nothing():
    N, world = 2817, 32
    rng = random.Random(1)
    w = [rng.choice([1 / 16, 1 / 10, 1 / 8]) for _ in range(N)]
    g = [rng.uniform(-1, 1) for _ in range(N)]
    got, D = _step_gradient(N, world, "merge", 21, w, g)
    rows = range(2688, 2817)
    assert got == pytest.approx(sum(w[i] * g[i] for i in rows) / sum(w[i] for i in rows), rel=1e-12)
    # the last row's share of that step: its weight over 129 rows' weights -- not 1
    assert w[2816] / D < 0.02
    # the original: the same row alone, its weight cancels and it is the whole step
    alone, D1 = _step_gradient(N, world, "own", 22, w, g)
    assert D1 == pytest.approx(w[2816]) and alone == pytest.approx(g[2816], rel=1e-12)
    # full steps are the same numbers in both modes
    assert _step_gradient(N, world, "merge", 3, w, g) == _step_gradient(N, world, "own", 3, w, g)


# ---- wiring ----
def test_trainer_wiring():
    assert 'p.add_argument("--tail_batch", default="own", choices=("own", "merge")' in SRC
    for old in ("range(gas)", "gas - 1", "/ gas", "S0 * gas", "math.ceil(N / args.global_batch)"):
        assert old not in SRC, old                                   # every per-step count goes through GS[s]
    assert SRC.count("GS[s]") >= 5 and "plan[sum(GS[:S0]):]" in SRC
    assert '"tail_batch": args.tail_batch,' in SRC                   # written to the mid checkpoint's meta.json
    assert 'MID.get("tail_batch", "own") == args.tail_batch' in SRC  # a checkpoint without the key = the original
    assert re.search(r'if args\.tail_batch == "merge" and args\.plan != "stride":\n    raise ValueError', SRC)
    assert "os._exit(14)" in SRC and 'args.tail_batch != "own" and reals[k] and b["idx"] not in old' in SRC


# ---- protective exits: the trainer's own lines, executed with fakes ----
def _block(first, n):
    """n source lines starting at the one that begins with `first` (after indentation), dedented"""
    import textwrap
    lines = SRC.splitlines()
    hits = [i for i, l in enumerate(lines) if l.strip().startswith(first)]
    assert len(hits) == 1, (first, hits)
    return textwrap.dedent("\n".join(lines[hits[0]:hits[0] + n]))


class _Args:
    def __init__(self, tail, plan="stride"):
        self.tail_batch, self.plan = tail, plan


@pytest.mark.parametrize("tail,plan,refused", [
    ("merge", "stride", False), ("merge", "balanced", True), ("own", "balanced", False), ("own", "stride", False)])
def test_unsupported_combinations_are_refused(tail, plan, refused):
    code = _block('if args.tail_batch == "merge" and args.plan', 2)
    ns = {"args": _Args(tail, plan)}
    if refused:
        with pytest.raises(ValueError, match="only --plan stride"):
            exec(code, ns)
    else:
        exec(code, ns)


@pytest.mark.parametrize("meta,tail,ok", [
    ({}, "own", True), ({"tail_batch": "own"}, "own", True), ({"tail_batch": "merge"}, "merge", True),
    ({}, "merge", False), ({"tail_batch": "own"}, "merge", False), ({"tail_batch": "merge"}, "own", False)])
def test_a_checkpoint_of_the_other_tail_mode_is_not_continued(meta, tail, ok):
    code = _block('assert MID.get("tail_batch", "own") == args.tail_batch', 1)
    ns = {"MID": meta, "args": _Args(tail)}
    if ok:
        exec(code, ns)
    else:
        with pytest.raises(AssertionError, match="tail_batch"):
            exec(code, ns)


@pytest.mark.parametrize("tail,real,have_old,code_", [
    ("merge", True, False, 14), ("merge", True, True, None), ("merge", False, False, None), ("own", True, False, None)])
def test_missing_old_logprob_stops_a_merge_run(tail, real, have_old, code_, capsys):
    class _Exit(Exception):
        pass

    class _Os:
        @staticmethod
        def _exit(c):
            raise _Exit(c)
    code = _block('if args.tail_batch != "own" and reals[k]', 3)
    ns = {"args": _Args(tail), "reals": [real], "k": 0, "b": {"idx": 7}, "old": ({7: 0} if have_old else {}), "rank": 3, "s": 21,
          "os": _Os, "print": print}
    if code_ is None:
        exec(code, ns)
    else:
        with pytest.raises(_Exit) as e:
            exec(code, ns)
        assert e.value.args == (code_,) and "no pi_old for sample 7 at step 22" in capsys.readouterr().out
