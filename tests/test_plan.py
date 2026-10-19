"""rl/plan.py: the balanced sample plan keeps every step's rows, gives every rank exactly gas entries and evens the load."""
import random
import sys

from conftest import ROOT

sys.path.insert(0, str(ROOT / "rl"))
from plan import balanced_plan, busiest_over_mean, stride_plan  # noqa: E402


def _tok(n, seed=0):
    rng = random.Random(seed)
    return [rng.randint(12_000, 120_000) for _ in range(n)]


def test_same_rows_per_step_and_gas_entries_per_rank():
    N, B, world = 5166, 128, 32
    steps = -(-N // B)
    tok = _tok(N)
    bal, stri = balanced_plan(tok, N, steps, B, world), stride_plan(N, steps, B, world)
    for s in range(steps):
        assert all(len(m) == B // world for m in bal[s])
        real_b = sorted(i for m in bal[s] for i in m if i < N)
        real_s = sorted(i for m in stri[s] for i in m if i < N)
        assert real_b == real_s == list(range(s * B, min((s + 1) * B, N)))
    # padding only in the partial last step
    assert all(i < N for s in range(steps - 1) for m in bal[s] for i in m)
    assert sum(i == N for m in bal[-1] for i in m) == steps * B - N


def test_deterministic_and_more_even():
    N, B, world = 1000, 128, 32
    steps = -(-N // B)
    tok = _tok(N, 3)
    assert balanced_plan(tok, N, steps, B, world) == balanced_plan(list(tok), N, steps, B, world)
    b = busiest_over_mean(balanced_plan(tok, N, steps, B, world), tok, N)
    s = busiest_over_mean(stride_plan(N, steps, B, world), tok, N)
    assert b < 1.1 < s


def test_small_world_and_short_round():
    tok = [5, 1, 1, 1, 1, 1, 1, 9]
    plan = balanced_plan(tok, 8, 1, 8, 2)   # 1 step, 2 ranks x 4
    assert sorted(len(m) for m in plan[0]) == [4, 4]
    loads = [sum(tok[i] for i in m) for m in plan[0]]
    assert max(loads) - min(loads) <= 4
