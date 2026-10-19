"""Which rank computes which sample of an optimizer step (rl/train_rl.py --plan). Pure functions, unit-tested in
tests/reward/test_plan.py.

Step s (0-based) uses rows [s*B, min((s+1)*B, N)). Every rank gets exactly gas = B // world entries per step; the
entry N means padding (the trainer replaces it by a weight-0 repeat, so FSDP collectives stay in step).

--tail_batch: what happens to a last batch with fewer than B rows. 'own' (the original): it is a step of
its own, normalised by its own weights -- with N = 2817, B = 128 that was one sample with a whole optimizer step.
'merge': it joins the step before it, so the last step has B + (N % B) rows and every rank runs
ceil(rows / world) entries in it (same number on every rank); no step has fewer than B rows unless N < B.
"""


def n_steps_for(N, B, tail="own"):
    """optimizer steps of a round: 'own' ceil(N / B); 'merge' N // B (one step when 0 < N < B)"""
    if tail == "merge":
        return N // B if N >= B else (1 if N > 0 else 0)
    return -(-N // B)


def step_rows(s, N, B, tail="own"):
    """(first, end) row of step s; under 'merge' the round's last step runs to N"""
    if tail == "merge" and s == n_steps_for(N, B, tail) - 1:
        return s * B, N
    return s * B, min((s + 1) * B, N)


def step_gas(s, N, B, world, tail="own"):
    """entries of every rank in step s: B // world, or ceil(rows / world) in the merged last step. Which step is the
    last one comes from N and B, never from a truncated step count (--max_steps)."""
    lo, hi = step_rows(s, N, B, tail)
    return max(B // world, -(-(hi - lo) // world))


def stride_plan(N, n_steps, B, world, tail="own"):
    """the original plan: rank r takes s*B + r + k*world (k < gas); indices >= N are padding (returned as N)"""
    assert tail == "own" or n_steps <= n_steps_for(N, B, tail), (n_steps, N, B)   # a step after the merged one would repeat its rows
    return [[[min(s * B + r + k * world, N) for k in range(step_gas(s, N, B, world, tail))] for r in range(world)]
            for s in range(n_steps)]


def balanced_plan(tok, N, n_steps, B, world):
    """the step's rows longest first (tok[i] = prompt + output tokens), each to the least-loaded rank that still has
    fewer than gas rows (ties: lower rank); deterministic for the same tok"""
    gas = B // world
    out = []
    for s in range(n_steps):
        idx = sorted(range(s * B, min((s + 1) * B, N)), key=lambda i: (-tok[i], i))
        mine, load = [[] for _ in range(world)], [0] * world
        for i in idx:
            r = min((r for r in range(world) if len(mine[r]) < gas), key=lambda r: (load[r], r))
            mine[r].append(i); load[r] += tok[i]
        out.append([m + [N] * (gas - len(m)) for m in mine])
    return out


def busiest_over_mean(plan, tok, N):
    """sum over steps of the busiest rank's tokens / sum of the mean rank's tokens (1.0 = perfectly even)"""
    smax = smean = 0.0
    for step in plan:
        load = [sum(tok[i] for i in m if i < N) for m in step]
        smax += max(load); smean += sum(load) / len(load)
    return smax / max(smean, 1e-9)
