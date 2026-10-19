"""Turn one round of RL rollouts into training samples (RLOO advantages, or success-only).

The builder is harness-agnostic: where the episodes are, which group each belongs to and how it ended come from a
harness adapter (--adapter, rl/adapters/; the interface is described in rl/adapters/__init__.py).

  1. verdict per episode (from the adapter):
       valid episode                           -> reward from the harness (1 solved, 0 not solved)
       infrastructure failure                  -> invalid (not a task failure)
     (an episode that left nothing at all counts as invalid: --group-size),
     plus a record check: the proxy must hold one record per model request
     (when the adapter knows the number of requests); otherwise the episode is invalid ("records");
     and every record must come from the round's policy version ("version").
  2. groups = the adapter's group key (same task, same initial state). A group with more than half of its
     episodes invalid is dropped; otherwise its valid episodes form the group (G_valid >= 2 needed).
  3. advantage per episode:
       rloo:    A_i = r_i - mean_{j != i} r_j          (no std division, Dr.GRPO / RLOO)
       success: A_i = r_i                              (SFT on the policy's own successes)
     Episodes with A = 0 are not trained (they carry no gradient).
  4. turns: up to --turns-per-episode records per episode, uniformly without replacement,
     with a RNG seeded by (round seed, episode key) -> identical after a restart.
     Sample weight inside the episode = 1 / (number of sampled turns): every episode counts
     the same in the loss whatever its length ("episode-mean").
     Optional exclusions (the drawn turn is replaced by an allowed turn that was not drawn):
     --max-output-tokens / --max-total-tokens (over-long turns) and --drop-post-success
     (turns of a solved episode after the harness first reported success, except calls of finish).

Writes samples.jsonl (one row per training turn) and round_stats.json.
  python rl/build_round.py --root <dir> --policy-version v000 --round-seed 1 --mode rloo --group-size 6 --out <dir>
"""
import argparse
import collections
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapters as _adapters  # noqa: E402

AD = None   # the harness adapter (rl/adapters/), set by main(): where episodes are, their groups and how they ended


def episode_rows(root: Path, policy_version: str):
    for ep, v in AD.find_episodes(root):
        turns = sorted((ep / "rl_turns").glob("turn_*.json.gz"))
        row = {**v, "episode_dir": str(ep), "n_records": len(turns)}
        if not v["valid"]:
            row["status"] = "invalid_infrastructure"
        elif row["requests"] is not None and row["requests"] != len(turns):
            row["status"] = "invalid_records"
        elif not turns:
            row["status"] = "invalid_records"
        else:
            versions = set()
            for tp in (turns[0], turns[-1]):
                versions.add(json.loads(gzip.decompress(tp.read_bytes())).get("policy_version"))
            row["status"] = "valid" if versions == {policy_version} else "invalid_version"
        row["reward"] = v["reward"] if row["status"] == "valid" else None
        row["reward_terminal"] = row["reward"]
        row.update(AD.episode_extras(turns))
        yield row


def _turn_tokens(turn_file: Path):
    """(prompt tokens, output tokens) of one rl_turns record"""
    rec = json.loads(gzip.decompress(turn_file.read_bytes()))
    return len(rec["prompt_token_ids"]), len(rec["output_token_ids"])


def _too_long(turn_file: Path, max_output: int, max_total: int) -> bool:
    """--max-output-tokens / --max-total-tokens (0 = that limit off)"""
    p, o = _turn_tokens(turn_file)
    return bool((max_output and o > max_output) or (max_total and p + o > max_total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--policy-version", required=True)
    ap.add_argument("--round-seed", required=True, type=int)
    ap.add_argument("--mode", choices=("rloo", "success"), required=True)
    ap.add_argument("--turns-per-episode", type=int, default=16)
    ap.add_argument("--group-size", type=int, required=True, help="episodes launched per group; missing ones count as invalid")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--adapter", default="rpent_libero",
                    help="harness adapter (rl/adapters/<name>.py): where the episodes are, their groups and how they ended")
    ap.add_argument("--max-output-tokens", type=int, default=0,
                    help="turns whose output has more tokens are not sampled (another turn of the episode is drawn instead); "
                         "0 = off. A 32,768-token output ran the trainer out of GPU memory")
    ap.add_argument("--drop-post-success", action="store_true",
                    help="solved episodes: turns after the harness first reported success are not sampled (another turn "
                         "is drawn), except turns calling finish. Post-success loops (e.g. re-reading one file 12 times) "
                         "would otherwise be reinforced by the positive advantage")
    ap.add_argument("--max-total-tokens", type=int, default=0,
                    help="same for prompt + output tokens (0 = off). A 229,807-token turn ran the trainer out of GPU memory")
    a = ap.parse_args()
    global AD
    try:
        AD = _adapters.load(a.adapter)
    except (ImportError, ValueError) as e:
        ap.error(f"--adapter {a.adapter}: {e}")
    a.out.mkdir(parents=True, exist_ok=True)

    # One row per attempt: an attempt relaunched after a crash can have several transcripts (under different
    # run ids); keep a valid one if any (the first by path), else the first invalid one. Duplicates are counted.
    by_attempt = collections.defaultdict(list)
    for r in episode_rows(a.root, a.policy_version):
        by_attempt[(*AD.group_key(r), r["attempt"])].append(r)
    rows, duplicates = [], 0
    for key in sorted(by_attempt):
        cand = sorted(by_attempt[key], key=lambda r: r["episode_dir"])
        valid = [r for r in cand if r["status"] == "valid"]
        duplicates += len(cand) - 1
        if len(valid) > 1:
            print(f"WARNING two valid transcripts for {key}: keeping {valid[0]['episode_dir']}", file=sys.stderr)
        rows.append(valid[0] if valid else cand[0])
    groups = collections.defaultdict(list)
    for r in rows:
        groups[AD.group_key(r)].append(r)

    samples, gstats = [], []
    long_turns = {}   # --max-output-tokens: episode -> turns left out of the draw
    post_success = {"by_episode": {}, "unaligned": []}   # --drop-post-success
    for gk in sorted(groups):
        eps = sorted(groups[gk], key=lambda r: r["attempt"])
        valid = [r for r in eps if r["status"] == "valid"]
        n_launched = max(a.group_size, len(eps))
        g = {"group": list(gk), "n": n_launched, "found": len(eps), "valid": len(valid), "solved": sum(int(r["reward_terminal"]) for r in valid)}
        if len(valid) * 2 < n_launched or len(valid) < 2:
            g["used"] = "dropped_invalid"
            gstats.append(g)
            continue
        rs = [r["reward"] for r in valid]
        for i, r in enumerate(valid):
            if a.mode == "rloo":
                others = rs[:i] + rs[i + 1:]
                r["advantage"] = rs[i] - sum(others) / len(others)
            else:
                r["advantage"] = r["reward"]
        g["used"] = "zero_advantage" if all(r["advantage"] == 0 for r in valid) else "trained"
        gstats.append(g)
        for r in valid:
            key = _adapters.episode_key(AD, gk, r["attempt"])
            if r["advantage"] == 0:
                continue  # a zero-advantage episode carries no gradient
            ep = Path(r["episode_dir"])
            turns = sorted((ep / "rl_turns").glob("turn_*.json.gz"))
            rng = random.Random(int(hashlib.sha256(f"{a.round_seed}|{key}".encode()).hexdigest()[:16], 16))
            k = min(a.turns_per_episode, len(turns))
            chosen = sorted(rng.sample(range(len(turns)), k))
            # the original draw stays; only drawn turns that are excluded (over a length limit, or after success in
            # a solved episode) are replaced, by allowed turns that were not drawn (an episode keeps min(k, allowed))
            excl = set()
            if a.max_output_tokens or a.max_total_tokens:
                long_ = [t for t in range(len(turns)) if _too_long(turns[t], a.max_output_tokens, a.max_total_tokens)]
                if long_:
                    long_turns[key] = long_
                    excl |= set(long_)
            if a.drop_post_success and r["reward_terminal"] == 1:
                ps = AD.post_success_turns(ep, len(turns))
                if ps is None:
                    post_success["unaligned"].append(key)
                elif ps["excluded"] or ps["finish_kept"]:
                    post_success["by_episode"][key] = ps
                    excl |= set(ps["excluded"])
            if excl:
                keep = [t for t in chosen if t not in excl]
                spare = [t for t in range(len(turns)) if t not in excl and t not in chosen]
                k = min(k, len(turns) - len(excl))
                chosen = sorted(keep + rng.sample(spare, k - len(keep)))
            wts = {t: 1.0 / k for t in chosen}
            for ti in chosen:
                samples.append({"episode": key, "episode_dir": str(ep), "turn_file": turns[ti].name, "turn": ti,
                                "n_turns": len(turns), "k": len(chosen), "group": list(gk), "reward": r["reward"],
                                "advantage": r["advantage"], "weight": wts[ti], "policy_version": a.policy_version})

    order = random.Random(a.round_seed).sample(range(len(samples)), len(samples))
    with open(a.out / "samples.jsonl", "w") as f:
        for i in order:
            f.write(json.dumps(samples[i]) + "\n")
    status = collections.Counter(r["status"] for r in rows)
    kinds = collections.Counter(AD.outcome_kind(r) for r in rows if r["status"] == "valid")
    stats = {
        "policy_version": a.policy_version, "mode": a.mode, "round_seed": a.round_seed,
        "episodes": len(rows), "duplicate_transcripts_dropped": duplicates, "status": dict(status), "valid_outcomes": dict(kinds),
        "success_rate_valid": (sum(r["reward_terminal"] for r in rows if r["status"] == "valid") /
                               max(1, sum(1 for r in rows if r["status"] == "valid"))),
        "groups": len(gstats), "groups_used": dict(collections.Counter(g["used"] for g in gstats)),
        "trained_episodes": len({s["episode"] for s in samples}), "samples": len(samples),
        "episode_elapsed_s": sorted(r.get("elapsed_s") or 0 for r in rows),
        "by_outcome": {name: {"episodes": len(sel), **{k: round(sum(r[k] or 0 for r in sel) / len(sel), 2) for k in
                                                        AD.OUTCOME_METRICS}}
                       for name, sel in (("solved", [r for r in rows if r["reward_terminal"] == 1]), ("failed", [r for r in rows if r["reward_terminal"] == 0]))
                       if sel},
        "max_output_tokens": a.max_output_tokens, "max_total_tokens": a.max_total_tokens,
        "drop_post_success": a.drop_post_success,
        "post_success_excluded": {"episodes": sum(1 for v in post_success["by_episode"].values() if v["excluded"]),
                                  "turns": sum(len(v["excluded"]) for v in post_success["by_episode"].values()),
                                  "finish_turns_kept": sum(len(v["finish_kept"]) for v in post_success["by_episode"].values()),
                                  "unaligned_solved_episodes": post_success["unaligned"], "by_episode": post_success["by_episode"]},
        "long_turns_excluded": {"episodes": len(long_turns), "turns": sum(len(v) for v in long_turns.values()), "by_episode": long_turns},
        "group_table": gstats,
        "episode_table": [{k: r.get(k) for k in AD.EPISODE_KEYS} for r in rows],
    }
    json.dump(stats, open(a.out / "round_stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("group_table", "episode_table", "episode_elapsed_s")}))


if __name__ == "__main__":
    main()
