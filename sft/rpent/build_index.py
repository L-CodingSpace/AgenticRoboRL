#!/usr/bin/env python3
"""Frozen training index + memory-dependency list from the replayed episodes.

  python build_index.py <replay dir> <manifest jsonl> <out dir> [seed=3407]

Writes
  train_index.jsonl           one row per training target, in the fixed training order
                              (global shuffle with the seed -> tasks interleaved)
  memory_dependent.jsonl      every teacher turn that is NOT a target, with the reason
  replay_summary.json         counts per suite / reason, replay problems, per-batch suite mix
Fails (exit 1) if a manifest episode has no replay file or a replay reported problems.
"""
import collections, json, os, random, statistics, sys

replay, manifest, out = sys.argv[1], sys.argv[2], sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 3407
os.makedirs(out, exist_ok=True)
rows = [json.loads(l) for l in open(manifest)]
targets, dependent, problems, benign = [], [], [], []
per_suite = collections.defaultdict(collections.Counter)
reasons = collections.Counter()
for r in rows:
    f = os.path.join(replay, r["key"] + ".json")
    if not os.path.exists(f):
        problems.append({"key": r["key"], "problem": "no replay file"})
        continue
    ep = json.load(open(f))
    m = ep["meta"]
    exhausted = (m.get("error") == "UnexpectedModelBehavior: Exceeded maximum output retries (1)"
                 and m["n_replayed"] == m["n_turns_teacher"] - m["dispositions"].get("drop_turn", 0))
    if exhausted:
        # teacher used all 100 turns without a finish call; every turn was replayed, the loop then got
        # the empty "transcript exhausted" reply and stopped -- nothing is missing
        benign.append(r["key"])
    elif m.get("error") or m.get("images_unresolved") or m.get("replay_mismatch"):
        problems.append({"key": r["key"], "problem": "replay", "error": m.get("error"), "images_unresolved": m.get("images_unresolved"),
                         "mismatch": m.get("replay_mismatch")[:3]})
    if m["suite"] != r["suite"] or m["task"] != r["task"] or m["seed"] != r["seed"]:
        problems.append({"key": r["key"], "problem": "meta does not match manifest"})
    s = per_suite[r["suite"]]
    s["episodes"] += 1
    s["teacher_turns"] += m["n_turns_teacher"]
    s["dropped_turns"] += m["dispositions"].get("drop_turn", 0)
    for pos, smp in enumerate(ep["samples"]):
        if smp["disposition"] == "target":
            targets.append({"key": f"{r['key']}:{smp['turn_index']}", "episode": r["key"], "pos": pos, "turn": smp["turn_index"],
                            "suite": r["suite"], "task": r["task"]})
            s["targets"] += 1
        else:
            dependent.append({"episode": r["key"], "turn": smp["turn_index"], "suite": r["suite"], "task": r["task"],
                              "reasons": smp["reasons"], "mem_keywords": smp["mem_keywords"], "mem_ngram_hits": smp["mem_ngram_hits"],
                              "numbers_from_removed": smp["numbers_from_removed"][:20],
                              "dropped_calls": smp["response"]["dropped_memory_calls"], "handling": "kept in history, not a target"})
            s["history_only"] += 1
            for x in smp["reasons"]:
                reasons[x] += 1
                s["reason:" + x] += 1
random.Random(seed).shuffle(targets)
with open(os.path.join(out, "train_index.jsonl"), "w") as fo:
    for t in targets:
        fo.write(json.dumps(t) + "\n")
with open(os.path.join(out, "memory_dependent.jsonl"), "w") as fo:
    for d in dependent:
        fo.write(json.dumps(d) + "\n")
mix = [len(set(t["suite"] for t in targets[i:i + 64])) for i in range(0, len(targets), 64)]
summary = {"episodes": len(rows), "targets": len(targets), "history_only": len(dependent), "reasons": dict(reasons),
           "per_suite": {k: dict(v) for k, v in sorted(per_suite.items())}, "problems": problems,
           "steps_at_64": -(-len(targets) // 64),
           "transcript_exhausted_after_all_turns": benign,
           "suites_per_batch": {"min": min(mix) if mix else 0, "median": statistics.median(mix) if mix else 0, "max": max(mix) if mix else 0},
           "seed": seed}
json.dump(summary, open(os.path.join(out, "replay_summary.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "problems"}, indent=1))
print("problems", len(problems), problems[:5])
sys.exit(1 if problems else 0)
