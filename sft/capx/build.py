#!/usr/bin/env python3
"""capx_sft_build.py --runs <capx_runs dir> --rounds 'pilot_tch27b_r*' --model <student dir> --out <data dir> [--seed 3407]
Teacher data -> SFT training set (run on a node, with the serving env's python).
Every SOLVED episode of the given rounds (<round>/n<i>/eps/<key>.json with "solved": true + <key>.tgz) is unpacked to
<out>/eps/<round>/n<i>/..., and every model call of it becomes one training target if
  - the call ended normally (finish_reason "stop", last output token <|im_end|>, no empty output),
  - the TEACHER's chat template renders the request to exactly the recorded prompt tokens (the record is intact),
  - the STUDENT's chat template renders it to the same tokens minus one inserted run right after "<|im_start|>system\n"
    (the teacher template's reasoning-effort sentence; the same run in every call),
  - every output token id exists in the student's vocabulary.
Writes <out>/index.jsonl (shuffled with the seed), <out>/summary.json. Exit 1 when any call fails the template or
vocabulary check (the data would not be what the student sees at inference), or no target is left.
"""
import argparse, collections, glob, gzip, json, os, random, subprocess, sys
from concurrent.futures import ProcessPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True); ap.add_argument("--rounds", required=True); ap.add_argument("--model", required=True); ap.add_argument("--teacher", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=3407); ap.add_argument("--workers", type=int, default=32)
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_proc = _tproc = None


def unpack(job):
    js, dst = job
    meta = json.load(open(js))
    ep = os.path.join(dst, meta["dir"])
    if not os.path.isdir(os.path.join(ep, "rl_turns")):
        os.makedirs(dst, exist_ok=True)
        subprocess.run(["tar", "-xzf", js[:-5] + ".tgz", "-C", dst], check=True)
    return meta, ep


def check(job):
    global _proc, _tproc
    meta, ep, rel = job
    if _proc is None:
        from transformers import AutoProcessor
        _proc = AutoProcessor.from_pretrained(a.model); _tproc = AutoProcessor.from_pretrained(a.teacher)
    import capx_sft_dataset as D
    tok = _proc.tokenizer; end = tok.convert_tokens_to_ids("<|im_end|>"); V = len(tok)
    res = json.load(open(os.path.join(ep, "result.json")))
    rows = []
    if not (res.get("valid") is True and res.get("reward") == 1.0):
        return [{"episode": rel, "why": "result.json is not a valid success"}]
    turns = sorted(glob.glob(os.path.join(ep, "rl_turns", "turn_*.json.gz")))
    for pos, tp in enumerate(turns):
        rec = json.loads(gzip.decompress(open(tp, "rb").read()))
        row = {"episode": rel, "pos": pos, "turn": pos, "task": meta["task"], "init": meta["init"], "n_turns": len(turns),
               "n_prompt": len(rec.get("prompt_token_ids") or []), "n_out": len(rec.get("output_token_ids") or [])}
        out = rec.get("output_token_ids") or []
        if os.path.basename(tp) != f"turn_{pos:03d}.json.gz" or rec.get("turn_index") != pos:
            row["why"] = "turn numbering"
        elif rec.get("finish_reason") != "stop" or not out or out[-1] != end:
            row["why"] = "skip: call did not end normally (%s)" % rec.get("finish_reason")
        elif max(out) >= V or min(out) < 0:
            row["why"] = "vocabulary"
        else:
            v = rec["prompt_token_ids"]; h = D.prefix_ids(_proc, rec).tolist(); k = len(v) - len(h)
            if D.prefix_ids(_tproc, rec).tolist() != v:
                row["why"] = "teacher template does not reproduce the recorded prompt"
            elif not (k > 0 and h[:3] == v[:3] and h[3:] == v[3 + k:]):
                row["why"] = "student template differs by more than the inserted sentence"
            else:
                row["inserted"] = _tproc.tokenizer.decode(v[3:3 + k]); row["n_prefix_student"] = len(h)
        rows.append(row)
    return rows


def main():
    # shared storage answers slowly per file: list and read the small per-episode files with many threads
    from concurrent.futures import ThreadPoolExecutor
    cands = []
    for rd in sorted(glob.glob(os.path.join(a.runs, a.rounds))):
        for js in sorted(glob.glob(os.path.join(rd, "n*", "eps", "*.json")) + glob.glob(os.path.join(rd, "eps", "*.json"))):
            node = os.path.basename(os.path.dirname(os.path.dirname(js)))
            cands.append((js, os.path.join(a.out, "eps", os.path.basename(rd), node if node.startswith("n") else "n0")))
    print("finished episodes listed:", len(cands), flush=True)

    def solved(c):
        try:
            return json.load(open(c[0])).get("solved") is True and os.path.exists(c[0][:-5] + ".tgz")
        except Exception:
            return False
    with ThreadPoolExecutor(64) as tp:
        jobs = [c for c, ok in zip(cands, tp.map(solved, cands)) if ok]
    print("solved episodes found:", len(jobs), flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        eps = list(ex.map(unpack, jobs, chunksize=8))
        todo = [(m, ep, os.path.relpath(ep, a.out)) for m, ep in eps]
        rows = [r for rs in ex.map(check, todo, chunksize=4) for r in rs]
    targets = [r for r in rows if "why" not in r]
    bad = [r for r in rows if "why" in r and not r["why"].startswith("skip")]
    skipped = [r for r in rows if "why" in r and r["why"].startswith("skip")]
    per = collections.defaultdict(lambda: {"episodes": set(), "inits": set(), "targets": 0})
    for r in targets:
        p = per[r["task"]]; p["episodes"].add(r["episode"]); p["inits"].add(r["init"]); p["targets"] += 1
    tot = sorted(r["n_prefix_student"] + r["n_out"] for r in targets) or [0]
    inserted = collections.Counter(r.pop("inserted") for r in targets)
    if len(inserted) > 1:
        bad.append({"why": "more than one inserted text", "texts": list(inserted)[:3]})
    random.Random(a.seed).shuffle(targets)
    with open(os.path.join(a.out, "index.jsonl"), "w") as f:
        for r in targets:
            f.write(json.dumps(r) + "\n")
    summary = {"teacher_template_inserts": dict(inserted), "solved_episodes": len(jobs), "episodes_with_targets": len({r["episode"] for r in targets}), "targets": len(targets),
               "skipped_calls": len(skipped), "failed_checks": len(bad), "failed_examples": bad[:10], "seed": a.seed,
               "steps_at_64": -(-len(targets) // 64), "target_tokens": sum(r["n_out"] for r in targets),
               "tokens_per_sample": {"median": tot[len(tot) // 2], "p95": tot[int(len(tot) * .95)], "max": tot[-1]},
               "turns_per_episode": dict(sorted(collections.Counter(r["n_turns"] for r in rows if r.get("pos") == 0).items())),
               "per_task": {k: {"episodes": len(v["episodes"]), "inits": len(v["inits"]), "targets": v["targets"]} for k, v in sorted(per.items())}}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}))
    print("CAPX_SFT_DATA_" + ("READY" if targets and not bad else "FAILED"))
    return 0 if targets and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
