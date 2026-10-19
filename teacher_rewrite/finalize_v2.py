#!/usr/bin/env python3
"""Rewrite v2, step 5: final decision per turn = automatic checks + 27B judge.
 target_clean                       -> target, unchanged
 history_* (reads memory, tool args cite a source, >= 3 flagged sentences) -> history (an original target is demoted)
 llm passing the checks          -> target only if the judge says OK
 judge_only (unedited)             -> target if the judge says OK or MINOR
 anything else                       -> history
Writes <dir>/changes.jsonl (rows only for turns whose disposition or text changes; input of apply_rewrite_to_shards.py),
<dir>/rewrite_log.jsonl.gz (every turn that was not target_clean: category, checks, judge verdict, before/after) and prints counts.
Usage: finalize_v2.py <dir> <judge.jsonl>"""
import collections, gzip, json, os, sys
def main():
    d, jf = sys.argv[1], sys.argv[2]
    judge = {}
    for l in open(jf):
        r = json.loads(l)
        if r.get("ok") or r["key"] not in judge: judge[r["key"]] = r
    summary = collections.Counter(); per_suite = collections.defaultdict(collections.Counter)
    with open(f"{d}/changes.jsonl", "w", encoding="utf-8") as fc, gzip.open(f"{d}/rewrite_log.jsonl.gz", "wt", encoding="utf-8") as fl:
        for l in open(f"{d}/checked.jsonl"):
            r = json.loads(l); c, before = r["category"], r["disposition_before"]; j = judge.get(r["key"])
            if r["decision"] == "to_judge":
                # edited turns need a clean OK; unedited turns may also pass with MINOR (v2 review: all 4 BAD of 60 were
                # edited turns the judge called MINOR; 0 BAD among 44 turns kept by this rule)
                ok = bool(j and j.get("ok") and j["verdict"] in (("OK",) if c == "llm" else ("OK", "MINOR")))
                final = "target" if ok else "history"
                why = f"judge_{j['verdict']}" if j and j.get("ok") else "judge_failed"
            else:
                final = r["decision"]; why = c if c != "llm" else "checks_failed"
            summary[(before, final, why)] += 1; per_suite[r["suite"]][(before, final)] += 1
            changed_text = final == "target" and r.get("after") is not None and (
                r["after"]["think"] != (r["before"]["think"] or "") or r["after"]["text"] != (r["before"]["text"] or ""))
            if ("target" if final == "target" else "keep_history_no_target") != before or changed_text:
                row = {"key": r["key"], "decision": final, "why": why, "think": None, "text": None}
                if changed_text: row.update(think=r["after"]["think"], text=r["after"]["text"])
                fc.write(json.dumps(row, ensure_ascii=False) + "\n")
            if c != "target_clean":
                fl.write(json.dumps({"key": r["key"], "category": c, "disposition_before": before, "decision": final, "why": why,
                                     "problems": r.get("problems"), "judge": {k: j.get(k) for k in ("verdict", "issues", "reason")} if j else None,
                                     "edits": r.get("edits"), "before": r.get("before"), "after": r.get("after") if final == "target" else None},
                                    ensure_ascii=False) + "\n")
    tot = collections.Counter()
    for (b, f, w), n in sorted(summary.items()): print(f"{b:>22} -> {f:<7} {w:<34} {n}"); tot[(b, f)] += n
    print({f"{b}->{f}": n for (b, f), n in tot.items()})
    json.dump({"counts": {f"{b}->{f} ({w})": n for (b, f, w), n in summary.items()},
               "per_suite": {s: {f"{b}->{f}": n for (b, f), n in v.items()} for s, v in per_suite.items()}},
              open(f"{d}/final_summary.json", "w"), indent=1)
if __name__ == "__main__":
    main()
