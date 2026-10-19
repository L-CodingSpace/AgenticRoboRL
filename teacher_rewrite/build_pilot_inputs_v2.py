#!/usr/bin/env python3
"""Rewrite v2, step 1: decide per turn what to do and build 27B edit requests (after the 60-turn review of v1).
Changes vs v1:
 - source rules v1.3; existing training targets are checked too (some leak memory file names / prior attempts);
 - hidden number = candidate number (see below) that is not in what the student saw before the turn (system prompt, user
   message, tool results, tool-call args; the agent's own earlier free text does not count; a visible value that rounds to
   it counts as visible) AND not in this turn's own tool-call args. Sentences holding one are flagged (rule "hidden_number");
 - gate: turns with >= 3 source-flagged sentences stay history (review: 29/31 such turns were BAD, 0 OK);
 - tool-call args (e.g. audit content, finish summary) that cite a source -> turn stays history;
 - history turns with nothing flagged are not edited but still go through the 27B judge ("judge_only");
 - every sentence is sent (flagged or not) and the model must also handle follow-ups of removed content.
Usage: build_pilot_inputs_v2.py <turn_plan_in.jsonl> <visibility.jsonl> <outdir>"""
import collections, gzip, json, os, re, sys
sys.path.insert(0, "."); from source_rules import classify
NUM = re.compile(r"(?<![\w.])-?\d+\.\d{2,}|(?<![\w.])-?\d{4,}(?![\w.])")
def numre(n): return re.compile(r"(?<![\d.])" + re.escape(n) + r"(?!\d)")
def main():
    src, visf, outdir = sys.argv[1:4]; os.makedirs(outdir, exist_ok=True)
    corpus = set(json.load(open("pilot/hidden_corpus_numbers.json")))
    nfr = json.load(open("pilot/numbers_from_removed_all.json"))
    vis = {}
    for l in open(visf):
        r = json.loads(l); vis[f'{r["episode"]}#{r["turn"]}'] = set(r["numbers"]) - set(r["found_in_visible"])
    plan, reqs, cat = [], [], collections.Counter()
    for l in (gzip.open(src, "rt", encoding="utf-8") if src.endswith(".gz") else open(src)):
        p = json.loads(l); reasons = set(p["reasons"])
        tc = json.dumps(p["tool_calls"], ensure_ascii=False)
        for s in p["sentences"]:
            f, r = classify(s["text"]); s["flagged"], s["rule"] = f, r
        n_src = sum(s["flagged"] for s in p["sentences"])
        # candidates: numbers the replay builder already found in removed memory content; in turns that cite a source, also
        # any number of the turn that occurs in the official memory corpus (in uncited turns the corpus matches the agent's
        # own estimates by chance too often: 865 distinct 3-decimal values in a small coordinate range)
        cand = set(nfr.get(p["key"]) or [])
        if n_src: cand |= {n for n in NUM.findall((p["think"] or "") + "\n" + (p["text"] or "")) if f"{float(n):.3f}" in corpus}
        hid = {n for n in cand & vis.get(p["key"], set()) if not numre(n).search(tc)}
        for s in p["sentences"]:
            s["hidden_numbers"] = sorted(n for n in hid if numre(n).search(s["text"]))
            if s["hidden_numbers"] and not s["flagged"]: s["flagged"], s["rule"] = True, "hidden_number"
        tool_src = [t["name"] for t in p["tool_calls"] if classify(json.dumps(t["arguments"], ensure_ascii=False))[0]]
        if "reads_memory" in reasons: c = "history_reads_memory"
        elif tool_src: c = "history_tool_args_cite_source"
        elif n_src >= 3: c = "history_gate_3plus_flagged"
        elif not any(s["flagged"] for s in p["sentences"]): c = "target_clean" if p["disposition"] == "target" else "judge_only"
        else: c = "llm"
        p.update(category_v2=c, hidden_numbers=sorted(hid), n_src_flagged=n_src, tool_args_cite_source=tool_src)
        cat[(p["disposition"], c)] += 1; plan.append(p)
        if c == "llm":
            reqs.append({"key": p["key"], "suite": p["suite"], "task": p["task"],
                         "sentences": [dict({"id": s["id"], "text": s["text"], "flagged": s["flagged"]},
                                            **({"hidden_numbers": s["hidden_numbers"]} if s["hidden_numbers"] else {})) for s in p["sentences"]],
                         "tool_calls_this_turn": [{"name": t["name"], "arguments": json.dumps(t["arguments"], ensure_ascii=False)[:300]} for t in p["tool_calls"]]})
    with (gzip.open(f"{outdir}/turn_plan.jsonl.gz", "wt", encoding="utf-8") if src.endswith(".gz") else open(f"{outdir}/turn_plan.jsonl", "w", encoding="utf-8")) as fo:
        for p in plan: fo.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(f"{outdir}/requests.jsonl", "w", encoding="utf-8") as fo:
        for q in reqs: fo.write(json.dumps(q, ensure_ascii=False) + "\n")
    for k, v in sorted(cat.items()): print(k, v)
    print("requests:", len(reqs))
if __name__ == "__main__":
    main()
