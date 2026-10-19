#!/usr/bin/env python3
"""Rewrite v2, step 3: apply 27B edits to the exact spans, verify, decide each turn (before the 27B judge pass).
A rewritten turn goes on to the judge only if ALL hold:
 - sentences holding a hidden number that the model did not remove are deleted automatically;
 - every flagged sentence got an entry; every id / "through" id exists
 - drop_source new_text: no source words (rules v1.3, no exceptions), no forbidden replacement words that were not in the
   original sentence, no result words (worked/fired/...) when the original cited a source, no new numbers, <= 3 new words
 - after applying: no sentence still source-flagged (except ones the model marked "keep"), no hidden number left,
   deleted characters <= 50% of think+text
Usage: check_edits_v2.py <outdir>   (reads outdir/turn_plan.jsonl, outdir/responses.jsonl; writes outdir/checked.jsonl)"""
import collections, json, re, sys
sys.path.insert(0, "."); from source_rules import classify
NUM = re.compile(r"-?\d+(?:\.\d+)?")
WORD = re.compile(r"[A-Za-z_]+")
REPL = re.compile(r"\b(guidance|solution|proven|documented|winning|winner|initial reference|baseline|known[- ]good)\b", re.I)
FACT = re.compile(r"\b(worked|works|fired|succeed\w*|success\w*|reached|proven|documented|winning|confirmed|verified|held|descended)\b", re.I)
def main():
    d = sys.argv[1]
    import gzip, os
    tp = f"{d}/turn_plan.jsonl.gz" if os.path.exists(f"{d}/turn_plan.jsonl.gz") else f"{d}/turn_plan.jsonl"
    plan = (json.loads(l) for l in (gzip.open(tp, "rt", encoding="utf-8") if tp.endswith(".gz") else open(tp)))
    resp = {}
    for l in open(f"{d}/responses.jsonl"):
        r = json.loads(l)
        if r.get("ok") or r["key"] not in resp: resp[r["key"]] = r
    summary, fails = collections.Counter(), collections.Counter(); out = open(f"{d}/checked.jsonl", "w", encoding="utf-8")
    for p in plan:
        key, c = p["key"], p["category_v2"]
        rec = {"key": key, "category": c, "suite": p["suite"], "task": p["task"], "disposition_before": p["disposition"]}
        if c == "judge_only":
            rec.update(before={"think": p["think"], "text": p["text"]}, after={"think": p["think"] or "", "text": p["text"] or ""},
                       tool_calls=p["tool_calls"], edits=[], problems=[], decision="to_judge")
            summary[f'{p["disposition"]} -> to_judge (judge_only)'] += 1; out.write(json.dumps(rec, ensure_ascii=False) + "\n"); continue
        if c != "llm":
            rec["decision"] = "target" if c == "target_clean" else "history"
            summary[f'{p["disposition"]} -> {rec["decision"]} ({c})'] += 1; out.write(json.dumps(rec, ensure_ascii=False) + "\n"); continue
        r = resp.get(key); problems = []
        if not r or not r.get("ok"):
            rec.update(decision="history", problems=["no_valid_response"]); summary[f'{p["disposition"]} -> history (no response)'] += 1
            fails["no_valid_response"] += 1; out.write(json.dumps(rec, ensure_ascii=False) + "\n"); continue
        order = {fld: [s["id"] for s in p["sentences"] if s["field"] == fld] for fld in ("think", "text")}
        sents = {s["id"]: s for s in p["sentences"]}; edits = {}
        for e in r["edits"]:
            if e["id"] not in sents: problems.append(f"unknown_id:{e['id']}"); continue
            edits[e["id"]] = dict(e)
        # expand header-block deletions ("through") into per-sentence deletes
        for sid, e in list(edits.items()):
            th = (e.get("through") or "").strip()
            if e["action"] != "delete" or not th or th == sid: continue
            ids = order[sents[sid]["field"]]
            if th not in sents or sents[th]["field"] != sents[sid]["field"] or ids.index(th) < ids.index(sid):
                problems.append(f"bad_through:{sid}->{th}"); continue
            for x in ids[ids.index(sid) + 1: ids.index(th) + 1]:
                edits.setdefault(x, {"id": x, "action": "delete", "through": "", "new_text": "", "why": f"block of {sid}"})
        # sentences flagged only for a hidden number that the model left unhandled (or kept the number in): delete them;
        # the judge then checks that the turn still reads coherently
        for sid, s in sents.items():
            e = edits.get(sid); hn = s.get("hidden_numbers") or []
            if hn and (e is None or e["action"] == "keep" or (e["action"] == "drop_source" and any(
                    re.search(r"(?<![\d.])" + re.escape(n) + r"(?!\d)", e["new_text"]) for n in hn))):
                edits[sid] = {"id": sid, "action": "delete", "through": "", "new_text": "", "why": "auto: hidden number"}
        for sid, s in sents.items():
            if s["flagged"] and sid not in edits: problems.append(f"flagged_without_entry:{sid}")
        for sid, e in edits.items():
            orig = sents[sid]["text"]
            if e["action"] == "drop_source":
                nt = e["new_text"]
                if classify(nt)[0]: problems.append(f"source_left:{sid}")
                if REPL.search(nt) and not REPL.search(orig): problems.append(f"replacement_word:{sid}")
                if sents[sid]["rule"] not in (None, "hidden_number") and FACT.search(nt): problems.append(f"result_as_fact:{sid}")
                if set(NUM.findall(nt)) - set(NUM.findall(orig)): problems.append(f"new_number:{sid}")
                if len(set(w.lower() for w in WORD.findall(nt)) - set(w.lower() for w in WORD.findall(orig))) > 3: problems.append(f"new_words:{sid}")
        after = {"think": p["think"] or "", "text": p["text"] or ""}; deleted = 0
        for fld in ("think", "text"):
            txt = after[fld]
            for s in sorted([s for s in p["sentences"] if s["field"] == fld], key=lambda s: -s["span"][0]):
                e = edits.get(s["id"])
                if not e or e["action"] == "keep": continue
                a, b = s["span"]; new = e["new_text"] if e["action"] == "drop_source" else ""
                deleted += max(0, (b - a) - len(new)); txt = txt[:a] + new + txt[b:]
            after[fld] = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", txt)).strip()
        total = len(p["think"] or "") + len(p["text"] or "")
        kept = {sents[k]["text"].strip() for k, e in edits.items() if e["action"] == "keep"}
        still = [s for s in re.split(r"(?<=[.!?。])\s+|\n+", after["think"] + "\n" + after["text"]) if s.strip() and s.strip() not in kept and classify(s)[0]]
        if still: problems.append(f"still_flagged:{len(still)}")
        both = after["think"] + "\n" + after["text"]
        left = [n for n in p["hidden_numbers"] if re.search(r"(?<![\d.])" + re.escape(n) + r"(?!\d)", both)]
        if left: problems.append(f"hidden_number_left:{len(left)}")
        ratio = deleted / max(1, total)
        if ratio > 0.5: problems.append(f"deleted_ratio:{ratio:.2f}")
        if not both.strip(): problems.append("empty_after")
        rec.update(before={"think": p["think"], "text": p["text"]}, after=after, tool_calls=p["tool_calls"], edits=list(edits.values()),
                   deleted_ratio=round(ratio, 3), n_flagged=sum(s["flagged"] for s in p["sentences"]), hidden_numbers=p["hidden_numbers"],
                   problems=problems, decision="to_judge" if not problems else "history")
        summary[f'{p["disposition"]} -> {rec["decision"]} (llm)'] += 1
        for pr in {x.split(":")[0] for x in problems}: fails[pr] += 1
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    for k, v in sorted(summary.items()): print(k, v)
    print("problem types (turn counts):", dict(fails))
if __name__ == "__main__":
    main()
