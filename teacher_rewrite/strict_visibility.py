#!/usr/bin/env python3
"""Like leftover_visibility.py, but stricter: "visible" = student system prompt, first user message, tool results and
tool-call arguments before this turn; the assistant's own earlier free text is NOT counted (it may itself repeat memory numbers).
A number also counts as visible when a visible number rounds to it at its own precision (e.g. -0.0241 -> -0.024).
Usage: strict_visibility.py <shard.tar> <numbers.jsonl> <out.jsonl>"""
import json, re, sys, tarfile
from number_provenance import norm_forms
NUMV = re.compile(r"-?\d+\.\d+")
def rounds_to(x, vals):
    if "." not in x: return False
    v, tol = float(x), 0.5 * 10 ** -len(x.split(".")[1]) + 1e-9
    return any(abs(u - v) <= tol for u in vals)
def visible_text(d, s):
    parts = [d.get("system_prompt") or "", d.get("user_message") or ""]
    for m in s["messages"]:
        c = m.get("content")
        if m.get("role") != "assistant":
            if isinstance(c, str): parts.append(c)
            elif isinstance(c, list): parts += [p.get("text", "") for p in c if isinstance(p, dict)]
        for tc in m.get("tool_calls") or []: parts.append(json.dumps(tc.get("function", tc)))
    return "\n".join(parts)
def main():
    src, lst, dst = sys.argv[1:4]; want = {}
    for l in open(lst):
        r = json.loads(l); want[(r["episode"], r["turn"])] = r["numbers"]
    eps = {e for e, _ in want}; n = 0
    with tarfile.open(src, mode="r|") as tf, open(dst, "w") as out:
        for mem in tf:
            if not (mem.isfile() and mem.name.endswith(".json") and "/images/" not in mem.name): continue
            if mem.name.rsplit("/", 1)[-1][:-5] not in eps: continue
            d = json.load(tf.extractfile(mem))
            for s in d["samples"]:
                nums = want.get((d["meta"]["episode"], s["turn_index"]))
                if nums is None: continue
                vis = visible_text(d, s)
                vals = [float(v) for v in NUMV.findall(vis)]
                found = [x for x in nums if any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?!\d)", vis) for f in norm_forms(x)) or rounds_to(x, vals)]
                out.write(json.dumps({"episode": d["meta"]["episode"], "turn": s["turn_index"], "numbers": nums,
                                      "found_in_visible": found, "all_found": len(found) == len(nums)}) + "\n"); n += 1
    print("DONE", src, "turns", n, flush=True)
if __name__ == "__main__":
    main()
