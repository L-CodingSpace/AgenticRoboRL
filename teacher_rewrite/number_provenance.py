#!/usr/bin/env python3
"""For history-only turns masked ONLY because of numbers that also appear in removed memory content
(reasons == {numbers_from_removed_content}), check whether each such number also appears in what the student
DOES see before that turn: the student system prompt, the first user message, or any earlier tool result / assistant text
in the replayed history. Writes a per-turn JSONL and prints counts. Usage: number_provenance.py <shard.tar> <out.jsonl>"""
import json, re, sys, tarfile
def norm_forms(n):
    s = str(n); forms = {s}
    try:
        v = float(s); forms |= {f"{v:.3f}", f"{v:.2f}", f"{v:.1f}", f"{v:g}"}
        if abs(v) < 1: forms |= {f.replace("0.", ".", 1) for f in list(forms) if f.startswith(("0.", "-0."))}
    except ValueError: pass
    return {f for f in forms if f}
def visible_text(d, s):
    parts = [d.get("system_prompt") or "", d.get("user_message") or ""]
    for m in s["messages"]:
        c = m.get("content")
        if isinstance(c, str): parts.append(c)
        elif isinstance(c, list): parts += [p.get("text", "") for p in c if isinstance(p, dict)]
        for tc in m.get("tool_calls") or []: parts.append(json.dumps(tc.get("function", tc)))
    return "\n".join(parts)
def main():
    src, dst = sys.argv[1], sys.argv[2]; n = 0
    with tarfile.open(src, mode="r|") as tf, open(dst, "w") as out:
        for mem in tf:
            if not (mem.isfile() and mem.name.endswith(".json") and "/images/" not in mem.name): continue
            d = json.load(tf.extractfile(mem))
            for s in d["samples"]:
                if s["disposition"] != "keep_history_no_target" or set(s.get("reasons") or []) != {"numbers_from_removed_content"}: continue
                vis = visible_text(d, s); nums = s.get("numbers_from_removed") or []
                found = [x for x in nums if any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?!\d)", vis) for f in norm_forms(x))]
                out.write(json.dumps({"episode": d["meta"]["episode"], "turn": s["turn_index"], "numbers": nums, "found_in_visible": found,
                                      "all_found": len(found) == len(nums)}) + "\n"); n += 1
    print("DONE", src, "turns", n, flush=True)
if __name__ == "__main__":
    main()
