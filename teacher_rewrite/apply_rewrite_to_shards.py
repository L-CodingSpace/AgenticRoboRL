#!/usr/bin/env python3
"""Full rewrite, last step: write a cleaned copy of one replay shard. The original shard is only read.
changes.jsonl: one row per turn whose disposition or text changes: {"key": "<episode>#<turn>", "decision": "target"|"history",
"think": str|null, "text": str|null, "why": str}. think/text null = keep the original.
Per episode json:
 - sample response.think/text replaced, disposition set, reason "rewrite_v2:<why>" appended (history) / reasons kept (target);
 - in every later sample of the episode, the history assistant message of a changed turn gets the new text. History
   assistant messages are matched to earlier samples in order by exact original text (usually message i == sample i, but a
   turn can be missing from later history, e.g. libero_10_task__t02__s043__a00 turn 63);
 - meta.dispositions / meta.n_targets recounted, meta.rewrite added. Images and every other member are copied byte-identical.
Usage: apply_rewrite_to_shards.py <in_shard.tar> <changes.jsonl> <out_shard.tar>"""
import collections, io, json, sys, tarfile, time
def main():
    src, chf, dst = sys.argv[1:4]; ch = {}
    for l in open(chf):
        r = json.loads(l); ch[r["key"]] = r
    stats = collections.Counter(); t0 = time.time()
    with tarfile.open(src, mode="r|") as tin, tarfile.open(dst + ".partial", mode="w|", format=tarfile.PAX_FORMAT) as tout:
        for mem in tin:
            if not mem.isfile():
                tout.addfile(mem); continue
            data = tin.extractfile(mem).read()
            if mem.name.endswith(".json") and "/images/" not in mem.name:
                d = json.loads(data); ep = d["meta"]["episode"]; S = d["samples"]; changed_text = {}
                orig = [s["response"]["text"] or "" for s in S]
                for i, s in enumerate(S):
                    c = ch.get(f'{ep}#{s["turn_index"]}')
                    if not c: continue
                    before = s["disposition"]
                    if c.get("think") is not None: s["response"]["think"] = c["think"]
                    if c.get("text") is not None and c["text"] != s["response"]["text"]:
                        s["response"]["text"] = c["text"]; changed_text[i] = c["text"]
                    s["disposition"] = "target" if c["decision"] == "target" else "keep_history_no_target"
                    if s["disposition"] != "target": s["reasons"] = sorted(set(s["reasons"]) | {"rewrite_v2:" + c.get("why", "")})
                    stats[f"{before}->{s['disposition']}"] += 1
                for j, s in enumerate(S):
                    if not changed_text: break
                    i = 0
                    for a in (m for m in s["messages"] if m["role"] == "assistant"):
                        k = i
                        while k < j and orig[k] != (a.get("content") or ""): k += 1
                        if k >= j: stats["history_msg_unmatched"] += 1; continue
                        if k in changed_text: a["content"] = changed_text[k]; stats["history_msgs_updated"] += 1
                        i = k + 1
                disp = collections.Counter(d["meta"].get("dispositions") or {})
                for k in ("target", "keep_history_no_target"): disp[k] = sum(1 for s in S if s["disposition"] == k)
                d["meta"]["dispositions"] = dict(disp); d["meta"]["n_targets"] = disp["target"]
                d["meta"]["rewrite"] = "rewrite_v2 (teacher reasoning cleaned of memory/seed-0 references; see rewrite_log)"
                data = json.dumps(d, ensure_ascii=False).encode("utf-8"); stats["episodes"] += 1
            info = tarfile.TarInfo(mem.name); info.size = len(data); info.mtime = mem.mtime; info.mode = mem.mode
            tout.addfile(info, io.BytesIO(data)); stats["members"] += 1
    import os; os.replace(dst + ".partial", dst)
    print("DONE", src, dict(stats), f"{time.time() - t0:.0f}s", flush=True)
if __name__ == "__main__":
    main()
