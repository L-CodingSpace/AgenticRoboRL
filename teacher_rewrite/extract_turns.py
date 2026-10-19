#!/usr/bin/env python3
"""Stream one replay shard (tar) and write every replayed turn's text as compact JSONL (gzip):
episode, turn_index, disposition, reasons, mem_keywords, mem_ngram_hits, numbers_from_removed, think, text, tool_calls.
Images and message histories are skipped. Usage: extract_turns.py <shard.tar> <out.jsonl.gz>"""
import gzip, json, sys, tarfile, time
def main():
    src, dst = sys.argv[1], sys.argv[2]; t0 = time.time(); n_ep = n_turn = 0
    with tarfile.open(src, mode="r|") as tf, gzip.open(dst + ".part", "wt", encoding="utf-8") as out:
        for m in tf:
            if not (m.isfile() and m.name.endswith(".json") and "/images/" not in m.name): continue
            d = json.load(tf.extractfile(m)); ep = d["meta"]["episode"]; n_ep += 1
            for s in d["samples"]:
                r = s["response"]
                out.write(json.dumps({"episode": ep, "suite": d["meta"]["suite"], "task": d["meta"]["task"], "turn_index": s["turn_index"],
                                      "disposition": s["disposition"], "reasons": s.get("reasons"), "mem_keywords": s.get("mem_keywords"),
                                      "mem_ngram_hits": s.get("mem_ngram_hits"), "numbers_from_removed": s.get("numbers_from_removed"),
                                      "think": r.get("think", ""), "text": r.get("text", ""),
                                      "tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in r.get("tool_calls", [])]},
                                     ensure_ascii=False) + "\n"); n_turn += 1
    import os; os.replace(dst + ".part", dst)
    print(f"DONE {src} episodes={n_ep} turns={n_turn} {time.time()-t0:.0f}s", flush=True)
if __name__ == "__main__":
    main()
