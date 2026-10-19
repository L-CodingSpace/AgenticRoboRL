#!/usr/bin/env python3
"""The two number lists the rewrite reads (steps 0 and 2):
  <out>/hidden_corpus_numbers.json    every decimal number (>= 2 decimals, or an integer of >= 4 digits) that occurs in the
                                      memory files the teacher read, as "%.3f", sorted
  <out>/numbers_from_removed_all.json per teacher turn ("<episode>#<turn>"), the numbers of removed memory content that the
                                      turn uses (the replay's numbers_from_removed field), for turns that have any
Usage: make_number_lists.py <memory dir, e.g. <memory>/resources/libero> <dir with turns_*.jsonl.gz (extract_turns.py)> <out dir>"""
import collections, glob, gzip, json, os, re, sys
NUM = re.compile(r"(?<![\w.])-?\d+\.\d+|(?<![\w.])-?\d{4,}(?![\w.])")
def main():
    mem, turns, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
    keys = collections.Counter()
    for f in glob.glob(os.path.join(mem, "**", "*"), recursive=True):
        try: s = open(f, encoding="utf-8").read()
        except Exception: continue
        for n in NUM.findall(s):
            v = float(n)
            if "." in n and len(n.split(".")[1]) < 2: continue
            keys[f"{v:.3f}"] += 1
    json.dump(sorted(keys), open(os.path.join(out, "hidden_corpus_numbers.json"), "w"))
    nfr = {}
    for f in glob.glob(os.path.join(turns, "*.jsonl.gz")):
        for l in gzip.open(f):
            t = json.loads(l); nfr[f"{t['episode']}#{t['turn_index']}"] = t["numbers_from_removed"]
    json.dump({k: v for k, v in nfr.items() if v}, open(os.path.join(out, "numbers_from_removed_all.json"), "w"))
    print(len(keys), "corpus numbers;", sum(1 for v in nfr.values() if v), "turns with numbers from removed content")
if __name__ == "__main__":
    main()
