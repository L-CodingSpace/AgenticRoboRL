#!/usr/bin/env python3
"""Full rewrite, step 0: sentence spans for ALL turns (same splitting as the pilot's build_pilot_inputs.py), plus the
candidate-number list for the visibility check on the replay shards.
Outputs: full_v2/turn_plan_base.jsonl.gz, full_v2/candidate_numbers.jsonl"""
import glob, gzip, json, os, re
SPLIT = re.compile(r"(?<=[.!?。])[ \t]+|\n+")
NUM = re.compile(r"(?<![\w.])-?\d+\.\d{2,}|(?<![\w.])-?\d{4,}(?![\w.])")
def spans(text):
    out, start = [], 0
    for m in SPLIT.finditer(text or ""):
        if m.start() > start: out.append((start, m.start()))
        start = m.end()
    if start < len(text or ""): out.append((start, len(text)))
    return [(a, b) for a, b in out if (text[a:b]).strip()]
def main():
    os.makedirs("full_v2", exist_ok=True)
    corpus = set(json.load(open("pilot/hidden_corpus_numbers.json"))); n = 0
    with gzip.open("full_v2/turn_plan_base.jsonl.gz", "wt", encoding="utf-8") as fo, open("full_v2/candidate_numbers.jsonl", "w") as fc:
        for f in sorted(glob.glob("turns/turns_*.jsonl.gz")):
            for l in gzip.open(f, "rt", encoding="utf-8"):
                r = json.loads(l); key = f'{r["episode"]}#{r["turn_index"]}'
                sents = [{"id": f"{tag}{k}", "field": fld, "span": [a, b], "text": r[fld][a:b]}
                         for fld, tag in (("think", "t"), ("text", "x")) for k, (a, b) in enumerate(spans(r[fld]))]
                fo.write(json.dumps({"key": key, "episode": r["episode"], "turn": r["turn_index"], "suite": r["suite"], "task": r["task"],
                                     "disposition": r["disposition"], "reasons": sorted(r.get("reasons") or []), "think": r["think"],
                                     "text": r["text"], "tool_calls": r["tool_calls"], "sentences": sents}, ensure_ascii=False) + "\n")
                nums = set(r.get("numbers_from_removed") or [])
                nums |= {x for x in NUM.findall((r["think"] or "") + "\n" + (r["text"] or "")) if f"{float(x):.3f}" in corpus}
                if nums: fc.write(json.dumps({"episode": r["episode"], "turn": r["turn_index"], "numbers": sorted(nums)}) + "\n")
                n += 1
    print("turns", n)
if __name__ == "__main__":
    main()
