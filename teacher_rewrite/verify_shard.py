#!/usr/bin/env python3
"""Compare a cleaned shard with its original: same member names/order; every non-episode member byte-identical;
per episode same samples/turns; changed turns match changes.jsonl; every training target in the new shard re-scanned with
source rules v1.3 (think + text + tool-call args); history assistant messages never still hold the old text of a changed turn.
Usage: verify_shard.py <orig.tar> <new.tar> <changes.jsonl>"""
import collections, hashlib, json, re, sys, tarfile
sys.path.insert(0, "."); from source_rules import classify
SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")
def members(path):
    with tarfile.open(path, mode="r|") as tf:
        for m in tf:
            if m.isfile(): yield m.name, tf.extractfile(m).read()
def main():
    o, n, chf = sys.argv[1:4]
    ch = {json.loads(l)["key"]: json.loads(l) for l in open(chf)}
    st = collections.Counter(); bad = []
    for (na, a), (nb, b) in zip(members(o), members(n)):
        if na != nb: bad.append(("name", na, nb)); break
        st["members"] += 1
        if not (na.endswith(".json") and "/images/" not in na):
            st["same_bytes" if hashlib.sha1(a).digest() == hashlib.sha1(b).digest() else "DIFF_bytes"] += 1; continue
        da, db = json.loads(a), json.loads(b); ep = da["meta"]["episode"]
        if [s["turn_index"] for s in da["samples"]] != [s["turn_index"] for s in db["samples"]]: bad.append(("turns", ep)); continue
        old_changed = {}
        for sa, sb in zip(da["samples"], db["samples"]):
            k = f"{ep}#{sa['turn_index']}"; c = ch.get(k)
            want = ("target" if c["decision"] == "target" else "keep_history_no_target") if c else sa["disposition"]
            if sb["disposition"] != want: bad.append(("disposition", k))
            if c and c.get("text") is not None:
                if sb["response"]["text"] != c["text"] or sb["response"]["think"] != c["think"]: bad.append(("text", k))
                if sa["response"]["text"] != c["text"]: old_changed[sa["turn_index"]] = sa["response"]["text"] or ""
            if not c and sa["response"] != sb["response"]: bad.append(("unexpected_change", k))
            st[f"{sa['disposition']}->{sb['disposition']}"] += 1
            if sb["disposition"] == "target":
                txt = (sb["response"]["think"] or "") + "\n" + (sb["response"]["text"] or "")
                hits = [s for s in SPLIT.split(txt) if s.strip() and classify(s)[0]]
                args = [t for t in sb["response"]["tool_calls"] if classify(json.dumps(t, ensure_ascii=False))[0]]
                if hits or args: st["target_rule_hit"] += 1; st["target_rule_hit_" + ("changed" if c else "unchanged")] += 1
        for sb in db["samples"]:
            for m in sb["messages"]:
                if m["role"] == "assistant" and old_changed and (m.get("content") or "") in old_changed.values() and (m.get("content") or ""):
                    st["history_still_old_text"] += 1
        st["episodes"] += 1
    print(json.dumps(dict(st))); print("problems", len(bad), bad[:5])
if __name__ == "__main__":
    main()
