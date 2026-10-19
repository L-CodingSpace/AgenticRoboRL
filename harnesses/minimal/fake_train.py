#!/usr/bin/env python3
"""A stand-in for the trainer (no GPU): checks the samples point at real proxy records and writes a "weights" dir.
The real command is rl/train_rl.py; see harnesses/rpent_libero/harness.yaml."""
import argparse
import gzip
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--samples", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = [json.loads(l) for l in Path(a.samples).read_text().splitlines()]
tokens = 0
for r in rows:
    rec = json.loads(gzip.decompress((Path(r["episode_dir"]) / "rl_turns" / r["turn_file"]).read_bytes()))
    assert len(rec["output_token_ids"]) == len(rec["output_logprobs"]) > 0 and rec["prompt_token_ids"], r
    tokens += len(rec["output_token_ids"])
out = Path(a.out); out.mkdir(parents=True)
(out / "info.json").write_text(json.dumps({"from": a.weights, "samples": len(rows), "output_tokens": tokens,
                                             "advantage_sum": sum(r["advantage"] * r["weight"] for r in rows)}))
