"""Gate A: does training see exactly what vLLM saw?

For every request recorded by rl/proxy.py under the given episode dirs:
  1. prompt: re-render the recorded messages + tools with the HF processor exactly as the
     SFT loader does (sft/chat_dataset.prepare_messages + apply_chat_template +
     processor(text, images)) and compare the ids with vLLM's prompt_token_ids, token by token;
  2. images: the number of image placeholder tokens in vLLM's prompt equals what the
     processor's image_grid_thw implies, and every referenced image file exists with its sha1;
  3. output: one logprob per generated token, the last token is an end token when
     finish_reason is "stop", and the sampling vLLM was sent is the RL sampling;
  4. (informative) re-rendering the parsed answer (think/text/tool calls) with the template
     vs the generated ids -- RL trains on the generated ids, not on a re-rendering.

  python rl/check_gate_a.py --model <checkpoint dir> <episode dir> [<episode dir> ...]
Prints one line per request and a summary; exit code 1 if any prompt or output check fails.
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sft"))
from chat_dataset import prepare_messages  # noqa: E402

from transformers import AutoProcessor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_sampling import RL_SAMPLING  # noqa: E402


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def check_images(ep_dir, messages):
    bad = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for it in c:
                if it.get("type") == "image_ref":
                    p = ep_dir / it["path"]
                    if not p.exists() or hashlib.sha1(p.read_bytes()).hexdigest() != it["sha1"]:
                        bad.append(it["path"])
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("episodes", nargs="+")
    ap.add_argument("--max-turns", type=int, default=0, help="0 = all")
    a = ap.parse_args()
    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    image_token_id = tok.convert_tokens_to_ids("<|image_pad|>")
    end_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    merge = getattr(proc.image_processor, "merge_size", 2)

    n = n_prompt_ok = n_out_ok = n_rerender_same = 0
    fails = []
    for ep in a.episodes:
        ep_dir = Path(ep)
        turns = sorted((ep_dir / "rl_turns").glob("turn_*.json.gz"))
        if a.max_turns:
            turns = turns[: a.max_turns]
        for tp in turns:
            rec = json.loads(gzip.decompress(tp.read_bytes()))
            n += 1
            tools = rec["request"].get("tools")
            msgs, images = prepare_messages({"messages": rec["messages"]}, root=str(ep_dir))
            prefix = proc.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)
            enc = proc(text=[prefix], images=images if images else None, return_tensors="pt")
            hf_ids = enc["input_ids"][0].tolist()
            v_ids = rec["prompt_token_ids"] or []
            d = first_diff(hf_ids, v_ids)
            n_img_tok = sum(1 for t in v_ids if t == image_token_id)
            want_img_tok = int((enc["image_grid_thw"].prod(-1) // (merge * merge)).sum()) if images else 0
            bad_img = check_images(ep_dir, rec["messages"])
            prompt_ok = d is None and n_img_tok == want_img_tok and not bad_img

            out_ids = rec["output_token_ids"] or []
            lps = rec["output_logprobs"] or []
            sampling_ok = all(rec["sent_sampling"].get(k) == v for k, v in RL_SAMPLING.items())
            end_ok = rec["finish_reason"] != "stop" or (out_ids and out_ids[-1] in end_ids)
            out_ok = len(out_ids) > 0 and len(lps) == len(out_ids) and all(x is not None for x in lps) and end_ok and sampling_ok

            msg = rec.get("message") or {}
            resp = {"think": msg.get("reasoning_content") or msg.get("reasoning") or "", "text": msg.get("content") or "",
                    "tool_calls": [{"name": t["function"]["name"], "arguments": json.loads(t["function"]["arguments"] or "{}")}
                                   for t in (msg.get("tool_calls") or [])]}
            tcs = [{"type": "function", "function": {"name": t["name"], "arguments": t["arguments"]}} for t in resp["tool_calls"]]
            am = {"role": "assistant", "content": resp["text"], "reasoning_content": resp["think"]}
            if tcs:
                am["tool_calls"] = tcs
            full = proc.apply_chat_template(msgs + [am], tools=tools, add_generation_prompt=False, tokenize=False)
            target = full[len(prefix):]
            target = target[:-1] if target.endswith("<|im_end|>\n") else target
            rer = tok(target, add_special_tokens=False)["input_ids"]
            same = rer == out_ids
            n_prompt_ok += prompt_ok
            n_out_ok += out_ok
            n_rerender_same += same
            print(f"{ep_dir.name}/{tp.name}: prompt {'OK' if prompt_ok else 'FAIL'} (hf {len(hf_ids)} vllm {len(v_ids)} "
                  f"first_diff {d} img_tok {n_img_tok}/{want_img_tok} imgs {len(images)} bad_img {len(bad_img)}) | "
                  f"output {'OK' if out_ok else 'FAIL'} ({len(out_ids)} tok, {len(lps)} lp, finish {rec['finish_reason']}, "
                  f"sampling {'RL' if sampling_ok else rec['sent_sampling']}) | rerender {'same' if same else f'differs ({len(rer)} vs {len(out_ids)})'}")
            if not (prompt_ok and out_ok):
                ctx = ""
                if d is not None:
                    ctx = f" hf[{d}:{d+8}]={tok.decode(hf_ids[d:d+8])!r} vllm[{d}:{d+8}]={tok.decode(v_ids[d:d+8])!r}"
                fails.append(f"{tp}{ctx}")
    print(f"SUMMARY requests={n} prompt_ok={n_prompt_ok} output_ok={n_out_ok} rerender_same={n_rerender_same}")
    for f in fails[:20]:
        print("FAIL", f)
    sys.exit(0 if n and n_prompt_ok == n and n_out_ok == n else 1)


if __name__ == "__main__":
    main()
