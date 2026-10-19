#!/usr/bin/env python3
"""Policy update for one RL round: PPO-style loss on per-turn samples built by rl/build_round.py.

Model / sharding / numerics are those of sft/rpent/train_fsdp.py (FSDP2 hybrid: shard in node, replicate
across nodes; fp32 master weights, bf16 compute, cast_forward_inputs=False, gradient checkpointing, LLM +
merger trained, ViT frozen, dummy-image pre-hook for text-only samples, one cross-node all-reduce per step).

A sample = one planner request of one episode: input = vLLM's prompt_token_ids + vLLM's output token ids
(exactly what was sampled; the prompt is re-rendered with the HF processor like the SFT loader and must be
identical to vLLM's ids -- gate A, checked on every sample), images through the same processor call.

Round = (1) pre-pass: log pi_old(token) under the round's starting weights for every sample (no grad);
gate-B statistics against the vLLM logprobs; (2) ceil(N / global_batch) optimizer steps, each sample used once
(--tail_batch merge: floor(N / global_batch) steps, the partial last batch joins the last step; see rl/plan.py).

Loss per token t of sample s (episode e, k_e sampled turns, T_s output tokens, advantage A_e):
    r_t   = exp(logp_t - logp_old_t)
    l_t   = max(-A r_t, -A clip(r_t, 1-eps_low, 1+eps_high)); for A < 0 also min(l_t, -A * dual_clip)
            (dual-clip PPO, Ye et al., AAAI 2020)
    w_t   = min(exp(logp_old_t - logp_vllm_t), tis_cap)            (truncated IS: trainer vs sampler)
    loss_step = sum_s (1/k_e) * mean_t (w_t l_t) / sum_s (1/k_e)    (episode-mean, turn-mean, token-mean)
No KL term, no entropy bonus. Constant learning rate.

--ce_chunk C (0 = off): long outputs (e.g. CaP-X, whose model calls return the whole reasoning, up to 32,768 tokens) do
not fit on a GPU with all logits at once (T_out x 248k vocabulary in fp32, plus their gradient). With C > 0 the output
head returns log p(token) itself, C positions at a time, each chunk checkpointed (its logits are recomputed in the
backward pass instead of kept); the entropy statistic is computed chunk by chunk without gradient. Same numbers, less
memory. The LIBERO-PRO runs used 0; the CaP-X runs 2048 (141 GB GPUs) or 512 (80 GB GPUs).

Mid-round checkpoints (--mid_every N): after every N-th optimizer step, <mid_dir>/step_NNN holds the sharded
model + optimizer (dcp), every rank's pi_old table (old_rank<r>.pt: the pre-pass logprobs of ITS samples under the
round's starting weights -- the PPO clip reference must stay the round start, never the checkpoint's weights), RNG
states and meta.json (step, world, gas, N, gate_b, history); COMPLETE last; mirrored to --mid_mirror; older ones deleted.
--mid_resume <dir> (same world size, same samples) continues after that step: no pre-pass, same data order.
"""
import argparse, gzip, json, math, os, shutil, subprocess, sys, time
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, get_model_state_dict, StateDictOptions
from transformers import AutoProcessor, AutoModelForImageTextToText

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sft"))
from chat_dataset import prepare_messages  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_loss import ppo_token_loss  # noqa: E402
from plan import balanced_plan, stride_plan, busiest_over_mean, n_steps_for, step_rows, step_gas  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--samples", required=True, help="samples.jsonl of rl/build_round.py")
p.add_argument("--model_path", required=True, help="HF dir: processor/config; weights when not resuming")
p.add_argument("--resume", default=None, help="checkpoint dir of the previous round (dcp/ with model + optimizer)")
p.add_argument("--output_dir", required=True, help="writes <output_dir>/dcp, model.safetensors, round_log.json")
p.add_argument("--ref_files", required=True, help="dir whose config.json / generation_config.json go into the output")
p.add_argument("--mirror_dir", default=None, help="copy of the output dir on shared storage (each node copies its own dcp shards)")
p.add_argument("--lr", type=float, default=1e-6)
p.add_argument("--global_batch", type=int, default=64)
p.add_argument("--gpus_per_node", type=int, default=4)
p.add_argument("--eps_low", type=float, default=0.2)
p.add_argument("--eps_high", type=float, default=0.28)
p.add_argument("--dual_clip", type=float, default=3.0)
p.add_argument("--tis_cap", type=float, default=2.0)
p.add_argument("--max_steps", type=int, default=-1, help="test only: stop after this many optimizer steps")
p.add_argument("--no_save", action="store_true")
p.add_argument("--dataloader_workers", type=int, default=4)
p.add_argument("--mid_every", type=int, default=0, help="mid-round checkpoint every N optimizer steps (0 = off)")
p.add_argument("--mid_dir", default=None, help="local dir of mid-round checkpoints (default <output_dir>/mid)")
p.add_argument("--mid_mirror", default=None, help="copy of --mid_dir on shared storage (survives a lost node)")
p.add_argument("--mid_resume", default=None, help="a mid-round checkpoint dir (<...>/step_NNN with COMPLETE) to continue from")
p.add_argument("--stop_after_step", type=int, default=0, help="test only: hard exit (no final save) after this step")
p.add_argument("--plan", default="stride", choices=("balanced", "stride"),
               help="which rank computes which sample of a step (see 'this rank's samples' below); stride = the original")
p.add_argument("--tail_batch", default="own", choices=("own", "merge"),
               help="a last batch with fewer than global_batch samples: own = a step of its own (the original); "
                    "merge = it joins the step before it (rl/plan.py)")
p.add_argument("--ce_chunk", type=int, default=0,
               help="log-probabilities in the output head, this many positions at a time (0 = all logits at once); see above")
args = p.parse_args()

dist.init_process_group("nccl", timeout=timedelta(minutes=60))
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local)
G = args.gpus_per_node
assert world % G == 0 and args.global_batch % world == 0, (world, G, args.global_batch)
nodes, gas = world // G, args.global_batch // world
# Timeout of the NCCL collectives (default 600 s = torch's own default). Cross-node collectives over TCP can hang and
# never recover; a smaller value (e.g. RL_NCCL_TIMEOUT_S=300) fails earlier so a controller can restart from the last
# mid-round checkpoint sooner.
import torch.distributed.distributed_c10d as _c10d
NCCL_T = int(os.environ.get("RL_NCCL_TIMEOUT_S", "600"))
_c10d.default_pg_nccl_timeout = timedelta(seconds=NCCL_T)
_slow = {"flag": False}
mesh = init_device_mesh("cuda", (nodes, G), mesh_dim_names=("replicate", "shard"))
cpu_pg = dist.new_group(backend="gloo", timeout=timedelta(minutes=60))
log0 = lambda *a: print(*a, flush=True) if rank == 0 else None

processor = AutoProcessor.from_pretrained(args.model_path)
rows = [json.loads(l) for l in open(args.samples)]
N = len(rows)
# Loss denominator of a step: the all-reduced sum of the step's sample weights (uniform 1/k weights per episode).
NORMALIZATION = "sum"
n_steps = n_steps_for(N, args.global_batch, args.tail_batch)
if args.max_steps > 0:
    n_steps = min(n_steps, args.max_steps)
log0(f"[rl] samples={N} world={world} nodes={nodes} gas={gas} steps={n_steps} normalization={NORMALIZATION}"
     + (f" tail_batch={args.tail_batch}" if args.tail_batch != "own" else ""))
if args.tail_batch == "merge" and args.plan != "stride":
    raise ValueError("--tail_batch merge: only --plan stride is implemented")
# entries of this rank in step s (rl/plan.py): gas, except the merged last step of --tail_batch merge
GS = [step_gas(s, N, args.global_batch, world, args.tail_batch) for s in range(n_steps)]


class RLDataset(torch.utils.data.Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        r = rows[i]
        ep = r["episode_dir"]
        rec = json.loads(gzip.decompress(open(os.path.join(ep, "rl_turns", r["turn_file"]), "rb").read()))
        msgs, images = prepare_messages({"messages": rec["messages"]}, root=ep)
        tools = rec["request"].get("tools")
        prefix = processor.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)
        enc = processor(text=[prefix], images=images if images else None, return_tensors="pt")
        p_ids = enc["input_ids"][0]
        ok = p_ids.tolist() == rec["prompt_token_ids"]  # gate A on every sample
        o_ids = torch.tensor(rec["output_token_ids"], dtype=p_ids.dtype)
        out = {"input_ids": torch.cat([p_ids, o_ids])[None], "attention_mask": torch.ones(1, p_ids.numel() + o_ids.numel(), dtype=torch.long),
               "n_prompt": p_ids.numel(), "lp_vllm": torch.tensor(rec["output_logprobs"], dtype=torch.float32),
               "advantage": float(r["advantage"]), "weight": float(r["weight"]) if ok else 0.0, "gate_a_ok": ok,
               "key": f"{r['episode']}:{r['turn']}", "idx": i}
        if images:
            out["pixel_values"] = enc["pixel_values"]
            out["image_grid_thw"] = enc["image_grid_thw"]
            if "mm_token_type_ids" in enc:
                mm = enc["mm_token_type_ids"][0]
                out["mm_token_type_ids"] = torch.cat([mm, torch.zeros_like(o_ids)])[None]
        return out


# ---- model: as sft/rpent/train_fsdp.py ----
model = AutoModelForImageTextToText.from_pretrained(args.model_path, dtype=torch.float32, attn_implementation="sdpa")
model.config.use_cache = False
for n, prm in model.named_parameters():
    prm.requires_grad_(not (".visual." in n and ".merger" not in n))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.to("cuda")
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
for mod in model.modules():
    if type(mod).__name__ in ("Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"):
        fully_shard(mod, mesh=mesh, mp_policy=mp)
fully_shard(model, mesh=mesh, mp_policy=mp)
trainable = [(n, prm) for n, prm in model.named_parameters() if prm.requires_grad]
opt = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

from PIL import Image as _Image
_d = processor(text=["<|vision_start|><|image_pad|><|vision_end|>"], images=[_Image.new("RGB", (64, 64), (0, 0, 0))], return_tensors="pt")


def _flat(x):
    if torch.is_tensor(x):
        return [x]
    if isinstance(x, (list, tuple)):
        return [t for y in x for t in _flat(y)]
    if isinstance(x, dict):
        return [t for y in x.values() for t in _flat(y)]
    return []


def _text_only_pre_hook(module, a, kw):
    if kw.get("pixel_values") is None and kw.get("inputs_embeds") is None and kw.get("input_ids") is not None:
        emb = module.get_input_embeddings()(kw["input_ids"])
        feats = module.get_image_features(_d["pixel_values"].to(emb.device, torch.bfloat16), _d["image_grid_thw"].to(emb.device))
        z = sum(t.float().sum() for t in _flat(feats)) * 0.0
        kw["inputs_embeds"] = emb + z.to(emb.dtype)
        kw["input_ids"] = None
        T = emb.shape[1]
        kw["position_ids"] = torch.arange(T, device=emb.device).view(1, 1, -1).expand(3, 1, -1)
        return a, kw
    return None


model.model.register_forward_pre_hook(_text_only_pre_hook, with_kwargs=True)

# ---- --ce_chunk: log p(token) computed inside the output head, chunk by chunk (see the module docstring) ----
_ce = {"y": None, "lp": None, "ent": 0.0}
if args.ce_chunk > 0:
    from torch.utils.checkpoint import checkpoint as _ckpt
    _head = model.lm_head
    _head_forward = _head.forward

    def _lp_chunk(h, w, y):
        return -F.cross_entropy(F.linear(h, w).float(), y, reduction="none")

    def _head_lp(h):
        y = _ce["y"]
        if y is None:
            return _head_forward(h)
        assert h.shape[0] == 1 and h.shape[1] == y.numel() and _head.bias is None, (h.shape, y.shape)
        lps, ent = [], 0.0
        for c in range(0, y.numel(), args.ce_chunk):
            hc, yc = h[0, c:c + args.ce_chunk], y[c:c + args.ce_chunk]
            lps.append(_ckpt(_lp_chunk, hc, _head.weight, yc, use_reentrant=False) if torch.is_grad_enabled() else _lp_chunk(hc, _head.weight, yc))
            with torch.no_grad():
                lg = F.linear(hc.detach(), _head.weight.detach()).float()
                ent += float((torch.logsumexp(lg, -1) - (lg.softmax(-1) * lg).sum(-1)).sum())
                del lg
        _ce["lp"], _ce["ent"] = torch.cat(lps), ent
        return h.new_zeros((1, 1, 1))  # nothing downstream uses the logits

    _head.forward = _head_lp

start_version = None
MID = None
if args.mid_resume:
    assert os.path.exists(os.path.join(args.mid_resume, "COMPLETE")), f"{args.mid_resume} has no COMPLETE"
    MID = json.load(open(os.path.join(args.mid_resume, "meta.json")))
    assert (MID["world"], MID["gas"], MID["N"], MID["global_batch"]) == (world, gas, N, args.global_batch), (MID, world, gas, N)
    # old_rank<r>.pt holds pi_old of the samples THIS rank had under the checkpoint's plan: the plan must be the same
    assert MID.get("plan", "stride") == args.plan, f"mid checkpoint plan {MID.get('plan', 'stride')} != --plan {args.plan}"
    assert MID.get("tail_batch", "own") == args.tail_batch, f"mid checkpoint tail_batch {MID.get('tail_batch', 'own')} != --tail_batch {args.tail_batch}"
    sd_m, sd_o = get_state_dict(model, opt)
    state = {"model": sd_m, "optim": sd_o}
    dcp.load(state, checkpoint_id=os.path.join(args.mid_resume, "dcp"), process_group=cpu_pg)
    set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
    log0(f"[rl] mid-round resume from {args.mid_resume}: after step {MID['step']} of {n_steps}")
elif args.resume:
    sd_m, sd_o = get_state_dict(model, opt)
    state = {"model": sd_m, "optim": sd_o}
    dcp.load(state, checkpoint_id=os.path.join(args.resume, "dcp"), process_group=cpu_pg)
    set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
    log0(f"[rl] resumed weights + optimizer from {args.resume}")
else:
    log0(f"[rl] fresh optimizer, weights from {args.model_path}")


# ---- this rank's samples: step s (0-based) uses rows [s*B, (s+1)*B) ----
# --plan stride: rank r takes s*B + r + k*world. --plan balanced: the step's rows, longest first (prompt + output
# tokens), each to the least-loaded rank that has fewer than gas rows -- the step waits for its busiest rank, and
# on round 2's samples (44k tokens median) stride gave busiest/mean 1.52, balanced 1.04. Same rows per step, so
# the step's loss/gradient is the same sum; only which rank computes which row changes.
# Every rank runs exactly gas forward(/backward) passes per step (FSDP collectives stay in step): a missing
# sample of the partial last step is replaced by a weight-0 repeat of this rank's first sample.
# --tail_batch merge: the last step also takes the rows after its own B (GS[s] passes on every rank instead of gas).
def _balanced_plan():
    box = [None]
    if rank == 0:   # rank 0 reads the turn files, every rank gets the same token counts -> the same plan
        box[0] = [len(rec["prompt_token_ids"]) + len(rec["output_token_ids"]) for rec in
                  (json.loads(gzip.decompress(open(os.path.join(r["episode_dir"], "rl_turns", r["turn_file"]), "rb").read()))
                   for r in rows[:min(N, n_steps * args.global_batch)])]
    dist.broadcast_object_list(box, src=0, group=cpu_pg)
    tok = box[0]
    out = balanced_plan(tok, N, n_steps, args.global_batch, world)
    log0(f"[rl] plan balanced: busiest/mean tokens per step {busiest_over_mean(out, tok, N):.3f} "
         f"(stride would be {busiest_over_mean(stride_plan(N, n_steps, args.global_batch, world), tok, N):.3f})")
    return out


BAL = _balanced_plan() if args.plan == "balanced" else None


def my_indices(s):   # entry N (or >= N) = padding
    if BAL is not None:
        return BAL[s][rank]
    return [s * args.global_batch + rank + k * world for k in range(GS[s])]


ds = RLDataset()
first_mine = rank if rank < N else 0
plan = [(i, True) if i < N else (first_mine, False) for s in range(n_steps) for i in my_indices(s)]
loader_kw = dict(batch_size=None, num_workers=args.dataloader_workers, pin_memory=True,
                 prefetch_factor=2 if args.dataloader_workers else None, persistent_workers=False)


def make_loader():
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, [i for i, _ in plan]), **loader_kw)


def to_cuda(b):
    return {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def _entropy_sum(logits):
    """sum over positions of the token entropy (no gradient), 2048 positions at a time"""
    e = 0.0
    with torch.no_grad():
        for c in range(0, logits.shape[0], 2048):
            lg = logits[c:c + 2048]
            e += float((torch.logsumexp(lg, -1) - (lg.softmax(-1) * lg).sum(-1)).sum())
    return e


def forward_logp(b):
    """log p(output token) for every output token of the sample (float32, shape [T_out]), and either the logits
    (--ce_chunk 0; the caller computes the entropy statistic from them after the backward pass) or None and the sum
    of the token entropies (--ce_chunk > 0, computed chunk by chunk in the output head)."""
    ids = b["input_ids"]
    P = b["n_prompt"]
    T = ids.shape[1]
    keep = torch.arange(P - 1, T - 1, device=ids.device)  # logits at t predict token t+1
    kw = {k: b[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids") if k in b}
    y = ids[0, P:]
    if args.ce_chunk > 0:
        _ce["y"] = y
        model(**kw, logits_to_keep=keep, use_cache=False)
        lp, ent = _ce["lp"], _ce["ent"]
        _ce["y"] = _ce["lp"] = None
        return lp, None, ent
    out = model(**kw, logits_to_keep=keep, use_cache=False)
    logits = out.logits[0].float()
    return -F.cross_entropy(logits, y, reduction="none"), logits, None


# ---- (1) pre-pass: pi_old under the round's starting weights; gate-B statistics ----
t0 = time.time()
model.eval()
old = {}
if MID is not None:
    mine = torch.load(os.path.join(args.mid_resume, f"old_rank{rank}.pt"))
    old = mine["old"]; torch.set_rng_state(mine["rng_cpu"]); torch.cuda.set_rng_state(mine["rng_cuda"])
    gate_b = MID["gate_b"] if rank == 0 else None
gateb_local = {"n_tok": 0, "sum_abs": 0.0, "max_abs": 0.0, "n_out_08_125": 0, "abs_list": [], "gate_a_bad": 0, "n_samples": 0}
with torch.no_grad():
    for (i, real), b in (zip(plan, make_loader()) if MID is None else []):
        b = to_cuda(b)
        lp, _, _ = forward_logp(b)
        lp = lp.cpu()
        if not real:
            continue
        old[b["idx"]] = lp
        d = (lp - b["lp_vllm"].cpu())
        a = d.abs()
        gateb_local["n_tok"] += a.numel(); gateb_local["sum_abs"] += float(a.sum()); gateb_local["max_abs"] = max(gateb_local["max_abs"], float(a.max()))
        gateb_local["n_out_08_125"] += int(((d.exp() < 0.8) | (d.exp() > 1.25)).sum())
        gateb_local["abs_list"].extend(a.tolist()[:4096])
        gateb_local["gate_a_bad"] += int(not b["gate_a_ok"]); gateb_local["n_samples"] += 1
# pad ranks so every rank runs the same number of forward/backward passes (FSDP collectives stay in step)
allb = [None] * world
dist.all_gather_object(allb, gateb_local, group=cpu_pg)
if MID is not None:
    pass
elif rank == 0:
    absl = sorted(x for g in allb for x in g["abs_list"])
    ntok = sum(g["n_tok"] for g in allb)
    gate_b = {"samples": sum(g["n_samples"] for g in allb), "gate_a_bad": sum(g["gate_a_bad"] for g in allb), "tokens": ntok,
              "mean_abs_dlogp": sum(g["sum_abs"] for g in allb) / max(ntok, 1),
              "p99_abs_dlogp": absl[int(0.99 * (len(absl) - 1))] if absl else None, "max_abs_dlogp": max(g["max_abs"] for g in allb),
              "frac_ratio_outside_0.8_1.25": sum(g["n_out_08_125"] for g in allb) / max(ntok, 1), "prepass_s": round(time.time() - t0, 1)}
    print("[gateB] " + json.dumps(gate_b), flush=True)
else:
    gate_b = None
model.train()


_mid_threads = []


def _mid_copy(step, d, h, base, t_local):
    """background (local rank 0 of every node): copy this node's files of checkpoint `step` to the mirror dir h, then
    drop NODE_<n>.ok there; rank 0 additionally waits for every node's marker, writes COMPLETE and deletes the older
    mirrored checkpoints. File copies only -- no collective calls from this thread. A failure leaves no COMPLETE (the
    previous complete checkpoint is kept, resume uses that)."""
    t0 = time.time()
    try:
        os.makedirs(os.path.join(h, "dcp"), exist_ok=True)
        for f in os.listdir(os.path.join(d, "dcp")):   # this node's dcp shards (+ .metadata on rank 0's node)
            subprocess.run(["cp", os.path.join(d, "dcp", f), os.path.join(h, "dcp", f + ".tmp")], check=True)
            os.rename(os.path.join(h, "dcp", f + ".tmp"), os.path.join(h, "dcp", f))
        for f in os.listdir(d):                          # meta.json (node 0), old_rank*.pt of this node's ranks
            if f != "dcp":
                subprocess.run(["cp", os.path.join(d, f), os.path.join(h, f + ".tmp")], check=True)
                os.rename(os.path.join(h, f + ".tmp"), os.path.join(h, f))
        open(os.path.join(h, f"NODE_{rank // G}.ok"), "w").write(f"{time.time() - t0:.1f}s\n")
        if rank == 0:
            deadline = time.time() + 1800
            while not all(os.path.exists(os.path.join(h, f"NODE_{n}.ok")) for n in range(nodes)):
                if time.time() > deadline:
                    raise RuntimeError(f"not every node copied its part of step {step} within 30 min")
                time.sleep(5)
            open(os.path.join(h, "COMPLETE"), "w").write(f"step {step}\n")
            for x in os.listdir(args.mid_mirror):   # keep only the newest complete mirrored checkpoint
                if x.startswith("step_") and x != f"step_{step:03d}":
                    shutil.rmtree(os.path.join(args.mid_mirror, x), ignore_errors=True)
            print(f"[rl] mid checkpoint step {step}: local {t_local:.1f}s (training waited), mirror copy {time.time() - t0:.1f}s in background, COMPLETE", flush=True)
        for x in os.listdir(base):                 # this node's older local copies
            if x.startswith("step_") and x != f"step_{step:03d}":
                shutil.rmtree(os.path.join(base, x), ignore_errors=True)
    except Exception as e:  # noqa: BLE001
        print(f"[rl] mid checkpoint step {step}: mirror copy FAILED on rank {rank}: {e!r}", flush=True)


def save_mid(step):
    """mid-round checkpoint after optimizer step `step` (1-based); see the module docstring. The local write is
    synchronous (every rank's shards must be on disk before training moves on); the mirror copy runs in background."""
    t0 = time.time()
    for th in _mid_threads:   # the previous checkpoint's copy must be finished before its local dir is reused/removed
        th.join()
    _mid_threads.clear()
    base = args.mid_dir or os.path.join(args.output_dir, "mid")
    d = os.path.join(base, f"step_{step:03d}")
    h = os.path.join(args.mid_mirror, f"step_{step:03d}") if args.mid_mirror else None
    if local == 0:
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    if rank == 0 and h:   # a stale incomplete dir of this step (from a crashed earlier attempt) must not be mixed in
        shutil.rmtree(h, ignore_errors=True); os.makedirs(os.path.join(h, "dcp"))
    dist.barrier()
    sd_m, sd_o = get_state_dict(model, opt)
    dcp.save({"model": sd_m, "optim": sd_o}, checkpoint_id=os.path.join(d, "dcp"), process_group=cpu_pg)
    torch.save({"old": old, "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, os.path.join(d, f"old_rank{rank}.pt"))
    if rank == 0:
        json.dump({"step": step, "n_steps": n_steps, "world": world, "gas": gas, "N": N, "global_batch": args.global_batch, "plan": args.plan,
                   "tail_batch": args.tail_batch,
                   "gate_b": gate_b, "history": history, "samples": os.path.abspath(args.samples)}, open(os.path.join(d, "meta.json"), "w"))
    dist.barrier()   # every file of this checkpoint is on local disk
    t_local = time.time() - t0
    if rank == 0:
        open(os.path.join(d, "COMPLETE"), "w").write(f"step {step}\n")
    if h and local == 0:
        import threading
        th = threading.Thread(target=_mid_copy, args=(step, d, h, base, t_local), daemon=False)
        th.start(); _mid_threads.append(th)
    elif not h and rank == 0:
        print(f"[rl] mid checkpoint step {step}: local {t_local:.1f}s (no mirror)", flush=True)


# ---- (2) updates ----
history = list(MID["history"]) if MID is not None else []
S0 = MID["step"] if MID is not None else 0
if S0:
    _rest = plan[sum(GS[:S0]):]
    it = zip(_rest, torch.utils.data.DataLoader(torch.utils.data.Subset(ds, [i for i, _ in _rest]), **loader_kw))
else:
    it = zip(plan, make_loader())
for s in range(S0, n_steps):
    ts = time.time()
    micro, weights, reals = [], [], []
    for _ in range(GS[s]):
        (i, real), b = next(it)
        micro.append(b); weights.append(b["weight"] if real else 0.0); reals.append(real)
    D = torch.tensor(float(sum(weights)), device="cuda"); dist.all_reduce(D)
    D_actual = float(D)
    D = D_actual
    D = max(float(D), 1e-8)
    stats = {"clipfrac": 0.0, "approx_kl": 0.0, "ratio_mean": 0.0, "tis_mean": 0.0}
    loss_sum = torch.zeros((), device="cuda", dtype=torch.float64)
    ent_sum = torch.zeros((), device="cuda", dtype=torch.float64)
    ntok = torch.zeros((), device="cuda", dtype=torch.float64)
    for k, b in enumerate(micro):
        model.set_requires_all_reduce(k == GS[s] - 1)
        b = to_cuda(b)
        lp, logits, ent_s = forward_logp(b)
        if args.tail_batch != "own" and reals[k] and b["idx"] not in old:   # plan and pre-pass table disagree: never train with ratio 1
            print(f"[rl] rank {rank}: no pi_old for sample {b['idx']} at step {s + 1} -> abort", flush=True)
            os._exit(14)
        lp_old = old[b["idx"]].cuda() if b["idx"] in old else lp.detach()
        lt, st = ppo_token_loss(lp, lp_old, b["lp_vllm"], b["advantage"], args.eps_low, args.eps_high, args.dual_clip, args.tis_cap)
        sample_loss = lt.mean() * weights[k]
        (sample_loss / D * world).backward()
        with torch.no_grad():
            ent_sum += (_entropy_sum(logits) if logits is not None else ent_s) * (weights[k] > 0)
            ntok += lp.numel() * (weights[k] > 0)
        loss_sum += sample_loss.detach().double()
        for kk in stats:
            stats[kk] += st[kk] / GS[s]
        del logits
    t = torch.tensor([float(loss_sum), float(ent_sum), float(ntok)] + [stats[k] for k in stats], device="cuda", dtype=torch.float64)
    dist.all_reduce(t)
    gn = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
    gn = float(gn.full_tensor() if hasattr(gn, "full_tensor") else gn)
    if not (math.isfinite(float(t[0])) and math.isfinite(gn)):
        # a timed-out collective can let the step finish with a non-finite loss: never apply or save that
        print(f"[rl] rank {rank}: non-finite loss/grad at step {s + 1} -> abort before the update", flush=True)
        os._exit(13)
    opt.step(); opt.zero_grad(set_to_none=True)
    if time.time() - ts > NCCL_T - 5:
        _slow["flag"] = True  # as long as a collective timeout: this step may have used incomplete gradients
    rec = {"step": s + 1, "loss": round(float(t[0]) / D, 6), "entropy": round(float(t[1]) / max(float(t[2]), 1), 4), "grad_norm": round(gn, 4),
           "clipfrac": round(float(t[3]) / world, 4), "approx_kl": round(float(t[4]) / world, 6), "ratio_mean": round(float(t[5]) / world, 4),
           "tis_mean": round(float(t[6]) / world, 4), "weight_sum": round(D_actual, 4), "denominator": round(D, 4), "step_s": round(time.time() - ts, 1)}
    if args.tail_batch != "own":
        _lo, _hi = step_rows(s, N, args.global_batch, args.tail_batch); rec["rows"] = _hi - _lo
    history.append(rec)
    log0("[rl] " + json.dumps(rec))
    if args.mid_every and (s + 1) % args.mid_every == 0 and s + 1 < n_steps:
        if _slow["flag"]:
            # after a timed-out collective the watchdog kills the process 1-2 min later: wait that long so a state built on
            # such a step never becomes a checkpoint (a genuinely slow step only costs this pause)
            print(f"[rl] rank {rank}: slow step since the last save; waiting {NCCL_T + 60}s before the mid checkpoint", flush=True)
            time.sleep(NCCL_T + 60); _slow["flag"] = False
        save_mid(s + 1)
    if args.stop_after_step and s + 1 == args.stop_after_step:
        log0(f"[rl] stop_after_step {s + 1}: hard exit (test)")
        os._exit(0)


def save():
    d = args.output_dir
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    sd_m, sd_o = get_state_dict(model, opt)
    dcp.save({"model": sd_m, "optim": sd_o}, checkpoint_id=os.path.join(d, "dcp"), process_group=cpu_pg)
    full = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank == 0:
        from safetensors.torch import save_file
        save_file({k: v.to(torch.bfloat16).contiguous() for k, v in full.items()}, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
        processor.save_pretrained(d)
        for f in ("config.json", "generation_config.json"):
            shutil.copy(os.path.join(args.ref_files, f), d)
        json.dump({"samples": N, "steps": n_steps, "gate_b": gate_b, "history": history, "args": vars(args), "normalization": NORMALIZATION},
                  open(os.path.join(d, "round_log.json"), "w"), indent=1)
    del full
    dist.barrier()
    if args.mirror_dir:
        if rank == 0:
            shutil.rmtree(args.mirror_dir, ignore_errors=True)
            os.makedirs(os.path.join(args.mirror_dir, "dcp"))
        dist.barrier()
        if local == 0:
            for f in os.listdir(os.path.join(d, "dcp")):
                subprocess.run(["cp", os.path.join(d, "dcp", f), os.path.join(args.mirror_dir, "dcp", f)], check=True)
        if rank == 0:
            for f in os.listdir(d):
                if f != "dcp":
                    subprocess.run(["cp", "-r", os.path.join(d, f), os.path.join(args.mirror_dir, f)], check=True)
        dist.barrier()
        if rank == 0:
            open(os.path.join(args.mirror_dir, "COMPLETE"), "w").write(f"steps {n_steps} dcp_files {len(os.listdir(os.path.join(args.mirror_dir, 'dcp')))}\n")
    if rank == 0:
        open(os.path.join(d, "COMPLETE"), "w").write(f"steps {n_steps}\n")
        print(f"[rl] saved {d} ({time.time()-t0:.0f}s)", flush=True)
    dist.barrier()


for th in _mid_threads:   # background mirror copies of the last mid checkpoint
    th.join()
if not args.no_save:
    if _slow["flag"]:
        print(f"[rl] rank {rank}: slow step since the last save; waiting {NCCL_T + 60}s before the final save", flush=True)
        time.sleep(NCCL_T + 60)
    save()
    if rank == 0 and args.mid_mirror:   # the round is complete: its mid checkpoints are not needed any more
        shutil.rmtree(args.mid_mirror, ignore_errors=True)
    if local == 0:
        shutil.rmtree(args.mid_dir or os.path.join(args.output_dir, "mid"), ignore_errors=True)
elif rank == 0:
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump({"samples": N, "steps": n_steps, "gate_b": gate_b, "history": history, "args": vars(args), "normalization": NORMALIZATION},
              open(os.path.join(args.output_dir, "round_log.json"), "w"), indent=1)
dist.barrier()
dist.destroy_process_group()
