#!/usr/bin/env python3
"""SFT trainer for CaP-X teacher data = sft/rpent/train_fsdp.py with: evaluation checkpoints take config.json from the
base model and write generation_config.json from the loaded model; log-probabilities computed chunk by chunk in the
output head (--ce_chunk, long reasoning targets); a shorter NCCL timeout with a pause before saving after a slow step.
Description of sft/rpent/train_fsdp.py:
SFT trainer (multi-node, FSDP2 hybrid sharding) -- the model is sharded over the 4 GPUs of a node and
replicated across nodes; inside an optimizer step every micro-batch reduce-scatters its gradient inside the
node only, and the nodes all-reduce ONCE, after the last micro-batch (set_requires_all_reduce).

Same training definition as train.py (DeepSpeed ZeRO-3, used for steps 1..N on one node):
  micro-batch 1 per GPU, global batch 64 (accumulation = 64 / world), fixed pre-shuffled index read in order
  (step s = index rows [(s-1)*64, s*64)), AdamW (torch) lr 1e-5 betas 0.9/0.999 eps 1e-8 wd 0, 10 warmup steps,
  cosine to 10% (transformers get_cosine_with_min_lr_schedule_with_warmup over steps/epoch), clip 1.0,
  fp32 master weights + optimizer, bf16 compute, gradient checkpointing, LLM + vision merger trained, ViT frozen,
  loss = CE summed over all supervised tokens of the global batch / their number (logits only at supervised
  positions), text-only micro-batches run the vision tower on a dummy image (same module order on every rank).
Start points: --init_universal <DeepSpeed universal checkpoint dir> --init_step N (fp32 weights AND Adam
moments of step N, exact continuation), or --resume <checkpoint dir of this script>, or the base weights.
Checkpoints: <output_dir>/checkpoint-<s>/ with dcp/ (sharded model+optimizer, reshardable), trainer_state.json,
and at keep steps model.safetensors (bf16) + config/processor files for evaluation; mirrored to --mirror_dir.
"""
import argparse, glob, hashlib, json, math, os, shutil, subprocess, time
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (get_state_dict, set_state_dict, get_model_state_dict,
                                                     set_model_state_dict, set_optimizer_state_dict, StateDictOptions)
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..")]
import capx_dataset as D  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--replay_dir", required=True, help="data dir written by sft/capx/build.py")
p.add_argument("--model_path", required=True)
p.add_argument("--output_dir", required=True)
p.add_argument("--index_file", required=True)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--warmup_steps", type=int, default=10)
p.add_argument("--global_batch", type=int, default=64)
p.add_argument("--gpus_per_node", type=int, default=4)
p.add_argument("--max_steps", type=int, default=-1, help="stop after this global step (the schedule still spans the epoch)")
p.add_argument("--save_steps", type=int, default=10)
p.add_argument("--keep_fraction", type=float, default=0.1)
p.add_argument("--keep_full", type=int, default=2)
p.add_argument("--mirror_dir", default=None)
p.add_argument("--init_universal", default=None)
p.add_argument("--init_step", type=int, default=0)
p.add_argument("--resume", default=None)
p.add_argument("--grad_check_out", default=None)
p.add_argument("--no_save", action="store_true")
p.add_argument("--dataloader_workers", type=int, default=4)
p.add_argument("--drop_partial_last", action="store_true", help="end the epoch one step early when the last step has fewer samples than there are ranks (see FINAL below)")
p.add_argument("--test_export", action="store_true", help="test only: write the evaluation weights at every save")
p.add_argument("--bf16_buffers", type=int, default=0, help="1 = cast floating buffers to bf16 (NOT what the DeepSpeed run did: from_pretrained keeps inv_freq in fp32; kept only for diagnosis)")
p.add_argument("--ref_files", default=None, help="checkpoint of the DeepSpeed run whose config.json / generation_config.json are copied into evaluation checkpoints (identical files, only the weights differ)")
args = p.parse_args()

from datetime import timedelta
# 60 min: at a start from a checkpoint, global rank 0 reads ~108 GB from shared storage while the other ranks wait in the
# first broadcast (the 10-min default timed out once when the storage was busy)
dist.init_process_group("nccl", timeout=timedelta(minutes=60))
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local)
G = args.gpus_per_node
assert world % G == 0 and args.global_batch % world == 0, (world, G, args.global_batch)
nodes, gas = world // G, args.global_batch // world
# the mesh process groups get torch's default NCCL timeout (10 min). Cross-node collectives over TCP hang now and then
# (never recover): fail after 4 min instead so the watcher can restart from the last checkpoint sooner.
import torch.distributed.distributed_c10d as _c10d
NCCL_T = int(os.environ.get("SFT_NCCL_TIMEOUT_S", "150"))
_c10d.default_pg_nccl_timeout = timedelta(seconds=NCCL_T)
mesh = init_device_mesh("cuda", (nodes, G), mesh_dim_names=("replicate", "shard"))
# checkpoint coordination (gather_object / broadcast of the save/load plans) over gloo: NCCL point-to-point
# gathers across nodes failed here with "PXN should not use host buffers" (NCCL internal error, step-190 save)
cpu_pg = dist.new_group(backend="gloo", timeout=timedelta(minutes=60))
log0 = lambda *a: print(*a, flush=True) if rank == 0 else None

processor = AutoProcessor.from_pretrained(args.model_path)
index = [(r["episode"], int(r["pos"]), int(r["turn"])) for r in map(json.loads, open(args.index_file))]
ds = D.SFTDataset(args.replay_dir, processor, index=index)
steps_per_epoch = math.ceil(len(index) / args.global_batch)
keep_steps = sorted(set(min(steps_per_epoch, round(k * args.keep_fraction * steps_per_epoch)) for k in range(1, int(round(1 / args.keep_fraction)) + 1)))
last_step = steps_per_epoch if args.max_steps < 0 else min(args.max_steps, steps_per_epoch)
# FINAL = the step that ends the run. A last step with fewer samples than ranks (2,370 = 37*64 + 2 on 8 ranks) is not
# supported by this trainer: the original code crashed there (IndexError), a stand-in sample on the empty ranks hung in
# a collective. With --drop_partial_last those few samples are left out and the
# run ends at the last full step; the learning-rate schedule still spans steps_per_epoch.
FINAL = steps_per_epoch
if args.drop_partial_last and 0 < len(index) % args.global_batch < world:
    FINAL = steps_per_epoch - 1
    keep_steps = sorted(set(min(s, FINAL) for s in keep_steps))
    last_step = min(last_step, FINAL)
    log0(f"[fsdp] --drop_partial_last: {len(index) % args.global_batch} samples of the last step are not used; the run ends at step {FINAL}")
log0(f"[fsdp] samples={len(index)} world={world} nodes={nodes} gas={gas} steps/epoch={steps_per_epoch} keep={keep_steps}")

# ---- model: fp32 master weights, bf16 compute; LLM + merger trained, ViT frozen ----
model = AutoModelForImageTextToText.from_pretrained(args.model_path, dtype=torch.float32, attn_implementation="sdpa")
model.config.use_cache = False
gen_cfg = model.generation_config  # written into evaluation checkpoints (the base snapshot has no generation_config.json)
if args.bf16_buffers:
    nb = 0
    for mod in model.modules():
        for bn, buf in list(mod.named_buffers(recurse=False)):
            if buf is not None and buf.is_floating_point() and buf.dtype != torch.bfloat16:
                mod._buffers[bn] = buf.to(torch.bfloat16); nb += 1
    print(f"[fsdp] {nb} floating buffers cast to bf16", flush=True) if int(os.environ.get("RANK", "0")) == 0 else None
for n, prm in model.named_parameters():
    prm.requires_grad_(not (".visual." in n and ".merger" not in n))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.to("cuda")
# cast_forward_inputs=False: FSDP's default casts every floating input of a wrapped module to bf16, which
# pushed the fp32 vision rotary embeddings into bf16 and changed the gradients of the merger / early layers by
# 20-100% on a single sample; without the cast the one-sample loss and gradients match the plain bf16 model
# (the numerics of a DeepSpeed ZeRO-3 run) to its own run-to-run noise.
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
for mod in model.modules():
    if type(mod).__name__ in ("Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"):
        fully_shard(mod, mesh=mesh, mp_policy=mp)
fully_shard(model, mesh=mesh, mp_policy=mp)
trainable = [(n, prm) for n, prm in model.named_parameters() if prm.requires_grad]
log0(f"[fsdp] trainable params {sum(p.numel() for _, p in trainable)/1e9:.3f}B")
opt = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
sched = get_cosine_with_min_lr_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps, num_training_steps=steps_per_epoch, min_lr_rate=0.1)

# ---- text-only micro-batches: run the vision tower on a dummy image (same all-gather order on every rank) ----
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

# ---- CaP-X teacher data: targets are long (the teacher's whole reasoning, up to ~32k tokens per sample). The logits of all
# supervised positions at once (n x 248k vocabulary, plus their fp32 copy and gradient) filled a 141 GB GPU. Same loss, less memory: the output head computes the summed cross-entropy itself, CE_CHUNK positions at a
# time, each chunk checkpointed (its logits are recomputed in the backward pass instead of being kept).
from torch.utils.checkpoint import checkpoint as _ckpt
CE_CHUNK = int(os.environ.get("SFT_CE_CHUNK", "2048"))
_ce = {"y": None, "loss": None}
_head = model.lm_head
_head_forward = _head.forward


def _ce_chunk(h, w, y):
    return F.cross_entropy(F.linear(h, w).float(), y, reduction="sum")


def _head_ce(h):
    y = _ce["y"]
    if y is None:
        return _head_forward(h)
    assert h.shape[0] == 1 and h.shape[1] == y.numel() and _head.bias is None, (h.shape, y.shape)
    tot = h.new_zeros((), dtype=torch.float32)
    for s in range(0, y.numel(), CE_CHUNK):
        tot = tot + _ckpt(_ce_chunk, h[0, s:s + CE_CHUNK], _head.weight, y[s:s + CE_CHUNK], use_reentrant=False)
    _ce["loss"] = tot
    return h.new_zeros((1, 1, 1))  # nothing downstream uses the logits


_head.forward = _head_ce

# ---- start point ----
start_step, history = 0, []
if args.resume:
    sd_m, sd_o = get_state_dict(model, opt)
    state = {"model": sd_m, "optim": sd_o}
    dcp.load(state, checkpoint_id=os.path.join(args.resume, "dcp"), process_group=cpu_pg)
    set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
    ts = json.load(open(os.path.join(args.resume, "trainer_state.json")))
    start_step, history = int(ts["global_step"]), ts.get("log_history", [])
    log0(f"[fsdp] resumed from {args.resume} at step {start_step}")
    dist.barrier()
    if (local == 0 and args.mirror_dir and os.path.exists(os.path.join(args.mirror_dir, f"checkpoint-{start_step}", "COMPLETE"))
            and os.path.realpath(args.resume).startswith(os.path.realpath(args.output_dir) + os.sep)):
        shutil.rmtree(args.resume, ignore_errors=True)  # a LOCAL copy of the mirrored checkpoint is not needed after loading
elif args.init_universal:
    # DeepSpeed universal checkpoint: zero/<param name>/{fp32,exp_avg,exp_avg_sq}.pt, flat fp32 tensors
    # only global rank 0 reads the files (fp32 weights + two Adam moments of the trained parameters, ~108 GB)
    # and broadcasts them; frozen ViT weights stay the base weights already loaded (strict=False)
    Z = os.path.join(args.init_universal, "zero")
    full_m, full_o = {}, {}
    if rank == 0:
        for n, prm in trainable:
            shape = prm.shape  # DTensor: global shape
            full_m[n] = torch.load(os.path.join(Z, n, "fp32.pt"), weights_only=False).view(shape)
            full_o[n] = {"step": torch.tensor(float(args.init_step)),
                         "exp_avg": torch.load(os.path.join(Z, n, "exp_avg.pt"), weights_only=False).view(shape),
                         "exp_avg_sq": torch.load(os.path.join(Z, n, "exp_avg_sq.pt"), weights_only=False).view(shape)}
    o = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False)
    set_model_state_dict(model, full_m, options=o)
    pg = [{**{k: v for k, v in opt.param_groups[0].items() if k != "params"}, "params": [n for n, _ in trainable]}]
    set_optimizer_state_dict(model, opt, optim_state_dict={"state": full_o, "param_groups": pg} if rank == 0 else {}, options=o)
    del full_m, full_o
    start_step = args.init_step
    log0(f"[fsdp] initialised from universal checkpoint {args.init_universal} (weights + Adam moments) at step {start_step}")
for _ in range(start_step):
    sched.step()
log0(f"[fsdp] start at step {start_step}, lr now {sched.get_last_lr()[0]:.4g}")


# ---- data: this rank's micro-batches of step s are index rows (s-1)*64 + rank + k*world, k < gas ----
class StepSampler(torch.utils.data.Sampler):
    def __iter__(self):
        for s in range(start_step + 1, last_step + 1):
            for k in range(gas):
                i = (s - 1) * args.global_batch + rank + k * world
                if i < len(index):
                    yield i

    def __len__(self):
        return (last_step - start_step) * gas


loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=StepSampler(), collate_fn=lambda b: (D.collate_single(b), b[0]["sample_key"]),
                                     num_workers=args.dataloader_workers, pin_memory=True, prefetch_factor=2 if args.dataloader_workers else None,
                                     persistent_workers=bool(args.dataloader_workers))
it = iter(loader)


def grad_sketch(out):
    sketch, tot, proj_total = {}, torch.zeros((), dtype=torch.float64, device="cuda"), torch.zeros(16, dtype=torch.float64, device="cuda")
    for n, prm in trainable:
        if prm.grad is None:
            continue
        g = prm.grad.full_tensor().detach().float().flatten()
        gen = torch.Generator(device="cuda"); gen.manual_seed(int(hashlib.md5(n.encode()).hexdigest()[:8], 16))
        pr = torch.zeros(16, dtype=torch.float64, device="cuda"); CH = 1 << 24
        for off in range(0, g.numel(), CH):
            gc_ = g[off:off + CH]
            R = torch.randint(0, 2, (16, gc_.numel()), generator=gen, device="cuda", dtype=torch.int8).float() * 2 - 1
            pr += (R @ gc_).double(); del R
        proj_total += pr; sq = (g.double() ** 2).sum(); tot += sq
        sketch[n] = {"norm": float(sq.sqrt()), "sum": float(g.double().sum()), "proj": pr.tolist()}
    if rank == 0:
        json.dump({"global_norm": float(tot.sqrt()), "proj_total": proj_total.tolist(), "params": sketch}, open(out, "w"))
        print("[gradcheck] wrote", out, "global_norm", float(tot.sqrt()), flush=True)


def save(step):
    d = os.path.join(args.output_dir, f"checkpoint-{step}")
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    sd_m, sd_o = get_state_dict(model, opt)
    dcp.save({"model": sd_m, "optim": sd_o}, checkpoint_id=os.path.join(d, "dcp"), process_group=cpu_pg)
    if step in keep_steps or args.test_export:  # evaluation weights, bf16, HF layout
        full = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if rank == 0:
            from safetensors.torch import save_file
            save_file({k: v.to(torch.bfloat16).contiguous() for k, v in full.items()}, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
            processor.save_pretrained(d)  # same processor/tokenizer files the HF Trainer wrote
            shutil.copy(os.path.join(args.model_path, "config.json"), d)  # base config: only the weights differ
            for f in ("chat_template.jinja",):
                if os.path.exists(os.path.join(args.model_path, f)) and not os.path.exists(os.path.join(d, f)):
                    shutil.copy(os.path.join(args.model_path, f), d)
            gen_cfg.save_pretrained(d)
        del full
    if rank == 0:
        json.dump({"global_step": step, "log_history": history}, open(os.path.join(d, "trainer_state.json"), "w"))
    dist.barrier()
    if args.mirror_dir:
        # every node wrote ITS ranks' dcp shard files to its own local disk: each node copies its files
        dst = os.path.join(args.mirror_dir, f"checkpoint-{step}")
        if rank == 0:
            os.makedirs(args.mirror_dir, exist_ok=True)
            shutil.rmtree(dst, ignore_errors=True)
            os.makedirs(os.path.join(dst, "dcp"))
        dist.barrier()
        if local == 0:
            for f in os.listdir(os.path.join(d, "dcp")):
                subprocess.run(["cp", os.path.join(d, "dcp", f), os.path.join(dst, "dcp", f)], check=True)
        if rank == 0:
            for f in os.listdir(d):
                if f != "dcp":
                    subprocess.run(["cp", "-r", os.path.join(d, f), os.path.join(dst, f)], check=True)
        dist.barrier()
        if rank == 0:
            n_files = len(os.listdir(os.path.join(dst, "dcp")))
            open(os.path.join(dst, "COMPLETE"), "w").write(f"step {step} dcp_files {n_files}\n")
            rem = sorted((x for x in glob.glob(os.path.join(args.mirror_dir, "checkpoint-*")) if os.path.exists(os.path.join(x, "COMPLETE"))), key=lambda x: int(x.rsplit("-", 1)[1]))
            for x in rem[:-args.keep_full]:
                s = int(x.rsplit("-", 1)[1])
                if s in keep_steps:
                    shutil.rmtree(os.path.join(x, "dcp"), ignore_errors=True); shutil.rmtree(os.path.join(x, "global_step%d" % s), ignore_errors=True)
                elif os.path.exists(os.path.join(x, "trainer_state.json")):
                    shutil.rmtree(x, ignore_errors=True)
            print(f"[mirror] checkpoint-{step} -> mirror ({n_files} dcp files), save+copy {time.time()-t0:.0f}s", flush=True)
    dist.barrier()
    if local == 0:  # node disks can be small: once a checkpoint is mirrored (COMPLETE) no local copy is kept; without a mirror keep two
        loc = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")), key=lambda x: int(x.rsplit("-", 1)[1]))
        for x in (loc if args.mirror_dir else loc[:-2]):
            shutil.rmtree(x, ignore_errors=True)
    dist.barrier()


model.train()
times = []
slow_since_save = False
for step in range(start_step + 1, last_step + 1):
    t0 = time.time()
    # the last step of the epoch is partial (27,110 = 423*64 + 38): this rank may own fewer than gas samples;
    # it then repeats its first sample with weight 0 so every rank runs gas forward/backward passes (the
    # collectives stay in step) while the loss and its denominator count real samples only
    n_real = sum(1 for k in range(gas) if (step - 1) * args.global_batch + rank + k * world < len(index))
    micro = [next(it) for _ in range(n_real)]
    if not micro:
        # the last step can hold fewer samples than there are ranks (2,370 = 37*64 + 2): a rank without a sample of its own
        # runs a stand-in with weight 0 (the original code repeated micro[0] and crashed here with an IndexError)
        micro = [(D.collate_single([ds[rank % len(ds)]]), "stand-in")]
    weights = [1.0] * n_real + [0.0] * (gas - n_real)
    micro += [micro[0]] * (gas - n_real)
    n_local = sum(int((b["labels"][:, 1:] != D.IGNORE).sum()) * w for (b, _), w in zip(micro, weights))
    N = torch.tensor(float(n_local), device="cuda"); dist.all_reduce(N)
    loss_sum = torch.zeros((), device="cuda", dtype=torch.float64)
    for k, (b, key) in enumerate(micro):
        model.set_requires_all_reduce(k == gas - 1)  # cross-node all-reduce only after the last micro-batch
        b = {kk: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for kk, v in b.items()}
        labels = b.pop("labels")
        sl = labels[:, 1:]; mask = sl != D.IGNORE
        keep = torch.nonzero(mask[0]).squeeze(1); y = sl[0][mask[0]]
        _ce["y"] = y
        model(**b, logits_to_keep=keep, use_cache=False)
        ce = _ce["loss"]; _ce["y"] = _ce["loss"] = None
        (ce * weights[k] / N * world).backward()  # FSDP averages over the world; the denominator is the global token count
        loss_sum += ce.detach().double() * weights[k]
    dist.all_reduce(loss_sum)
    if args.grad_check_out:
        grad_sketch(args.grad_check_out)
        break
    gn = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
    gn = float(gn.full_tensor() if hasattr(gn, "full_tensor") else gn)
    if not (math.isfinite(float(loss_sum / N)) and math.isfinite(gn)):
        # a timed-out cross-node collective can let the step finish with loss nan. Never apply or save that.
        print(f"[fsdp] rank {rank}: non-finite loss/grad at step {step} (loss {float(loss_sum / N)}, grad {gn}) -> abort before the update", flush=True)
        os._exit(13)
    lr = sched.get_last_lr()[0]
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    dt = time.time() - t0; times.append(dt)
    if dt > NCCL_T - 5:
        slow_since_save = True  # as long as a collective timeout: this step may have used incomplete gradients
    rec = {"loss": f"{float(loss_sum / N):.4f}", "grad_norm": f"{gn:.4g}", "learning_rate": f"{lr:.4g}", "epoch": f"{step/steps_per_epoch:.4g}", "step": step, "step_s": round(dt, 1)}
    history.append(rec)
    log0(str({k: v for k, v in rec.items()}))
    if not args.no_save and (step % args.save_steps == 0 or step in keep_steps or step == FINAL):
        if slow_since_save:
            # after a timed-out collective the step still "finishes" with a normal-looking loss and
            # the process is killed by the watchdog ~1-2 min later. Wait that long before writing a checkpoint, so a state
            # built on a timed-out step never reaches the mirror; a genuinely slow step only costs this pause.
            print(f"[fsdp] rank {rank}: a step since the last save took > {NCCL_T - 5}s; waiting {NCCL_T + 60}s before saving step {step}", flush=True)
            time.sleep(NCCL_T + 60)
        save(step); slow_since_save = False
if rank == 0:
    w = times[2:] if len(times) > 4 else times
    print(json.dumps({"steps": len(times), "mean_step_s_after_warmup": sum(w) / max(len(w), 1), "all_step_s": [round(t, 1) for t in times],
                      "peak_mem_GB": torch.cuda.max_memory_allocated() / 2**30}), flush=True)
if not args.no_save and not args.grad_check_out and last_step == FINAL and rank == 0 and args.mirror_dir:
    src = os.path.join(args.mirror_dir, f"checkpoint-{FINAL}")
    subprocess.run(["cp", "-r", src, os.path.join(args.mirror_dir, "last")], check=False)
    open(os.path.join(args.mirror_dir, "last", "COMPLETE"), "w").write("last\n")
dist.barrier()
dist.destroy_process_group()
