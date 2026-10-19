<div align="center">

<img src="assets/logo.png" alt="" width="16%">

<img src="assets/wordmark.png" alt="AgenticRoboRL" width="42%">

<h3>Bringing Coding-Agent Post-Training to Agentic Robotics</h3>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](#-prerequisites)

</div>

## 📑 Table of Contents

- [📖 Introduction](#-introduction)
- [🧭 Overview](#-overview)
- [📊 Main Results](#-main-results)
- [📦 Checkpoints](#-checkpoints)
- [📁 Repository Layout](#-repository-layout)
- [🧰 Prerequisites](#-prerequisites)
- [🚀 Quick Start](#-quick-start)
- [📚 Supervised Fine-Tuning](#-supervised-fine-tuning)
- [🔁 Reinforcement Learning](#-reinforcement-learning)
- [🎯 Evaluation](#-evaluation)
- [🙌 Acknowledgements](#-acknowledgements)
- [📜 License](#-license)

## 📖 Introduction

AgenticRoboRL trains the planner model of a robot harness inside the harness itself. A harness pairs a language model
planner with robot tools: the planner may call a VLA policy as a skill, or write Python against the harness's robot
APIs. The harness keeps its tools and its control loop, and only the planner is trained.

- **SFT** on a stronger teacher's successful trajectories, each turn rendered as the request the student would receive.
- **Multi-turn RL** with the simulator's success verdict as the reward. Rollouts run in the harness, and a proxy between
  the harness and the model server records every planner call with its token ids and log-probabilities.
- **One pipeline for different harnesses.** A harness is connected through a short adapter that reports which episodes
  form a group and how each one ended. This repository connects two harnesses, one that calls a VLA policy (RPent) and
  one that generates control code (CaP-X).

## 🧭 Overview

| Harness | What the planner does | Adapter | Config |
|---|---|---|---|
| [RPent](https://github.com/RLinf/RPent) (Harness VLA) on LIBERO-PRO | calls tools, including a frozen VLA policy | `rl/adapters/rpent_libero.py` | `harnesses/rpent_libero/` |
| [CaP-X](https://github.com/capgym/cap-x) on LIBERO-PRO | writes Python against the harness's robot APIs | `rl/adapters/result_json.py` | `harnesses/capx_libero/` |

### Workflow at a glance

```
harness (planner) ──> rl/proxy.py ──> vLLM model server
                          │ records prompt / output token ids and log-probabilities per call
                          v
episodes + adapter ──> rl/build_round.py ──> samples.jsonl ──> rl/train_rl.py ──> new weights ──> next round
                     (groups, success, advantages)           (dual-clip PPO)
```

An **adapter** is a small module that tells `build_round.py` where a harness keeps its episodes, how to group them
and how each one ended (`rl/adapters/__init__.py` lists the interface). A harness that writes a `result.json` per episode
needs no adapter of its own (`result_json`). A **harness.yaml** says how to start the model server, the harness's
services and one episode (`rl/harness_cfg.py` documents every field).

### Quick links

- Try the whole loop on a laptop, then on one or several machines: [Quick Start](#-quick-start)
- Train: [Supervised Fine-Tuning](#-supervised-fine-tuning), then [Reinforcement Learning](#-reinforcement-learning)
- Evaluate a checkpoint: [Evaluation](#-evaluation)

## 📊 Main Results

Success rate (%) of a Qwen3.5-9B planner.

| Harness / Benchmark | Qwen3.5-9B | + SFT | + SFT + RL |
|---|---|---|---|
| RPent / LIBERO-PRO | 18.1 | 65.1 | **71.0** |
| RPent / LIBERO-Plus | 19.2 | 66.7 | **72.2** |

Each LIBERO-PRO number is one evaluation of 800 episodes (8 suites x 10 tasks x initial states 1-10). The LIBERO-Plus
numbers are one evaluation of its 10,030 configurations with the checkpoints trained on LIBERO-PRO, without further
training. For reference, the Qwen3.8-27B teacher with Harness VLA's memory reaches 74.0% on RPent / LIBERO-PRO.

## 📦 Checkpoints

Both checkpoints are Qwen3.5-9B planners for the RPent harness, trained on LIBERO-PRO.

| Checkpoint | Training | LIBERO-PRO | LIBERO-Plus | Download |
|---|---|---|---|---|
| SFT, step 563 | SFT on the teacher's trajectories | 65.1 | 66.7 | coming soon |
| RL, round 6 (v006) | 6 RL rounds from the SFT checkpoint | 71.0 | 72.2 | coming soon |

## 📁 Repository Layout

```
rl/
  proxy.py  rl_sampling.py         recording proxy and the sampling it enforces
  build_round.py  adapters/        episodes -> training samples
  train_rl.py  rl_loss.py  plan.py policy update
  check_gate_a.py                  checks that training re-renders exactly the tokens the server saw
  harness_cfg.py                   harness.yaml format
  local_ctl.py                     controller for one machine (train / eval / merge-eval)
  multi_node_ctl.py                controller for several machines that share a directory (train / eval)
  cluster.example.yaml
sft/
  chat_dataset.py                  renders a recorded request with the student's template (shared with rl/)
  rpent/                           RPent (LIBERO) teacher transcripts -> student requests -> SFT
  capx/                            CaP-X teacher model calls -> SFT (chunked cross-entropy)
teacher_rewrite/                   removing references to teacher-only memory from teacher reasoning (LIBERO)
harnesses/
  minimal/                         a toy harness with a fake model server and trainer: the whole loop on a laptop
  rpent_libero/  capx_libero/      the two harnesses of the paper
patches/rpent/                     our changes to RPent (planner switches, the `proven` knowledge profile, verdict fields)
assets/                            images used by this README
tests/                             no GPU, simulator or network needed
```

## 🧰 Prerequisites

### Software versions

| Component | Version |
|---|---|
| Python | 3.11 |
| Training (`TRAIN_ENV`) | torch 2.13.0, transformers 5.17.0, flash-linear-attention 0.5.2; multi-GPU training with PyTorch FSDP2 |
| Model server (`PLANNER_ENV`) | vLLM 0.28.0 |
| RPent (`RPENT_ENV`) | transformers 4.57.6, pydantic-ai-slim 2.49.0 |

### Environment variables

| Variable | Harness | Meaning |
|---|---|---|
| `RPENT_DIR` | RPent | an RPent checkout at `f70d720` with `patches/rpent/rpent_f70d720.patch` applied |
| `RPENT_ENV` | RPent | RPent's Python environment (with LIBERO-PRO) |
| `PI05_CHECKPOINT` | RPent | the frozen VLA (RLinf-Pi05-LIBERO-130-fullshot-SFT) |
| `SAM3_CHECKPOINT` | RPent | `sam3.pt` |
| `CAPX_DIR` | CaP-X | a CaP-X checkout with its LIBERO environment installed |
| `CAPX_PYTHON` | CaP-X | the Python of that environment (`.venv-libero/bin/python`) |
| `PLANNER_ENV` | both | an environment with vLLM 0.28 |
| `TRAIN_ENV` | both | the environment of `rl/train_rl.py` |

### RPent

1. Clone RPent, check out `f70d720`, and apply `patches/rpent/rpent_f70d720.patch` (see `patches/rpent/README.md`).
2. Set `RPENT_DIR`, `RPENT_ENV`, `PLANNER_ENV`, `TRAIN_ENV`, `PI05_CHECKPOINT` and `SAM3_CHECKPOINT` as described at the
   top of `harnesses/rpent_libero/harness.yaml`.

### CaP-X

Install CaP-X with its LIBERO environment (its `docs/libero-tasks.md`) and set `CAPX_DIR` and `CAPX_PYTHON`.
`harnesses/capx_libero/run_episode.py` runs one CaP-X trial for a given task and trial number with a given model URL,
without changing the CaP-X checkout: retries off, and CaP-X's per-request and per-trial time limits raised at run time.

## 🚀 Quick Start

### Without a GPU

```bash
pip install pyyaml fastapi uvicorn httpx requests pytest
python rl/local_ctl.py train --config harnesses/minimal/harness.yaml --run-dir /tmp/minimal_run --weights w0 --rounds 2 --groups 6
python -m pytest -q tests
```

### One machine

```bash
# validate a checkpoint on the validation inits (one controller per GPU; --shard splits the tasks)
python rl/local_ctl.py eval --config harnesses/capx_libero/harness.yaml --run-dir runs/capx/g0 --weights <ckpt> --tag val_v000 \
       --init-key val_init_ids --shard 0/8 --var gpu=0 --var model_port=18110 --var proxy_port=18210
python rl/local_ctl.py merge-eval --out runs/capx/val_v000.json --tag val_v000 runs/capx/g*/eval/val_v000/report.json

# train: tasks whose success rate on the validation inits lies in [0.1, 0.9] form the pool
python rl/local_ctl.py train --config harnesses/capx_libero/harness.yaml --run-dir runs/capx/train --weights <ckpt> \
       --rounds 15 --groups 64 --pool-report runs/capx/val_v000.json --var gpu=0 --var model_port=18110 --var proxy_port=18210
```

The run directory holds `state.json`; starting the same command again resumes. `touch <run dir>/STOP` stops between
episodes.

### Several machines

`rl/multi_node_ctl.py` spreads the episodes of a round (or of an evaluation) over machines that share one directory,
one worker per GPU, and runs the training on one of them. Machines are reached through a command template (ssh by
default), so nothing in it depends on a particular cluster; copy `rl/cluster.example.yaml` and fill in your hosts.

```bash
python rl/multi_node_ctl.py eval  --cluster cluster.yaml --config harnesses/rpent_libero/harness.yaml \
       --run-dir /shared/runs/rpent --weights /shared/ckpt/sft --tag val_v000 --init-key val_init_ids
python rl/multi_node_ctl.py train --cluster cluster.yaml --config harnesses/rpent_libero/harness.yaml \
       --run-dir /shared/runs/rpent --weights /shared/ckpt/sft --rounds 6 --groups 64 --pool-report /shared/runs/rpent/eval/val_v000/report.json
```

A worker claims episodes with an exclusive file create in the shared directory (up to `episodes_parallel` at a time,
sharing one model server) and beats by creating new numbered files. If a worker stops beating for `lease_s`, the
controller hands its episodes out again, into a new directory, so a late writer can never mix into the new run. The
shared directory must support exclusive create and atomic rename; file modification times are never used (they are
not updated on some network file systems, e.g. HDFS mounted through FUSE). The module docstring
describes every file it uses.

## 📚 Supervised Fine-Tuning

The teacher trajectories used for SFT are not released. RL on RPent can start from the released SFT checkpoint
(see [Checkpoints](#-checkpoints)).

### RPent

Teacher: Qwen3.8-27B with Harness VLA's memory. Student: Qwen3.5-9B without it.

```
teacher transcripts (successful episodes on initial states 11-44)
  -> sft/rpent/replay_build.py     each teacher turn replayed through the student's harness configuration: the exact
                                   request the student would receive, with tool calls the student may not make removed
  -> teacher_rewrite/              references to memory content removed from the teacher's reasoning (27B rewriter + judge)
  -> sft/rpent/build_index.py      fixed training order
  -> sft/rpent/train_fsdp.py       lr 1e-5, 10 warm-up steps, cosine to 10%, global batch 64
```

### CaP-X

`sft/capx/build.py` turns the teacher's recorded model calls (through the proxy) into samples rendered with the
student's chat template, then `sft/capx/train_fsdp.py`.

## 🔁 Reinforcement Learning

### How one RL round works

```
rollout   the harness runs episodes; every model call goes through rl/proxy.py, which forces plain temperature-1
          sampling (rl/rl_sampling.py) and stores prompt token ids, output token ids and their log-probabilities
            -> <episode dir>/rl_turns/turn_NNN.json.gz
build     rl/build_round.py + an adapter: which episodes belong to one group (same task, same initial state), which
          succeeded, which are infrastructure failures; leave-one-out advantages inside a group; up to 16 sampled model
          calls per episode, each weighted 1/k -> samples.jsonl
train     rl/train_rl.py: dual-clip PPO with truncated importance sampling (rl/rl_loss.py), re-renders each
          recorded prompt with the student's chat template and checks it is token-identical to what the server saw
```

### Settings of the paper's runs

| | RPent / LIBERO-PRO | CaP-X / LIBERO-PRO |
|---|---|---|
| start | SFT checkpoint (step 563) | SFT checkpoint (step 37) |
| groups per round x episodes per group | 64 x 6 | 64 x 6 |
| turns per episode in the samples | 16 | 16 |
| sample exclusions | `--max-output-tokens 20000 --max-total-tokens 160000 --drop-post-success` (from the 6th round) | none |
| learning rate, global batch | 1e-6, 128 | 1e-6, 128 |
| PPO | eps 0.2 / 0.28, dual clip 3.0, truncated IS cap 2.0, no KL, no entropy bonus | same, `--plan balanced`, `--ce_chunk 2048` |

The exact commands are in the two `harness.yaml` files.

## 🎯 Evaluation

Every task in `tasks.jsonl` has three disjoint lists of initial states: `init_ids` for training rollouts,
`val_init_ids` for choosing checkpoints and the training pool, and `eval_init_ids` for the reported numbers. `eval`
has no default for `--init-key`: every evaluation names the inits it runs.

```bash
# evaluate the checkpoint chosen on the validation inits
python rl/local_ctl.py eval --config harnesses/capx_libero/harness.yaml --run-dir runs/capx/g0 --weights <ckpt> --tag v015 \
       --init-key eval_init_ids --shard 0/8 --var gpu=0 --var model_port=18110 --var proxy_port=18210
```

On RPent, evaluation follows the protocol of Harness VLA: the planner talks to the model server directly with its own
sampling settings (`eval.direct`), 8 LIBERO-PRO suites x 10 tasks x initial states 1-10. Training rollouts use initial
states 11-44; checkpoints and the training pool are chosen on initial states 45-49.

## 🙌 Acknowledgements

This repository trains and evaluates models inside two existing robot harnesses. CaP-X is used unchanged, and our
changes to RPent are listed in `patches/rpent/README.md`. We thank the authors of the following projects for releasing
their code and models, and ask you to cite their papers if you use the corresponding parts of this repository.

- Harnesses: [RPent](https://github.com/RLinf/RPent) (Harness VLA) and [CaP-X](https://github.com/capgym/cap-x)
- Benchmarks: LIBERO, LIBERO-PRO and LIBERO-Plus
- Models: pi0.5 (openpi), SAM 3 and the Qwen models
- Serving and training: vLLM and flash-linear-attention

## 📜 License

Apache License 2.0 (`LICENSE`). Third-party code and its licenses are listed in `NOTICE`.
