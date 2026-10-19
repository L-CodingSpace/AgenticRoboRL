# Harnesses

The RL core (`rl/proxy.py` -> `rl/build_round.py` -> `rl/train_rl.py`) trains the model *inside* a harness without
changing the harness. It needs three things from it, and nothing else:

| What | How the harness provides it |
|---|---|
| The model calls of each episode | It talks to the OpenAI-compatible URL it is given (`{model_url}`, one per episode). The recording proxy behind that URL stores every request: prompt token ids, output token ids, chosen-token log-probabilities, images. Chat completions, non-streaming. |
| Which episodes belong together | It can be started for a given task and initial state (`{task_id}`, `{init_id}`): the controller runs `group_size` episodes of the same pair and compares them with each other. |
| How each episode ended | A `result.json` in the episode directory: `{"reward": 1.0, "valid": true}` (optional: `"requests"`, the number of model calls; `"end_turn"`, the call after which the task was already done). |

A harness is described by a `harness.yaml` (format: `rl/harness_cfg.py`) and a `tasks.jsonl`, and read by an adapter
(`rl/adapters/`): `result_json` for the contract above, or one written for a harness that already has its own records.

| Directory | Harness | Adapter |
|---|---|---|
| `minimal/` | 60-line toy agent, stand-in model server and trainer (no GPU); runs in the test suite | `result_json` |
| `rpent_libero/` | RPent on LIBERO-PRO (needs `patches/rpent/`) | `rpent_libero` |
| `capx_libero/` | [CaP-X](https://github.com/capgym/cap-x) on its LIBERO-PRO tasks: the model writes Python over the harness's APIs, no VLA | `result_json` |

Try the whole loop on a laptop:

```bash
python rl/local_ctl.py train --config harnesses/minimal/harness.yaml --run-dir /tmp/minimal_run --weights w0 --rounds 2 --groups 6
```

## Adding a harness

1. Make it startable by one command that takes a task, an initial state, a model URL and an output directory.
2. Have it (or a small `episode.result_cmd` script) write `result.json`.
3. List its tasks and their initial states in `tasks.jsonl`; keep the evaluation ones (`eval_init_ids`) and the
   validation ones (`val_init_ids`, used to choose checkpoints and the training pool) apart from the training ones
   (`init_ids`).
4. Write `harness.yaml`; use `adapter: result_json`.

Limits today: the proxy speaks OpenAI chat completions without streaming, the model server must return token ids and
log-probabilities the way vLLM does, and `rl/train_rl.py` has only been validated with Qwen3.5-9B.
