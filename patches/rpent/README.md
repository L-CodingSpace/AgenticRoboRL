# Changes to RPent

`rpent_f70d720.patch` applies to RPent at commit `f70d720` and gives the RPent code all evaluations of the paper used
(8 files, +303 / -9 lines). The planner's tools and RPent's control loop are not changed.

```bash
git clone https://github.com/RLinf/RPent.git && cd RPent
git checkout f70d720
git apply --check /path/to/AgenticRoboRL/patches/rpent/rpent_f70d720.patch
git apply /path/to/AgenticRoboRL/patches/rpent/rpent_f70d720.patch
git add -A && git write-tree        # 11c44fcdedab126962be13ed3f477e0212932323
```

| Change | Files |
|---|---|
| `--knowledge-profile proven`: the zero-shot setting of Harness VLA. The planner gets the cross-task "proven levers" text whole and the guides, but no task memory and no global memory; its file tools can read only the guides and the episode's own output directory | `robots/libero/prompts/knowledge_eval.py` (new), `robots/libero/prompt_bundle.py`, `robots/libero/robot_spec.py`, `robots/libero/toolkit.py`, `rpent/cli/main.py`, `rpent/cli/dashboard.py` |
| `--no-send-back-thinking`: the planner's earlier reasoning is not sent back to the model in later turns; `--temperature`, `--top-p`, `--top-k`, `--presence-penalty`, `--generation-seed` passed to the model server | `rpent/planner/api_loop.py`, `rpent/planner/base.py`, `rpent/cli/main.py`, `rpent/cli/dashboard.py` |
| The transcript records `solved` (the simulator's success flag) and `agent_error`; `rl/adapters/rpent_libero.py` reads the reward from them | `rpent/cli/main.py` |

The RL rollouts of the paper ran on a later version of these files that additionally recorded simulator state for
an experiment not reported in the paper (off unless an environment variable is set; nothing of it reaches the planner).
That recording is not part of this patch.

sha256 of `rpent_f70d720.patch`: `74380a09251aa42f516457b1f410415d7d7cd1aee84d661886c164e18b0860b7`
