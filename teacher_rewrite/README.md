# Removing teacher-only information from teacher reasoning (RPent SFT data)

The RPent teacher (Qwen3.8-27B) reads Harness VLA's memory; the student does not. After `sft/rpent/replay_build.py`,
a teacher turn whose reasoning refers to memory content is kept as history only, never as a training target. These
scripts rewrite such turns so that more of them can be used: sentences that cite content the student cannot see are
deleted or rewritten by the 27B model, every rewrite is checked automatically and then judged by the 27B model with
reasoning on, and only turns that pass become targets.

Input: the replay shards of `sft/rpent/replay_build.py` (tar files of per-episode JSON) and the memory files the
teacher read. Order of the steps (as they were run; the working directory holds `turns/`, `pilot/` and `full_v2/`):

| Step | Script | What it does |
|---|---|---|
| prep | `extract_turns.py` | one shard -> `turns/turns_<i>.jsonl.gz`: every teacher turn's text, tool calls and replay fields |
| prep | `make_number_lists.py` | `pilot/hidden_corpus_numbers.json` (numbers in the memory files) and `pilot/numbers_from_removed_all.json` (numbers of removed memory content per turn) |
| 0 | `make_turn_plan_full.py` | sentence spans of every teacher turn, candidate numbers for the visibility check |
| 1 | `number_provenance.py`, `strict_visibility.py` | is a number in a turn visible to the student before that turn (system prompt, first user message, tool results, tool-call arguments)? |
| 2 | `build_pilot_inputs_v2.py` | per turn: keep, history only, or a rewrite request; `source_rules.py` flags sentences that cite unseen content |
| 3 | `rewrite_client_v2.py` | the 27B model edits the flagged sentences (`serve_gpu.sh` starts one server per GPU) |
| 4 | `check_edits_v2.py` | automatic checks of every edit (source words, invented results, new or hidden numbers, too much deleted) |
| 5 | `judge_client.py` | the 27B model judges each rewritten turn; only OK (rewritten) or OK / MINOR (unchanged) become targets |
| 6 | `finalize_v2.py` | final decision per turn -> `changes.jsonl` |
| 7 | `apply_rewrite_to_shards.py`, `verify_shard.py` | write a cleaned copy of each replay shard (later history updated too) and check it against the original |

Then `sft/rpent/build_index.py` on the cleaned shards. In the paper's data this raised the number of training targets
from 27,110 to 40,047 of 79,973 teacher turns.

The two prep scripts were checked against the files the paper's run used: `make_number_lists.py` reproduces both
number lists byte for byte, and `extract_turns.py` reproduces the extracted turns of a shard exactly (same content).
