"""Harness adapters: everything rl/build_round.py needs to know about ONE harness.

The RL core (proxy -> build_round -> train_rl) only relies on three things per episode:

  1. the proxy records of the episode's model calls   <episode dir>/rl_turns/turn_NNN.json.gz   (rl/proxy.py)
  2. which group the episode belongs to                (same task, same initial state => one group)
  3. how it ended                                      (reward, valid or an infrastructure failure)

An adapter supplies 2 and 3 (and a few optional extras) for its harness. It is a module with:

  NAME                 str
  find_episodes(root)  -> iterable of (episode dir: Path, verdict: dict). The verdict must hold
                            the fields group_key() reads, "attempt" (int), "valid" (bool; False = infrastructure
                            failure, the episode is not a task failure), "reward" (float; ignored when not valid),
                            "requests" (int | None: model calls the harness made; None = unknown, not checked)
                          and may hold anything else the adapter wants in round_stats.json.
  group_key(row)       -> tuple, the episode's group
  group_name(gk)       -> str, the group's name in samples.jsonl / round_stats.json
  post_success_turns(episode dir, n_turns) -> None | {"success_turn", "excluded", "finish_kept"}
                          (--drop-post-success; None = cannot tell, nothing is filtered)
  episode_extras(turn files) -> dict of per-episode statistics for round_stats.json
  outcome_kind(row)    -> str, the label of a valid episode in round_stats.json "valid_outcomes"
  EPISODE_KEYS         tuple of row fields written to round_stats.json "episode_table"
  OUTCOME_METRICS      tuple of numeric row fields averaged in round_stats.json "by_outcome"

rpent_libero reads RPent's own transcripts (LIBERO / LIBERO-PRO); result_json is the generic one for any harness
that writes a result.json per episode (used for CaP-X).
"""
from __future__ import annotations

import importlib

REQUIRED = ("NAME", "find_episodes", "group_key", "group_name", "post_success_turns", "episode_extras",
            "outcome_kind", "EPISODE_KEYS", "OUTCOME_METRICS")


def load(name: str):
    """The adapter module `name` (rl/adapters/<name>.py), checked for the interface above."""
    if not name.replace("_", "").isalnum():
        raise ValueError(f"bad adapter name {name!r}")
    mod = importlib.import_module(f"adapters.{name}")
    missing = [x for x in REQUIRED if not hasattr(mod, x)]
    if missing:
        raise ValueError(f"adapter {name!r} lacks {missing}")
    return mod


def episode_key(adapter, gk, attempt) -> str:
    """Name of one episode: its group's name + the attempt."""
    return f"{adapter.group_name(gk)}/a{attempt}"
