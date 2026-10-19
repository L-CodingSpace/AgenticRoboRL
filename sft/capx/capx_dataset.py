"""Per-turn SFT samples for Qwen3.5-9B from the teacher's recorded CaP-X model calls (rl_turns/turn_*.json.gz).

A sample = the request's messages rendered with the STUDENT's chat template (add_generation_prompt=True) -- what the
student sees at inference -- followed by the token ids the teacher produced (reasoning + answer + <|im_end|>), exactly as
the serving engine recorded them. The teacher (Qwen3.8-27B) and the student (Qwen3.5-9B) share the vocabulary, but the
teacher's template puts one sentence ("Reasoning effort is set to xhigh. ...") in front of the system prompt; the student's
does not. capx_sft_build.py checks for every sample that this sentence is the ONLY difference between the student's
rendering and the prompt the teacher really got. Labels: -100 on the prefix, ids on the target.
Same interface as sft/chat_dataset.py (the FSDP trainer imports this module as D). Text only (ground-truth-state setting).
"""
import gzip, json, os
import torch
from chat_dataset import prepare_messages, collate_single  # noqa: F401  (collate_single is used by the trainer)

IGNORE = -100


def load_turn(root, episode, pos):
    return json.loads(gzip.decompress(open(os.path.join(root, episode, "rl_turns", f"turn_{pos:03d}.json.gz"), "rb").read()))


def prefix_ids(processor, rec):
    msgs, images = prepare_messages({"messages": rec["messages"]})
    assert not images, "text-only data expected"
    prefix = processor.apply_chat_template(msgs, tools=rec["request"].get("tools"), add_generation_prompt=True, tokenize=False)
    return processor(text=[prefix], return_tensors="pt")["input_ids"][0]


class SFTDataset(torch.utils.data.Dataset):
    """index = list of (episode dir relative to the data root, turn number, turn number)."""

    def __init__(self, replay_dir, processor, index):
        self.root, self.processor, self.index = replay_dir, processor, index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ep, pos, turn = self.index[i]
        rec = load_turn(self.root, ep, pos)
        p = prefix_ids(self.processor, rec)
        o = torch.tensor(rec["output_token_ids"], dtype=p.dtype)
        ids = torch.cat([p, o])
        return {"input_ids": ids, "labels": torch.cat([torch.full_like(p, IGNORE), o]), "attention_mask": torch.ones_like(ids),
                "n_prefix": int(p.numel()), "n_target": int(o.numel()), "n_images": 0, "sample_key": f"{ep}:{turn}"}
