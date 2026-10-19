"""Per-turn SFT samples for Qwen3.5-9B from the replayed teacher requests (RPent, LIBERO-PRO).
prepare_messages() is also what rl/train_rl.py and rl/check_gate_a.py use to re-render a recorded request.

Episodes are read from disk on demand (2,450 replay
files do not fit comfortably in RAM once per rank and dataloader worker); image paths are
relative to the replay root; the sample list comes from a precomputed index (episode key,
sample position, turn index) so no process has to parse every episode.

A sample = (prefix rendered with the model's own chat template + tools + images,
add_generation_prompt=True) followed by the teacher turn rendered as the template renders an
assistant message after the last user query (think + text + XML tool calls + <|im_end|>).
Labels: -100 on the prefix, token ids on the target.  History assistant tool-call arguments are
json-decoded before templating, as vLLM's _postprocess_messages does at inference.
"""
import json, os, glob, copy
from PIL import Image
import torch

IGNORE = -100


def load_episodes(replay_dir):
    eps = []
    for f in sorted(glob.glob(os.path.join(replay_dir, "*.json"))):
        eps.append(json.load(open(f)))
    return eps


def list_samples(eps, targets_only=True):
    out = []
    for ei, ep in enumerate(eps):
        for si, s in enumerate(ep["samples"]):
            if targets_only and s["disposition"] != "target":
                continue
            out.append((ei, si))
    return out


def prepare_messages(sample, root=""):
    """Return (messages for the template, PIL images in order)."""
    msgs = copy.deepcopy(sample["messages"])
    images = []
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                a = tc["function"].get("arguments")
                if isinstance(a, str):
                    try:
                        tc["function"]["arguments"] = json.loads(a)
                    except Exception:
                        tc["function"]["arguments"] = {}
            if m.get("content") is None:
                m["content"] = ""
        c = m.get("content")
        if isinstance(c, list):
            for it in c:
                if it.get("type") == "image_ref":
                    images.append(Image.open(os.path.join(root, it["path"])).convert("RGB"))
                    it.clear()
                    it["type"] = "image"
    return msgs, images


def assistant_message(resp):
    tcs = [{"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in resp["tool_calls"]]
    m = {"role": "assistant", "content": resp["text"], "reasoning_content": resp["think"]}
    if tcs:
        m["tool_calls"] = tcs
    return m


def render(processor, sample, root=""):
    msgs, images = prepare_messages(sample, root)
    tools = sample["tools"]
    prefix = processor.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)
    full = processor.apply_chat_template(msgs + [assistant_message(sample["response"])], tools=tools, add_generation_prompt=False, tokenize=False)
    assert full.startswith(prefix), "template prefix mismatch"
    target = full[len(prefix):]
    if target.endswith("<|im_end|>\n"):
        target = target[:-1]
    return prefix, target, images


def encode(processor, sample, root=""):
    prefix, target, images = render(processor, sample, root)
    enc = processor(text=[prefix], images=images if images else None, return_tensors="pt")
    prefix_ids = enc["input_ids"][0]
    target_ids = torch.tensor(processor.tokenizer(target, add_special_tokens=False)["input_ids"], dtype=prefix_ids.dtype)
    input_ids = torch.cat([prefix_ids, target_ids])
    labels = torch.cat([torch.full_like(prefix_ids, IGNORE), target_ids])
    out = {"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones_like(input_ids)}
    if images:
        out["pixel_values"] = enc["pixel_values"]
        out["image_grid_thw"] = enc["image_grid_thw"]
        if "mm_token_type_ids" in enc:
            mm = enc["mm_token_type_ids"][0]
            out["mm_token_type_ids"] = torch.cat([mm, torch.zeros_like(target_ids)])
    out["n_prefix"] = int(prefix_ids.numel()); out["n_target"] = int(target_ids.numel()); out["n_images"] = len(images)
    return out


class SFTDataset(torch.utils.data.Dataset):
    """index = list of (episode key, sample position in the replay file, teacher turn index)."""

    def __init__(self, replay_dir, processor, index):
        self.root = replay_dir
        self.processor = processor
        self.index = index
        self._cache_key, self._cache = None, None

    def __len__(self):
        return len(self.index)

    def _episode(self, key):
        if key != self._cache_key:
            self._cache = json.load(open(os.path.join(self.root, key + ".json")))
            self._cache_key = key
        return self._cache

    def __getitem__(self, i):
        key, si, turn = self.index[i]
        s = self._episode(key)["samples"][si]
        assert s["turn_index"] == turn and s["disposition"] == "target", (key, si, turn)
        out = encode(self.processor, s, self.root)
        out["sample_key"] = f"{key}:{turn}"
        if os.environ.get("DEBUG_KEYS"):
            print(f"[keys] rank={os.environ.get('RANK','?')} key={out['sample_key']} n={out['input_ids'].numel()}", flush=True)
        return out


def collate_single(batch):
    """micro-batch = 1 sample (variable length, no padding)."""
    assert len(batch) == 1
    b = batch[0]
    out = {"input_ids": b["input_ids"][None], "labels": b["labels"][None], "attention_mask": b["attention_mask"][None]}
    if "pixel_values" in b:
        out["pixel_values"] = b["pixel_values"]
        out["image_grid_thw"] = b["image_grid_thw"]
        if "mm_token_type_ids" in b:
            out["mm_token_type_ids"] = b["mm_token_type_ids"][None]
    return out
