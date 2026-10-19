#!/usr/bin/env python3
"""Rebuild, per assistant turn, the exact request the STUDENT (Qwen3.5-9B, `proven`
profile, echo-off) would see, by replaying the teacher transcript through the frozen
rpent code + pydantic-ai (same ApiAgentLoop, same tool wrappers, same image pruning,
same OpenAI message/tool mapping), with a patched OpenAIChatModel that records the
mapped request and returns the teacher's recorded response.

Memory handling:
- tool calls the student's `proven` toolkit would refuse -- memory library / seed-0 references
  ("memory") and any other path outside the guides and the run's own output directory
  ("out_of_scope") -- are removed from history together with their results;
- turns consisting only of such calls are dropped entirely;
- turns that made such a call, and ANY turn whose think/text/tool arguments refer to memory content
  (keywords or a verbatim 8-gram of the memory files), or whose think/text/tool
  arguments contain a decimal number that appeared in a removed result but nowhere the student
  could have seen it (system/user prompt, earlier kept tool results, earlier visible text and
  tool arguments), are replayed (kept in history) but NOT used as training targets.

Usage (rpent env, on the node):
  RPENT_DIR=<RPent checkout with patches/rpent applied> python sft/rpent/replay_build.py <cell dir> <out_dir> <episode key>
Writes <out_dir>/<episode key>.json with {"meta", "system_prompt", "user_message", "samples": [...]}.
"""
import asyncio, base64, hashlib, json, os, re, sys, glob, collections
from pathlib import Path

CKPT = os.path.expanduser(os.environ["RPENT_DIR"])
sys.path.insert(0, CKPT)
os.chdir(CKPT)

from rpent.planner.api_loop import ApiAgentLoop  # noqa: E402
from rpent.dashboard.events import NullDashboardEventSink  # noqa: E402
from rpent.tools.state import EnvState  # noqa: E402
from rpent.tools import common  # noqa: E402
from rpent.utils.templates import substitute  # noqa: E402
from rpent.utils.config import get_memory_dir  # noqa: E402
from robots.libero import robot_spec as rs  # noqa: E402
from robots.libero import tools as libero_tools  # noqa: E402
from robots.libero.prompts import knowledge_eval  # noqa: E402

from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart, ToolCallPart  # noqa: E402
from pydantic_ai.models.openai import OpenAIChatModel  # noqa: E402
from pydantic_ai.profiles.openai import OpenAIModelProfile  # noqa: E402
from pydantic_ai.providers.openai import OpenAIProvider  # noqa: E402

STUDENT_MODEL = "Qwen/Qwen3.5-9B"
STATEFUL = {"move_to", "pi0_pick", "pi0_doubled", "release", "set_gripper", "rotate_wrist",
            "rotate_pitch", "move_pose", "reset"}

# ---------------- memory classification (same as audit_memory.py) ----------------
STRONG = {"seed-0", "seed 0", "seed0", "s0 reference", "seed-0 reference", "memory.md", "task-specific memory",
          "global memory", "reference recipe", "recipe file", "_pert", "memory says", "memory notes",
          "per the notes", "from the notes"}
KW = re.compile(r"\b(memory|MEMORY\.md|seed[- ]?0|recipe file|reference recipe|results_[a-z0-9_]+|_pert|task[- ]specific memory|global memory|from the notes|per the notes|memory says|memory notes|the recipe|s0 reference|seed-0 reference)\b", re.I)


GUIDES = (Path(CKPT) / "robots" / "libero" / "guides").resolve()


def classify_path(p: str, output_dir: str) -> str:
    """What the student's proven toolkit does with this read: 'guide' / 'own_run' are readable
    (same rule as knowledge_eval.proven_read_roots), 'memory' / 'out_of_scope' are refused."""
    p = str(p or "")
    target = (Path(p) if Path(p).is_absolute() else Path(CKPT) / p) if p else Path(output_dir)
    target = Path(os.path.normpath(str(target)))
    run = Path(os.path.normpath(output_dir))
    if target == GUIDES or GUIDES in target.parents or "robots/libero/guides" in p.replace("\\", "/"):
        return "guide"  # incl. 23 reads through the teacher checkout's absolute path (same files)
    if target == run or run in target.parents:
        return "own_run"
    q = p.replace("\\", "/")
    if "resources/libero" in q or "results_" in q or "memory" in q.lower():
        return "memory"
    return "out_of_scope"


REFUSED = ("memory", "out_of_scope")
NUM = re.compile(r"(?<![\w.])-?\d+\.\d{2,}(?![\w])")


def nums(text: str) -> set:
    out = set()
    for x in NUM.findall(text or ""):
        try:
            out.add(f"{round(float(x), 3):.3f}")
        except ValueError:
            pass
    return out


def ngrams(text, n=8):
    toks = re.findall(r"[A-Za-z0-9_.\-]+", text)
    return set(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def load_corpus(root):
    texts = []
    for f in glob.glob(os.path.join(root, "**", "*.md"), recursive=True) + glob.glob(os.path.join(root, "**", "*.json*"), recursive=True):
        try:
            texts.append(open(f, errors="ignore").read())
        except Exception:
            pass
    return "\n".join(texts)


MEM_NG = ngrams(load_corpus(os.path.join(CKPT, "resources", "libero"))) - ngrams(load_corpus(os.path.join(CKPT, "robots", "libero", "guides")))


# ---------------- tool specs exactly as LiberoToolkit registers them (evaluation mode) ----------------
def student_tool_specs():
    specs = []
    for spec in common.TOOLS_SPEC:
        specs.append(spec)
    for spec in libero_tools.TOOLS_SPEC:
        name = spec["name"]
        if name == "reset":
            continue  # evaluation mode
        if name in ("view_env_state", "view_camera_meta", "back_project", "segment"):
            specs.append(spec)
            continue
        if getattr(libero_tools.LiberoPrimitives, name, None) is None:
            continue
        specs.append(spec)
    return specs


class ReplayToolkit:
    """Toolkit stand-in: recorded tool texts + images reloaded from the cell artifacts."""

    def __init__(self, state: EnvState, specs, image_index: dict):
        self.state = state
        self._specs = specs
        self.turn_calls = []  # list of dicts {id, name, input, result_text}, for the current turn
        self.image_index = image_index
        self.mismatch = []

    def get_tools_spec(self):
        return substitute(list(self._specs))

    def cancel_active_and_wait(self):
        pass

    def _images_for(self, name, text):
        step = None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                step = obj.get("step")
                overlay = obj.get("overlay_artifact")
        except Exception:
            obj = None
            m = re.search(r'"step":\s*(\d+)', text)
            step = int(m.group(1)) if m else None
            overlay = None
        blobs = []
        if step is None:
            return blobs
        if name in STATEFUL or name == "view_env_state":
            try:
                out = libero_tools.view_env_state(step, state=self.state)
            except Exception as e:
                self.mismatch.append(f"view_env_state({step}) failed: {e}")
                return blobs
            for key in ("_image_bytes", "_image_cam_bytes", "_image_nav_bytes", "_image_wrist_bytes"):
                if out.get(key):
                    blobs.append(out[key])
        elif name == "segment" and overlay:
            try:
                blobs.append(self.state.load_bytes(overlay, step=step))
            except Exception as e:
                self.mismatch.append(f"segment overlay {overlay}@{step} failed: {e}")
        return blobs

    def execute_tool(self, name, kwargs):
        rec = None
        for c in self.turn_calls:
            if not c.get("used") and c["name"] == name and c["input"] == kwargs:
                rec = c
                break
        if rec is None:
            for c in self.turn_calls:
                if not c.get("used") and c["name"] == name:
                    rec = c
                    self.mismatch.append(f"arg mismatch for {name}: {kwargs} vs {c['input']}")
                    break
        if rec is None:
            raise RuntimeError(f"no recorded result for {name}({kwargs})")
        rec["used"] = True
        text = rec["result_text"]
        blocks = [{"type": "text", "text": text}]
        for b in self._images_for(name, text):
            self.image_index[hashlib.sha1(b).hexdigest()] = (name, len(b))
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(b).decode()}})

        class R:
            content_blocks = blocks
        return R()


def parse_transcript(path, output_dir):
    tr = json.load(open(path))
    ms = tr["messages"]
    turns = []
    results = {}
    for m in ms:
        if m["role"] == "assistant":
            t = {"think": "", "text": "", "calls": []}
            for p in m["content"]:
                if p["type"] == "thinking":
                    t["think"] += p["thinking"]
                elif p["type"] == "text":
                    t["text"] += p["text"]
                elif p["type"] == "tool_use":
                    t["calls"].append({"id": p["id"], "name": p["name"], "input": p["input"]})
            turns.append(t)
        elif m["role"] == "tool":
            results[m["tool_call_id"]] = m["content"]
    for t in turns:
        for c in t["calls"]:
            c["result_text"] = results.get(c["id"])
            c["kind"] = classify_path(c["input"].get("path", ""), output_dir) if c["name"] in ("read_text_file", "list_dir") else "other"
    return tr, turns


def annotate(turns, visible_prompt_text):
    first_mem = None
    for i, t in enumerate(turns):
        t["reads_memory"] = any(c["kind"] in REFUSED for c in t["calls"])
        t["only_memory"] = t["reads_memory"] and all(c["kind"] in REFUSED for c in t["calls"])
        t["refused_kinds"] = sorted(set(c["kind"] for c in t["calls"] if c["kind"] in REFUSED))
        if t["reads_memory"] and first_mem is None:
            first_mem = i
    visible = nums(visible_prompt_text)   # numbers the student can see so far
    removed = set()                       # numbers in results the student never sees (up to this turn)
    for i, t in enumerate(turns):
        kept_calls = [c for c in t["calls"] if c["kind"] not in REFUSED]
        # memory references are looked for in EVERY turn and also in the tool-call arguments (e.g. the
        # write_text_file audit note "MEMORY consulted: MEMORY.md ...") -- found after a step-10 trial
        # episode in which the student wrote about reading resources/libero/memory/MEMORY.md
        said = t["think"] + " " + t["text"] + " " + json.dumps([c["input"] for c in kept_calls])
        kws = set(k.lower() for k in KW.findall(said))
        ng = len(ngrams(said) & MEM_NG)
        own = nums(t["think"] + " " + t["text"] + " " + json.dumps([c["input"] for c in kept_calls]))
        num_dep = sorted((own & removed) - visible)
        t["mem_keywords"] = sorted(kws)
        t["mem_ngram_hits"] = ng
        t["numbers_from_removed"] = num_dep
        t["refers_after"] = bool(kws) or ng > 0   # name kept for the record format; now any turn
        reasons = []
        if t["reads_memory"]:
            reasons.append("reads_" + "+".join(t["refused_kinds"]))
        if t["refers_after"]:
            reasons.append("memory_reference")
        if num_dep:
            reasons.append("numbers_from_removed_content")
        t["reasons"] = reasons
        if t["only_memory"]:
            t["disposition"] = "drop_turn"
        elif reasons:
            t["disposition"] = "keep_history_no_target"
        else:
            t["disposition"] = "target"
        # what later turns can see: this turn's visible text + kept tool arguments + kept results
        # (thinking is not sent back); what they cannot: the removed results
        if t["disposition"] != "drop_turn":
            visible |= nums(t["text"] + " " + json.dumps([c["input"] for c in kept_calls]))
            for c in kept_calls:
                visible |= nums(str(c.get("result_text") or ""))
        for c in t["calls"]:
            if c["kind"] in REFUSED:
                removed |= nums(str(c.get("result_text") or ""))


def build_prompts(tr, output_dir):
    suite, task, seed = tr["suite"], tr["task"], tr["seed"]
    recipe_tag = f"{suite.replace('libero_', '')}_t{task}_s{seed}"
    memory_dir = get_memory_dir("libero")
    prompt_vars = {
        "suite": suite, "task": task, "seed": seed, "recipe_tag": recipe_tag, "mode": "eval",
        "memory_profile": "hf", "knowledge_profile": "proven", "memory_dir": str(memory_dir),
        "reference_tag": f"{suite.replace('libero_', '')}_t{task}_s0",
        "memory_inbox": str(memory_dir / "_inbox" / recipe_tag), "session_number": 1, "session_max": 1,
        "output_dir": output_dir,
    }
    spec = rs.get_robot_spec()
    return spec.prompts.render("system", variables=prompt_vars), spec.prompts.render("user", variables=prompt_vars), recipe_tag


def main(cell_dir, out_dir, ep):
    cell_dir = os.path.abspath(cell_dir)
    tf = glob.glob(os.path.join(cell_dir, "transcript_*.json"))[0]
    tr0 = json.load(open(tf))
    m = re.search(r"- output_dir: (\S+)", tr0["messages"][0]["content"])
    assert m, "teacher output_dir not found in the first user message"
    teacher_output_dir = m.group(1)
    tr, turns = parse_transcript(tf, teacher_output_dir)
    system_prompt, user_msg, recipe_tag = build_prompts(tr, teacher_output_dir)
    annotate(turns, system_prompt + "\n" + user_msg)
    import rpent.utils.logging as rl
    from pathlib import Path
    rl._output_dir = Path(teacher_output_dir)  # tool specs substitute {{output_dir}} from here

    state = EnvState(cell_dir)
    # EnvState.reset() only clears memory; reload the recorded step trace from states.json
    from rpent.tools.state import StepRecord
    manifest = json.load(open(os.path.join(cell_dir, "states.json")))
    state._run_artifacts = set(manifest.get("run_artifacts", []))
    state._steps = [StepRecord(step_idx=b["step_idx"], state=b["state"], terminated=b.get("terminated", False),
                               truncated=b.get("truncated", False), artifacts=set(b.get("artifacts", [])),
                               command=b.get("command"), result=b.get("result"), elapsed_s=b.get("elapsed_s"),
                               extras=b.get("extras", {})) for b in manifest["steps"]]
    image_index = {}
    toolkit = ReplayToolkit(state, student_tool_specs(), image_index)

    # replay schedule: turns not dropped
    schedule = [t for t in turns if t["disposition"] != "drop_turn"]
    records = []
    cursor = {"i": 0}

    async def fake_request(self, messages, model_settings, model_request_parameters):
        model_settings, mrp = self.prepare_request(model_settings, model_request_parameters)
        tools, _tc = self._get_tool_choice(model_settings, mrp)
        openai_messages = await self._map_messages(messages, mrp, model_settings=model_settings)
        i = cursor["i"]
        if i >= len(schedule):
            # teacher transcript exhausted: end the run
            cursor["i"] += 1
            return ModelResponse(parts=[TextPart("")])
        t = schedule[i]
        cursor["i"] += 1
        calls = [c for c in t["calls"] if c["kind"] not in REFUSED and c["result_text"] is not None]
        toolkit.turn_calls = [dict(c) for c in calls]
        parts = []
        if t["think"].strip():
            parts.append(ThinkingPart(content=t["think"]))
        if t["text"].strip():
            parts.append(TextPart(content=t["text"]))
        for c in calls:
            parts.append(ToolCallPart(tool_name=c["name"], args=c["input"], tool_call_id=c["id"]))
        records.append({
            "turn_index": turns.index(t), "disposition": t["disposition"], "reasons": t["reasons"],
            "mem_keywords": t["mem_keywords"], "mem_ngram_hits": t["mem_ngram_hits"],
            "numbers_from_removed": t["numbers_from_removed"],
            "messages": json.loads(json.dumps(openai_messages, default=str)),
            "tools": json.loads(json.dumps(tools, default=str)),
            "response": {"think": t["think"], "text": t["text"],
                         "tool_calls": [{"id": c["id"], "name": c["name"], "arguments": c["input"]} for c in calls],
                         "dropped_memory_calls": [c["kind"] + ":" + c["name"] + ":" + str(c["input"].get("path")) for c in t["calls"] if c["kind"] in REFUSED]},
        })
        return ModelResponse(parts=parts)

    OpenAIChatModel.request = fake_request  # type: ignore[assignment]
    os.environ.setdefault("OPENAI_API_KEY", "replay")
    api_model = OpenAIChatModel(STUDENT_MODEL, provider=OpenAIProvider(base_url="http://127.0.0.1:9/v1", api_key="replay"),
                                profile=OpenAIModelProfile(openai_chat_send_back_thinking_parts=False))
    loop = ApiAgentLoop(model=api_model, max_tokens=32768, no_images=False, reasoning_effort="low",
                        dashboard_events=NullDashboardEventSink(), timeout_s=None,
                        sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5})
    result = loop.solve(system_prompt=system_prompt, user_message=user_msg, toolkit=toolkit, max_turns=100)

    # replace image data URLs by references
    n_img = 0
    for r in records:
        for msg in r["messages"]:
            c = msg.get("content")
            if isinstance(c, list):
                for item in c:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        b = base64.b64decode(url.split(",", 1)[1])
                        item.clear()
                        item["type"] = "image_ref"
                        item["sha1"] = hashlib.sha1(b).hexdigest()
                        item["bytes"] = len(b)
                        n_img += 1
    # store the image bytes once per episode (deduplicated by sha1)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    # re-derive: walk artifacts and hash every png so refs resolve to files
    sha_to_path = {}
    for png in glob.glob(os.path.join(cell_dir, "*.png", "*.png")):
        b = open(png, "rb").read()
        sha_to_path[hashlib.sha1(b).hexdigest()] = png
    missing = 0
    import shutil
    for r in records:
        for msg in r["messages"]:
            c = msg.get("content")
            if isinstance(c, list):
                for item in c:
                    if isinstance(item, dict) and item.get("type") == "image_ref":
                        p = sha_to_path.get(item["sha1"])
                        if p is None:
                            missing += 1
                            item["path"] = None
                            continue
                        rel = os.path.join("images", item["sha1"] + ".png")  # relative to the replay root
                        if not os.path.exists(os.path.join(out_dir, rel)):
                            shutil.copyfile(p, os.path.join(out_dir, rel))
                        item["path"] = rel
    out = {
        "meta": {"episode": ep, "cell_dir": cell_dir, "suite": tr["suite"], "task": tr["task"], "seed": tr["seed"],
                 "recipe_tag": recipe_tag, "teacher_output_dir": teacher_output_dir, "student_model": STUDENT_MODEL,
                 "n_turns_teacher": len(turns), "n_replayed": len(records),
                 "n_targets": sum(1 for r in records if r["disposition"] == "target"),
                 "dispositions": dict(collections.Counter(t["disposition"] for t in turns)),
                 "reasons": dict(collections.Counter(r for t in turns for r in t["reasons"])),
                 "refused_calls": dict(collections.Counter(c["kind"] for t in turns for c in t["calls"] if c["kind"] in REFUSED)),
                 "out_of_scope_paths": sorted(set(str(c["input"].get("path")) for t in turns for c in t["calls"] if c["kind"] == "out_of_scope"))[:20],
                 "transcript_sha256": hashlib.sha256(open(tf, "rb").read()).hexdigest(),
                 "finish": result.finish_result, "error": result.error, "replay_mismatch": toolkit.mismatch[:50],
                 "n_images_in_requests": n_img, "images_unresolved": missing,
                 "system_prompt_sha1": hashlib.sha1(system_prompt.encode()).hexdigest()},
        "system_prompt": system_prompt, "user_message": user_msg,
        "samples": records,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, ep + ".json"), "w") as f:
        json.dump(out, f)
    print(json.dumps({k: v for k, v in out["meta"].items() if k not in ("replay_mismatch",)}), "mismatch", len(toolkit.mismatch))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
