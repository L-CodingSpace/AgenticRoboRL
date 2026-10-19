#!/usr/bin/env python3
"""Rewrite v2, step 2: send every request to the local 27B server (async, JSON-schema output, thinking off, temperature 0).
v2 prompt: whole-turn view, header+list deletion ("through"), follow-up sentences, no turning hidden results into facts,
hidden numbers. Resumable: keys already in <out.jsonl> are skipped and new lines are appended.
Usage: rewrite_client_v2.py <requests.jsonl> <student_system_prompt.txt> <out.jsonl> <port> [concurrency]"""
import asyncio, json, os, sys, time
from openai import AsyncOpenAI
RULES = """You edit the reasoning of a robot-manipulation planning agent so it can become training data for a STUDENT model.

The STUDENT will NOT have any of these (call them HIDDEN material):
- the memory library (resources/libero/memory/*.md, MEMORY.md, feedback_* files, "the memory", "memories", "the fragile-box memory", ...);
- the seed-0 reference results and recipes (results_*_pert/*_s0.json, recipe_*_s0.jsonl), which the agent calls "seed-0", "s0", "the reference", "the recipe", "the audit", "the notes", "the solution", "the documented/winning/proven approach";
- records of the authors' earlier attempts ("attempt 86", "41 prior attempts", "they did X", "that run").
The STUDENT WILL have: the system prompt quoted at the end, the tool results and its own tool calls in THIS episode, and it writes its own audit/recipe files at the end.

You receive ONE turn split into numbered sentences (all of them). "flagged": true = a keyword rule matched. "hidden_numbers" = numbers that come from HIDDEN material and that the student has not seen.
Goal: the edited turn must read as reasoning the student could have produced from what it can see. Return one entry for EVERY sentence you change; sentences you do not list stay unchanged.
Actions:
- "delete": the sentence states content from HIDDEN material that is not in the student's system prompt: seed-0 coordinates, what another run/the authors did or observed, results ("worked", "succeeded", "fired", "reached", "descended fine", "qpos=0.006"), parameter values and budgets not in the prompt, or it is about reading/finding memory or reference files. new_text = "".
  If the sentence is a header that introduces a list, quote or block (e.g. "The seed-0 recipe:", "Key takeaways from memory:"), set "through" to the id of the LAST sentence of that block so the whole block is deleted.
- "drop_source": ONLY when the content is (a) a plan or decision for this episode ("I will ...", "let me try ..."), or (b) stated in the student's system prompt, or (c) shown by this episode's own tool results. Remove only the attribution words; keep the rest verbatim; do not replace them with another source word ("the guidance", "the solution", "proven", "documented", "the initial reference" are forbidden replacements). Do not add any fact, number or claim.
- "keep": the flag is a mistake (the sentence is about this episode's own history or its own output files, or uses the word generically). new_text = the sentence unchanged.
Also required:
1. After every delete or drop_source, look at the next sentences: a sentence that continues the removed content ("they ...", "that run ...", "that worked", "so it fired", "similar", "the same", "that's the site center") must also be deleted or fixed. A sentence whose subject or comparison object was deleted must be deleted too.
2. Never turn "the source says X" into "X" when X is a result observed elsewhere or a value not in the prompt; delete it instead.
3. The edited turn must contain none of the "hidden_numbers". Remove them by delete, or by drop_source that drops the number (a vague phrase of at most 3 words such as "a bit higher" may replace it; never another number).
4. Do not delete content that is in the student's system prompt; use drop_source for it.
5. Every number in a drop_source new_text must already appear in that original sentence.
"through" = "" unless you delete a block. In "why", give a short reason (under 20 words).

=== STUDENT SYSTEM PROMPT (use it to decide whether content is available to the student) ===
"""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["edits"], "properties": {"edits": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "action", "through", "new_text", "why"],
    "properties": {"id": {"type": "string"}, "action": {"type": "string", "enum": ["drop_source", "delete", "keep"]},
                   "through": {"type": "string"}, "new_text": {"type": "string"}, "why": {"type": "string"}}}}}}
async def one(client, sysmsg, q, sem, out, stats):
    async with sem:
        user = json.dumps({"suite": q["suite"], "task": q["task"], "tool_calls_this_turn": q["tool_calls_this_turn"], "sentences": q["sentences"]}, ensure_ascii=False)
        err = ""
        for attempt in range(3):
            t0 = time.time()
            try:
                r = await client.chat.completions.create(model="qwen27", temperature=0, max_tokens=8192,
                    messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
                    response_format={"type": "json_schema", "json_schema": {"name": "edits", "schema": SCHEMA, "strict": True}},
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                parsed = json.loads(r.choices[0].message.content)
                out.write(json.dumps({"key": q["key"], "ok": True, "attempt": attempt, "edits": parsed["edits"], "usage": r.usage.model_dump() if r.usage else None,
                                      "latency_s": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n"); out.flush(); stats["ok"] += 1; return
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:300]
        out.write(json.dumps({"key": q["key"], "ok": False, "error": err}) + "\n"); out.flush(); stats["fail"] += 1
async def main():
    reqs = [json.loads(l) for l in open(sys.argv[1])]; sysmsg = RULES + open(sys.argv[2]).read()
    done = set()
    if os.path.exists(sys.argv[3]):
        done = {json.loads(l)["key"] for l in open(sys.argv[3]) if l.strip() and json.loads(l).get("ok")}
    reqs = [q for q in reqs if q["key"] not in done]
    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{sys.argv[4]}/v1", api_key="EMPTY", timeout=1800)
    sem = asyncio.Semaphore(int(sys.argv[5]) if len(sys.argv) > 5 else 512); stats = {"ok": 0, "fail": 0}; t0 = time.time()
    with open(sys.argv[3], "a", encoding="utf-8") as out:
        await asyncio.gather(*[one(client, sysmsg, q, sem, out, stats) for q in reqs])
    print(f"DONE requests={len(reqs)} skipped={len(done)} ok={stats['ok']} fail={stats['fail']} wall={time.time()-t0:.0f}s", flush=True)
if __name__ == "__main__":
    asyncio.run(main())
