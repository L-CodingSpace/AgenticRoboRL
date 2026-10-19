#!/usr/bin/env python3
"""Rewrite v2, step 4: 27B judges each rewritten turn with the 4 questions of the 60-turn review; only OK/MINOR become targets.
Input: checked.jsonl lines with decision == "to_judge" (or any line with before/after when --all). Resumable (appends, skips done keys).
Usage: judge_client.py <checked.jsonl> <student_system_prompt.txt> <out.jsonl> <port> [concurrency] [think]"""
import asyncio, json, os, sys, time
from openai import AsyncOpenAI
RULES = """You audit training data for a STUDENT robot-planning model. A teacher agent wrote the BEFORE turn while it could read HIDDEN material:
memory notes (memory/*.md, feedback_* files), seed-0 reference results/recipes/audits, and records of the authors' earlier attempts.
The student sees only: the system prompt quoted at the end, this episode's tool results and its own tool calls. An editor removed the hidden
material, giving the AFTER turn, which will be trained as the student's own reasoning.

Judge the AFTER turn (compare with BEFORE to know what was hidden):
1. fabrication: AFTER asserts something the student could not know: a result observed in another run or in seed 0 stated as fact
   ("this descended to 0.960 successfully", "grasping lower gives qpos=0.006"), seed-0 coordinates, parameter values or budgets
   that are not in the system prompt or tool results. A plan or guess ("I will try z=0.96") is fine.
2. leftover: any remaining hint of hidden material: source words or synonyms ("the guidance", "the solution", "proven", "documented",
   "the reference", "as before", "they did", "that run"), memory file names, recipe step lists left without their header.
3. coherence: dangling references whose antecedent was deleted ("that's the site center", "similar", "they"), or the tool call of this
   turn is no longer motivated by the reasoning.
4. over-deletion: legitimate reasoning (from the system prompt, tool results or own derivation) was deleted and the turn is hollow.
Verdict: "OK" (clean), "MINOR" (small blemish, still fine to train on), "BAD" (any fabrication, any leftover hint of hidden material,
or a coherence break that changes the meaning). Be strict: when unsure between MINOR and BAD, choose BAD.

=== STUDENT SYSTEM PROMPT ===
"""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["issues", "verdict", "reason"], "properties": {
    "issues": {"type": "array", "items": {"type": "string", "enum": ["fabrication", "leftover", "coherence", "over-deletion"]}},
    "verdict": {"type": "string", "enum": ["OK", "MINOR", "BAD"]}, "reason": {"type": "string"}}}
async def one(client, sysmsg, r, sem, out, stats, think):
    async with sem:
        user = json.dumps({"BEFORE": r["before"], "AFTER": r["after"],
                           "tool_calls_this_turn": [{"name": t["name"], "arguments": json.dumps(t["arguments"], ensure_ascii=False)[:600]} for t in r.get("tool_calls") or []]},
                          ensure_ascii=False)
        err = ""
        for attempt in range(3):
            t0 = time.time()
            try:
                kw = {"messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}], "model": "qwen27", "temperature": 0,
                      "max_tokens": 16384 if think else 2048,
                      "response_format": {"type": "json_schema", "json_schema": {"name": "judge", "schema": SCHEMA, "strict": True}},
                      "extra_body": {"chat_template_kwargs": {"enable_thinking": bool(think)}}}
                x = await client.chat.completions.create(**kw)
                parsed = json.loads(x.choices[0].message.content)
                out.write(json.dumps({"key": r["key"], "ok": True, **parsed, "usage": x.usage.model_dump() if x.usage else None,
                                      "latency_s": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n"); out.flush(); stats["ok"] += 1; return
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:300]
        out.write(json.dumps({"key": r["key"], "ok": False, "error": err}) + "\n"); out.flush(); stats["fail"] += 1
async def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    rows = [r for r in rows if r.get("decision") == "to_judge" or (os.environ.get("JUDGE_ALL") and r.get("after"))]
    sysmsg = RULES + open(sys.argv[2]).read(); done = set()
    if os.path.exists(sys.argv[3]):
        done = {json.loads(l)["key"] for l in open(sys.argv[3]) if l.strip() and json.loads(l).get("ok")}
    rows = [r for r in rows if r["key"] not in done]
    think = len(sys.argv) > 6 and sys.argv[6] == "think"
    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{sys.argv[4]}/v1", api_key="EMPTY", timeout=3600)
    sem = asyncio.Semaphore(int(sys.argv[5]) if len(sys.argv) > 5 else 512); stats = {"ok": 0, "fail": 0}; t0 = time.time()
    with open(sys.argv[3], "a", encoding="utf-8") as out:
        await asyncio.gather(*[one(client, sysmsg, r, sem, out, stats, think) for r in rows])
    print(f"DONE judged={len(rows)} skipped={len(done)} ok={stats['ok']} fail={stats['fail']} wall={time.time()-t0:.0f}s", flush=True)
if __name__ == "__main__":
    asyncio.run(main())
