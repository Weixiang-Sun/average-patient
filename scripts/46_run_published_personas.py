"""Administer a questionnaire with published simulator personas (TalkDep, CACTUS, CACTUS with
its case formulation).

usage: python scripts/46_run_published_personas.py PROVIDER MODEL [--gad7] [talkdep|cactus|cactus_cm ...]
Output: outputs/runs/persona_<source>__<tag>.jsonl (PHQ-9) or persona_<source>_gad7__<tag>.jsonl
(GAD-7); resumable. Reasoning is off as in the main runs (scripts/65): GPT-5-mini at
minimal, Gemini 2.5 Flash and Kimi-K2.5 at none; Gemini 2.5 Pro cannot turn it off and runs at
its minimum thinking budget (128 tokens)."""
import sys, json, time
sys.path.insert(0, "src")
from mental_act import llm_runner, published_personas as PP
from mental_act.config import RUNS

OFF = {"gpt-5-mini": "minimal", "gemini-2.5-flash": "none", "Kimi-K2.5": "none", "gemini-2.5-pro": "budget128"}
args = sys.argv[1:]; instrument = "gad7" if "--gad7" in args else "phq9"; args = [a for a in args if a != "--gad7"]
prov, model = args[0], args[1]; sources = args[2:] or ["talkdep", "cactus", "cactus_cm"]
tag = model.replace(".", "").replace("-", "").lower()
for src in sources:
    out = PP.run_name(src, tag, instrument); t0 = time.time()
    llm_runner.run(PP.build_trials(src, instrument=instrument), model=model, provider=prov, out_name=out,
                   workers=12 if prov != "gemini" or "pro" not in model else 16, reasoning_effort=OFF.get(model))
    rs = [json.loads(l) for l in (RUNS / f"{out}.jsonl").read_text().splitlines() if l.strip()]
    ok = {r["trial_id"] for r in rs if r["status"] == "ok"}
    print(f"[{tag}] {out}: {len(ok)} ok, {sum(r['status'] != 'ok' for r in rs)} error records, {time.time() - t0:.0f}s", flush=True)
