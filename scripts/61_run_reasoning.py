"""Reasoning effort as a variable (review W6): every questionnaire experiment that
GPT-5-mini, Gemini 2.5 Flash or Kimi-K2.5 took part in, at several reasoning efforts.
Since 2026-10-08 the main runs have reasoning off (GPT-5-mini minimal, Gemini 2.5 Flash
and Kimi-K2.5 none; scripts/65_reasoning_off_main.py swaps them in) and the API-default
runs are kept as eff__<exp>__<tag>__default.

usage: python scripts/61_run_reasoning.py PROVIDER MODEL EFFORT EXP [EXP ...]
  EXP: phq9 gad7 pcl5 x_k6 x_ocir x_phq15 x_itq x_edeq x_pcl5c fixes persona_<talkdep|cactus|cactus_cm>[_gad7]
  EFFORT: an OpenAI reasoning effort, or budget<N> for a Gemini thinking budget of N tokens
Output: outputs/runs/eff__<exp>__<tag>__<effort>.jsonl (resumable). The eff__ prefix keeps
these runs out of every analysis that globs the main runs."""
import sys, json
sys.path.insert(0, "src")
from mental_act import llm_runner, personas, pcl5, extra, fixes, published_personas as PP
from mental_act.config import RUNS

def trials(exp):
    if exp in ("phq9", "gad7"): return personas.build_trials(n_per_cell=50, instrument=exp)
    if exp == "pcl5": return pcl5.build_trials(n_per_cell=60)
    if exp.startswith("x_"): return extra.build_trials(exp[2:], n_per_cell=50)
    if exp == "fixes": return fixes.build_trials(["label", "hetero", "numeric", "biography", "fewshot"], n_per_cell=40)
    if exp.startswith("persona_"):   # persona_<source>[_gad7]
        src = exp.split("_", 1)[1]; gad = src.endswith("_gad7")
        return PP.build_trials(src[:-5] if gad else src, instrument="gad7" if gad else "phq9")
    raise ValueError(exp)

prov, model, effort, exps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
tag = model.replace(".", "").replace("-", "").lower()
for exp in exps:
    t = trials(exp); out = f"eff__{exp}__{tag}__{effort}"
    if exp == "fixes":   # few-shot exemplars, for the copy rate
        (RUNS / f"{out}.meta.json").write_text(json.dumps({x["trial_id"]: x["meta"] for x in t}))
    llm_runner.run(t, model=model, provider=prov, out_name=out, workers=12, reasoning_effort=effort)
    rs = [json.loads(l) for l in (RUNS / f"{out}.jsonl").read_text().splitlines() if l.strip()]
    print(f"{out}: {len({r['trial_id'] for r in rs if r['status'] == 'ok'})}/{len(t)} ok", flush=True)
