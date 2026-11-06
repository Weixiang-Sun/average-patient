"""Run a list of (instrument, provider, model, tag) jobs sequentially."""
import sys, time, json
sys.path.insert(0, "src")
from mental_act import llm_runner, pcl5, personas
from mental_act.config import RUNS

JOBS = {
 "RETRY": [("gad7","azure","gpt-4.1"),("pcl5","azure","gpt-4.1")],
 "GEMFL": [("phq9","gemini","gemini-2.5-flash-lite"),("gad7","gemini","gemini-2.5-flash-lite"),("pcl5","gemini","gemini-2.5-flash-lite")],
 "GEMF": [("phq9","gemini","gemini-2.5-flash"),("gad7","gemini","gemini-2.5-flash"),("pcl5","gemini","gemini-2.5-flash")],
 "GEMPRO": [("phq9","gemini","gemini-2.5-pro"),("gad7","gemini","gemini-2.5-pro"),("pcl5","gemini","gemini-2.5-pro")],
 "G54": [("gad7","azure","gpt-5.4-mini"),("pcl5","azure","gpt-5.4-mini")],   # deployed on Azure 2026-09-28
 "PHI": [("phq9","azure","Phi-4"),("gad7","azure","Phi-4"),("pcl5","azure","Phi-4")],
 "A": [("pcl5","azure","gpt-4o-mini"),("pcl5","azure","gpt-4o"),
       ("phq9","azure","gpt-4o"),("gad7","azure","gpt-4o"),
       ("phq9","azure","gpt-4.1-nano"),("gad7","azure","gpt-4.1-nano"),("pcl5","azure","gpt-4.1-nano"),
       ("phq9","azure","gpt-4.1-mini"),("gad7","azure","gpt-4.1-mini"),("pcl5","azure","gpt-4.1-mini"),
       ("phq9","azure","gpt-4.1"),("gad7","azure","gpt-4.1"),("pcl5","azure","gpt-4.1")],
 "B": [("pcl5","anthropic","claude-haiku-4-5"),("pcl5","azure","gpt-5.2"),
       ("phq9","azure","gpt-5-mini"),("gad7","azure","gpt-5-mini"),("pcl5","azure","gpt-5-mini"),
       ("phq9","azure","Phi-4"),("gad7","azure","Phi-4"),("pcl5","azure","Phi-4")],
 "C": [("pcl5","azure","grok-4-1-fast-non-reasoning"),("pcl5","azure","Llama-3.3-70B-Instruct"),
       ("phq9","azure","Mistral-Large-3"),("gad7","azure","Mistral-Large-3"),("pcl5","azure","Mistral-Large-3"),
       ("pcl5","azure","Kimi-K2.5")],
}
TAG = lambda inst, m: f"{inst}__{m.replace('.','').replace('-','').lower()}"

def trials(inst):
    if inst == "pcl5": return pcl5.build_trials(n_per_cell=60)
    return personas.build_trials(n_per_cell=50, instrument=inst)

job = sys.argv[1]
for inst, prov, model in JOBS[job]:
    out = TAG(inst, model); t0 = time.time()
    try:
        llm_runner.run(trials(inst), model=model, provider=prov, out_name=out,
                       workers=4 if job == "RETRY" else 12 if "kimi" not in model.lower() else 16)
        rs = [json.loads(l) for l in (RUNS/f"{out}.jsonl").read_text().splitlines() if l.strip()]
        ok = sum(r["status"] == "ok" for r in rs)
        print(f"[{job}] DONE {out}: {ok}/{len(rs)} ok in {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"[{job}] FAIL {out}: {type(e).__name__}: {e}", flush=True)
print(f"[{job}] ALL DONE", flush=True)
