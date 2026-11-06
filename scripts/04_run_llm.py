"""Step 4: administer the PHQ-9 to the LLM.  *** THE EXPERIMENT ***

Not run during preparation. Invoke explicitly:
    uv run python scripts/04_run_llm.py --n-per-cell 50
Resumable: re-running skips trial_ids already present in the JSONL log.
"""
import argparse, sys
sys.path.insert(0, "src")
from mental_act import llm_runner
from mental_act.personas import build_trials

p = argparse.ArgumentParser()
p.add_argument("--model", default="gpt-4o-mini")
p.add_argument("--provider", default="openai", choices=["openai","anthropic","azure"])
p.add_argument("--n-per-cell", type=int, default=50)
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--limit", type=int, default=None, help="cap trials (smoke test)")
p.add_argument("--out", default="pilot_gpt4omini")
p.add_argument("--mode", default="block", choices=["block","peritem"])
p.add_argument("--style", default="label", choices=["label","impairment","chart"])
p.add_argument("--workers", type=int, default=12)
p.add_argument("--instrument", default="phq9", choices=["phq9","gad7"])
a = p.parse_args()

trials = build_trials(n_per_cell=a.n_per_cell, mode=a.mode, style=a.style, instrument=a.instrument)
print(f"{len(trials)} trials, model={a.model}, temp={a.temperature}")
path = llm_runner.run(trials, model=a.model, provider=a.provider,
                      temperature=a.temperature,
                      out_name=a.out, limit=a.limit, workers=a.workers)
print(f"wrote {path}")
