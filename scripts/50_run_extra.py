"""Administer the extension instruments to one model.

usage: python scripts/50_run_extra.py PROVIDER MODEL [instr ...] [--limit N] [--workers W]
Output: outputs/runs/x_<instr>__<modeltag>.jsonl (resumable; failed trials are retried).
"""
import argparse, json, sys, time
sys.path.insert(0, "src")
from mental_act import extra, llm_runner
from mental_act.config import RUNS

p = argparse.ArgumentParser()
p.add_argument("provider"); p.add_argument("model"); p.add_argument("instr", nargs="*")
p.add_argument("--limit", type=int); p.add_argument("--workers", type=int, default=8)
p.add_argument("--n", type=int, default=50)
a = p.parse_args()
tag = a.model.replace(".", "").replace("-", "").lower()
for instr in (a.instr or list(extra.SPEC)):
    out = f"x_{instr}__{tag}"; t0 = time.time()
    try:
        llm_runner.run(extra.build_trials(instr, n_per_cell=a.n), model=a.model,
                       provider=a.provider, out_name=out, limit=a.limit, workers=a.workers)
        rs = [json.loads(l) for l in (RUNS / f"{out}.jsonl").read_text().splitlines() if l.strip()]
        ok = {r["trial_id"] for r in rs if r["status"] == "ok"}
        errs = [r.get("error", "")[:90] for r in rs if r["status"] != "ok"][-2:]
        print(f"[{tag}] {out}: {len(ok)} ok trials, {sum(r['status']!='ok' for r in rs)} error records, "
              f"{time.time()-t0:.0f}s {errs}", flush=True)
    except Exception as e:
        print(f"[{tag}] FAIL {out}: {type(e).__name__}: {e}", flush=True)
print(f"[{tag}] ALL DONE", flush=True)
