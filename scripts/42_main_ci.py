"""Bootstrap 95% CIs for the headline descriptive numbers (no cross-validation
involved, so resampling with replacement is valid here)."""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A
from mental_act.config import RUNS, TABLES

B = 1000
def ci(fn, X, seed=0):
    rng = np.random.default_rng(seed); n = len(X)
    v = [fn(X[rng.integers(0, n, n)]) for _ in range(B)]
    return fn(X), *np.percentile(v, [2.5, 97.5])

def div(X, lo, hi, k=200):
    rng = np.random.default_rng(1); out = []
    for t in range(lo, hi + 1):
        P = X[X.sum(1) == t]
        if len(P) < 5: continue
        P = P[rng.choice(len(P), min(len(P), k), replace=False)]
        dm = np.abs(P[:, None] - P[None]).sum(-1); out.append(dm[np.triu_indices(len(P), 1)].mean())
    return float(np.mean(out)) if out else np.nan

rows = []
for instr, col in (("phq9", 1), ("gad7", 2)):
    J = A.INSTR[instr]["J"]; lo, hi = A.INSTR[instr]["band"]
    Xh = A.human_X(instr); Hb = A.in_band(Xh, instr)
    groups = [("REAL", Hb, Xh)]
    for m in A.ALL_MODELS:
        run = m[col]
        if run and (RUNS / f"{run}.jsonl").exists():
            d = A.load_run(run)
            if len(d) >= 100:
                X = np.array(d["ordered"].tolist(), int); groups.append((m[0], A.in_band(X, instr), X))
    for lab, M, full in groups:
        if len(M) < 30: continue
        r = dict(instrument=instr, model=lab, n=len(M))
        for name, fn in [("breadth", lambda X: A.breadth(X).mean()),
                         ("all_items", lambda X: 100 * (A.breadth(X) == J).mean()),
                         ("zero", lambda X: 100 * (X == 0).mean()),
                         ("uneven", lambda X: A.spikiness(X).mean())]:
            est, l, h = ci(fn, M); r.update({name: est, f"{name}_lo": l, f"{name}_hi": h})
        if instr == "phq9":   # SI endorsement at the same totals as every other measure (band 15-24)
            est, l, h = ci(lambda X: 100 * (X[:, 8] > 0).mean(), M); r.update(si=est, si_lo=l, si_hi=h)
            # every one of items 1-8 endorsed (the measure the text and the interviews use)
            est, l, h = ci(lambda X: 100 * (A.breadth(X[:, :8]) == 8).mean(), M); r.update(all8=est, all8_lo=l, all8_hi=h)
        rows.append(r)
T = pd.DataFrame(rows); T.to_csv(TABLES / "main_ci.csv", index=False)
fmt = lambda r, k: f"{r[k]:6.2f} [{r[k+'_lo']:.2f},{r[k+'_hi']:.2f}]"
for instr in ("phq9", "gad7"):
    print(f"\n=== {instr.upper()} (in-band), estimate [95% bootstrap CI] ===")
    for _, r in T[T.instrument == instr].iterrows():
        extra = f"  SI {r['si']:5.1f}% [{r['si_lo']:.1f},{r['si_hi']:.1f}]" if instr == "phq9" and pd.notna(r.get("si")) else ""
        print(f"{r['model']:18s} n={r['n']:5d}  breadth {fmt(r,'breadth')}  all {r['all_items']:5.1f}% [{r['all_items_lo']:.1f},{r['all_items_hi']:.1f}]  zero {r['zero']:5.1f}% [{r['zero_lo']:.1f},{r['zero_hi']:.1f}]{extra}")
