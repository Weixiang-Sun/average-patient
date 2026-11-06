"""Profile measures in every severity band, with the total held fixed.

Models and people are both grouped by their actual total score (not by the
severity a model was told). Within a band, a model's totals are not distributed
like the real respondents' totals, so the real reference for each model is the
expectation over real respondents with exactly the same totals, weighted by that
model's totals. Diversity is already computed within exact totals."""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A
from mental_act.config import TABLES

Xh = A.human_X("phq9"); th = Xh.sum(1)
BANDS = [("0-4 (none)", 0, 4), ("5-9 (mild)", 5, 9), ("10-14 (moderate)", 10, 14),
         ("15-19 (mod.-severe)", 15, 19), ("20-27 (severe)", 20, 27)]
MEAS = {"breadth": lambda X: A.breadth(X).mean(), "zero": lambda X: 100 * (X == 0).mean(),
        "uneven": lambda X: A.spikiness(X).mean(), "all0": lambda X: 100 * (A.breadth(X) == 0).mean()}

def human_at(totals, fn):
    """Real-respondent value at the same totals, weighted by the model's totals."""
    t, c = np.unique(totals, return_counts=True)
    v = np.array([fn(Xh[th == x]) for x in t])
    return float(np.sum(v * c) / c.sum())

per_model, rows = [], []
for lab, lo, hi in BANDS:
    H = Xh[(th >= lo) & (th <= hi)]
    rows.append(dict(band=lab, who="NHANES", n=len(H), **{k: f(H) for k, f in MEAS.items()}, diversity=A.diversity(H, lo, hi)))
    for m in A.ALL_MODELS:
        X = A.run_X(m[1]); X = X[(X.sum(1) >= lo) & (X.sum(1) <= hi)]
        if len(X) < 20: continue
        r = dict(band=lab, model=m[0], n=len(X), diversity=A.diversity(X, lo, hi), diversity_h=A.diversity_same_totals(X, Xh, min_n=8))   # real respondents over the same totals
        for k, f in MEAS.items():
            r[k] = f(X); r[k + "_h"] = human_at(X.sum(1), f)
        per_model.append(r)
PM = pd.DataFrame(per_model); PM.to_csv(TABLES / "by_severity_models.csv", index=False)
for lab, lo, hi in BANDS:
    q = PM[PM.band == lab]
    rows.append(dict(band=lab, who="NHANES at the models' totals (median over models)", n=len(q),
                     **{k: q[k + "_h"].median() for k in MEAS}, diversity=q.diversity_h.median()))
    rows.append(dict(band=lab, who=f"LLMs (n={len(q)}) median", n=len(q), **{k: q[k].median() for k in MEAS}, diversity=q.diversity.median()))
T = pd.DataFrame(rows); T.to_csv(TABLES / "by_severity.csv", index=False)
print(T.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
print("\nmodel minus same-totals human, breadth, by band (median, range):")
for lab, _, _ in BANDS:
    q = PM[PM.band == lab]; d = q.breadth - q.breadth_h
    print(f"  {lab:20s} n={len(q):2d}  {d.median():+.2f}  [{d.min():+.2f}, {d.max():+.2f}]")
