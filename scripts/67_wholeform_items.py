"""Per-item answer distributions from the whole-form runs (all 17 models), and a control for
the item-by-item sampling argument.

  per model    entropy (bits) of each PHQ-9 item's answers across the 50 naive-arm forms of a
               severity cell, averaged over items and the four symptomatic severities
               (the format of the main experiment; no log-probabilities needed); share of
               "not at all" answers; 400 forms drawn item by item from these per-item
               answer distributions: share of distinct forms and symptoms endorsed
  real         the same estimator on 50 real respondents drawn from the matching score band,
               averaged over 200 draws (same sample size, so the plug-in bias is comparable)
  real_t       the same estimators on real respondents with the same totals as each cell's forms
               (one per form, nearest total, 200 draws); lock = share of items whose most common
               answer equals the severity level (mild 1, moderate 2, severe 3)
  real_indep   400 forms drawn item by item, independently, from real respondents' per-item
               answer distributions in each band: share of distinct forms and symptoms
               endorsed, against the real respondents themselves
-> wholeform_items.csv (models), wholeform_items_real.csv (bands)
"""
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A
from mental_act.config import RUNS, TABLES

BAND = {"mild": (5, 9), "moderate": (10, 14), "moderately_severe": (15, 19), "severe": (20, 27)}
LEVEL = {"mild": 1, "moderate": 2, "severe": 3}          # severity word -> answer position
Xh = A.human_X("phq9")

def H(col):
    p = np.bincount(col, minlength=4) / len(col); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

rng = np.random.default_rng(20261012); rows = []
for lab, run, *_ in A.ALL_MODELS:
    d = pd.DataFrame([json.loads(l) for l in open(RUNS / f"{run}.jsonl") if l.strip()])
    d = d[(d.status == "ok") & (d.arm == "naive")].drop_duplicates("trial_id")
    r = dict(model=lab)
    for s in BAND:
        X = np.array(d[d.severity == s].ordered.tolist(), int)
        r[f"H_{s}"] = np.mean([H(X[:, j]) for j in range(X.shape[1])]); r[f"n_{s}"] = len(X)
        r[f"P0_{s}"] = (X == 0).mean()
        P = np.array([np.bincount(X[:, j], minlength=4) / len(X) for j in range(9)])
        S = np.stack([rng.choice(4, size=400, p=P[j]) for j in range(9)], axis=1)
        r[f"distinct_{s}"] = 100 * len({tuple(x) for x in S}) / len(S); r[f"breadth_{s}"] = (S > 0).sum(1).mean()
    # real respondents at this cell's totals (one per form, nearest total, 200 draws): same estimators
    for s in BAND:
        X = np.array(d[d.severity == s].ordered.tolist(), int)
        if s in LEVEL: r[f"lock_{s}"] = np.mean(np.array([np.bincount(X[:, j], minlength=4) for j in range(9)]).argmax(1) == LEVEL[s])
        Rs = [Xh[A.nearest_total_match(X, Xh, rng)] for _ in range(200)]
        r[f"Ht_{s}"] = np.mean([np.mean([H(R[:, j]) for j in range(9)]) for R in Rs])
        Rp = np.vstack(Rs); r[f"P0t_{s}"] = (Rp == 0).mean()
        Pt = np.array([np.bincount(Rp[:, j], minlength=4) / len(Rp) for j in range(9)])
        St = np.stack([rng.choice(4, size=400, p=Pt[j]) for j in range(9)], axis=1); r[f"distinctt_{s}"] = 100 * len({tuple(x) for x in St}) / len(St)
        if s in LEVEL: r[f"lockt_{s}"] = np.mean(np.array([np.bincount(Rp[:, j], minlength=4) for j in range(9)]).argmax(1) == LEVEL[s])
    for k in ("H", "distinct", "breadth", "Ht", "distinctt"): r[k] = np.mean([r[f"{k}_{s}"] for s in BAND])
    for k in ("lock", "lockt"): r[k] = np.mean([r[f"{k}_{s}"] for s in LEVEL])
    r["distinct_max"] = max(r[f"distinct_{s}"] for s in BAND); rows.append(r)
M = pd.DataFrame(rows); M.to_csv(TABLES / "wholeform_items.csv", index=False)

Xh = A.human_X("phq9"); th = Xh.sum(1); out = []
for s, (lo, hi) in BAND.items():
    Hb = Xh[(th >= lo) & (th <= hi)]
    h50 = np.mean([np.mean([H(Hb[rng.choice(len(Hb), 50, replace=False)][:, j]) for j in range(9)]) for _ in range(200)])
    P = np.array([np.bincount(Hb[:, j], minlength=4) / len(Hb) for j in range(9)])
    S = np.stack([rng.choice(4, size=400, p=P[j]) for j in range(9)], axis=1)
    R = Hb[rng.choice(len(Hb), min(400, len(Hb)), replace=False)]
    out.append(dict(band=s, n_real=len(Hb), H_real50=h50, P0_real=(Hb == 0).mean(), indep_distinct=100 * len({tuple(x) for x in S}) / len(S),
                    indep_breadth=(S > 0).sum(1).mean(), real_distinct=100 * len({tuple(x) for x in R}) / len(R),
                    real_breadth=(Hb > 0).sum(1).mean()))
B = pd.DataFrame(out); B.to_csv(TABLES / "wholeform_items_real.csv", index=False)
pd.set_option("display.width", 200)
print(M.round(2).to_string(index=False)); print(B.round(2).to_string(index=False))
