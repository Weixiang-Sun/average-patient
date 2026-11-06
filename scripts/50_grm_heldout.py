"""GRM parameters for the sampled-profile experiment, fit on held-out data.

Fit the PHQ-9 GRM on NHANES 2005-2012 only (cycles D-G) and check the sampled
profiles against the 2013-2023 respondents (cycles H-L), which the fit never saw.
Writes data/processed/grm_heldout.json (used by grm_sampler) and
outputs/tables/grm_heldout.csv. Profiles are drawn to match the held-out
total-score distribution exactly (totals >= 5), so every comparison is at the same totals.
"""
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A, irt, grm_sampler as G
from mental_act.config import DATA_PROC, TABLES, PHQ9_ITEMS

FIT, TEST = ("D", "E", "F", "G"), ("H", "I", "J", "L")
df = pd.read_csv(DATA_PROC / "nhanes_phq9.csv")
Xfit = df[df.cycle.isin(FIT)][PHQ9_ITEMS].to_numpy(int); Xte = df[df.cycle.isin(TEST)][PHQ9_ITEMS].to_numpy(int)
fit = irt.fit_grm(Xfit, n_cat=4, max_iter=100)
json.dump({"fit_cycles": list(FIT), "fit_years": "2005-2012", "n_fit": len(Xfit), "test_cycles": list(TEST), "n_test": len(Xte),
           "loglik": float(fit["loglik"]), "grm_a": fit["a"].tolist(), "grm_b": fit["b"].tolist()},
          open(DATA_PROC / "grm_heldout.json", "w"), indent=1)

def matched(Xsrc, t_target, rng):
    """Rows of Xsrc drawn without replacement to reproduce the target total counts."""
    ts = Xsrc.sum(1); out = []
    for t, c in zip(*np.unique(t_target, return_counts=True)):
        P = Xsrc[ts == t]; out.append(P[rng.choice(len(P), c, replace=False)])
    return np.vstack(out)

rng = np.random.default_rng(20261012)
Te = Xte[Xte.sum(1) >= 5]; tt = Te.sum(1)
pool = G.sample(2_000_000, rng, fit["a"], fit["b"])
Grm = matched(pool, tt, rng)
Fr = Xfit[Xfit.sum(1) >= 5]
def row(name, X, reweight):
    t = X.sum(1); auc, sd = A.separability_rep(X, Te) if X is not Te else (np.nan, np.nan)
    # the 2005-2012 respondents are reweighted to the held-out total distribution
    w = np.array([np.mean(tt == u) / np.mean(t == u) for u in t]) if reweight else np.ones(len(X))
    avg = lambda v: float(np.average(v, weights=w))
    return dict(source=name, n=len(X), breadth=avg(A.breadth(X)), all9=avg(100 * (A.breadth(X) == 9)), zero=avg(100 * (X == 0).mean(1)),
                si=avg(100 * (X[:, 8] > 0)), diversity=A.diversity(X, 5, 27), auc=auc, auc_sd=sd)
T = pd.DataFrame([row("held-out 2013-2023", Te, False), row("GRM fit 2005-2012", Grm, False), row("real 2005-2012", Fr, True)])
T.to_csv(TABLES / "grm_heldout.csv", index=False)
print(f"fit on {len(Xfit)} ({', '.join(FIT)}), held out {len(Xte)} ({', '.join(TEST)}); a = {np.round(fit['a'], 2)}")
print(T.round(3).to_string(index=False))
