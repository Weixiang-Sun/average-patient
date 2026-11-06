"""How separable are two REAL samples under the same test used for the models?

Same pipeline as the model comparison (A.separability: total-matched 1:1,
5-fold logistic regression on the item pattern), same sample size as a typical
model (250 responses in the band), 10 draws each.
  - NHIS PHQ-8 vs NHANES items 1-8: same construct, different survey and mode
  - NHANES 2005-2010 vs 2017-2023: same survey, a decade apart
  - NHANES vs NHANES: random draws from one population (chance level)
"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A
from mental_act.config import DATA_PROC, PHQ9_ITEMS, TABLES

nh = pd.read_csv(DATA_PROC / "nhanes_phq9.csv"); X9 = nh[PHQ9_ITEMS].to_numpy(int)
p8 = pd.read_csv(DATA_PROC / "nhis_phq8.csv"); P8 = p8[[f"PHQ8{i}_A" for i in range(1, 9)]].to_numpy(int)
band = lambda X, lo, hi: X[(X.sum(1) >= lo) & (X.sum(1) <= hi)]
N = 250

def run(name, A_, B_, draws=10):
    a = []
    for k in range(draws):
        rng = np.random.default_rng(k)
        a.append(A.separability(A_[rng.choice(len(A_), min(N, len(A_)), replace=False)], B_, seed=k))
    return dict(comparison=name, n=min(N, len(A_)), auc=float(np.mean(a)), auc_sd=float(np.std(a, ddof=1)))

rows = []
# PHQ-8 has no item 9; PHQ-8 totals 12-21 cover roughly the PHQ-9 band 15-24
rows.append(run("NHIS PHQ-8 vs NHANES items 1-8 (different survey and mode)", band(P8, 12, 21), band(X9[:, :8], 12, 21)))
early = X9[nh.years.isin(["2005-2006", "2007-2008", "2009-2010"]).values]
late = X9[nh.years.isin(["2017-2018", "2021-2023"]).values]
rows.append(run("NHANES 2017-2023 vs 2005-2010 (same survey, a decade apart)", A.in_band(late, "phq9"), A.in_band(early, "phq9")))
Hb = A.in_band(X9, "phq9"); rng = np.random.default_rng(99); idx = rng.permutation(len(Hb))
rows.append(run("NHANES vs NHANES (one population; chance)", Hb[idx[: len(Hb) // 2]], Hb[idx[len(Hb) // 2:]]))
T = pd.DataFrame(rows); T.to_csv(TABLES / "human_baselines.csv", index=False)
print(T.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
