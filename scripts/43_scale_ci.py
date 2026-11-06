"""Is 'larger models are less diverse' within a family more than noise?

Intervals come from subsampling 80% of responses WITHOUT replacement. A
with-replacement bootstrap duplicates responses, and two copies of one response
are at distance 0, which pulls diversity down in every resample (the same
problem as duplicated rows in cross-validated AUC)."""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A
from mental_act.config import TABLES


LADDERS = {"GPT-4.1": ["GPT-4.1-nano", "GPT-4.1-mini", "GPT-4.1"],
           "GPT-4o": ["GPT-4o-mini", "GPT-4o"], "GPT-5": ["GPT-5-mini", "GPT-5.2"],
           "Gemini-2.5": ["Gemini-2.5-Flash-Lite", "Gemini-2.5-Flash", "Gemini-2.5-Pro"]}
REG = {m[0]: m[1] for m in A.ALL_MODELS}
rows = []; boots = {}
for fam, ms in LADDERS.items():
    for lab in ms:
        X = np.array(A.load_run(REG[lab])["ordered"].tolist(), int)
        rng = np.random.default_rng(7); b = []
        for _ in range(400):
            b.append(A.diversity(X[rng.choice(len(X), int(.8 * len(X)), replace=False)], 15, 24, seed=int(rng.integers(1e6))))
        b = np.array(b); boots[lab] = b
        rows.append(dict(family=fam, model=lab, diversity=A.diversity(X, 15, 24), lo=np.nanpercentile(b, 2.5),
                         hi=np.nanpercentile(b, 97.5)))
T = pd.DataFrame(rows)
print(T.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
print("\npairwise: share of subsample draws in which the smaller model is more diverse")
pairs = []
for fam, ms in LADDERS.items():
    for a, b in zip(ms, ms[1:]):
        p = float(np.nanmean(boots[a] > boots[b])); pairs.append(dict(family=fam, smaller=a, larger=b, p_smaller_more_diverse=p))
        print(f"  {fam:8s} {a:22s} > {b:22s}  {p:.3f}")
pd.DataFrame(pairs).to_csv(TABLES / "scale_pairs.csv", index=False)
T.to_csv(TABLES / "scale_ci.csv", index=False)
