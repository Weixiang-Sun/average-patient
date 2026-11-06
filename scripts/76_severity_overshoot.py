"""Do simulated patients present at the severity they are told? PHQ-9 totals by instructed severity.

For each model, arm (naive, coached) and instructed severity, the share of whole-form PHQ-9 totals
inside the matching band (mild 5-9, moderate 10-14, moderately severe 15-19, severe 20-27), above
it and below it, and the mean and median total. framing = label (the persona names the severity) for every
model; framing = impairment (daily-life impact only, no severity word) for the two robustness runs.
-> outputs/tables/severity_overshoot.csv
"""
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A
from mental_act.config import TABLES, RUNS

CUT = {"mild": (5, 9), "moderate": (10, 14), "moderately_severe": (15, 19), "severe": (20, 27)}
IMPAIRMENT = {"GPT-4o-mini": "4om_impairment", "GPT-5.2": "g52_impairment"}
rows = []
runs = [(lab, "label", A.load_run(run)) for lab, run, *_ in A.ALL_MODELS]
runs += [(lab, "impairment", pd.DataFrame([json.loads(l) for l in open(RUNS / f"{f}.jsonl")])) for lab, f in IMPAIRMENT.items()]
for lab, framing, d in runs:
    d = d[d.status == "ok"]
    for arm in ("naive", "coached"):
        for sev, (lo, hi) in CUT.items():
            t = np.array([sum(o) for o in d[(d.arm == arm) & (d.severity == sev)].ordered])
            if not len(t): continue
            rows.append(dict(model=lab, framing=framing, arm=arm, severity=sev, n=len(t), mean_total=t.mean(), median_total=np.median(t),
                             in_band=100 * np.mean((t >= lo) & (t <= hi)), above=100 * np.mean(t > hi), below=100 * np.mean(t < lo)))
R = pd.DataFrame(rows); R.to_csv(TABLES / "severity_overshoot.csv", index=False)
pd.set_option("display.width", 200)
print(R[R.framing == "impairment"].groupby(["model", "arm", "severity"])[["median_total", "above"]].first().to_string())
R = R[R.framing == "label"]
print(R.groupby(["arm", "severity"])[["median_total", "in_band", "above", "below"]].agg(["min", "median", "max"]).round(0).to_string())
nm = R[(R.severity != "severe")].groupby(["arm", "model"]).above.mean()
print("above the band, mild to moderately severe, mean over the three bands per model:\n", nm.groupby("arm").agg(["min", "median", "max"]).round(0))
