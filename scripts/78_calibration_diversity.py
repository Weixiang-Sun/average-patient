"""Calibration against diversity for every PHQ-9 configuration (Figure 9e; read by 56 and 60).

calibration  % of trials whose total falls in the instructed severity band
diversity    totals 5-27, as a percentage of real respondents with the same totals
Configurations: the 17 models with the default prompt, the robustness variants (other framing,
answer format or temperature), and the intervention conditions of scripts/38 (fixes.csv).
-> outputs/tables/calibration_diversity.csv
"""
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A, figlong as FL
from mental_act.config import TABLES, RUNS

TARGET = {"none": (0, 4), "mild": (5, 9), "moderate": (10, 14), "moderately_severe": (15, 19), "severe": (20, 27)}
def load(run):
    d = pd.DataFrame([json.loads(l) for l in open(RUNS / f"{run}.jsonl") if l.strip()])
    return d[d.status == "ok"].drop_duplicates("trial_id")
def calib(d):
    t = np.array(d["ordered"].tolist()).sum(1)
    return 100 * np.mean([TARGET[s][0] <= x <= TARGET[s][1] for x, s in zip(t, d.severity)])

FX = pd.read_csv(TABLES / "fixes.csv"); FX = FX[FX.cond != "-"]
FXTAG = {t: l for l, t in ((m[0], m[3]) for m in FL.MODELS) if t in set(FX.model)}
FIXES = ["hetero", "biography", "numeric", "fewshot"]     # the four changes to the one-line persona
REG = {m[0]: m[1] for m in A.ALL_MODELS}
Xh = A.human_X("phq9")
pts = []
for lab in FL.ORDER:
    dd = load(REG[lab]); X = np.array(dd["ordered"].tolist(), int)
    pts.append(dict(kind="base", label=lab, calib=calib(dd), div=A.diversity(X, 5, 27), div_h=A.diversity_same_totals(X[(X.sum(1) >= 5) & (X.sum(1) <= 27)], Xh, min_n=8)))
for run, lab in [("4om_impairment", "GPT-4o-mini"), ("4om_chart", "GPT-4o-mini"), ("g52_impairment", "GPT-5.2"),
                 ("g52_chart", "GPT-5.2"), ("peritem", "GPT-4o-mini"), ("temp00", "GPT-4o-mini"), ("temp05", "GPT-4o-mini")]:
    dd = load(run); X = np.array(dd["ordered"].tolist(), int)
    pts.append(dict(kind="variant", label=lab, calib=calib(dd), div=A.diversity(X, 5, 27), div_h=A.diversity_same_totals(X[(X.sum(1) >= 5) & (X.sum(1) <= 27)], Xh, min_n=8)))
for tag, lab in FXTAG.items():
    for _, q in FX[(FX.model == tag) & FX.cond.isin(FIXES)].iterrows():
        pts.append(dict(kind="fewshot" if q.cond == "fewshot" else "fix", label=lab, calib=q.calib, div=q.diversity, div_h=q.diversity_h))
Pt = pd.DataFrame(pts); Pt["divp"] = 100 * Pt["div"] / Pt.div_h
Pt.to_csv(TABLES / "calibration_diversity.csv", index=False)
print(Pt.groupby("kind")[["calib", "div", "divp"]].agg(["min", "max"]).round(2).to_string())
