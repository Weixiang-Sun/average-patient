"""Every model, both instruments: profile measures, calibration, refusals, scale."""
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A
from mental_act.config import RUNS, TABLES

TARGET = {"none":(0,4),"mild":(5,9),"moderate":(10,14),"moderately_severe":(15,19),"severe":(20,27)}

rows = []
for instr, col in (("phq9", 1), ("gad7", 2)):
    Xh = A.human_X(instr); Hb = A.in_band(Xh, instr); lo, hi = A.INSTR[instr]["band"]
    J = A.INSTR[instr]["J"]
    hdiv = A.diversity(Xh, lo, hi)
    rows.append(dict(instrument=instr, model="REAL", vendor="-", family="-", size=0,
        n=len(Hb), breadth=A.breadth(Hb).mean(), all_items=100*(A.breadth(Hb)==J).mean(),
        uneven=A.spikiness(Hb).mean(), top=100*(Hb==3).mean(), zero=100*(Hb==0).mean(),
        diversity=hdiv, rho=1.0, auc=0.5))
    for m in A.ALL_MODELS:
        lab, run, vend, fam, size = m[0], m[col], m[3], m[4], m[5]
        if run is None: continue
        f = RUNS / f"{run}.jsonl"
        if not f.exists(): continue
        raw = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        d = pd.DataFrame(raw)
        n_all = d.trial_id.nunique()
        d = d[d.status == "ok"].drop_duplicates("trial_id")
        if len(d) < 100: continue
        X = np.array(d["ordered"].tolist(), int); M = A.in_band(X, instr)
        err = n_all - len(d)
        r = dict(instrument=instr, model=lab, vendor=vend, family=fam, size=size, n=len(d),
                 errors=err, n_band=len(M))
        if instr == "phq9":
            tot = X.sum(1)
            r["calib"] = 100*np.mean([TARGET[s][0] <= t <= TARGET[s][1] for t, s in zip(tot, d.severity)])
            ds = d[d.severity.isin(["moderately_severe","severe"])]
            r["si"] = 100*(np.array(ds["ordered"].tolist())[:, 8] > 0).mean()
        if len(M) >= 30:
            r.update(breadth=A.breadth(M).mean(), all_items=100*(A.breadth(M)==J).mean(),
                     uneven=A.spikiness(M).mean(), top=100*(M==3).mean(), zero=100*(M==0).mean(),
                     diversity=A.diversity(X, lo, hi), rho=A.order_agreement(X, Xh),
                     **dict(zip(("auc", "auc_sd"), A.separability_rep(M, Xh))))
            # real respondents at exactly this model's totals (the band alone does not fix the total)
            tt, cc = np.unique(M.sum(1), return_counts=True); th = Xh.sum(1)
            w = lambda fn: float(np.sum([fn(Xh[th == t]) * c for t, c in zip(tt, cc)]) / cc.sum())
            r.update(breadth_h=w(lambda H: A.breadth(H).mean()), all_items_h=w(lambda H: 100 * (A.breadth(H) == J).mean()),
                     zero_h=w(lambda H: 100 * (H == 0).mean()),
                     # real diversity over the same totals (each total that has >= 8 responses of this model)
                     diversity_h=A.diversity_same_totals(M, Xh, min_n=8),
                     # PHQ-9: all of items 1-8 endorsed (the all-nine rate is close to the item-9 rate for most models)
                     **(dict(all8=100 * (A.breadth(M[:, :8]) == 8).mean(), all8_h=w(lambda H: 100 * (A.breadth(H[:, :8]) == 8).mean())) if instr == "phq9" else {}))
        rows.append(r)
T = pd.DataFrame(rows); T.to_csv(TABLES / "all_models.csv", index=False)
for instr in ("phq9", "gad7"):
    print(f"\n=== {instr.upper()} ===")
    c = ["model","vendor","n","errors","n_band","breadth","all_items","uneven","zero","diversity","rho","auc"] + (["calib","si"] if instr=="phq9" else [])
    print(T[T.instrument==instr][[x for x in c if x in T]].to_string(index=False, float_format=lambda v: f"{v:6.2f}", na_rep="-"))

print("\n=== within-family scale ladders (PHQ-9) ===")
S = T[(T.instrument=="phq9") & T.family.isin(["GPT-4.1","GPT-4o","GPT-5"])].sort_values(["family","size"])
print(S[["family","model","breadth","all_items","uneven","diversity","auc","calib","si"]].to_string(index=False, float_format=lambda v: f"{v:6.2f}", na_rep="-"))
