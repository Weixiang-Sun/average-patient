"""Item-position effects in whole-form administration (item order was
randomised per trial). Residualise each response on (model, instructed
severity, item), then average the residual by the position the item was shown
at. A composition process that anchors on early answers or drifts across the
form would show up as a slope."""
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import linregress
from mental_act import analysis as A
from mental_act.config import RUNS, TABLES, PHQ9_ITEMS

recs = []
for m in A.ALL_MODELS:
    lab, run = m[0], m[1]
    if not (RUNS/f"{run}.jsonl").exists(): continue
    d = A.load_run(run)
    d = d[d.arm.isin(["naive", "coached"])]
    if "item_order" not in d or len(d) < 100: continue
    for _, r in d.iterrows():
        for pos, it in enumerate(r["item_order"]):
            recs.append(dict(model=lab, sev=r["severity"], item=it, pos=pos + 1,
                             y=r["responses"][it], trial=r["trial_id"]))
D = pd.DataFrame(recs)
D["resid"] = D.y - D.groupby(["model", "sev", "item"]).y.transform("mean")
P = D.groupby(["model", "pos"]).resid.mean().unstack()
print("mean residual by presentation position (rows: model; cols: position 1..9)")
print(P.to_string(float_format=lambda v: f"{v:+.3f}"))
out = []
for mdl, row in P.iterrows():
    lr = linregress(row.index, row.values)
    out.append(dict(model=mdl, slope_per_position=lr.slope, first_minus_rest=row.iloc[0] - row.iloc[1:].mean(),
                    p=lr.pvalue))
O = pd.DataFrame(out); O.to_csv(TABLES / "position_effects.csv", index=False)
print("\nslope of residual across positions (response units per position):")
print(O.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
print(f"\npooled slope: {linregress(D.pos, D.resid).slope:+.4f} per position "
      f"(over 8 steps = {8*linregress(D.pos, D.resid).slope:+.3f} response units)")

# anchoring: does the first-shown answer predict the rest of the form beyond item/severity?
anc = []
for mdl, g in D.groupby("model"):
    f = g[g.pos == 1].set_index("trial").resid
    rest = g[g.pos > 1].groupby("trial").resid.mean()
    j = pd.concat([f, rest], axis=1, keys=["first", "rest"]).dropna()
    if len(j) > 30: anc.append(dict(model=mdl, r_first_rest=np.corrcoef(j["first"], j["rest"])[0, 1]))
print("\nanchoring: corr(residual of first-shown item, mean residual of the rest)")
AN = pd.DataFrame(anc); AN.to_csv(TABLES / "anchoring.csv", index=False)
print(AN.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
