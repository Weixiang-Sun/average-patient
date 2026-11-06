"""Item-level view of the extension instruments: which items do simulated
patients endorse, relative to real respondents at the same totals?

For each instrument, per item: endorsement rate (response > 0) among simulated
responses (pooled over models, each model weighted equally) and the
total-matched human expectation (same bins and weights as 52_extra_analysis.py).
"""
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import extra as E
from mental_act.config import RUNS, TABLES
from mental_act.figlong import MODELS

W = {"k6": 1, "ocir": 2, "phq15": 1, "itq": 1, "edeq": 4, "pcl5c": 3}
rows, pm_rows = [], []
for instr, w in W.items():
    H = E.human_X(instr); bh = H.sum(1) // w
    hc = pd.Series(bh).value_counts(); ok_bins = set(hc[hc >= 5].index)
    items = E.SPEC[instr]["items"]
    per_model_m, per_model_h, tags = [], [], []
    for _, _, _, tag in MODELS:
        f = RUNS / f"x_{instr}__{tag}.jsonl"
        if not f.exists(): continue
        d = pd.DataFrame([json.loads(l) for l in f.read_text().splitlines() if l.strip()])
        d = d[(d.status == "ok") & d.arm.isin(["naive", "coached"])].drop_duplicates("trial_id")
        if len(d) < 30: continue
        M = np.array(d["ordered"].tolist(), int); bm = M.sum(1) // w
        keep = np.isin(bm, list(ok_bins)); M, bm = M[keep], bm[keep]
        if len(M) < 30: continue
        wts = pd.Series(bm).value_counts()
        per_model_m.append((M > 0).mean(0)); tags.append(tag)
        per_model_h.append(sum(wts[b] * (H[bh == b] > 0).mean(0) for b in wts.index) / wts.sum())
    m, h = 100 * np.mean(per_model_m, 0), 100 * np.mean(per_model_h, 0)
    for j, it in enumerate(items):
        text = [s for _, _, dd in E.SPEC[instr]["sections"] for k, s in dd.items() if k == it][0]
        rows.append(dict(instrument=instr, item=it, text=text[:70], human=h[j], models=m[j], gap=m[j] - h[j]))
    for tag, v in zip(tags, per_model_m):
        pm_rows += [dict(instrument=instr, item=it, tag=tag, rate=100 * v[j]) for j, it in enumerate(items)]
T = pd.DataFrame(rows); T.to_csv(TABLES / "extra_items.csv", index=False)
pd.DataFrame(pm_rows).to_csv(TABLES / "extra_items_by_model.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 70)
for instr in ("ocir", "phq15"):
    print(f"\n=== {instr} (sorted by human endorsement) ===")
    print(T[T.instrument == instr].sort_values("human")[["text", "human", "models", "gap"]].round(1).to_string(index=False))
