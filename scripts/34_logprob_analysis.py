"""Analyse per-item response distributions read from logprobs."""
import sys, json, glob
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A
from mental_act.config import PHQ9_ITEMS, RUNS, TABLES

BAND = {"none": (0, 4), "mild": (5, 9), "moderate": (10, 14),
        "moderately_severe": (15, 19), "severe": (20, 27)}
LEVEL = {"mild": 1, "moderate": 2, "severe": 3}          # label -> Likert position
Xh = A.human_X("phq9")

def human_dist(sev):
    lo, hi = BAND[sev]; M = Xh[(Xh.sum(1) >= lo) & (Xh.sum(1) <= hi)]
    return np.array([[(M[:, j] == k).mean() for k in range(4)] for j in range(9)])

HD = {s: human_dist(s) for s in BAND}
ent = lambda P: -(P * np.log2(np.clip(P, 1e-12, 1))).sum(-1)

rows, sims = [], []
rng = np.random.default_rng(0)
for f in sorted(glob.glob(str(RUNS / "logprobs__*.jsonl"))):
    tag = f.split("logprobs__")[1][:-6]
    d = pd.DataFrame([json.loads(l) for l in open(f)]).query("status=='ok'")
    for frame in ("label", "impairment"):
        for sev in BAND:
            s = d[(d.frame == frame) & (d.severity == sev)]
            if len(s) < 9: continue
            P = np.array([s[s.item == it].p.iloc[0] for it in PHQ9_ITEMS])   # 9 x 4
            H = HD[sev]
            mode = P.argmax(1)
            rows.append(dict(model=tag, frame=frame, severity=sev,
                ent_model=ent(P).mean(), ent_human=ent(H).mean(),
                p0_model=P[:, 0].mean(), p0_human=H[:, 0].mean(),
                maxprob=P.max(1).mean(),
                label_lock=(np.mean(mode == LEVEL[sev]) if sev in LEVEL else np.nan),
                same_mode_all_items=float(len(set(mode)) == 1),
                exp_model_sd=(P @ np.arange(4)).std(),      # spread of expected score across items
                exp_human_sd=(H @ np.arange(4)).std()))
            # independent per-item sampling -> simulated profiles
            if frame == "label" and sev != "none":
                S = np.array([[rng.choice(4, p=P[j] / P[j].sum()) for j in range(9)] for _ in range(400)])
                sims.append(dict(model=tag, severity=sev, breadth=A.breadth(S).mean(),
                                 uneven=A.spikiness(S).mean(),
                                 distinct=len({tuple(r) for r in S}) / len(S)))
R = pd.DataFrame(rows); R.to_csv(TABLES / "logprob_mechanism.csv", index=False)

print("=== per-item distributions, 'label' framing, averaged over models ===")
g = R[R.frame == "label"].groupby("severity")[["ent_model","ent_human","p0_model","p0_human",
                                                 "maxprob","label_lock","same_mode_all_items",
                                                 "exp_model_sd","exp_human_sd"]].mean()
print(g.reindex(list(BAND)).to_string(float_format=lambda v: f"{v:7.3f}"))

print("\n=== label-lock rate by model (share of items whose modal answer = severity word position) ===")
ll = R[(R.frame == "label") & R.severity.isin(LEVEL)].groupby("model").label_lock.mean()
lli = R[(R.frame == "impairment") & R.severity.isin(LEVEL)].groupby("model").label_lock.mean()
print(pd.DataFrame({"label framing": ll, "impairment framing": lli}).to_string(float_format=lambda v: f"{v:7.3f}"))

print("\n=== entropy per item (bits): model vs human, 'label' framing ===")
e = R[R.frame == "label"].groupby("model")[["ent_model","ent_human"]].mean()
print(e.to_string(float_format=lambda v: f"{v:7.3f}"))

print("\n=== profiles sampled independently from per-item distributions (no composition) ===")
Sd = pd.DataFrame(sims)
print(Sd.groupby("severity")[["breadth","uneven","distinct"]].mean().reindex(["mild","moderate","moderately_severe","severe"])
      .to_string(float_format=lambda v: f"{v:7.3f}"))
hrows = []
for sev in ("mild","moderate","moderately_severe","severe"):
    lo, hi = BAND[sev]; M = Xh[(Xh.sum(1) >= lo) & (Xh.sum(1) <= hi)]
    hrows.append(dict(severity=sev, breadth=A.breadth(M).mean(), uneven=A.spikiness(M).mean(),
                      distinct=len({tuple(r) for r in M[:400]}) / min(400, len(M)),
                      # real analogue of label lock: items whose most common answer equals the severity level
                      label_lock=float(np.mean(HD[sev].argmax(1) == LEVEL[sev])) if sev in LEVEL else np.nan))
    print(f"   human {sev:18s} breadth={hrows[-1]['breadth']:.2f} uneven={hrows[-1]['uneven']:.3f} distinct={hrows[-1]['distinct']:.3f}")
Sd.to_csv(TABLES / "logprob_sampled_profiles.csv", index=False)
pd.DataFrame(hrows).to_csv(TABLES / "logprob_human_profiles.csv", index=False)
