"""Three-pole comparison on the PCL-5.

Humans (OSF 6b8jf): genuinely injured / uninjured honest / instructed feigners.
LLMs, same instrument and roles: patient / honest / feigner.

Question: is an LLM playing a patient closer to genuine patients than a human
who is deliberately faking is? We report profile measures, the PCL-5 Rare Items
validity index, and total-matched separability from genuine patients.
"""
import sys, json, glob, re
sys.path.insert(0, "src")
import numpy as np, pandas as pd, pyreadstat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from mental_act import analysis as A
from mental_act.config import RUNS, TABLES, ROOT
from mental_act.pcl5 import PCL5_ITEMS

J, TOP = 20, 4
df, _ = pyreadstat.read_sav(ROOT/"data/external/osf_6b8jf_GOD_feigning.sav")
grp = df["Groups_1_injured_2uninjuredhonest_3simulators"].to_numpy()
H = df[[f"PCL_5_NOW_{i}" for i in range(1, 21)]].to_numpy(int)
HUM = {"human genuine": H[grp == 1], "human honest": H[grp == 2],
       "human feigner": H[grp == 3]}
GEN = HUM["human genuine"]

def pri(X):      # PCL-5 Rare Items, scoring reverse-engineered from the dataset
    i = lambda k: X[:, k-1]
    return ((i(3)==4).astype(int)+(i(8)>=3)+(i(9)==4)+(i(10)==4)+(i(15)==4)+(i(16)>=3))

# verify PRI reconstruction against the dataset's own total
assert np.array_equal(pri(H), df["PCL_5_NOW_PRI_total"].to_numpy(int)), "PRI mismatch"
_flag = df[[c for c in df.columns if c.startswith("PCL_5_NOW_PRI_over")][0]].to_numpy(int)
PRI_CUT = min(k for k in range(1, 7) if np.array_equal((pri(H) >= k).astype(int), _flag))
print(f"PRI reconstruction verified on {len(H)} respondents; over-report cut-off = PRI >= {PRI_CUT}\n")

def load_llm():
    out = {}
    for f in sorted(glob.glob(str(RUNS/"pcl5__*.jsonl"))):
        tag = f.split("pcl5__")[1].replace(".jsonl", "")
        rs = [json.loads(l) for l in open(f) if l.strip()]
        d = pd.DataFrame(rs).query("status=='ok'").drop_duplicates("trial_id")
        for arm in ("patient", "feigner", "honest"):
            s = d[d.arm == arm]
            if len(s) >= 20:
                out[(tag, arm)] = np.array(s["ordered"].tolist(), int)
    return out

def profile(X):
    b = (X > 0).sum(1)
    dep = np.array([r[r > 0].mean() if (r > 0).any() else np.nan for r in X])
    return dict(n=len(X), total=X.sum(1).mean(), breadth=b.mean(),
                all20=100*(b == J).mean(), depth=np.nanmean(dep),
                uneven=X.std(1).mean(), top=100*(X == TOP).mean(),
                zero=100*(X == 0).mean(), pri=pri(X).mean(),
                pri_flag=100*(pri(X) >= PRI_CUT).mean())

def matched(X, ref, rng):
    """1:1 nearest-total draw from `ref` to X's totals (with replacement)."""
    return ref[A.nearest_total_match(X, ref, rng)]

def sep_auc(X, ref, seed=0):
    rng = np.random.default_rng(seed)
    idx = A.nearest_total_match(X, ref, rng); R = ref[idx]
    Z = np.vstack([X, R]); y = np.r_[np.ones(len(X)), np.zeros(len(R))]
    groups = np.r_[np.arange(len(X)), len(X) + idx]          # a genuine patient drawn twice stays in one fold
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    k = max(3, min(5, int(min(len(X), len(set(idx))) // 4)))
    return float(cross_val_score(clf, Z, y, groups=groups, cv=A.grouped_folds(y, groups, seed, k), scoring="roc_auc").mean())

def diversity(X, ref=None):
    """Mean pairwise L1 distance among responses, residualised on total score:
    compared within total-score bins of width 4 to hold severity fixed."""
    out = []
    t = X.sum(1)
    for lo in range(0, 81, 4):
        P = X[(t >= lo) & (t < lo + 4)]
        if len(P) < 4: continue
        dm = np.abs(P[:, None] - P[None]).sum(-1)
        out.append(dm[np.triu_indices(len(P), 1)].mean())
    return float(np.mean(out)) if out else np.nan

rows = []
for name, X in HUM.items():
    r = profile(X); r.update(group=name, source="human",
        sep_vs_genuine=np.nan if name == "human genuine" else sep_auc(X, GEN),
        rho_vs_genuine=1.0 if name == "human genuine" else
            spearmanr(X.mean(0), GEN.mean(0)).statistic,
        diversity=diversity(X)); rows.append(r)

for (tag, arm), X in load_llm().items():
    r = profile(X); r.update(group=f"{tag} / {arm}", source="llm", model=tag, arm=arm,
        sep_vs_genuine=sep_auc(X, GEN),
        rho_vs_genuine=spearmanr(X.mean(0), GEN.mean(0)).statistic,
        diversity=diversity(X)); rows.append(r)

T = pd.DataFrame(rows)
T.to_csv(TABLES/"pcl5_triangle.csv", index=False)
cols = ["group","n","total","breadth","all20","uneven","top","zero","pri","pri_flag",
        "diversity","rho_vs_genuine","sep_vs_genuine"]
print(T[cols].to_string(index=False, float_format=lambda v: f"{v:6.2f}"))

print("\n=== summary: separability from genuine patients (AUC; 0.5 = indistinguishable) ===")
hf = T[T.group == "human feigner"].sep_vs_genuine.iloc[0]
print(f"  human feigner            {hf:.2f}")
for arm in ("patient", "feigner", "honest"):
    s = T[(T.source == "llm") & (T.arm == arm)].sep_vs_genuine
    if len(s): print(f"  LLM {arm:9s} (n={len(s)} models)  median {s.median():.2f}  range {s.min():.2f}-{s.max():.2f}")
