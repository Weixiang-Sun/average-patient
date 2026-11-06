"""Is the separability of simulated patients from genuinely injured students
(OSF PCL-5 study) just a difference in total level?

For each group (people told to feign, each model as a genuine patient), with the
same nearest-total matching and classifier as 37_triangle_ci:
  total gap     mean |total - matched genuine total|, and % matched exactly
  auc_items     the headline item-pattern AUC
  auc_total     AUC from the total alone (should be at chance if matching works)
  auc_ipsative  AUC from item scores divided by the total (profile shape only)
  auc_overlap   item AUC using only responses whose total a genuine patient has
  descriptives  breadth and "not at all" share of the group and of the matched
                genuine patients (so the comparison is at equal totals)
"""
import sys, json, glob, zlib
sys.path.insert(0, "src")
import numpy as np, pandas as pd, pyreadstat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mental_act import analysis as A
from mental_act.config import RUNS, TABLES, ROOT
DATA_EXT = ROOT / "data" / "external"
from mental_act.figlong import TAG2LABEL

df, _ = pyreadstat.read_sav(DATA_EXT / "osf_6b8jf_GOD_feigning.sav")
g = df["Groups_1_injured_2uninjuredhonest_3simulators"].to_numpy()
H = df[[f"PCL_5_NOW_{i}" for i in range(1, 21)]].to_numpy(int)
GEN = H[g == 1]; TG = GEN.sum(1)

def matched(X, rng):
    """Indices of genuine patients at the nearest total (with replacement)."""
    return A.nearest_total_match(X, GEN, rng)

def auc_ungrouped(X, idx, seed):
    """The same test with repeats of one genuine patient free to fall in different folds (the old pipeline)."""
    R = GEN[idx]; Z = np.vstack([X, R]); y = np.r_[np.ones(len(X)), np.zeros(len(R))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    p = cross_val_predict(clf, Z, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)

def auc(X, idx, feats, seed):
    R = GEN[idx]; Z = np.vstack([X, R]); y = np.r_[np.ones(len(X)), np.zeros(len(R))]
    groups = np.r_[np.arange(len(X)), len(X) + idx]          # repeats of one genuine patient stay in one fold
    F = {"items": Z, "total": Z.sum(1, keepdims=True), "ipsative": Z / np.maximum(Z.sum(1, keepdims=True), 1)}[feats]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    p = cross_val_predict(clf, F, y, groups=groups, cv=A.grouped_folds(y, groups, seed), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)

groups = {"human feigner": H[g == 3]}
for f in sorted(glob.glob(str(RUNS / "pcl5__*.jsonl"))):
    d = pd.DataFrame([json.loads(l) for l in open(f)]).query("status=='ok' and arm=='patient'").drop_duplicates("trial_id")
    groups[TAG2LABEL[f.split("pcl5__")[1][:-6]]] = np.array(d.ordered.tolist(), int)

rows = []
for name, X in groups.items():
    base = zlib.crc32(name.encode())
    Is = [matched(X, np.random.default_rng(base + s)) for s in range(10)]; Rs = [GEN[i] for i in Is]
    inside = X[np.isin(X.sum(1), TG)]
    m = lambda feats, XX=X: float(np.mean([auc(XX, Is[s] if XX is X else matched(XX, np.random.default_rng(base + s)), feats, s)
                                           for s in range(len(Is))]))
    R0 = np.vstack(Rs)
    rows.append(dict(group=name, n=len(X), total_median=float(np.median(X.sum(1))),
                     total_gap=float(np.mean([np.abs(X.sum(1) - R.sum(1)).mean() for R in Rs])),
                     exact_pct=float(np.mean([100 * (X.sum(1) == R.sum(1)).mean() for R in Rs])),
                     auc_items=m("items"), auc_total=m("total"), auc_ipsative=m("ipsative"),
                     auc_items_ungrouped=float(np.mean([auc_ungrouped(X, Is[s], s) for s in range(len(Is))])),
                     distinct_genuine=float(np.mean([len(set(i)) for i in Is])), max_reuse=float(np.mean([np.bincount(i).max() for i in Is])),
                     n_overlap=len(inside), auc_overlap=m("items", inside) if len(inside) >= 20 else np.nan,
                     breadth=(X > 0).sum(1).mean(), breadth_matched_genuine=(R0 > 0).sum(1).mean(),
                     zero=100 * (X == 0).mean(), zero_matched_genuine=100 * (R0 == 0).mean()))
T = pd.DataFrame(rows); T.to_csv(TABLES / "pcl5_feign_checks.csv", index=False)
print(f"genuine totals: median {np.median(TG):.0f}, IQR {np.percentile(TG, 25):.0f}-{np.percentile(TG, 75):.0f}, max {TG.max()}, n>=50 {int((TG >= 50).sum())}")
print(T.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
