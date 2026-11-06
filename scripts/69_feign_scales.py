"""Feigning comparison on further scales (sixth review).

1. Log odds ratio of answering "not at all" relative to genuine patients with the same
   total: a logistic model of P(answer = 0 | item, total) is fit on the genuine patients
   (item effects plus a cubic spline in the total); a group's log odds ratio is the shift
   that best fits its answers given that model. Responses outside the genuine range of
   totals are left out. Paired bootstrap (genuine patients, the group and the feigners
   resampled in the same draw); one-sided p for "model below the feigners", Holm over models.
2. The same within the feigners' 10th-90th percentile of totals, with model responses
   reweighted to the feigners' distribution of totals (bins of 5 points).
3. Separability against the genuine-genuine baseline at the same totals: in each of 200
   random halvings of the genuine patients, a group's AUC against one half and the AUC of
   genuine patients drawn from the other half at the group's totals (feign.real_baseline);
   p = share of halvings in which the group is not more separable; Holm over models.
"""
import sys, json, glob, zlib
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer
from mental_act import feign as FG
from mental_act.figlong import TAG2LABEL
from mental_act.config import RUNS, TABLES

B = 1000
GEN, FEI = FG.GEN, FG.FEIGNER
TLO, THI = GEN.sum(1).min(), GEN.sum(1).max()
FLO, FHI = FG.FEIGNER_RANGE

GROUPS = {"human feigner": FEI}
NAME37 = {"human feigner": "human feigner"}   # group names used by scripts/37 (they set its seeds)
for f in sorted(glob.glob(str(RUNS / "pcl5__*.jsonl"))):
    tag = f.split("pcl5__")[1][:-6]
    if tag.startswith("eff") or tag not in TAG2LABEL: continue
    d = pd.DataFrame([json.loads(l) for l in open(f)]).query("status == 'ok'").drop_duplicates("trial_id")
    GROUPS[TAG2LABEL[tag]] = np.array(d[d.arm == "patient"]["ordered"].tolist(), int); NAME37[TAG2LABEL[tag]] = f"LLM: {tag}"


def long(X):
    """Item-level rows: item index, respondent total, answered 0."""
    J = X.shape[1]
    return np.tile(np.arange(J), len(X)), np.repeat(X.sum(1), J), (X == 0).ravel().astype(float)


def design(item, tot, spl):
    D = np.zeros((len(item), 19)); m = item > 0; D[np.where(m)[0], item[m] - 1] = 1
    return np.hstack([D, spl.transform(tot[:, None])])


def fit_gen(G):
    item, tot, y = long(G)
    spl = SplineTransformer(n_knots=4, degree=3, extrapolation="constant").fit(np.r_[TLO, THI, tot][:, None])
    m = LogisticRegression(C=1e4, max_iter=5000).fit(design(item, tot, spl), y)
    return lambda X: m.decision_function(design(*long(X)[:2], spl))


def shift(X, off, w=None):
    """MLE of b in logit P(0) = offset + b (Newton), optional per-row weights."""
    y = (X == 0).ravel().astype(float); w = np.ones_like(y) if w is None else np.repeat(w, X.shape[1]); b = 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-(off + b))); g = np.sum(w * (y - p)); h = np.sum(w * p * (1 - p))
        if h <= 0: break
        step = g / h; b += step
        if abs(step) < 1e-8: break
    return b


def weights(X):
    """Reweight responses inside the feigners' range to the feigners' distribution of totals."""
    edges = np.arange(FLO, FHI + 5, 5); bf = np.digitize(FEI.sum(1), edges); bx = np.digitize(X.sum(1), edges)
    pf = {k: np.mean(bf == k) for k in np.unique(bf)}; px = {k: np.mean(bx == k) for k in np.unique(bx)}
    return np.array([pf.get(k, 0) / px[k] for k in bx])


def stats(G, groups):
    f = fit_gen(G); out = {}
    for k, X in groups.items():
        Xa = X[(X.sum(1) >= TLO) & (X.sum(1) <= THI)]
        Xr = X[(X.sum(1) >= FLO) & (X.sum(1) <= FHI)]
        out[k] = (shift(Xa, f(Xa)) if len(Xa) >= 10 else np.nan,
                  shift(Xr, f(Xr), weights(Xr)) if len(Xr) >= 10 else np.nan, len(Xa), len(Xr))
    return out


point = stats(GEN, GROUPS)


def draw(s):
    r = np.random.default_rng(10_000 + s)
    G = GEN[r.integers(0, len(GEN), len(GEN))]
    gb = {k: X[r.integers(0, len(X), len(X))] for k, X in GROUPS.items()}
    return {k: v[:2] for k, v in stats(G, gb).items()}


BS = Parallel(n_jobs=22)(delayed(draw)(s) for s in range(B))
rows = []
for k in GROUPS:
    a = np.array([b[k] for b in BS], float)                     # B x 2 (all, in range)
    fe = np.array([b["human feigner"] for b in BS], float)
    r = dict(group=k, n_all=point[k][2], n_range=point[k][3], logor=point[k][0], logor_range=point[k][1])
    for j, nm in ((0, ""), (1, "_range")):
        v = a[:, j]
        r[f"logor{nm}_lo"], r[f"logor{nm}_hi"] = np.nanpercentile(v, [2.5, 97.5])
        if k != "human feigner" and not np.isnan(point[k][j]):
            d = v - fe[:, j]
            r[f"diff{nm}"] = point[k][j] - point["human feigner"][j]
            r[f"diff{nm}_p"] = float(np.nanmean(d >= 0))              # one-sided: model not below the feigners
    rows.append(r)
T = pd.DataFrame(rows)
T["none_zero"] = [bool((X == 0).sum() == 0) for X in GROUPS.values()]
for c in ("logor", "logor_lo", "logor_hi", "logor_range", "logor_range_lo", "logor_range_hi"):
    T[c.replace("logor", "or")] = np.exp(T[c])


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); m = len(p); run = 0
    for i, j in enumerate(o):
        run = max(run, min(1, (m - i) * p[j])); adj[j] = run
    return adj


M = T.group != "human feigner"
for nm in ("", "_range"):
    ok = M & T[f"diff{nm}_p"].notna()
    T.loc[ok, f"diff{nm}_p_holm"] = holm(T.loc[ok, f"diff{nm}_p"])

# ---- 3. separability against the genuine-genuine baseline at the same totals, per halving
par = Parallel(n_jobs=22)
for k, X in GROUPS.items():
    seed = zlib.crc32(NAME37[k].encode())                                              # the halvings of scripts/37
    v = np.array(par(delayed(FG._base)(X, seed + 400_000 + b) for b in range(FG.S_BASE)))
    T.loc[T.group == k, ["base_group", "base_real"]] = v.mean(0)
    T.loc[T.group == k, "base_p"] = float(np.mean(v[:, 0] <= v[:, 1]))
T.loc[M, "base_p_holm"] = holm(T.loc[M, "base_p"])
T.to_csv(TABLES / "pcl5_scales.csv", index=False)
pd.set_option("display.width", 250)
print(T.round(3).to_string(index=False))
