"""Checks on the separability test (sixth review). No API calls.

1. Real-vs-real baselines under the nine-reference protocol of Table 1 (scripts/52 `compare`):
   NHIS GAD-7 2022 vs 2019, NHIS K6 2016 vs 2018, EDE-Q study 1 vs study 2, NHANES PHQ-9
   2017-2023 vs 2005-2010; 400 respondents from the first sample play the model, 5 seeds.
2. Wide bins (OCI-R width 2, postpartum PCL-5 width 3, EDE-Q width 4): within the same
   balanced bin samples, AUC from the exact total alone, and item AUC after subtracting the
   real respondents' item means at each exact total (so within-bin total differences cannot help).
3. One-hot classifier (one weight per answer level of each item) in place of linear item
   scores, same total matching and grouped folds as A.separability: PHQ-9 and GAD-7 models
   (totals 15-24 / 12-19), the PCL-5 feigners and the GRM-sampled profiles.
"""
import sys, json, zlib
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mental_act import analysis as A, nhis, extra as E, feign as FG, grm_sampler as G
from mental_act.config import GAD7_ITEMS, DATA_PROC, PHQ9_ITEMS, TABLES

src52 = open("scripts/52_extra_analysis.py").read().split("\nrows = []")[0]
ns = {"__name__": "s52"}; exec(compile(src52, "52_extra_analysis", "exec"), ns)
compare, REFS, load52, cv_auc52 = ns["compare"], ns["REFS"], ns["load"], ns["cv_auc"]
MODELS = [m[0] for m in ns["MODELS"]]

# ---- 1. real-vs-real baselines -------------------------------------------------------
g19 = A.human_X("gad7")
g22 = nhis.load(GAD7_ITEMS, years=[2022])[GAD7_ITEMS].to_numpy(int)
import zipfile
def k6_year(y):
    cols = ["ASISAD", "ASINERV", "ASIRSTLS", "ASIHOPLS", "ASIEFFRT", "ASIWTHLS"]
    z = zipfile.ZipFile(E.DATA_RAW / "nhis" / f"samadult{y}.zip")
    d = pd.read_csv(z.open(z.namelist()[0]), usecols=lambda c: c.upper() in cols, low_memory=False)
    d.columns = [c.upper() for c in d.columns]; d = d[cols].where(d[cols].isin([1, 2, 3, 4, 5])).dropna()
    return (5 - d).to_numpy(int)
ed = pd.read_excel(E.EXT / "edeq_contreras2022.xlsx", sheet_name="Data")
src = [f"edeq_i{i:02d}" for i in list(range(1, 13)) + list(range(19, 29))]
ed = ed[["study"] + src].dropna(); ed = ed[ed[src].isin(range(7)).all(1)]
nh = pd.read_csv(DATA_PROC / "nhanes_phq9.csv"); X9 = nh[PHQ9_ITEMS].to_numpy(int)
PAIRS = [("GAD-7: NHIS 2022 vs 2019", g22, g19, 3, 1),
         ("K6: NHIS 2016 vs 2018", k6_year(2016), k6_year(2018), 4, 1),
         ("EDE-Q: study 1 vs study 2", ed[ed.study == sorted(ed.study.unique())[0]][src].to_numpy(int), ed[ed.study == sorted(ed.study.unique())[1]][src].to_numpy(int), 6, 4),
         ("PHQ-9: NHANES 2017-2023 vs 2005-2010", X9[nh.years.isin(["2017-2018", "2021-2023"]).values], X9[nh.years.isin(["2005-2006", "2007-2008", "2009-2010"]).values], 3, 1)]
base = []
for name, Xa, Xb, K, w in PAIRS:
    Xa = Xa[Xa.sum(1) > 0]                      # the models' responses have symptoms; drop all-zero forms
    for s in range(5):
        rng = np.random.default_rng(s); M = Xa[rng.choice(len(Xa), min(400, len(Xa)), replace=False)]
        r = compare(M, Xb, K, w, 1000 + s); base.append(dict(pair=name, seed=s, auc=r.get("auc"), auc_null=r.get("auc_null"), n_a=len(Xa), n_b=len(Xb)))
B = pd.DataFrame(base).groupby("pair").agg(auc=("auc", "mean"), auc_sd=("auc", "std"), auc_null=("auc_null", "mean"), n_a=("n_a", "first"), n_b=("n_b", "first")).reset_index()
B.to_csv(TABLES / "separability_baselines.csv", index=False)
print(B.round(3).to_string(index=False))


# ---- 2. wide bins: total alone, and item AUC with the real item means at each total removed
def binned(M, H, w, rng, draws=20):
    bm, bh = M.sum(1) // w, H.sum(1) // w
    hc = pd.Series(bh).value_counts(); keep = np.isin(bm, hc[hc >= ns["MIN_H"]].index); M, bm = M[keep], bm[keep]
    th = H.sum(1); mu = {t: H[th == t].mean(0) for t in np.unique(th)}
    resid = lambda Z: Z - np.array([mu.get(t, Z[i]) for i, t in enumerate(Z.sum(1))])
    tot, res, raw = [], [], []
    for _ in range(draws):
        ia, ib = [], []
        for b in pd.Series(bm).value_counts().index:
            pm, ph = np.where(bm == b)[0], np.where(bh == b)[0]; k = min(len(pm), len(ph))
            ia += list(rng.choice(pm, k, replace=False)); ib += list(rng.choice(ph, k, replace=False))
        if len(ia) < 40: return None
        Ma, Hb = M[ia], H[ib]
        tot.append(cv_auc52(Ma.sum(1, keepdims=True).astype(float), Hb.sum(1, keepdims=True).astype(float), rng))
        res.append(cv_auc52(resid(Ma), resid(Hb), rng)); raw.append(cv_auc52(Ma, Hb, rng))
    return dict(auc_items=np.mean(raw), auc_total_only=np.mean(tot), auc_residual=np.mean(res))


wb = []
for key in ("ocir", "pcl5c", "edeq"):
    lab, dis, H, K, w, runf, arms = REFS[key]
    for m in MODELS:
        M, _ = load52(runf(m), arms)
        if M is None or len(M) < 30: continue
        r = binned(M, H, w, np.random.default_rng(zlib.crc32(f"{key}|{m}".encode())))
        if r: wb.append(dict(ref=key, model=m, **r))
W = pd.DataFrame(wb); W.to_csv(TABLES / "separability_wide_bins.csv", index=False)
print(W.groupby("ref")[["auc_items", "auc_total_only", "auc_residual"]].agg(["min", "max"]).round(2).to_string())


# ---- 3. one-hot classifier -------------------------------------------------------------
def sep_onehot(Xm, Xh, seed, K):
    rng = np.random.default_rng(seed)
    picks = A.nearest_total_match(Xm, Xh, rng)
    X = np.vstack([Xm, Xh[picks]]); y = np.r_[np.ones(len(Xm)), np.zeros(len(picks))]
    groups = np.r_[np.arange(len(Xm)), len(Xm) + picks]
    enc = OneHotEncoder(categories=[list(range(K + 1))] * X.shape[1], handle_unknown="ignore", drop="first")
    clf = make_pipeline(enc, LogisticRegression(max_iter=3000, C=0.5))
    return float(cross_val_score(clf, X, y, groups=groups, cv=A.grouped_folds(y, groups, seed), scoring="roc_auc").mean())


oh = []
for instr, K in (("phq9", 3), ("gad7", 3)):
    Xh = A.human_X(instr); Hb = A.in_band(Xh, instr)
    for m in A.ALL_MODELS:
        run = m[1] if instr == "phq9" else m[2]
        M = A.in_band(A.run_X(run), instr)
        if len(M) < 30: continue
        oh.append(dict(set=instr, group=m[0], auc_linear=A.separability_rep(M, Xh)[0],
                       auc_onehot=float(np.mean([sep_onehot(M, Xh, k, K) for k in range(10)]))))
# PCL-5 feigners against the genuine patients (feign.py classifier and matching, 50 matchings)
fe = [FG.cv_auc(FG.FEIGNER, FG.GEN, np.random.default_rng(k)) for k in range(50)]
oh.append(dict(set="pcl5", group="human feigner", auc_linear=float(np.mean(fe)),
               auc_onehot=float(np.mean([sep_onehot(FG.FEIGNER, FG.GEN, k, 4) for k in range(50)]))))
# GRM-sampled profiles against 2013-2023 respondents (totals >= 5), as in scripts/50
df = pd.read_csv(DATA_PROC / "nhanes_phq9.csv"); Te = df[df.cycle.isin(["H", "I", "J", "L"])][PHQ9_ITEMS].to_numpy(int); Te = Te[Te.sum(1) >= 5]
pool = G.sample(400_000, np.random.default_rng(7)); pool = pool[pool.sum(1) >= 5]
rng = np.random.default_rng(8); S = np.vstack([pool[pool.sum(1) == t][rng.integers(0, (pool.sum(1) == t).sum())] for t in Te[rng.choice(len(Te), 600, replace=False)].sum(1)])
oh.append(dict(set="phq9_grm", group="GRM-sampled profiles", auc_linear=A.separability_rep(S, Te)[0],
               auc_onehot=float(np.mean([sep_onehot(S, Te, k, 3) for k in range(10)]))))
O = pd.DataFrame(oh); O.to_csv(TABLES / "separability_onehot.csv", index=False)
print(O.round(3).to_string(index=False))
