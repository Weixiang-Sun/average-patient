"""Generality: the flat-profile signature on every instrument with a public
item-level human reference, under ONE total-matched protocol.

For each (reference, model):
  * simulated responses = the non-honest arms (naive + coached, all severities);
    for the OSF PCL-5 reference, the 'patient' role
  * totals are binned (exact totals for narrow scales, width-w bins for wide ones);
    only simulated responses whose bin holds >= MIN_H real respondents are kept
    ("coverage" reports the share kept)
  * human values are the expectation over humans in the same bins, weighted by the
    model's bin counts -- i.e. what 1:1 total matching would give, without the
    duplicate draws that sampling with replacement introduces
  * diversity = mean pairwise L1 within a bin, per item, averaged over bins where
    both groups have >= 8 responses (weighted by model counts); div_bins and div_n
    record how many bins and simulated responses that is
  * separability = 5-fold CV logistic AUC on a balanced, bin-matched sample drawn
    WITHOUT replacement, averaged over draws; a label-shuffled run of the same
    pipeline gives its chance level
The OSF PCL-5 reference (69 injured students) is not in this table: simulated
patients land at totals almost no real respondent reaches, so matching keeps
too few; that comparison is analysed separately (37_triangle_ci.py).
Measures are normalised so instruments can share a table: endorsed = % of items
above 0; zero = % of item responses at 0; uneven = within-person SD / max category.
"""
import sys, json, zlib
sys.path.insert(0, "src")
import numpy as np, pandas as pd, pyreadstat
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from mental_act import analysis as A, extra as E
from mental_act.config import ROOT, RUNS, TABLES, DATA_PROC, PHQ9_ITEMS
from mental_act.figlong import MODELS

MIN_H, MIN_DIV, DRAWS = 5, 8, 20

def load(name, arms=("naive", "coached")):
    f = RUNS / f"{name}.jsonl"
    if not f.exists(): return None, 0
    rs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    d = pd.DataFrame(rs); n_trials = d.trial_id.nunique()
    d = d[d.status == "ok"].drop_duplicates("trial_id")
    n_filtered = n_trials - len(d)
    d = d[d.arm.isin(arms)]
    if len(d) == 0: return None, n_filtered
    return np.array(d["ordered"].tolist(), int), n_filtered

osf, _ = pyreadstat.read_sav(ROOT / "data/external/osf_6b8jf_GOD_feigning.sav")
osf_g = osf["Groups_1_injured_2uninjuredhonest_3simulators"].to_numpy()
OSF_GEN = osf[[f"PCL_5_NOW_{i}" for i in range(1, 21)]].to_numpy(int)[osf_g == 1]
X_H = {"phq9": A.human_X("phq9"), "gad7": A.human_X("gad7")}
coach = pd.read_csv(DATA_PROC / "extra_phq9_coach.csv")[PHQ9_ITEMS].to_numpy(int)

TAG = {m[0]: m[3] for m in MODELS}
PHQ9_RUN = {m[0]: m[1] for m in A.ALL_MODELS}
GAD7_RUN = {m[0]: m[2] for m in A.ALL_MODELS}

# reference key -> (label, disorder, human matrix, max category, bin width, run-name fn, arms)
REFS = {
    "phq9_nhanes": ("PHQ-9 · NHANES", "depression", X_H["phq9"], 3, 1, lambda m: PHQ9_RUN.get(m), ("naive", "coached")),
    "phq9_coach":  ("PHQ-9 · COACH (clinical)", "depression", coach, 3, 1, lambda m: PHQ9_RUN.get(m), ("naive", "coached")),
    "gad7_nhis":   ("GAD-7 · NHIS", "anxiety", X_H["gad7"], 3, 1, lambda m: GAD7_RUN.get(m), ("naive", "coached")),
    "k6":          ("K6 · NHIS", "distress", E.human_X("k6"), 4, 1, lambda m: f"x_k6__{TAG[m]}", ("naive", "coached")),
    "pcl5c":       ("PCL-5 · postpartum (clinical)", "PTSD", E.human_X("pcl5c"), 4, 3, lambda m: f"x_pcl5c__{TAG[m]}", ("naive", "coached")),
    "itq":         ("ITQ · C19PRC", "PTSD", E.human_X("itq"), 4, 1, lambda m: f"x_itq__{TAG[m]}", ("naive", "coached")),
    "ocir":        ("OCI-R · C19PRC", "OCD", E.human_X("ocir"), 4, 2, lambda m: f"x_ocir__{TAG[m]}", ("naive", "coached")),
    "phq15":       ("PHQ-15 · C19PRC", "somatic", E.human_X("phq15"), 2, 1, lambda m: f"x_phq15__{TAG[m]}", ("naive", "coached")),
    "edeq":        ("EDE-Q · Mexico", "eating", E.human_X("edeq"), 6, 4, lambda m: f"x_edeq__{TAG[m]}", ("naive", "coached")),
}


def measures(X, K):
    J = X.shape[1]
    return dict(endorsed=100 * (X > 0).mean(1), all_items=100 * (X > 0).all(1),
                zero=100 * (X == 0).mean(1), uneven=X.std(1) / K)


def pairwise(P, rng, k=300):
    if len(P) > k: P = P[rng.choice(len(P), k, replace=False)]
    d = np.abs(P[:, None] - P[None]).sum(-1)
    return d[np.triu_indices(len(P), 1)].mean() / P.shape[1]


def cv_auc(Xa, Xb, rng, shuffle=False):
    Z = np.vstack([Xa, Xb]); y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]
    if shuffle: y = rng.permutation(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    cv = StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1e9)))
    return roc_auc_score(y, cross_val_predict(clf, Z, y, cv=cv, method="predict_proba")[:, 1])


def compare(M, H, K, w, seed):
    """Total-matched comparison of simulated responses M against humans H."""
    rng = np.random.default_rng(seed)
    bm, bh = M.sum(1) // w, H.sum(1) // w
    hc = pd.Series(bh).value_counts()
    keep = np.isin(bm, hc[hc >= MIN_H].index)
    cov = keep.mean(); M, bm = M[keep], bm[keep]
    if len(M) < 30: return dict(coverage=100 * cov, n=len(M))
    mm, hm_rows = measures(M, K), []
    wts = pd.Series(bm).value_counts()
    out = dict(coverage=100 * cov, n=len(M))
    hmeas = measures(H, K)
    for key in mm:
        out[f"{key}_m"] = float(np.mean(mm[key]))
        out[f"{key}_h"] = float(sum(wts[b] * hmeas[key][bh == b].mean() for b in wts.index) / wts.sum())
    # diversity within bins where both have enough responses
    dm, dh, ww = [], [], []
    for b, n in wts.items():
        P, Q = M[bm == b], H[bh == b]
        if len(P) >= MIN_DIV and len(Q) >= MIN_DIV:
            dm.append(pairwise(P, rng)); dh.append(pairwise(Q, rng)); ww.append(n)
    if ww:
        out["div_m"] = float(np.average(dm, weights=ww)); out["div_h"] = float(np.average(dh, weights=ww))
    out["div_bins"], out["div_n"] = len(ww), int(sum(ww))     # totals and simulated responses behind the diversity
    # separability: balanced within-bin samples without replacement
    aucs, nulls = [], []
    for _ in range(DRAWS):
        ia, ib = [], []
        for b in wts.index:
            pm, ph = np.where(bm == b)[0], np.where(bh == b)[0]
            k = min(len(pm), len(ph))
            ia += list(rng.choice(pm, k, replace=False)); ib += list(rng.choice(ph, k, replace=False))
        if len(ia) < 40: break
        aucs.append(cv_auc(M[ia], H[ib], rng)); nulls.append(cv_auc(M[ia], H[ib], rng, shuffle=True))
    if aucs:
        out.update(auc=float(np.mean(aucs)), auc_null=float(np.mean(nulls)), n_auc=len(ia) // 1)
    out["rho"] = float(spearmanr(M.mean(0), H.mean(0)).statistic)
    return out


rows = []
for key, (lab, dis, H, K, w, runf, arms) in REFS.items():
    for m in [x[0] for x in MODELS]:
        name = runf(m)
        if not name: continue
        M, nf = load(name, arms)
        if M is None or len(M) < 30: continue
        r = compare(M, H, K, w, zlib.crc32(f"{key}|{m}".encode()))
        rows.append(dict(ref=key, label=lab, disorder=dis, model=m, n_sim=len(M), filtered=nf, J=H.shape[1], K=K, **r))
    print(f"done {key}", flush=True)

# a real group against the community reference: are clinical profiles community-like?
r = compare(coach, X_H["phq9"], 3, 1, 1)
rows.append(dict(ref="coach_vs_nhanes", label="COACH patients vs NHANES", disorder="depression",
                 model="(real clinical sample)", n_sim=len(coach), filtered=0, J=9, K=3, **r))
T = pd.DataFrame(rows); T.to_csv(TABLES / "extra_generality.csv", index=False)

pd.set_option("display.width", 250)
S = []
for key, g in T[T.ref != "coach_vs_nhanes"].groupby("ref", sort=False):
    g = g.dropna(subset=["endorsed_m"])
    rng_ = lambda c: f"{g[c].min():.1f}-{g[c].max():.1f}"
    S.append(dict(ref=REFS[key][0], models=len(g), coverage=f"{g.coverage.median():.0f}%",
                  endorsed_h=round(g.endorsed_h.median(), 1), endorsed_m=rng_("endorsed_m"),
                  all_h=round(g.all_items_h.median(), 1), all_m=rng_("all_items_m"),
                  zero_h=round(g.zero_h.median(), 1), zero_m=rng_("zero_m"),
                  div_ratio=f"{(g.div_h / g.div_m).min():.1f}-{(g.div_h / g.div_m).max():.1f}x",
                  auc=f"{g.auc.min():.2f}-{g.auc.max():.2f}", null=round(g.auc_null.mean(), 2),
                  filtered=int(g.filtered.sum())))
print(pd.DataFrame(S).to_string(index=False))
print("\nclinical vs community (PHQ-9, total-matched):")
print(T[T.ref == "coach_vs_nhanes"][["n", "coverage", "endorsed_m", "endorsed_h", "all_items_m", "all_items_h",
                                     "zero_m", "zero_h", "div_m", "div_h", "auc", "auc_null"]].round(3).to_string(index=False))
