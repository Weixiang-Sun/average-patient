"""Is the missing variation structured? Item covariance at a fixed PHQ-9 total.

Residuals: each response minus the mean item scores of its group at the same total (totals 5-24).
Their correlations show which symptoms go together once severity is fixed. Summaries:
  contrast    mean residual correlation within the somatic (sleep, fatigue, appetite,
              concentration, psychomotor) and the non-somatic (interest, mood, worthlessness,
              suicidal thoughts) items minus between them (the best-fitting two-factor split in
              Elhai et al. 2012)
  similarity  Pearson correlation of a group's 36 residual correlations with those of all
              NHANES respondents
References: real respondents drawn at each model's totals, one per model response, 100 draws
(same size and totals); a graded response model (one trait, independent items); and each
model with independent noise added (each item moves one level up or down with probability p,
p chosen so that diversity reaches that of real respondents with the same totals).
Structure-free nulls at each model's size and totals (residuals at a fixed total sum to zero, so
residual correlations depend on the item distributions as well as on covariation):
  null_own    the maximum-entropy distribution at each total with the model's own item
              distributions there (no covariation beyond the fixed total), one per model response,
              100 draws
  null_grm    GRM-sampled profiles drawn at the model's totals (one per model response), 100 draws
Interviews: the same summaries for the PHQ-9 rated from real- and sampled-profile interviews
(primary rater) and for the profiles given, pooled over the five patient models (which were given
the same 80 profiles), with 80 real respondents drawn at the given totals as the reference.
Pairs: the 36 residual correlations of NHANES, of real respondents drawn at the models' totals
and of the models (median over models), to show where the models differ.
-> outputs/tables/residual_structure.csv, residual_structure_interviews.csv, residual_structure_pairs.csv
"""
import sys, json, glob
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A, grm_sampler as G
from mental_act.config import TABLES, RUNS

SOM, NON = [2, 3, 4, 6, 7], [0, 1, 5, 8]
Xh = A.human_X("phq9"); Xh = Xh[(Xh.sum(1) >= 5) & (Xh.sum(1) <= 24)]
iu = np.triu_indices(9, 1)


def rcorr(X):
    X = X[(X.sum(1) >= 5) & (X.sum(1) <= 24)].astype(float); t = X.sum(1); R = X.copy()
    for u in np.unique(t): R[t == u] -= X[t == u].mean(0)
    sd = R.std(0); M = np.full((9, 9), np.nan); ok = np.flatnonzero(sd > 1e-9)
    if len(ok) > 1: M[np.ix_(ok, ok)] = np.corrcoef(R[:, ok].T)
    return M


def contrast(M):
    w = [M[i, j] for g in (SOM, NON) for i in g for j in g if i < j]; b = [M[i, j] for i in SOM for j in NON]
    return np.nanmean(w) - np.nanmean(b)


MH = rcorr(Xh)
def similarity(M):
    a, b = M[iu], MH[iu]; ok = ~np.isnan(a)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 5 else np.nan


def summarise(M): return dict(contrast=contrast(M), similarity=similarity(M))


rows = [dict(group="NHANES", n=len(Xh), **summarise(MH))]
MODEL_M, MATCHED_M = {}, {}; rng_pairs = np.random.default_rng(20261012)
rng = np.random.default_rng(20261010)
# random halves of NHANES
hv = []
for _ in range(100):
    p = rng.permutation(len(Xh)); a, b = rcorr(Xh[p[:len(p) // 2]]), rcorr(Xh[p[len(p) // 2:]])
    hv.append(dict(contrast=contrast(a), similarity=float(np.corrcoef(a[iu], b[iu])[0, 1])))
hv = pd.DataFrame(hv); rows.append(dict(group="NHANES halves", n=len(Xh) // 2, contrast=hv.contrast.mean(), similarity=hv.similarity.mean(),
                                        contrast_lo=hv.contrast.quantile(.025), contrast_hi=hv.contrast.quantile(.975)))
# graded response model (fit to NHANES 2005-2012)
S = G.sample(200_000, rng); S = S[(S.sum(1) >= 5) & (S.sum(1) <= 24)][:20_000]
rows.append(dict(group="GRM-sampled profiles", n=len(S), **summarise(rcorr(S))))


POOL = G.sample(400_000, np.random.default_rng(20261013)); POOL_T = POOL.sum(1)
BY_T = {u: POOL[POOL_T == u] for u in range(5, 25)}


CONF = np.stack(np.meshgrid(*[np.arange(4)] * 9, indexing="ij"), -1).reshape(-1, 9); CONF_T = CONF.sum(1)


def maxent_at_total(P, iters=500):
    """Maximum-entropy distribution over the responses with P's total whose item distributions
    equal those of P (iterative proportional fitting); items are as independent as the fixed
    total allows."""
    u = int(P[0].sum()); C = CONF[CONF_T == u]; n = len(P)
    target = np.stack([np.bincount(P[:, k], minlength=4) / n for k in range(9)])
    w = np.ones(len(C)) * np.all(target[np.arange(9), C] > 0, 1)
    for _ in range(iters):
        for k in range(9):
            cur = np.bincount(C[:, k], weights=w, minlength=4) / w.sum()
            f = np.divide(target[k], cur, out=np.zeros(4), where=cur > 0); w = w * f[C[:, k]]
    return C, w / w.sum()


def null_own(X):
    """Sampler of structure-free responses with X's item distributions at each total (one per response)."""
    t = X.sum(1); parts = [(*maxent_at_total(X[t == u]), int((t == u).sum())) for u in np.unique(t)]
    return lambda r: np.vstack([C[r.choice(len(C), n, p=p)] for C, p, n in parts])


def null_grm(X, r):
    return np.vstack([BY_T[u][r.integers(0, len(BY_T[u]))] for u in X.sum(1)])


def jitter(X, p, r):
    move = r.random(X.shape) < p; step = r.choice([-1, 1], size=X.shape)
    return np.clip(X + move * step, 0, 3)


for lab, run, *_ in A.ALL_MODELS:
    X = A.run_X(run); X = X[(X.sum(1) >= 5) & (X.sum(1) <= 24)]
    r = dict(group=lab, n=len(X), **summarise(rcorr(X)))
    bs = [contrast(rcorr(X[rng.integers(0, len(X), len(X))])) for _ in range(200)]
    r["contrast_lo"], r["contrast_hi"] = np.nanpercentile(bs, [2.5, 97.5])
    # real respondents at this model's totals, same size
    mt = []
    for _ in range(100):
        R = Xh[A.nearest_total_match(X, Xh, rng)]; M = rcorr(R); mt.append((contrast(M), similarity(M)))
    mt = np.array(mt); r.update(matched_contrast=np.nanmean(mt[:, 0]), matched_contrast_lo=np.nanpercentile(mt[:, 0], 2.5),
                                matched_contrast_hi=np.nanpercentile(mt[:, 0], 97.5), matched_similarity=np.nanmean(mt[:, 1]),
                                matched_similarity_lo=np.nanpercentile(mt[:, 1], 2.5))
    # independent noise up to real diversity at the same totals
    target = A.diversity_same_totals(X, Xh, min_n=8)
    best = None
    for p in np.arange(0.05, 0.95, 0.05):
        Xn = jitter(X, p, np.random.default_rng(1)); d = A.diversity(Xn, 5, 24)
        if best is None or abs(d - target) < abs(best[1] - target): best = (p, d)
    nz = [summarise(rcorr(jitter(X, best[0], np.random.default_rng(k)))) for k in range(20)]
    r.update(noise_p=best[0], noise_diversity=best[1], real_diversity=target,
             noise_contrast=np.nanmean([z["contrast"] for z in nz]), noise_similarity=np.nanmean([z["similarity"] for z in nz]))
    # structure-free nulls at this model's size and totals (own generator: main results unchanged)
    rn = np.random.default_rng(20261014)
    for name, fn in (("null_own", null_own(X)), ("null_grm", lambda r: null_grm(X, r))):
        z = np.array([[f(rcorr(fn(rn))) for f in (contrast, similarity)] for _ in range(100)])
        r.update({f"{name}_contrast": np.nanmean(z[:, 0]), f"{name}_contrast_hi": np.nanpercentile(z[:, 0], 97.5),
                  f"{name}_similarity": np.nanmean(z[:, 1]), f"{name}_similarity_lo": np.nanpercentile(z[:, 1], 2.5),
                  f"{name}_similarity_hi": np.nanpercentile(z[:, 1], 97.5)})
    rows.append(r)
    # own generator, so that these draws leave the main results above unchanged
    MODEL_M[lab] = rcorr(X); MATCHED_M[lab] = np.nanmean([rcorr(Xh[A.nearest_total_match(X, Xh, rng_pairs)]) for _ in range(20)], 0)
T = pd.DataFrame(rows); T.to_csv(TABLES / "residual_structure.csv", index=False)
pd.set_option("display.width", 250)
print(T.round(3).to_string(index=False))

# ---- interviews: rated PHQ-9 (primary rater) and the given profiles, pooled over patient models.
# The five patient models were given the same 80 profiles per condition, so the given profiles are
# 80 distinct respondents; the reference is 80 real respondents drawn at their totals (200 draws),
# and intervals for the interviews resample profiles with all their interviews (200 draws).
PRIMARY = lambda tag: "claude-haiku-4-5" if tag == "gpt41" else "gpt-4.1"
pool = {}
for f in sorted(glob.glob(str(RUNS / "interview__*__*.jsonl"))):
    _, cond, tag = f[:-6].split("/")[-1].split("__")
    if tag.startswith("eff") or cond not in ("oracle", "grm"): continue
    d = A.drop_filtered_interviews(pd.DataFrame([json.loads(l) for l in open(f)]).query("status == 'ok'").drop_duplicates("trial_id"))
    pool.setdefault(cond, []).append(pd.DataFrame(dict(trial=d.trial_id.values, given=d.true_profile.tolist(), rated=d[f"rated__{PRIMARY(tag)}"].tolist())))
irows = []
for cond, lab in (("oracle", "real profiles"), ("grm", "sampled profiles")):
    D = pd.concat(pool[cond]); G = D.drop_duplicates("trial").set_index("trial").given.apply(np.array)
    Xg = np.vstack(G.values); Xg = Xg[(Xg.sum(1) >= 5) & (Xg.sum(1) <= 24)]
    Xr = np.array(D.rated.tolist(), int); keep = (Xr.sum(1) >= 5) & (Xr.sum(1) <= 24)
    Mg, Mr = rcorr(Xg), rcorr(Xr[keep])
    ref = np.array([[f(rcorr(Xh[A.nearest_total_match(Xg, Xh, rng)])) for f in (contrast, similarity)] for _ in range(200)])
    trials = D.trial.unique(); bs = []
    for _ in range(200):
        pick = rng.choice(trials, len(trials)); B = pd.concat([D[D.trial == t] for t in pick]); Xb = np.array(B.rated.tolist(), int)
        bs.append([f(rcorr(Xb[(Xb.sum(1) >= 5) & (Xb.sum(1) <= 24)])) for f in (contrast, similarity)])
    bs = np.array(bs)
    irows.append(dict(group=lab, n_profiles=len(Xg), n_interviews=int(keep.sum()), given_contrast=contrast(Mg), given_similarity=similarity(Mg),
                      rated_contrast=contrast(Mr), rated_contrast_lo=np.nanpercentile(bs[:, 0], 2.5), rated_contrast_hi=np.nanpercentile(bs[:, 0], 97.5),
                      rated_similarity=similarity(Mr), rated_similarity_lo=np.nanpercentile(bs[:, 1], 2.5), rated_similarity_hi=np.nanpercentile(bs[:, 1], 97.5),
                      rated_vs_given=float(np.corrcoef(Mr[iu], Mg[iu])[0, 1]),
                      ref80_contrast=np.nanmean(ref[:, 0]), ref80_contrast_lo=np.nanpercentile(ref[:, 0], 2.5), ref80_contrast_hi=np.nanpercentile(ref[:, 0], 97.5),
                      ref80_similarity=np.nanmean(ref[:, 1]), ref80_similarity_lo=np.nanpercentile(ref[:, 1], 2.5), ref80_similarity_hi=np.nanpercentile(ref[:, 1], 97.5)))
TI = pd.DataFrame(irows); TI.to_csv(TABLES / "residual_structure_interviews.csv", index=False)
print(TI.round(2).T.to_string())

# ---- the 36 residual correlations: NHANES, real at the models' totals, models (median)
NAMES = ["interest", "mood", "sleep", "fatigue", "appetite", "worthless", "concentration", "psychomotor", "suicidal"]
med = np.nanmedian(np.stack(list(MODEL_M.values())), 0); mat = np.nanmean(np.stack(list(MATCHED_M.values())), 0)
P = pd.DataFrame([dict(item_a=NAMES[i], item_b=NAMES[j], somatic_a=i in SOM, somatic_b=j in SOM, nhanes=MH[i, j], matched_real=mat[i, j],
                       models_median=med[i, j], models_below=int(sum(MODEL_M[k][i, j] < MATCHED_M[k][i, j] for k in MODEL_M)))
                  for i, j in zip(*iu)])
P["diff"] = P.models_median - P.matched_real; P = P.sort_values("diff"); P.to_csv(TABLES / "residual_structure_pairs.csv", index=False)
print(P.round(2).to_string(index=False))
