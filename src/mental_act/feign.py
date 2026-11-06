"""Tests for the OSF PCL-5 feigning comparison, shared by scripts/37 and scripts/63.

Separability (cross-validated AUC at matched totals):
  subsample_interval  80% subsamples of both groups without replacement
  permutation         labels of a group and the genuine patients shuffled; observed
                      value and every null value are the same statistic (mean of K
                      matchings and fold splits), so the test is exact
  direct_test         labels of model and feigner responses shuffled; statistic is
                      AUC(model) - AUC(feigner), both against all genuine patients
  real_baseline       genuine patients drawn at a group's totals from one random half
                      of the genuine group, compared with the other half

Profile at matched totals:
  profile_ci          "not at all" share relative to matched genuine patients (ratio),
                      symptoms endorsed minus matched genuine, with a bootstrap that
                      resamples the group, the feigners and the genuine patients in the
                      same draw (so the difference from the feigners is paired)
  shift_ci            feigning minus patient role (total, % "extremely"), bootstrap

Every function takes an explicit seed; parallel draws use seed + offset per draw, so
results do not depend on scheduling.
"""
import numpy as np, pyreadstat
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from . import analysis as A
from .config import ROOT

_df, _ = pyreadstat.read_sav(ROOT / "data/external/osf_6b8jf_GOD_feigning.sav")
_g = _df["Groups_1_injured_2uninjuredhonest_3simulators"].to_numpy()
_H = _df[[f"PCL_5_NOW_{i}" for i in range(1, 21)]].to_numpy(int)
GEN, HONEST, FEIGNER = _H[_g == 1], _H[_g == 2], _H[_g == 3]
FEIGNER_RANGE = tuple(np.percentile(FEIGNER.sum(1), [10, 90]))   # feigners' 10th-90th percentile of totals
K, B_PERM, B_DIRECT, K_DIRECT, S_BASE, B_BOOT = 50, 1000, 1000, 10, 200, 4000
JOBS = 22


def cv_auc(X, ref, rng, xgroups=None):
    """Nearest-total matching with replacement; a genuine patient drawn more than once
    stays in one fold. xgroups: source ids when X itself was drawn with replacement."""
    idx = A.nearest_total_match(X, ref, rng); R = ref[idx]
    Z = np.vstack([X, R]); y = np.r_[np.ones(len(X)), np.zeros(len(R))]
    gx = np.arange(len(X)) if xgroups is None else len(X) + len(ref) + np.asarray(xgroups)
    groups = np.r_[gx, len(X) + idx]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    cv = A.grouped_folds(y, groups, int(rng.integers(1e9)))
    return roc_auc_score(y, cross_val_predict(clf, Z, y, groups=groups, cv=cv, method="predict_proba")[:, 1])


def _stat(X, ref, rng, k=K):
    return float(np.mean([cv_auc(X, ref, rng) for _ in range(k)]))


def _sub(X, s, frac=0.8):
    rng = np.random.default_rng(s)
    xs = X[rng.choice(len(X), int(frac * len(X)), replace=False)]
    rs = GEN[rng.choice(len(GEN), int(frac * len(GEN)), replace=False)]
    return cv_auc(xs, rs, rng)


def _perm(X, s):
    rng = np.random.default_rng(s); pool = np.vstack([X, GEN]); i = rng.permutation(len(pool))
    return _stat(pool[i[:len(X)]], pool[i[len(X):]], rng)


def separability(X, seed, n_sub=60, par=None):
    """Observed AUC (mean of K), 95% interval over 80% subsamples, exact permutation p."""
    par = par or Parallel(n_jobs=JOBS)
    obs = _stat(X, GEN, np.random.default_rng(seed))
    sub = np.array(par(delayed(_sub)(X, seed + 1 + b) for b in range(n_sub)))
    null = np.array(par(delayed(_perm)(X, seed + 100_000 + b) for b in range(B_PERM)))
    k = int(np.floor(0.05 * (B_PERM + 1)))          # p < 0.05  <=>  obs above the k-th largest null value
    return dict(auc=obs, sub_lo=np.percentile(sub, 2.5), sub_hi=np.percentile(sub, 97.5),
                null_mean=null.mean(), null_hi=np.percentile(null, 97.5), null_crit05=np.sort(null)[-k],
                p=(np.sum(null >= obs) + 1) / (B_PERM + 1), null=null)


def _direct_stat(X, F, rng):
    return float(np.mean([cv_auc(X, GEN, rng) - cv_auc(F, GEN, rng) for _ in range(K_DIRECT)]))


def _direct_perm(X, s):
    rng = np.random.default_rng(s); pool = np.vstack([X, FEIGNER]); i = rng.permutation(len(pool))
    return _direct_stat(pool[i[:len(X)]], pool[i[len(X):]], rng)


def direct_test(X, seed, par=None):
    """One-sided: is the group more separable from genuine patients than the feigners are?"""
    par = par or Parallel(n_jobs=JOBS)
    obs = _direct_stat(X, FEIGNER, np.random.default_rng(seed + 200_000))
    null = np.array(par(delayed(_direct_perm)(X, seed + 300_000 + b) for b in range(B_DIRECT)))
    return dict(direct_diff=obs, direct_p=(np.sum(null >= obs) + 1) / (B_DIRECT + 1))


def _base(X, s):
    rng = np.random.default_rng(s); p = rng.permutation(len(GEN)); a, b = GEN[p[:len(GEN) // 2]], GEN[p[len(GEN) // 2:]]
    src = A.nearest_total_match(X, a, rng)
    return cv_auc(X, b, rng), cv_auc(a[src], b, rng, xgroups=src)


def real_baseline(X, seed, par=None):
    par = par or Parallel(n_jobs=JOBS)
    v = np.array(par(delayed(_base)(X, seed + 400_000 + b) for b in range(S_BASE)))
    return dict(base_group=v[:, 0].mean(), base_real=v[:, 1].mean())


naa = lambda Z: (Z == 0).mean()
breadth = lambda Z: (Z > 0).sum(1).mean()


def _matched(X, ref, rng):
    R = ref[A.nearest_total_match(X, ref, rng)]
    return naa(R), breadth(R)


def _ratio_gap(X, ref, rng):
    n, b = _matched(X, ref, rng)
    return naa(X) / n, breadth(X) - b


def profile_ci(X, seed, alpha_adj=0.05 / 17):
    """Group and matched-genuine "not at all" share and breadth (genuine averaged over 50
    matchings; ratio = group / genuine), 95% bootstrap intervals, and the paired
    difference from the feigners' ratio with a Bonferroni-adjusted interval."""
    rng = np.random.default_rng(seed)
    gn, gb = np.mean([_matched(X, GEN, rng) for _ in range(50)], 0)
    lo_t, hi_t = FEIGNER_RANGE; Xr = X[(X.sum(1) >= lo_t) & (X.sum(1) <= hi_t)]
    in_range = naa(Xr) / np.mean([_matched(Xr, GEN, rng)[0] for _ in range(50)]) if len(Xr) >= 10 else np.nan
    bx, bf = [], []
    for _ in range(B_BOOT):
        G = GEN[rng.integers(0, len(GEN), len(GEN))]
        bx.append(_ratio_gap(X[rng.integers(0, len(X), len(X))], G, rng))
        bf.append(_ratio_gap(FEIGNER[rng.integers(0, len(FEIGNER), len(FEIGNER))], G, rng)[0])
    # a draw whose matched genuine patients have no "not at all" answer (only at very high
    # totals) leaves the ratio undefined; such draws are dropped and counted
    bx, bf = np.array(bx), np.array(bf); d = bf - bx[:, 0]; P = np.nanpercentile
    return dict(naa=100 * naa(X), naa_gen=100 * gn, naa_ratio=naa(X) / gn, undefined_draws=int(np.isnan(bx[:, 0]).sum()),
                naa_lo=P(bx[:, 0], 2.5), naa_hi=P(bx[:, 0], 97.5),
                naa_ratio_in_range=in_range, n_in_range=len(Xr),
                feigner_minus_lo=P(d, 100 * alpha_adj / 2), feigner_minus_hi=P(d, 100 - 100 * alpha_adj / 2),
                breadth=breadth(X), breadth_gen=gb, breadth_gap=breadth(X) - gb,
                gap_lo=np.percentile(bx[:, 1], 2.5), gap_hi=np.percentile(bx[:, 1], 97.5),
                **absolute_gaps(X, seed + 500_000, alpha_adj))


def _abs(X, ref, rng):
    """Matched-genuine minus group share of "not at all" (percentage points), and breadth gap."""
    n, b = _matched(X, ref, rng)
    return 100 * (n - naa(X)), breadth(X) - b


def absolute_gaps(X, seed, alpha_adj=0.05 / 17):
    """Absolute "not at all" gap overall and within the feigners' 10th-90th percentile of totals,
    with 95% bootstrap intervals, and the paired in-range difference from the feigners' gap
    (one-sided bootstrap p; the feigners and the genuine patients are resampled in the same draw)."""
    rng = np.random.default_rng(seed)
    lo_t, hi_t = FEIGNER_RANGE; inr = lambda Z: Z[(Z.sum(1) >= lo_t) & (Z.sum(1) <= hi_t)]
    Xr, Fr = inr(X), inr(FEIGNER)
    pa = np.mean([_abs(X, GEN, rng) for _ in range(50)], 0); pr = np.mean([_abs(Xr, GEN, rng) for _ in range(50)], 0)
    ba, br, d = [], [], []
    for _ in range(B_BOOT // 2):
        G = GEN[rng.integers(0, len(GEN), len(GEN))]
        ba.append(_abs(X[rng.integers(0, len(X), len(X))], G, rng)[0])
        r = _abs(Xr[rng.integers(0, len(Xr), len(Xr))], G, rng); br.append(r)
        d.append(r[0] - _abs(Fr[rng.integers(0, len(Fr), len(Fr))], G, rng)[0])
    ba, br, d = np.array(ba), np.array(br), np.array(d)
    return dict(naa_abs=pa[0], naa_abs_lo=np.percentile(ba, 2.5), naa_abs_hi=np.percentile(ba, 97.5),
                naa_abs_in_range=pr[0], naa_abs_in_range_lo=np.percentile(br[:, 0], 2.5), naa_abs_in_range_hi=np.percentile(br[:, 0], 97.5),
                breadth_gap_in_range=pr[1], gap_in_range_lo=np.percentile(br[:, 1], 2.5), gap_in_range_hi=np.percentile(br[:, 1], 97.5),
                abs_minus_feigner=np.mean(d), abs_minus_feigner_p=float(np.mean(d <= 0)))


def shift_ci(Xf, Xp, seed, B=B_BOOT):
    """Feigning minus patient role: total and % "extremely", with 95% bootstrap intervals."""
    rng = np.random.default_rng(seed)
    tot = lambda Z: Z.sum(1).mean(); ext = lambda Z: 100 * (Z == 4).mean()
    v = np.array([(tot(a) - tot(b), ext(a) - ext(b)) for a, b in
                  ((Xf[rng.integers(0, len(Xf), len(Xf))], Xp[rng.integers(0, len(Xp), len(Xp))]) for _ in range(B))])
    return dict(dtotal=tot(Xf) - tot(Xp), dtotal_lo=np.percentile(v[:, 0], 2.5), dtotal_hi=np.percentile(v[:, 0], 97.5),
                dextreme=ext(Xf) - ext(Xp), dext_lo=np.percentile(v[:, 1], 2.5), dext_hi=np.percentile(v[:, 1], 97.5))
