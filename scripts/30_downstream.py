"""Screening model trained on simulated patients: predicting PHQ-9 item 9 from items 1-8.

Split (src/mental_act/downstream.py): real training data, the recalibration set
and the GRM all come from NHANES 2005-2012; the test set is every 2013-2023
respondent. Main metrics on test respondents with items 1-8 total >= 5.

  sources     real NHANES, GRM-sampled profiles, each model's simulated patients, all models pooled
  controls    for each simulated training set, real training sets of the same size drawn
              (rand)   at random,
              (sev)    with the same items 1-8 totals,
              (sevlab) with the same items 1-8 totals and the same item-9 answers;
              against sevlab the only difference is how items 1-8 are composed
  recal       each simulated-trained model recalibrated (logistic, intercept and slope)
              on 1,000 held-out real respondents
  intervals   paired bootstrap: resample the simulated training set and the test set,
              draw fresh controls, refit everything
  augment     a small real training set plus pooled simulated patients
"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, brier_score_loss
from mental_act import analysis as A, downstream as DS, grm_sampler as G
from mental_act.config import TABLES

N_CTRL, N_BOOT = 50, 1000
D = DS.split()
Fs, Ys = D.Fte[D.sym], D.Yte[D.sym]
print(f"train pool (2005-2012) n={len(D.Ftr):,}, SI {100 * D.Ytr.mean():.2f}%; recalibration n={len(D.Yval):,}; "
      f"test (2013-2023) n={len(D.Yte):,}, items 1-8 >= {DS.SYM}: n={len(Ys):,}, SI {100 * Ys.mean():.1f}%")

SIM = {}
for lab, run, *_ in A.ALL_MODELS:
    X = np.array(A.load_run(run)["ordered"].tolist(), int)
    if len(np.unique(X[:, 8] > 0)) == 2: SIM[lab] = X
    else: print(f"{lab}: item 9 constant in the simulated data; no classifier fits")
SIM["All models pooled"] = np.vstack(list(SIM.values()))
real = DS.fit(D.Ftr, D.Ytr)
MATCH = DS.Matcher(D.Ftr, D.Ytr)


def row(name, m, kind, n_train, pos):
    p = m.predict_proba(Fs)[:, 1]
    r = dict(train=name, kind=kind, n_train=n_train, train_pos=pos, n_test=len(Ys), **DS.metrics(p, Ys))
    rc = DS.recalibrator(m, D.Fval, D.Yval)(p)
    r.update({f"recal_{k}": v for k, v in DS.metrics(rc, Ys).items() if k in ("brier", "citl", "slope", "fpr", "sens")})
    return r


rows = [row("Real (NHANES)", real, "real", len(D.Ytr), 100 * D.Ytr.mean())]
Sg = G.sample(20_000, np.random.default_rng(20261012))
rows.append(row("GRM-sampled profiles", DS.fit(Sg[:, :8], (Sg[:, 8] > 0).astype(int)), "grm", len(Sg), 100 * (Sg[:, 8] > 0).mean()))
MOD = {k: DS.fit(X[:, :8], (X[:, 8] > 0).astype(int)) for k, X in SIM.items()}
for k, X in SIM.items():
    rows.append(row(k, MOD[k], "pooled" if k == "All models pooled" else "sim", len(X), 100 * (X[:, 8] > 0).mean()))
T = pd.DataFrame(rows)
T.to_csv(TABLES / "downstream_si.csv", index=False)
pd.set_option("display.width", 220)
print(T[["train", "n_train", "train_pos", "auc", "brier", "citl", "slope", "fpr", "sens", "recal_brier", "recal_slope"]]
      .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))


# ---- matched real controls and paired bootstrap ------------------------------------
def controls(k, X, seed):
    rng = np.random.default_rng(seed)
    t8, y = X[:, :8].sum(1), (X[:, 8] > 0).astype(int); n = len(X)
    draws = {"rand": lambda: rng.choice(len(D.Ytr), n, replace=False),
             "sev": lambda: MATCH.draw(t8, y, rng), "sevlab": lambda: MATCH.draw(t8, y, rng, label=True)}
    # point estimates: the simulated model against the mean over N_CTRL control draws
    ms = MOD[k]; ps = ms.predict_proba(Fs)[:, 1]
    out = dict(train=k, n_train=n, sim_pos=100 * y.mean(), auc_sim=roc_auc_score(Ys, ps))
    rc = DS.recalibrator(ms, D.Fval, D.Yval)
    out.update(brier_sim=brier_score_loss(Ys, ps), brier_sim_recal=brier_score_loss(Ys, rc(ps)),
               slope_sim=DS.calibration_slope(ps, Ys), slope_sim_recal=DS.calibration_slope(rc(ps), Ys))
    for c, dr in draws.items():
        ms_ = []
        for _ in range(N_CTRL):
            j = dr(); m = DS.fit(D.Ftr[j], D.Ytr[j]); p = m.predict_proba(Fs)[:, 1]
            ms_.append(dict(auc=roc_auc_score(Ys, p), brier=brier_score_loss(Ys, p), slope=DS.calibration_slope(p, Ys),
                            pos=100 * D.Ytr[j].mean()))
        mm = pd.DataFrame(ms_).mean()
        out.update({f"{s}_{c}": mm[s] for s in ("auc", "brier", "slope", "pos")})
    # paired bootstrap of the AUC differences and of the recalibrated Brier difference against sev
    bs = {c: [] for c in draws}; bb = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n); te = rng.integers(0, len(Ys), len(Ys)); yb = Ys[te]
        Xb = X[i]
        if len(np.unique(Xb[:, 8] > 0)) < 2: continue
        mb = DS.fit(Xb[:, :8], (Xb[:, 8] > 0).astype(int)); pb = mb.predict_proba(Fs[te])[:, 1]; a = roc_auc_score(yb, pb)
        for c, dr in draws.items():
            j = dr(); pc = DS.fit(D.Ftr[j], D.Ytr[j]).predict_proba(Fs[te])[:, 1]
            bs[c].append(a - roc_auc_score(yb, pc))
            if c == "sev": bb.append(brier_score_loss(yb, DS.recalibrator(mb, D.Fval, D.Yval)(pb)) - brier_score_loss(yb, pc))
    for c in draws:
        out[f"d_{c}"] = out["auc_sim"] - out[f"auc_{c}"]
        out[f"d_{c}_lo"], out[f"d_{c}_hi"] = np.percentile(bs[c], [2.5, 97.5])
    out["d_brier_recal_sev"] = out["brier_sim_recal"] - out["brier_sev"]
    out["d_brier_recal_sev_lo"], out["d_brier_recal_sev_hi"] = np.percentile(bb, [2.5, 97.5])
    return out


C = pd.DataFrame(Parallel(n_jobs=min(18, len(SIM)))(delayed(controls)(k, X, s) for s, (k, X) in enumerate(SIM.items())))
C.to_csv(TABLES / "downstream_controls.csv", index=False)
print("\n=== simulated vs same-size real controls (AUC; paired 95% bootstrap intervals) ===")
print(C[["train", "n_train", "sim_pos", "pos_sev", "pos_sevlab", "auc_sim", "auc_rand", "auc_sev", "auc_sevlab",
         "d_sev", "d_sev_lo", "d_sev_hi", "d_sevlab", "d_sevlab_lo", "d_sevlab_hi"]].to_string(index=False, float_format=lambda v: f"{v:6.3f}"))
print(C[["train", "brier_sim", "brier_sim_recal", "brier_sev", "d_brier_recal_sev_lo", "d_brier_recal_sev_hi",
         "slope_sim", "slope_sim_recal", "slope_sev"]].to_string(index=False, float_format=lambda v: f"{v:6.3f}"))

# ---- calibration by items 1-8 total (all 2013-2023 respondents) -----------------
strata = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24)]
tot8 = D.Fte.sum(1)
cal = []
models = {"real": real, "GRM": DS.fit(Sg[:, :8], (Sg[:, 8] > 0).astype(int)), **{("LLM pooled" if k == "All models pooled" else k): m for k, m in MOD.items()}}
for lo, hi in strata:
    m = (tot8 >= lo) & (tot8 <= hi)
    cal.append(dict(lo=lo, hi=hi, n=int(m.sum()), n_si=int(D.Yte[m].sum()), observed=100 * D.Yte[m].mean(),
                    **{f"pred_{k}": 100 * mdl.predict_proba(D.Fte[m])[:, 1].mean() for k, mdl in models.items()}))
CAL = pd.DataFrame(cal); CAL.to_csv(TABLES / "downstream_calibration.csv", index=False)
print("\n=== observed and predicted SI rate by items 1-8 total, all 2013-2023 respondents ===")
print(CAL[["lo", "hi", "n", "n_si", "observed", "pred_real", "pred_GRM", "pred_LLM pooled"]].round(1).to_string(index=False))

# ---- augmentation: a small real training set plus pooled simulated patients ------------
S = SIM["All models pooled"]; rng = np.random.default_rng(0)
aug = []
for n_real in (50, 100, 200, 500):
    for rep in range(100):
        idx = rng.choice(len(D.Ytr), n_real, replace=False)
        if len(np.unique(D.Ytr[idx])) < 2: continue
        base = DS.metrics(DS.fit(D.Ftr[idx], D.Ytr[idx]).predict_proba(Fs)[:, 1], Ys)
        for k in (1, 2, 4):
            j = rng.choice(len(S), k * n_real, replace=False)
            Xa = np.vstack([D.Ftr[idx], S[j, :8]]); ya = np.r_[D.Ytr[idx], (S[j, 8] > 0).astype(int)]
            r = DS.metrics(DS.fit(Xa, ya).predict_proba(Fs)[:, 1], Ys)
            aug.append(dict(n_real=n_real, ratio=k, rep=rep, base_auc=base["auc"], base_brier=base["brier"], base_citl=base["citl"],
                            **{f"d_{s}": r[s] - base[s] for s in ("auc", "brier", "citl", "slope")}))
AG = pd.DataFrame(aug)
# 95% interval of the mean change over draws (normal approximation)
ci = lambda s: (lambda x: x.mean() + s * 1.96 * x.std(ddof=1) / np.sqrt(len(x)))
ag = AG.groupby(["n_real", "ratio"]).agg(draws=("rep", "size"), base_auc=("base_auc", "mean"), base_brier=("base_brier", "mean"), base_citl=("base_citl", "mean"),
                                          d_auc=("d_auc", "mean"), d_auc_lo=("d_auc", ci(-1)), d_auc_hi=("d_auc", ci(1)),
                                          d_brier=("d_brier", "mean"), d_brier_lo=("d_brier", ci(-1)), d_brier_hi=("d_brier", ci(1)),
                                          d_citl=("d_citl", "mean"), d_citl_lo=("d_citl", ci(-1)), d_citl_hi=("d_citl", ci(1)),
                                          p_auc_up=("d_auc", lambda x: 100 * (x > 0).mean())).reset_index()
ag.to_csv(TABLES / "downstream_augmentation.csv", index=False)
print("\n=== augmentation (mean change and its 95% interval over draws) ===")
print(ag.to_string(index=False, float_format=lambda v: f"{v:+7.3f}"))
