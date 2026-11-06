"""Shared setup for the screening experiment (scripts/30_downstream.py, 63).

Task: predict whether PHQ-9 item 9 (thoughts of being better off dead or of
self-harm) is endorsed from items 1-8. This is a measurement task, not a
prediction of suicide risk.

Split. Everything fit on real data uses NHANES 2005-2012 (cycles D-G), the
cycles the GRM sampler is fit on (scripts/50_grm_heldout.py); the test set is
every 2013-2023 respondent (cycles H-L), so no test respondent informs any
training source. 1,000 respondents with items 1-8 total >= 5 are held out of the
2005-2012 data for recalibration and never enter a training set.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from .config import DATA_PROC, PHQ9_ITEMS

FIT, TEST = ("D", "E", "F", "G"), ("H", "I", "J", "L")
SYM = 5          # items 1-8 total defining the screening population
N_VAL = 1000     # held-out 2005-2012 respondents (items 1-8 >= SYM) for recalibration


@dataclass
class Split:
    Ftr: np.ndarray; Ytr: np.ndarray      # 2005-2012 training pool
    Fval: np.ndarray; Yval: np.ndarray    # 2005-2012 recalibration set (symptomatic)
    Fte: np.ndarray; Yte: np.ndarray      # 2013-2023, all respondents
    sym: np.ndarray                       # test mask: items 1-8 >= SYM


def split(seed: int = 0) -> Split:
    df = pd.read_csv(DATA_PROC / "nhanes_phq9.csv")
    X = df[PHQ9_ITEMS].to_numpy(int); F, Y = X[:, :8], (X[:, 8] > 0).astype(int)
    fit_ = df.cycle.isin(FIT).to_numpy(); te = df.cycle.isin(TEST).to_numpy()
    cand = np.flatnonzero(fit_ & (F.sum(1) >= SYM))
    val = np.random.default_rng(seed).choice(cand, N_VAL, replace=False)
    tr = np.setdiff1d(np.flatnonzero(fit_), val)
    return Split(F[tr], Y[tr], F[val], Y[val], F[te], Y[te], F[te].sum(1) >= SYM)


def fit(X, y):
    return LogisticRegression(max_iter=2000, C=1.0).fit(X, y)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def calibration_slope(p, y):
    """Slope of the logistic regression of the outcome on the logit of the prediction (1 = calibrated)."""
    return float(LogisticRegression(C=np.inf, max_iter=2000).fit(_logit(p)[:, None], y).coef_[0, 0])


def metrics(p, y):
    f = p >= 0.5
    return dict(auc=roc_auc_score(y, p), brier=brier_score_loss(y, p), mean_pred=100 * p.mean(), observed=100 * y.mean(),
                citl=100 * (p.mean() - y.mean()), slope=calibration_slope(p, y), flagged=100 * f.mean(),
                sens=100 * f[y == 1].mean(), fpr=100 * f[y == 0].mean(), ppv=100 * (f & (y == 1)).sum() / max(f.sum(), 1))


def recalibrator(model, Fval, Yval):
    """Logistic recalibration (intercept and slope on the logit) fit on the held-out real set."""
    r = LogisticRegression(C=np.inf, max_iter=2000).fit(_logit(model.predict_proba(Fval)[:, 1])[:, None], Yval)
    return lambda p: r.predict_proba(_logit(p)[:, None])[:, 1]


class Matcher:
    """Draw real training respondents matched to a simulated training set.

    sev: same items 1-8 total as each simulated response;
    sevlab: same items 1-8 total and the same item-9 answer (present/absent).
    The nearest total is used when no real respondent has the exact one.
    """
    def __init__(self, F, Y):
        self.t, self.y = F.sum(1), Y
        self.idx = {}
        for lab in (None, 0, 1):
            m = np.ones(len(Y), bool) if lab is None else (Y == lab)
            ts = np.unique(self.t[m])
            self.idx[lab] = {u: np.flatnonzero(m & (self.t == ts[np.abs(ts - u).argmin()])) for u in range(25)}

    def draw(self, t8, y, rng, label=False):
        keys = y if label else [None] * len(t8)
        return np.array([rng.choice(self.idx[k][int(u)]) for u, k in zip(t8, keys)])
