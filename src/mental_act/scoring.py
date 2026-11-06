"""Score any response matrix against the FROZEN human norms."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from . import indices, irt
from .config import DATA_PROC


def load_norms(path=None, tag: str = "") -> dict:
    """tag='' loads the PHQ-9 norms; tag='gad7'/'phq8' load NHIS norms."""
    if path is None:
        name = f"norms_{tag}_frozen.pkl" if tag else "norms_frozen.pkl"
        p = DATA_PROC / name
    else:
        p = path
    return pickle.loads(p.read_bytes())


METRICS = ["lz", "guttman_normed", "rare_symptom_surprisal",
           "symptom_combo_surprisal", "extreme_severity"]


def score(X: np.ndarray, norms: dict) -> pd.DataFrame:
    """All frozen indices for an (n, 9) response matrix coded 0..3."""
    X = np.asarray(X, int)
    out = indices.compute(X, norms["reference"])
    out["lz"] = irt.lz(X, norms["grm"])
    out["theta"] = irt.eap_theta(X, norms["grm"])
    out["guttman"] = irt.guttman_errors(X, norms["item_order"])
    out["guttman_normed"] = irt.guttman_normed(X, norms["item_order"])
    return out


def matched_human_reference(human: pd.DataFrame, totals, rng=None,
                            n_per: int = 1) -> pd.DataFrame:
    """Draw humans with the SAME PHQ-9 totals as the rows being compared.

    Comparing unmatched groups would confound the indices with severity: a
    genuinely severe patient endorses more, more extremely, and in rarer
    combinations than a mild one. Matching on total removes that.
    """
    rng = rng or np.random.default_rng(0)
    col = "total" if "total" in human.columns else "phq9_total"
    picks = []
    for t in np.asarray(totals):
        pool = human.index[human[col] == t]
        if len(pool) == 0:                      # nearest available total
            d = (human[col] - t).abs()
            pool = human.index[d == d.min()]
        picks.extend(rng.choice(pool, size=n_per, replace=len(pool) < n_per))
    return human.loc[picks].reset_index(drop=True)


def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def auc(pos, neg) -> float:
    """P(a random 'pos' scores higher than a random 'neg')."""
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    r = rankdata(np.concatenate([pos, neg]))
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def n_for_power(d: float, power: float = 0.8, alpha: float = 0.05) -> float:
    """Per-group N for a two-sample t-test at the given effect size."""
    from scipy.stats import norm
    if d == 0:
        return np.inf
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(np.ceil(2 * (z / d) ** 2))
