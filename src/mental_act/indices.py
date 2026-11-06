"""Base-rate validity probes, calibrated on NHANES.

These operationalise three of the published detection strategies from the
malingering literature WITHOUT using any proprietary instrument:

  rare_symptom       -- endorsing symptom levels that real respondents at the
                        same severity rarely endorse   ("Rare Symptoms")
  symptom_combination-- endorsing symptom PAIRS that rarely co-occur at the
                        same severity                  ("Rare Combinations")
  extreme_severity   -- answering at the top category more often than real
                        respondents with the same total ("Extreme Symptomatology")

Everything is conditioned on the respondent's own total score. Without that
conditioning the indices would simply re-measure severity, and a severely
depressed real patient would look like a faker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PHQ9_ITEMS, SEVERITY_BANDS

SMOOTH = 0.5          # Laplace smoothing so unseen cells give finite surprisal
MIN_BIN = 200         # collapse to severity band if an exact total is sparser


def band_of(total: int, bands=None) -> str:
    for lo, hi, name in (bands or SEVERITY_BANDS):
        if lo <= total <= hi:
            return name
    return "unknown"


# --------------------------------------------------------------- reference


def build_reference(df: pd.DataFrame, items: list[str] = PHQ9_ITEMS,
                    bands=None) -> dict:
    """Empirical conditional distributions from the human reference sample.

    Conditioning variable is the exact PHQ-9 total where there are enough
    respondents, otherwise the severity band.
    """
    X = df[items].to_numpy(int)
    totals = X.sum(axis=1)
    J, K = len(items), 4

    counts = pd.Series(totals).value_counts()
    use_exact = {int(t) for t, n in counts.items() if n >= MIN_BIN}

    def key(t: int) -> str:
        return f"t{int(t)}" if int(t) in use_exact else f"b{band_of(int(t), bands)}"

    keys = np.array([key(t) for t in totals])

    ref: dict[str, dict] = {}
    for k in np.unique(keys):
        m = keys == k
        Xk = X[m]
        n = len(Xk)

        # P(item j = category c)
        item_cat = np.zeros((J, K))
        for j in range(J):
            for c in range(K):
                item_cat[j, c] = ((Xk[:, j] == c).sum() + SMOOTH) / (n + SMOOTH * K)

        # P(item i >= 2 AND item j >= 2)
        hi = Xk >= 2
        pair = np.zeros((J, J))
        for i in range(J):
            for j in range(J):
                pair[i, j] = (np.logical_and(hi[:, i], hi[:, j]).sum() + SMOOTH) / (n + SMOOTH * 2)

        # distribution of "share of endorsed items answered at the top category"
        endorsed = (Xk > 0).sum(axis=1)
        top = (Xk == K - 1).sum(axis=1)
        es = np.divide(top, endorsed, out=np.zeros(n, float), where=endorsed > 0)

        ref[k] = {"n": int(n), "item_cat": item_cat, "pair": pair, "es_dist": es}

    return {"ref": ref, "items": items, "use_exact": sorted(use_exact),
            "bands": bands or SEVERITY_BANDS, "n_total": int(len(X))}


def _lookup(reference: dict, total: int) -> dict:
    k = f"t{int(total)}"
    if k not in reference["ref"]:
        k = f"b{band_of(int(total), reference.get('bands'))}"
    if k not in reference["ref"]:                     # fall back to nearest band
        k = min(reference["ref"], key=lambda kk: reference["ref"][kk]["n"] * -1)
    return reference["ref"][k]


# ----------------------------------------------------------------- indices


def compute(X: np.ndarray, reference: dict) -> pd.DataFrame:
    """All three base-rate indices for each row of X (n, 9), 0..3 coded."""
    X = np.asarray(X, int)
    N, J = X.shape
    totals = X.sum(axis=1)

    rs_surp = np.zeros(N)      # rare-symptom surprisal, -sum log P(x_j | total)
    rs_cnt = np.zeros(N)       # count of cells with base rate < 5%
    sc_surp = np.zeros(N)      # rare-combination surprisal over endorsed pairs
    sc_cnt = np.zeros(N)
    es_raw = np.zeros(N)       # share of endorsed items at the top category
    es_pct = np.zeros(N)       # its percentile among real respondents, same total

    for i in range(N):
        r = _lookup(reference, totals[i])
        ic, pr = r["item_cat"], r["pair"]

        p = ic[np.arange(J), X[i]]
        rs_surp[i] = -np.log(p).sum()
        rs_cnt[i] = (p < 0.05).sum()

        hi = np.where(X[i] >= 2)[0]
        if len(hi) >= 2:
            ps = [pr[a, b] for k, a in enumerate(hi) for b in hi[k + 1:]]
            sc_surp[i] = -np.log(ps).sum()
            sc_cnt[i] = sum(1 for q in ps if q < 0.05)

        endorsed = (X[i] > 0).sum()
        es_raw[i] = (X[i] == 3).sum() / endorsed if endorsed else 0.0
        es_pct[i] = float((r["es_dist"] <= es_raw[i]).mean())

    return pd.DataFrame({
        "total": totals,
        "phq9_total": totals,   # backwards-compatible alias
        "rare_symptom_surprisal": rs_surp,
        "rare_symptom_count": rs_cnt,
        "symptom_combo_surprisal": sc_surp,
        "symptom_combo_count": sc_cnt,
        "extreme_severity": es_raw,
        "extreme_severity_pctile": es_pct,
    })
