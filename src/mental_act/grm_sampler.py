"""Symptom profiles drawn from a graded response model instead of real rows.

Uses only the 9 discriminations and 27 thresholds of the PHQ-9 GRM (the kind of
parameters published in IRT studies); no item-level data are needed at
generation time. theta ~ N(0, 1), items conditionally independent given theta,
rejection-sampled into a total-score band. Parameters are fit on NHANES
2005-2012 only (scripts/50_grm_heldout.py), so 2013-2023 is held out.
"""
from __future__ import annotations
import json
import numpy as np
from .config import DATA_PROC

def params():
    d = json.load(open(DATA_PROC / "grm_heldout.json"))
    return np.array(d["grm_a"]), np.array(d["grm_b"])

def sample(n: int, rng, a=None, b=None, theta=None) -> np.ndarray:
    if a is None: a, b = params()
    th = rng.standard_normal(n) if theta is None else theta
    # P(X >= k) for k = 1..3, then category probabilities
    ge = 1 / (1 + np.exp(-a[None, :, None] * (th[:, None, None] - b[None, :, :])))   # n x 9 x 3
    cum = np.concatenate([np.ones((n, len(a), 1)), ge, np.zeros((n, len(a), 1))], axis=2)
    p = cum[:, :, :-1] - cum[:, :, 1:]                                                 # n x 9 x 4
    u = rng.random((n, len(a), 1))
    return (u > np.cumsum(p, axis=2)).sum(2).astype(int)

def sample_band(lo: int, hi: int, n: int, rng, batch: int = 200_000) -> np.ndarray:
    out = []
    while sum(len(x) for x in out) < n:
        X = sample(batch, rng); t = X.sum(1); out.append(X[(t >= lo) & (t <= hi)])
    return np.vstack(out)[:n]
