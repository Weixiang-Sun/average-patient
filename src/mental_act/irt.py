"""Graded Response Model (Samejima) + person-fit statistics.

Two families, deliberately:
  * lz   -- parametric, model-based (needs the GRM fit)
  * G^p  -- nonparametric Guttman errors (needs only item ordering)

Both are referenced against the EMPIRICAL NHANES distribution rather than
asymptotic theory. With ~42k real respondents we do not need the N(0,1)
approximation for lz (which is anyway biased when theta is estimated).
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ GRM


def _pstar(theta: np.ndarray, a: float, b: np.ndarray) -> np.ndarray:
    """P(X >= k | theta) for k = 1..K-1.  shape (n_theta, K-1)"""
    z = a * (theta[:, None] - b[None, :])
    return 1.0 / (1.0 + np.exp(-z))


def cat_probs(theta: np.ndarray, a: float, b: np.ndarray) -> np.ndarray:
    """P(X = k | theta) for k = 0..K-1.  shape (n_theta, K)"""
    ps = _pstar(theta, a, b)
    ones = np.ones((len(theta), 1))
    zeros = np.zeros((len(theta), 1))
    upper = np.hstack([ones, ps])        # P(X>=0)=1, P(X>=1), ...
    lower = np.hstack([ps, zeros])       # P(X>=1), ..., P(X>=K)=0
    return np.clip(upper - lower, 1e-12, 1.0)


def _quad(n_nodes: int = 61, lo: float = -5.0, hi: float = 5.0):
    x = np.linspace(lo, hi, n_nodes)
    w = np.exp(-0.5 * x**2)
    return x, w / w.sum()


def fit_grm(X: np.ndarray, n_cat: int = 4, n_nodes: int = 61,
            max_iter: int = 200, tol: float = 1e-5, verbose: bool = False):
    """Marginal-ML (EM) fit of the graded response model.

    X : (N, J) integer responses in 0..n_cat-1
    Returns dict with a (J,) and b (J, n_cat-1).
    """
    from scipy.optimize import minimize

    N, J = X.shape
    nodes, prior = _quad(n_nodes)

    a = np.ones(J)
    b = np.tile(np.linspace(-1.0, 1.0, n_cat - 1), (J, 1))

    prev = -np.inf
    for it in range(max_iter):
        # ---- E step: posterior over nodes, and expected category counts
        loglik_nodes = np.zeros((N, n_nodes))
        for j in range(J):
            P = cat_probs(nodes, a[j], b[j])          # (nodes, K)
            loglik_nodes += np.log(P[:, X[:, j]].T)   # (N, nodes)

        m = loglik_nodes.max(axis=1, keepdims=True)
        post = np.exp(loglik_nodes - m) * prior[None, :]
        ll = float((np.log(post.sum(axis=1)) + m.ravel()).sum())
        post /= post.sum(axis=1, keepdims=True)

        # r[j, node, k] = expected count of category k at node, item j
        r = np.zeros((J, n_nodes, n_cat))
        for j in range(J):
            for k in range(n_cat):
                r[j, :, k] = post[X[:, j] == k].sum(axis=0)

        # ---- M step: per-item maximisation
        for j in range(J):
            rj = r[j]

            def nll(par):
                aj = np.exp(par[0])                       # a > 0
                bj = np.cumsum(np.concatenate([[par[1]], np.exp(par[2:])]))  # ordered
                P = cat_probs(nodes, aj, bj)
                return -float((rj * np.log(P)).sum())

            x0 = np.concatenate([[np.log(a[j])], [b[j, 0]],
                                 np.log(np.maximum(np.diff(b[j]), 1e-3))])
            res = minimize(nll, x0, method="Nelder-Mead",
                           options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4})
            a[j] = np.exp(res.x[0])
            b[j] = np.cumsum(np.concatenate([[res.x[1]], np.exp(res.x[2:])]))

        if verbose and it % 5 == 0:
            print(f"  EM iter {it:3d}  loglik={ll:,.1f}")
        if abs(ll - prev) < tol * abs(ll):
            break
        prev = ll

    return {"a": a, "b": b, "n_cat": n_cat, "loglik": ll, "n_iter": it + 1}


def eap_theta(X: np.ndarray, fit: dict, n_nodes: int = 61) -> np.ndarray:
    """Expected a-posteriori theta for each response vector."""
    nodes, prior = _quad(n_nodes)
    N, J = X.shape
    ll = np.zeros((N, n_nodes))
    for j in range(J):
        P = cat_probs(nodes, fit["a"][j], fit["b"][j])
        ll += np.log(P[:, X[:, j]].T)
    m = ll.max(axis=1, keepdims=True)
    post = np.exp(ll - m) * prior[None, :]
    post /= post.sum(axis=1, keepdims=True)
    return post @ nodes


# ------------------------------------------------------- person fit: lz


def lz(X: np.ndarray, fit: dict, theta: np.ndarray | None = None) -> np.ndarray:
    """Standardised log-likelihood person-fit statistic.

    Negative values = response pattern less likely than the model expects
    (aberrant); strongly positive = *too* consistent ("over-fitting" pattern,
    itself a known signature of manufactured responses).
    """
    theta = eap_theta(X, fit) if theta is None else theta
    N, J = X.shape
    l0 = np.zeros(N)
    exp_l = np.zeros(N)
    var_l = np.zeros(N)

    for j in range(J):
        P = cat_probs(theta, fit["a"][j], fit["b"][j])     # (N, K)
        logP = np.log(P)
        l0 += logP[np.arange(N), X[:, j]]
        e = (P * logP).sum(axis=1)
        e2 = (P * logP**2).sum(axis=1)
        exp_l += e
        var_l += e2 - e**2

    return (l0 - exp_l) / np.sqrt(np.maximum(var_l, 1e-12))


# ------------------------------------ person fit: polytomous Guttman errors


def guttman_errors(X: np.ndarray, item_order: np.ndarray) -> np.ndarray:
    """Raw polytomous Guttman error count G^p (Emons 2008).

    item_order : item indices sorted from most-endorsed ("easiest") to least.
    A Guttman error is scoring higher on a harder item than on an easier one.
    """
    Xo = X[:, item_order]
    N, J = Xo.shape
    G = np.zeros(N)
    for i in range(J - 1):
        for j in range(i + 1, J):
            # item i is easier than item j -> expect Xo[:,i] >= Xo[:,j]
            G += np.maximum(0, Xo[:, j] - Xo[:, i])
    return G


def _max_guttman(total: int, J: int, K: int) -> float:
    """Max attainable G^p for a given total score, by placing all the score
    mass on the hardest items."""
    resp = np.zeros(J)
    remaining = total
    for j in range(J - 1, -1, -1):          # fill hardest items first
        put = min(K - 1, remaining)
        resp[j] = put
        remaining -= put
        if remaining <= 0:
            break
    G = 0.0
    for i in range(J - 1):
        for j in range(i + 1, J):
            G += max(0, resp[j] - resp[i])
    return G


def guttman_normed(X: np.ndarray, item_order: np.ndarray, n_cat: int = 4) -> np.ndarray:
    """G^p normalised by the maximum attainable at the same total score.

    Normalising matters: without it the statistic is confounded with severity,
    and we would simply be re-measuring the PHQ-9 total.
    """
    G = guttman_errors(X, item_order)
    J = X.shape[1]
    totals = X.sum(axis=1)
    gmax = np.array([_max_guttman(int(t), J, n_cat) for t in totals])
    out = np.divide(G, gmax, out=np.zeros_like(G, dtype=float), where=gmax > 0)
    return out
