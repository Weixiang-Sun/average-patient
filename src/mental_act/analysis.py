"""Shared loading + derived measures for the deep analysis."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import (DATA_PROC, GAD7_ITEMS, GAD7_LABELS, PHQ8_ITEMS,
                     PHQ9_ITEMS, PHQ9_LABELS, RUNS)

# ---------------------------------------------------------------- registry
PHQ9_MODELS = [("GPT-4o-mini", "gpt4omini_objschema", "OpenAI"),
               ("GPT-5.4-mini", "gpt54mini", "OpenAI"),
               ("GPT-5.2", "gpt52", "OpenAI"),
               ("Claude Haiku 4.5", "haiku", "Anthropic"),
               ("Grok-4.1-fast", "grok41", "xAI"),
               ("Llama-3.3-70B", "llama33_70b", "Meta"),
               ("Kimi-K2.5", "kimi_k25", "Moonshot")]
GAD7_MODELS = [("GPT-4o-mini", "gad_4om", "OpenAI"),
               ("GPT-5.2", "gad_g52", "OpenAI"),
               ("Claude Haiku 4.5", "gad_haiku", "Anthropic"),
               ("Grok-4.1-fast", "gad_grok", "xAI"),
               ("Llama-3.3-70B", "gad_llama", "Meta"),
               ("Kimi-K2.5", "gad_kimi", "Moonshot")]
# Extended registry: every model with PHQ-9 / GAD-7 data, incl. the Azure runs.
# family / size_rank support within-family scale comparisons.
ALL_MODELS = [
    # label,              phq9 run,               gad7 run,              vendor,     family,   size
    ("GPT-4o-mini",       "gpt4omini_objschema",  "gad_4om",             "OpenAI",   "GPT-4o", 1),
    ("GPT-4o",            "phq9__gpt4o",          "gad7__gpt4o",         "OpenAI",   "GPT-4o", 2),
    ("GPT-4.1-nano",      "phq9__gpt41nano",      "gad7__gpt41nano",     "OpenAI",   "GPT-4.1", 1),
    ("GPT-4.1-mini",      "phq9__gpt41mini",      "gad7__gpt41mini",     "OpenAI",   "GPT-4.1", 2),
    ("GPT-4.1",           "phq9__gpt41",          "gad7__gpt41",         "OpenAI",   "GPT-4.1", 3),
    ("GPT-5-mini",        "phq9__gpt5mini",       "gad7__gpt5mini",      "OpenAI",   "GPT-5",  1),
    ("GPT-5.2",           "gpt52",                "gad_g52",             "OpenAI",   "GPT-5",  2),
    ("GPT-5.4-mini",      "gpt54mini",            "gad7__gpt54mini",     "OpenAI",   "GPT-5.4", 1),
    ("Claude Haiku 4.5",  "haiku",                "gad_haiku",           "Anthropic","Claude", 1),
    ("Grok-4.1-fast",     "grok41",               "gad_grok",            "xAI",      "Grok",   1),
    ("Llama-3.3-70B",     "llama33_70b",          "gad_llama",           "Meta",     "Llama",  1),
    ("Kimi-K2.5",         "kimi_k25",             "gad_kimi",            "Moonshot", "Kimi",   1),
    ("Mistral-Large-3",   "phq9__mistrallarge3",  "gad7__mistrallarge3", "Mistral",  "Mistral",1),
    ("Phi-4",             "phq9__phi4",           "gad7__phi4",          "Microsoft","Phi",    1),
    ("Gemini-2.5-Flash-Lite", "phq9__gemini25flashlite", "gad7__gemini25flashlite", "Google", "Gemini-2.5", 1),
    ("Gemini-2.5-Flash",  "phq9__gemini25flash",  "gad7__gemini25flash", "Google",   "Gemini-2.5", 2),
    ("Gemini-2.5-Pro",    "phq9__gemini25pro",    "gad7__gemini25pro",   "Google",   "Gemini-2.5", 3),
]
# (label, run, vendor) lists over all 14 models, for the per-instrument analyses.
PHQ9_ALL = [(m[0], m[1], m[3]) for m in ALL_MODELS]
GAD7_ALL = [(m[0], m[2], m[3]) for m in ALL_MODELS]
VENDOR_COLOR = {"OpenAI": "#0072B2", "Anthropic": "#E69F00", "xAI": "#CC79A7",
                "Meta": "#009E73", "Moonshot": "#D55E00", "Mistral": "#56B4E9",
                "Microsoft": "#8C6D1F", "Google": "#6A3D9A"}

INSTR = {
    "phq9": dict(items=PHQ9_ITEMS, labels=PHQ9_LABELS, J=9, band=(15, 24),
                 human_csv="nhanes_phq9.csv", idx_csv="human_indices.csv",
                 norms="", survey="NHANES", condition="depression", maxs=27),
    "gad7": dict(items=GAD7_ITEMS, labels=GAD7_LABELS, J=7, band=(12, 19),
                 human_csv="nhis_gad7.csv", idx_csv="human_indices_gad7.csv",
                 norms="gad7", survey="NHIS", condition="anxiety", maxs=21),
    "phq8": dict(items=PHQ8_ITEMS, labels=None, J=8, band=(13, 21),
                 human_csv="nhis_phq8.csv", idx_csv="human_indices_phq8.csv",
                 norms="phq8", survey="NHIS", condition="depression", maxs=24),
}


def load_run(name: str) -> pd.DataFrame | None:
    f = RUNS / f"{name}.jsonl"
    if not f.exists():
        return None
    r = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    # filter to successes BEFORE de-duplicating: a trial that failed (e.g. rate
    # limit) and later succeeded on retry must keep its successful record
    d = pd.DataFrame(r)
    return d[d.status == "ok"].drop_duplicates("trial_id", keep="first").reset_index(drop=True)


def run_X(name: str) -> np.ndarray:
    return np.array(load_run(name)["ordered"].tolist(), int)


def diversity(X: np.ndarray, lo: int, hi: int, min_n: int = 8, cap: int = 400, seed: int = 0, groups=None) -> float:
    """Mean pairwise L1 distance between responses with the same total, averaged
    over the totals lo..hi that have at least min_n responses (at most cap per
    total, drawn without replacement). The one definition used everywhere.
    groups: the persona of each response when personas describe individuals and are
    answered repeatedly (TalkDep); only pairs from different personas then count, and
    totals without such a pair are skipped."""
    rng = np.random.default_rng(seed); out = []
    G = None if groups is None else np.asarray(groups)
    for t in range(lo, hi + 1):
        m = X.sum(1) == t; P = X[m]
        if len(P) < min_n: continue
        idx = rng.choice(len(P), min(len(P), cap), replace=False); P = P[idx]
        dm = np.abs(P[:, None] - P[None]).sum(-1); iu = np.triu_indices(len(P), 1); v = dm[iu]
        if G is not None:
            g = G[m][idx]; v = v[g[iu[0]] != g[iu[1]]]
            if not len(v): continue
        out.append(v.mean())
    return float(np.mean(out)) if out else np.nan


def at_same_totals(X: np.ndarray, H: np.ndarray, fn) -> float:
    """fn over real respondents H at exactly X's totals, weighted to X's totals
    (totals that no real respondent has are skipped)."""
    t, th = X.sum(1), H.sum(1); tt, cc = np.unique(t, return_counts=True)
    ok = np.array([(th == u).any() for u in tt]); tt, cc = tt[ok], cc[ok]
    return float(np.sum([fn(H[th == u]) * c for u, c in zip(tt, cc)]) / cc.sum())


def diversity_same_totals(X: np.ndarray, H: np.ndarray, min_n: int = 3, groups=None) -> float:
    """Real diversity over the totals where X has at least min_n responses (and, with
    groups, responses from at least two groups, as in diversity)."""
    t, th = X.sum(1), H.sum(1); tt, cc = np.unique(t, return_counts=True)
    two = (lambda u: True) if groups is None else (lambda u: len(set(np.asarray(groups)[t == u])) > 1)
    v = [diversity(H[th == u], u, u, min_n=min_n) for u, c in zip(tt, cc) if c >= min_n and (th == u).sum() >= min_n and two(u)]
    return float(np.mean(v)) if v else np.nan


def human_X(instr: str) -> np.ndarray:
    spec = INSTR[instr]
    return pd.read_csv(DATA_PROC / spec["human_csv"])[spec["items"]].to_numpy(int)


def in_band(X: np.ndarray, instr: str) -> np.ndarray:
    lo, hi = INSTR[instr]["band"]
    t = X.sum(1)
    return X[(t >= lo) & (t <= hi)]


# ------------------------------------------------------- derived measures
def breadth(X):        return (X > 0).sum(1)
def depth(X):          return np.array([r[r > 0].mean() if (r > 0).any() else np.nan for r in X])
def spikiness(X):      return X.std(axis=1, ddof=0)          # within-person profile SD
def top_share(X):      return (X == 3).sum(1) / np.maximum((X > 0).sum(1), 1)


def difficulty_order(X: np.ndarray) -> np.ndarray:
    """Mean item score = empirical 'easiness' ordering."""
    return X.mean(0)


def order_agreement(Xm: np.ndarray, Xh: np.ndarray) -> float:
    """Spearman rho between model and human item-easiness orderings.

    If a model endorses everything uniformly the human difficulty hierarchy
    (fatigue common, suicidality rare) collapses and rho drops toward 0.
    """
    return float(spearmanr(difficulty_order(Xm), difficulty_order(Xh)).statistic)


def corr_matrix(X: np.ndarray) -> np.ndarray:
    C = np.corrcoef(X.T)
    return np.nan_to_num(C, nan=0.0)


def corr_distance(Xm, Xh) -> float:
    """Frobenius distance between symptom covariance structures (off-diagonal)."""
    A, B = corr_matrix(Xm), corr_matrix(Xh)
    m = ~np.eye(A.shape[0], dtype=bool)
    return float(np.sqrt(((A[m] - B[m]) ** 2).mean()))


def nearest_total_match(X: np.ndarray, ref: np.ndarray, rng) -> np.ndarray:
    """Row indices into `ref`, one per row of X: a random reference respondent with
    the same total, or the nearest total if none. Drawn with replacement, so one
    person can be picked several times; callers must keep repeats in one CV fold."""
    tr = ref.sum(1); out = []
    for t in X.sum(1):
        d = np.abs(tr - t); out.append(rng.choice(np.where(d == d.min())[0]))
    return np.array(out, int)


def grouped_folds(y: np.ndarray, groups: np.ndarray, seed: int, n_splits: int = 5):
    """Stratified, shuffled folds that never split one group (one real respondent
    picked more than once) across training and test."""
    from sklearn.model_selection import StratifiedGroupKFold
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def separability(Xm: np.ndarray, Xh: np.ndarray, seed: int = 0, mgroups=None) -> float:
    """Cross-validated AUC for telling model responses from human responses
    using ONLY the item pattern, with the two groups matched on total score.
    A real respondent matched to several model responses stays in one fold, and so do
    the responses of one persona when mgroups gives it (TalkDep, answered repeatedly).

    1.0 = trivially distinguishable; 0.5 = indistinguishable.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    picks = nearest_total_match(Xm, Xh, rng)          # 1:1 on total
    X = np.vstack([Xm, Xh[picks]])
    y = np.r_[np.ones(len(Xm)), np.zeros(len(picks))]
    gm = np.arange(len(Xm)) if mgroups is None else pd.factorize(np.asarray(mgroups))[0]
    groups = np.r_[gm, len(Xm) + picks]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return float(cross_val_score(clf, X, y, groups=groups, cv=grouped_folds(y, groups, seed), scoring="roc_auc").mean())


def separability_rep(Xm: np.ndarray, Xh: np.ndarray, reps: int = 10, mgroups=None) -> tuple[float, float]:
    """Mean and s.d. of separability over `reps` matchings and fold splits.
    A single run varies by a few hundredths at a few hundred responses."""
    a = [separability(Xm, Xh, seed=k, mgroups=mgroups) for k in range(reps)]
    return float(np.mean(a)), float(np.std(a, ddof=1))


def boot_ci(x, fn=np.mean, n=2000, seed=0, q=(2.5, 97.5)):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    s = [fn(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return tuple(np.percentile(s, q))


def drop_filtered_interviews(d: pd.DataFrame) -> pd.DataFrame:
    """Exclude interviews in which the platform filter removed any answer to
    PHQ-9 items 1-8. Those items are missing (the rater could only infer them
    from context), so the whole interview is left out of the item-level
    analyses. A removed item-9 answer is handled separately: that interview
    is kept for items 1-8 and left out of the suicidality rate."""
    fi = d["filtered_items"] if "filtered_items" in d else pd.Series([[]] * len(d), index=d.index)
    keep = [not any(i != "DPQ090" for i in (v if isinstance(v, list) else [])) for v in fi]
    return d[keep]
