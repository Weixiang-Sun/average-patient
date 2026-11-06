"""Do richer, published simulator personas remove the flat, uniform profile?

For every model and both questionnaires (PHQ-9 against NHANES, GAD-7 against NHIS),
compares the responses under
  one-line   the main experiment's severity persona (all arms)
  talkdep    12 clinician-validated TalkDep depression personas, 30 runs each
  cactus     400 distinct CACTUS counselling-client intake forms, 1 run each
  cactus_cm  the same 400 clients with the automatic thought and cognitive distortions
             of their CACTUS case formulation
with real respondents at the same totals. Only responses with total >= 5.
  breadth / all_items / zero : model value, and real respondents weighted to the model's totals
  diversity             : mean pairwise L1 at the same total, over totals where the model
                          has >= 8 responses; real value over the same totals. TalkDep
                          personas describe individuals answered 30 times each, so only
                          pairs from different personas count and a persona stays in one
                          CV fold
  distinct              : % distinct answer patterns; real respondents drawn at the same totals
  auc                   : separability from total-matched real respondents, 10 runs
-> published_personas.csv (one row per model x persona x instrument)
"""
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A, published_personas as PP
from mental_act.config import RUNS, TABLES, PHQ9_ITEMS, ROOT
from mental_act.figlong import MODELS

Xh = A.human_X("phq9"); th = Xh.sum(1)
TAG = {m[0]: m[3] for m in MODELS}                       # all 17 models
ONE = {(m[0], "phq9"): m[1] for m in A.ALL_MODELS} | {(m[0], "gad7"): m[2] for m in A.ALL_MODELS}
REF = {"phq9": A.human_X("phq9"), "gad7": A.human_X("gad7")}

def load(path):
    p = RUNS / f"{path}.jsonl"
    if not p.exists(): return None
    d = pd.DataFrame([json.loads(l) for l in open(p) if l.strip()])
    return d[d.status == "ok"].drop_duplicates("trial_id").reset_index(drop=True)

def pairwise(P, cap=400, rng=None, g=None):
    """Mean pairwise L1; with g (persona per response), only pairs from different personas."""
    if len(P) > cap:
        i = rng.choice(len(P), cap, replace=False); P = P[i]; g = None if g is None else g[i]
    dm = np.abs(P[:, None] - P[None]).sum(-1); iu = np.triu_indices(len(P), 1); v = dm[iu]
    if g is not None: v = v[g[iu[0]] != g[iu[1]]]
    return v.mean() if len(v) else np.nan

def distinct_real(X, H, rng, draws=50):
    """% distinct answer patterns among real respondents drawn at X's totals, one per response,
    without replacement within a total (two people can still give the same pattern)."""
    t, th_ = X.sum(1), H.sum(1); out = []
    for _ in range(draws):
        R = np.vstack([H[rng.choice(np.flatnonzero(th_ == u), c, replace=(th_ == u).sum() < c)]
                       for u, c in zip(*np.unique(t, return_counts=True)) if (th_ == u).any()])
        out.append(100 * len({tuple(r) for r in R}) / len(R))
    return float(np.mean(out))

def measures(X, H, g=None):
    """g: persona of each response for TalkDep (individual personas answered repeatedly):
    diversity then uses pairs from different personas, and a persona stays in one CV fold."""
    k5 = X.sum(1) >= 5; X = X[k5]; g = None if g is None else np.asarray(g)[k5]
    t = X.sum(1); th_ = H.sum(1); J = H.shape[1]; rng = np.random.default_rng(0)
    tt, cc = np.unique(t, return_counts=True); keep = [(u, c) for u, c in zip(tt, cc) if (th_ == u).any()]
    w = lambda fn: float(np.sum([fn(H[th_ == u]) * c for u, c in keep]) / sum(c for _, c in keep))
    bins = [u for u, c in zip(tt, cc) if c >= 8 and (th_ == u).sum() >= 8 and (g is None or len(set(g[t == u])) > 1)]
    dm = [pairwise(X[t == u], rng=rng, g=None if g is None else g[t == u]) for u in bins]; dh = [pairwise(H[th_ == u], rng=rng) for u in bins]
    auc, sd = A.separability_rep(X, H, mgroups=g)
    r = dict(n=len(X), mean_total=t.mean(), breadth=A.breadth(X).mean(), breadth_h=w(lambda Q: A.breadth(Q).mean()),
             all_items=100 * (A.breadth(X) == J).mean(), all_items_h=w(lambda Q: 100 * (A.breadth(Q) == J).mean()),
             zero=100 * (X == 0).mean(), zero_h=w(lambda Q: 100 * (Q == 0).mean()),
             n_bins=len(bins), diversity=np.mean(dm) if dm else np.nan, diversity_h=np.mean(dh) if dh else np.nan,
             distinct=100 * len({tuple(r) for r in X}) / len(X), distinct_h=distinct_real(X, H, np.random.default_rng(1)), auc=auc, auc_sd=sd)
    if J == 9: r.update(si=100 * (X[:, 8] > 0).mean(), si_h=w(lambda Q: 100 * (Q[:, 8] > 0).mean()))
    return r

rows = []
for lab, tag in TAG.items():
    for ins in ("phq9", "gad7"):
        for src in ("one-line", "talkdep", "cactus", "cactus_cm"):
            d = load(ONE[(lab, ins)] if src == "one-line" else PP.run_name(src, tag, ins))
            if d is None or len(d) < 50: continue
            rows.append(dict(model=lab, instrument=ins, persona=src, n_all=len(d),
                             **measures(np.array(d.ordered.tolist(), int), REF[ins], g=d.severity.to_numpy() if src == "talkdep" else None)))
T = pd.DataFrame(rows); T["div_ratio"] = T.diversity_h / T.diversity
T.to_csv(TABLES / "published_personas.csv", index=False)
pd.set_option("display.width", 250)
print(T.round(2).to_string(index=False))

# TalkDep: persona-level view. Key symptoms mapped to PHQ-9 items (approximate; symptoms
# with no PHQ-9 counterpart, e.g. irritability or worry, are left out)
KEY = {"Alex": ["DPQ070", "DPQ030", "DPQ050"], "Elena": ["DPQ020", "DPQ040"], "Ethan": ["DPQ010", "DPQ030", "DPQ070"],
       "Gabriel": ["DPQ060", "DPQ050"], "James": ["DPQ040", "DPQ060", "DPQ070"], "Laura": ["DPQ020", "DPQ060", "DPQ040", "DPQ070"],
       "Linda": ["DPQ060", "DPQ020", "DPQ070", "DPQ040"], "Marco": ["DPQ060", "DPQ080", "DPQ010", "DPQ070"],
       "Maria": ["DPQ020", "DPQ040", "DPQ060", "DPQ010"], "Maya": ["DPQ080", "DPQ060", "DPQ040"],
       "Noah": ["DPQ060", "DPQ010", "DPQ030"], "Priya": ["DPQ030", "DPQ060", "DPQ010"]}
BDI = {l.split()[0]: int(l.split()[1]) for l in open(ROOT / "data/external/talkdep/golden_truth_BDI_depression_ranking.txt") if l.strip()}
# real respondents at each response's exact total (A.at_same_totals): symptoms endorsed, and how
# often they endorse the persona's "other" items
Xh9 = A.human_X("phq9")
pr = []
for lab, tag in TAG.items():
    d = load(PP.run_name("talkdep", tag))
    if d is None: continue
    for name, g in d.groupby("severity"):
        X = np.array(g.ordered.tolist(), int); key = [PHQ9_ITEMS.index(i) for i in KEY[name]]; other = [j for j in range(9) if j not in key]
        pr.append(dict(model=lab, persona=name, bdi=BDI.get(name), n=len(X), total_mean=X.sum(1).mean(), total_sd=X.sum(1).std(),
                       breadth=A.breadth(X).mean(), breadth_real=A.at_same_totals(X, Xh9, lambda H: A.breadth(H).mean()),
                       at_one=100 * (X == 1).sum() / max((X > 0).sum(), 1),
                       at_one_real=A.at_same_totals(X, Xh9, lambda H: 100 * (H == 1).sum() / max((H > 0).sum(), 1)),
                       key_endorsed=100 * (X[:, key] > 0).mean(), other_endorsed=100 * (X[:, other] > 0).mean(),
                       other_real=A.at_same_totals(X, Xh9, lambda H, o=other: 100 * (H[:, o] > 0).mean()),
                       distinct=100 * len({tuple(r) for r in X}) / len(X)))
P = pd.DataFrame(pr)
P.to_csv(TABLES / "published_personas_talkdep.csv", index=False)
# CACTUS intake forms that mention suicidal thoughts or self-harm (the persona text given to the model)
import re
from mental_act import published_personas as PP
SUI = re.compile(r"suicid|self[- ]harm|kill (my|him|her)self|end (my|his|her) life|better off dead|want(s|ed)? to die|"
                 r"wish(es|ed)? (i|he|she) (was|were) dead|(don't|do not|not) want to (live|be alive)|hurt(ing)? (my|him|her)self|"
                 r"take (my|his|her) (own )?life|no reason to live|ending it", re.I)
forms = PP.cactus_forms()
pd.DataFrame([dict(source="cactus", n_forms=len(forms), n_mention_suicidality=sum(bool(SUI.search(f)) for f in forms))]).to_csv(
    TABLES / "published_personas_si_mentions.csv", index=False)
print("\nTalkDep, per persona (median over models):")
print(P.groupby("persona")[["bdi", "total_mean", "total_sd", "breadth", "key_endorsed", "other_endorsed", "other_real", "distinct"]].median().round(1).to_string())
