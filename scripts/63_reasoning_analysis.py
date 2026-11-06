"""Reasoning effort as a variable (review W6), for every experiment that GPT-5-mini
or Gemini 2.5 Flash took part in, Kimi-K2.5 with reasoning off and on, and Gemini 2.5 Pro
at its minimum thinking budget and its API default. The main run has reasoning off, or at
the minimum for Gemini 2.5 Pro (scripts/65); the API-default run is eff__<exp>__<tag>__default;
the other efforts are the eff__ runs (scripts/61, 62). Every measure is computed as
in the main analysis:
  reasoning_effort.csv            PHQ-9 (Table 3 measures, calibration, anchoring),
                                  GAD-7, PCL-5 roles ("not at all" ratio and grouped-CV
                                  AUC against the genuine patients, permutation p, as in
                                  scripts/37 via mental_act.feign; feigning shift),
                                  screening model trained on the PHQ-9 responses
  reasoning_effort_refs.csv       nine-reference protocol of Table 1 (scripts/52)
  reasoning_effort_gemini.csv     Gemini Flash and Pro: interventions, published personas,
                                  interviews (rater GPT-4.1, as in Figure 5)
"""
import sys, json, zlib
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mental_act import analysis as A, feign as FG
from joblib import Parallel
PAR = Parallel(n_jobs=FG.JOBS)
from mental_act.config import RUNS, TABLES, DATA_PROC, PHQ9_ITEMS

L52 = {"gpt5mini": "GPT-5-mini", "gemini25flash": "Gemini-2.5-Flash", "kimik25": "Kimi-K2.5", "gemini25pro": "Gemini-2.5-Pro"}   # model names used for seeds in scripts/52
# (label, effort): None = the main run (reasoning off, scripts/65); "default" = the API default, kept as eff__..__default
MODELS = [("GPT-5-mini", "gpt5mini", [("minimal", None), ("low", "low"), ("medium", "default"), ("high", "high")]),
          ("Gemini 2.5 Flash", "gemini25flash", [("off", None), ("low", "low"), ("dynamic", "default"), ("high", "high")]),
          ("Kimi-K2.5", "kimik25", [("off", None), ("on", "default")]),
          # Gemini 2.5 Pro cannot turn thinking off: main run at its minimum budget (128 tokens, scripts/68)
          ("Gemini 2.5 Pro", "gemini25pro", [("min", None), ("dynamic", "default")])]
MAIN_RUN = {(m[0], g): r for m in A.ALL_MODELS for g, r in (("phq9", m[1]), ("gad7", m[2]))}   # main PHQ-9/GAD-7 run names
from mental_act.figlong import TAG2LABEL
TARGET = {"none": (0, 4), "mild": (5, 9), "moderate": (10, 14), "moderately_severe": (15, 19), "severe": (20, 27)}

def name(exp, tag, eff):
    if eff is None and exp in ("phq9", "gad7"): return MAIN_RUN[(TAG2LABEL[tag], exp)]
    if exp.startswith("interview-"):
        return f"eff__{exp}__{tag}__{eff}" if eff else f"interview__{exp[10:]}__{tag}"
    return f"eff__{exp}__{tag}__{eff}" if eff else f"{exp}__{tag}"

def load(n):
    p = RUNS / f"{n}.jsonl"
    if not p.exists(): return None
    d = pd.DataFrame([json.loads(l) for l in open(p) if l.strip()])
    return d[d.status == "ok"].drop_duplicates("trial_id").reset_index(drop=True)

def tokens(u, tag=""):
    """Reasoning tokens as itemised by the API; for Kimi (not itemised) the generated tokens."""
    u = u if isinstance(u, dict) else {}
    if tag == "kimik25": return u.get("completion_tokens", np.nan)
    det = u.get("completion_tokens_details") or {}
    if det.get("reasoning_tokens") is not None: return det["reasoning_tokens"]
    if u.get("total_tokens") is not None: return u["total_tokens"] - u.get("prompt_tokens", 0) - u.get("completion_tokens", 0)
    return np.nan

def anchoring(d):
    d = d[d.arm.isin(["naive", "coached"])]
    rec = [dict(sev=r.severity, item=it, pos=p + 1, y=r.responses[it], trial=r.trial_id)
           for _, r in d.iterrows() for p, it in enumerate(r.item_order)]
    if not rec: return np.nan
    D = pd.DataFrame(rec); D["resid"] = D.y - D.groupby(["sev", "item"]).y.transform("mean")
    f = D[D.pos == 1].set_index("trial").resid; rest = D[D.pos > 1].groupby("trial").resid.mean()
    j = pd.concat([f, rest], axis=1, keys=["first", "rest"]).dropna()
    return float(np.corrcoef(j["first"], j["rest"])[0, 1])

# ---- references ----
Xh9 = A.human_X("phq9"); Hb9 = A.in_band(Xh9, "phq9"); Hb7 = A.in_band(A.human_X("gad7"), "gad7")
# screening test set as in scripts/30_downstream.py: NHANES 2013-2023, items 1-8 total >= 5
from mental_act import downstream as DSP
_DS = DSP.split(); Fte, Yte, sym = _DS.Fte, _DS.Yte, _DS.sym
nh = pd.read_csv(DATA_PROC / "nhanes_phq9.csv"); HOLD = nh[nh.SEQN % 2 == 1][PHQ9_ITEMS].to_numpy(int)
def sep(M, H, mgroups=None):
    return A.separability_rep(M, H, mgroups=mgroups)[0] if len(M) >= 30 else np.nan

rows, refs, gem = [], [], []
# the nine-reference protocol of Table 1, reused from scripts/52
src52 = open("scripts/52_extra_analysis.py").read().split("\nrows = []")[0]
ns52 = {}; exec(compile(src52, "52_extra_analysis", "exec"), ns52)

for model, tag, levels in MODELS:
    for label, eff in levels:
        r = dict(model=model, effort=label, main=eff is None, api_default=eff == "default")
        d = load(name("phq9", tag, eff))
        if d is not None and len(d) >= 100:
            X = np.array(d.ordered.tolist(), int); M = A.in_band(X, "phq9"); t = X.sum(1)
            r.update(n=len(d), tokens=float(np.nanmean([tokens(u, tag) for u in d.usage])),
                     calib=100 * np.mean([TARGET[s][0] <= v <= TARGET[s][1] for v, s in zip(t, d.severity)]),
                     anchoring=anchoring(d), breadth=A.breadth(M).mean(), all_items=100 * (A.breadth(M) == 9).mean(),
                     diversity=A.diversity(M, 15, 24), auc=sep(M, Hb9))
            mdl = LogisticRegression(max_iter=2000).fit(X[:, :8], (X[:, 8] > 0).astype(int)) if len(np.unique(X[:, 8] > 0)) == 2 else None
            if mdl is not None:
                p = mdl.predict_proba(Fte[sym])[:, 1]; flag = p >= 0.5
                r.update(ds_auc=roc_auc_score(Yte[sym], p), ds_fpr=100 * (flag & (Yte[sym] == 0)).sum() / (Yte[sym] == 0).sum())
        d = load(name("gad7", tag, eff))
        if d is not None and len(d) >= 100:
            X = np.array(d.ordered.tolist(), int); M = A.in_band(X, "gad7")
            r.update(gad7_all_items=100 * (A.breadth(M) == 7).mean() if len(M) else np.nan, gad7_diversity=A.diversity(M, 12, 19), gad7_auc=sep(M, Hb7))
        d = load(name("pcl5", tag, eff))
        if d is not None and len(d) >= 100:
            P = d[d.arm == "patient"]; Fg = d[d.arm == "feigner"]
            XP, XF = np.array(P.ordered.tolist(), int), np.array(Fg.ordered.tolist(), int)
            seed = zlib.crc32((f"LLM: {tag}" + (f"|{eff}" if eff else "")).encode())   # default: the seed of scripts/37
            sp = FG.separability(XP, seed, n_sub=60, par=PAR); pr = FG.profile_ci(XP, seed)
            r.update(pcl5_auc=sp["auc"], pcl5_lo=sp["sub_lo"], pcl5_hi=sp["sub_hi"], pcl5_p=sp["p"],
                     pcl5_naa_ratio=pr["naa_ratio"], pcl5_naa_lo=pr["naa_lo"], pcl5_naa_hi=pr["naa_hi"],
                     pcl5_feigner_minus_lo=pr["feigner_minus_lo"], pcl5_undefined_draws=pr["undefined_draws"],
                     pcl5_dtotal=XF.sum(1).mean() - XP.sum(1).mean(),
                     pcl5_dextreme=100 * (XF == 4).mean() - 100 * (XP == 4).mean())
        rows.append(r)
        # Table 1 protocol on the nine references
        for key, (lab, dis, H, K, w, runf, arms) in ns52["REFS"].items():
            exp = {"phq9_nhanes": "phq9", "phq9_coach": "phq9", "gad7_nhis": "gad7"}.get(key, f"x_{key}")
            M, _ = ns52["load"](name(exp, tag, eff), arms)
            if M is None or len(M) < 30: continue
            c = ns52["compare"](M, H, K, w, zlib.crc32((f"{key}|{L52[tag]}" + (f"|{eff}" if eff else "")).encode()))   # default: seed of scripts/52
            refs.append(dict(model=model, effort=label, ref=key, all_items_m=c.get("all_items_m"), all_items_h=c.get("all_items_h"),
                             div_ratio=c["div_h"] / c["div_m"] if c.get("div_m") else np.nan, div_bins=c.get("div_bins"), auc=c.get("auc")))
        if tag not in ("gemini25flash", "gemini25pro"): continue
        g = dict(model=model, effort=label)
        d = load(name("fixes", tag, eff))
        if d is not None:
            for cond in ["label", "hetero", "biography", "numeric", "fewshot"]:
                s = d[d.arm == cond]
                if len(s) < 40: continue
                X = np.array(s.ordered.tolist(), int); M = X[(X.sum(1) >= 15) & (X.sum(1) <= 24)]
                g[f"fix_{cond}_div"] = A.diversity(X, 5, 27); g[f"fix_{cond}_auc"] = sep(M, HOLD)
        for src in ["talkdep", "cactus"]:
            d = load(name(f"persona_{src}", tag, eff))
            if d is None: continue
            X = np.array(d.ordered.tolist(), int); band = (X.sum(1) >= 15) & (X.sum(1) <= 24); M = X[band]
            # TalkDep: pairs of different personas, a persona in one CV fold (as in scripts/38, 49)
            gp = d.severity.to_numpy() if src == "talkdep" else None
            g.update({f"{src}_div": A.diversity(X, 5, 27, groups=gp), f"{src}_auc": sep(M, HOLD, None if gp is None else gp[band]),
                      f"{src}_si": 100 * (M[:, 8] > 0).mean() if len(M) else np.nan})
        H8 = Xh9[:, :8]
        for cond in ["free", "oracle", "grm"]:
            d = load(name(f"interview-{cond}", tag, eff))
            if d is None or len(d) < 20: continue
            d = A.drop_filtered_interviews(d); R = np.array(d["rated__gpt-4.1"].tolist(), int)
            g.update({f"iv_{cond}_n": len(d), f"iv_{cond}_all8": 100 * (A.breadth(R[:, :8]) == 8).mean(),
                      f"iv_{cond}_all8_h": A.at_same_totals(R[:, :8], H8, lambda H: 100 * (A.breadth(H) == 8).mean()),
                      f"iv_{cond}_div": A.diversity(R[:, :8], 0, 24, min_n=3), f"iv_{cond}_div_h": A.diversity_same_totals(R[:, :8], H8)})
            if cond != "free":
                T = np.array(d.true_profile.tolist(), int)
                g.update({f"iv_{cond}_item_r": float(np.corrcoef(T.ravel(), R.ravel())[0, 1]),
                          f"iv_{cond}_si_agree": 100 * ((T[:, 8] > 0) == (R[:, 8] > 0)).mean()})
            else:
                g["iv_free_si_mild"] = 100 * (R[(d.severity == "mild").values][:, 8] > 0).mean()
        gem.append(g)
    print("done", model, flush=True)

pd.set_option("display.width", 250)
for nm, rr in (("reasoning_effort", rows), ("reasoning_effort_refs", refs), ("reasoning_effort_gemini", gem)):
    T = pd.DataFrame(rr); T.to_csv(TABLES / f"{nm}.csv", index=False)
    print(f"\n=== {nm} ==="); print(T.round(2).to_string(index=False))
