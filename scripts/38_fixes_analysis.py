"""Do targeted interventions fix the flat profile?

Evaluation reference is the ODD-SEQN half of NHANES; few-shot exemplars came
only from the even half, so no exemplar can be matched back to itself.
Published simulator personas (TalkDep, CACTUS; scripts 46/47) replace the
one-line persona and are scored the same way as conditions "talkdep", "cactus", "cactus_cm";
they have no instructed severity, so calibration and SI by severity are not defined."""
import sys, json, glob
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A
from mental_act.config import DATA_PROC, PHQ9_ITEMS, RUNS, TABLES

df = pd.read_csv(DATA_PROC / "nhanes_phq9.csv")
HOLD = df[df.SEQN % 2 == 1][PHQ9_ITEMS].to_numpy(int)
EVEN = df[df.SEQN % 2 == 0][PHQ9_ITEMS].to_numpy(int)
HB = HOLD[(HOLD.sum(1) >= 15) & (HOLD.sum(1) <= 24)]

def null_auc(M, draws=5):
    """Chance level of this separability test at this sample size: real
    respondents from the other half, matched to the model's totals, against
    the held-out half."""
    th = EVEN.sum(1); out = []
    for k in range(draws):
        rng = np.random.default_rng(k)
        picks = [rng.choice(np.where(th == t)[0]) for t in M.sum(1)]
        out.append(A.separability(EVEN[picks], HOLD, seed=k))
    return float(np.mean(out))

def band_shape(M):
    """Totals 15-24: "not at all" on items 1-8, suicidality endorsed, all nine endorsed."""
    return dict(zero18=100 * (M[:, :8] == 0).mean(), si_band=100 * (M[:, 8] > 0).mean(),
                all9_band=100 * (A.breadth(M) == 9).mean(), breadth_band=A.breadth(M).mean())
BANDS = {"mild": (5, 9), "moderate": (10, 14), "moderately_severe": (15, 19), "severe": (20, 27)}


# diversity over all symptomatic totals: 160 trials per cell leave too few per total inside 15-24
HDIV = A.diversity(HOLD, 5, 27)
rows = []
for f in sorted(glob.glob(str(RUNS / "fixes__*.jsonl"))):
    tag = f.split("fixes__")[1][:-6]
    d = pd.DataFrame([json.loads(l) for l in open(f)])
    meta = json.loads(open(f.replace(".jsonl", ".meta.json")).read()) if glob.glob(f.replace(".jsonl", ".meta.json")) else {}
    for cond in ["label", "hetero", "numeric", "biography", "fewshot"]:
        s = d[d.arm == cond]; n_all = s.trial_id.nunique()
        s = s[s.status == "ok"].drop_duplicates("trial_id")
        if len(s) < 40: continue
        X = np.array(s["ordered"].tolist(), int); tot = X.sum(1)
        hit = np.mean([BANDS[v][0] <= t <= BANDS[v][1] for t, v in zip(tot, s.severity)]) * 100
        M = X[(tot >= 15) & (tot <= 24)]
        r = dict(model=tag, cond=cond, n=len(s), errors=n_all - len(s), calib=hit,
                 breadth=A.breadth(X).mean(), zero=100 * (M == 0).mean(),
                 uneven=A.spikiness(X).mean(), diversity=A.diversity(X, 5, 27),
                 diversity_h=A.diversity_same_totals(X[(tot >= 5) & (tot <= 27)], HOLD, min_n=8),
                 si_modsev=100 * (X[s.severity.isin(["moderately_severe", "severe"]).values][:, 8] > 0).mean(),
                 **band_shape(M),
                 auc_null=null_auc(M) if len(M) >= 30 else np.nan)
        # separability averaged over 10 matchings / fold splits; the s.d. is the
        # run-to-run noise of this test at this sample size
        if len(M) >= 30:
            a = [A.separability(M, HOLD, seed=k) for k in range(10)]
            r.update(auc=float(np.mean(a)), auc_sd=float(np.std(a, ddof=1)))
        else:
            r.update(auc=np.nan, auc_sd=np.nan)
        if cond == "label" and len(M) >= 30:
            r["auc18"] = float(np.mean([A.separability(M[:, :8], HOLD[:, :8], seed=k) for k in range(10)]))
        if cond == "fewshot":
            copies = 0
            for tid, v in zip(s.trial_id, X):
                ex = meta.get(tid) or []
                copies += any(list(v) == list(e) for e in ex)
            r["copy_rate"] = 100 * copies / len(s)
        rows.append(r)

for tag in sorted({r["model"] for r in rows}):
    for cond in ["talkdep", "cactus", "cactus_cm"]:
        p = RUNS / f"persona_{cond}__{tag}.jsonl"
        if not p.exists(): continue
        d = pd.DataFrame([json.loads(l) for l in open(p)]); n_all = d.trial_id.nunique()
        s = d[d.status == "ok"].drop_duplicates("trial_id")
        X = np.array(s["ordered"].tolist(), int); tot = X.sum(1); band = (tot >= 15) & (tot <= 24); M = X[band]
        # TalkDep personas describe individuals and are answered 30 times each: diversity counts
        # only pairs from different personas, and a persona's responses stay in one CV fold
        g = s.severity.to_numpy() if cond == "talkdep" else None
        sub = (tot >= 5) & (tot <= 27)
        r = dict(model=tag, cond=cond, n=len(s), errors=n_all - len(s), breadth=A.breadth(X).mean(),
                 zero=100 * (M == 0).mean(), uneven=A.spikiness(X).mean(), diversity=A.diversity(X, 5, 27, groups=g),
                 diversity_h=A.diversity_same_totals(X[sub], HOLD, min_n=8, groups=None if g is None else g[sub]),
                 n_band=len(M), auc_null=null_auc(M) if len(M) >= 30 else np.nan, **band_shape(M))
        gM = None if g is None else g[band]
        a = [A.separability(M, HOLD, seed=k, mgroups=gM) for k in range(10)]
        r.update(auc=float(np.mean(a)), auc_sd=float(np.std(a, ddof=1)))
        # the same test without item 9 (suicidality), matched on the total of items 1-8
        r["auc18"] = float(np.mean([A.separability(M[:, :8], HOLD[:, :8], seed=k, mgroups=gM) for k in range(10)]))
        rows.append(r)

rows.insert(0, dict(model="REAL (held-out half)", cond="-", n=len(HB), breadth=A.breadth(HB).mean(),
                    zero=100 * (HB == 0).mean(), uneven=A.spikiness(HB).mean(), diversity=HDIV, **band_shape(HB)))
T = pd.DataFrame(rows); T.to_csv(TABLES / "fixes.csv", index=False)
T = T[T.cond != "-"]
print(f"held-out NHANES reference: diversity {HDIV:.2f}, zero% {100*(HOLD[HOLD.sum(1)>=5]==0).mean():.1f}, "
      f"uneven {A.spikiness(HOLD[HOLD.sum(1)>=5]).mean():.2f}, breadth(>=5) {A.breadth(HOLD[HOLD.sum(1)>=5]).mean():.2f}\n")
print(T.to_string(index=False, float_format=lambda v: f"{v:6.2f}", na_rep="-"))

# mean change vs the label baseline across models, with a 95% t-interval over models
from scipy import stats
summ = []
base = T[T.cond == "label"].set_index("model")
for c in ["hetero", "biography", "numeric", "fewshot", "talkdep", "cactus", "cactus_cm"]:
    x = T[T.cond == c].set_index("model")
    for k in ["calib", "diversity", "zero", "auc", "si_modsev"]:
        dd = (x[k] - base[k]).dropna()
        h = stats.t.ppf(.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else np.nan
        summ.append(dict(cond=c, measure=k, n_models=len(dd), mean=dd.mean(), lo=dd.mean() - h, hi=dd.mean() + h))
pd.DataFrame(summ).to_csv(TABLES / "fixes_summary.csv", index=False)
print(pd.DataFrame(summ).to_string(index=False, float_format=lambda v: f"{v:+7.3f}"))

print("\n=== mean change vs label baseline, across models ===")
base = T[T.cond == "label"].set_index("model")
for c in ["hetero", "numeric", "biography", "fewshot"]:
    x = T[T.cond == c].set_index("model")
    common = x.index.intersection(base.index)
    dd = (x.loc[common, ["calib","breadth","zero","uneven","diversity","auc","si_modsev"]] -
          base.loc[common, ["calib","breadth","zero","uneven","diversity","auc","si_modsev"]]).mean()
    print(f"  {c:10s} " + "  ".join(f"{k} {v:+6.2f}" for k, v in dd.items()))
