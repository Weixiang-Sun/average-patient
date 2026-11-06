"""Conversational modality: rater-coded PHQ-9 from interview transcripts.

free    patient persona by severity (how simulators are used)
oracle  patient given a real NHANES profile; tests whether the rater pipeline
        can recover uneven profiles at all
talkdep patient given a full TalkDep persona (own table, interview_talkdep.csv)
When the patient and a rater are the same model, that rater is not used for
that patient (self-rating confound)."""
import sys, json, glob
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from mental_act import analysis as A
from mental_act.published_personas import talkdep_severity
from mental_act.config import RUNS, TABLES

Xh = A.human_X("phq9")
BANDS = {"mild": (5, 9), "moderate": (10, 14), "moderately_severe": (15, 19), "severe": (20, 27)}
BANDS5 = {"none": (0, 4), **BANDS}
TD = talkdep_severity()   # TalkDep persona -> (BDI-II, PHQ-9 band of its severity label)
RATERS = ["gpt-4.1", "claude-haiku-4-5"]
PATIENT_ID = {"gpt41": "gpt-4.1", "claudehaiku45": "claude-haiku-4-5"}

def diversity(X, groups=None):
    """Shared definition on items 1-8; bins need only 3 interviews because each
    condition has at most 80. groups: TalkDep persona, so that only interviews of
    different personas are compared."""
    return A.diversity(X[:, :8], 0, 24, min_n=3, groups=groups)

def prof(X, si_mask=None):
    """Profile stats. Where the platform filtered the patient's suicidality
    answer, item 9 is unobservable (the rater would infer it from context), so
    breadth/zero/unevenness use items 1-8 for those rows and SI excludes them."""
    si_mask = np.ones(len(X), bool) if si_mask is None else si_mask
    X8 = X[:, :8]
    return dict(n=len(X), total8=X8.sum(1).mean(), breadth8=A.breadth(X8).mean(),
                all8=100*(A.breadth(X8)==8).mean(), zero=100*(X8==0).mean(),
                uneven=A.spikiness(X8).mean(),
                si=100*(X[si_mask, 8] > 0).mean() if si_mask.any() else np.nan,
                si_filtered=100*(~si_mask).mean())

H8 = Xh[:, :8]
def same_totals(X, groups=None):
    """Real NHANES respondents at exactly the interviews' item 1-8 totals, weighted to
    those totals: all eight endorsed, "not at all", breadth, and diversity over the same totals."""
    X = X[:, :8]
    return dict(all8_h=A.at_same_totals(X, H8, lambda H: 100 * (A.breadth(H) == 8).mean()),
                zero_h=A.at_same_totals(X, H8, lambda H: 100 * (H == 0).mean()),
                breadth8_h=A.at_same_totals(X, H8, lambda H: A.breadth(H).mean()),
                diversity_h=A.diversity_same_totals(X, H8, groups=groups))
rows = []
Hs = Xh[Xh.sum(1) >= 5]
rows.append(dict(source="NHANES (self-report, total>=5)", rater="-", **prof(Hs), diversity=diversity(Hs)))
agree = []; sens = []
for f in sorted(glob.glob(str(RUNS / "interview__*.jsonl"))):
    _, cond, tag = f[:-6].split("__")
    d = pd.DataFrame([json.loads(l) for l in open(f)])
    d_all = d[d.status == "ok"].drop_duplicates("trial_id")
    d = A.drop_filtered_interviews(d_all)
    if len(d) < 20: continue
    if len(d) < len(d_all):
        for rater in RATERS:
            if PATIENT_ID.get(tag) == rater or f"rated__{rater}" not in d_all: continue
            Xa, Xk = (np.array(q[f"rated__{rater}"].tolist(), int) for q in (d_all, d))
            sens.append(dict(source=f"{cond}: {tag}", rater=rater, n_all=len(Xa), n_kept=len(Xk),
                             breadth8_all=A.breadth(Xa[:, :8]).mean(), breadth8_kept=A.breadth(Xk[:, :8]).mean(),
                             all8_all=100*(A.breadth(Xa[:, :8])==8).mean(), all8_kept=100*(A.breadth(Xk[:, :8])==8).mean(),
                             div_all=diversity(Xa), div_kept=diversity(Xk)))
    for rater in RATERS:
        if PATIENT_ID.get(tag) == rater or f"rated__{rater}" not in d: continue
        X = np.array(d[f"rated__{rater}"].tolist(), int)
        fi = d["filtered_items"] if "filtered_items" in d else pd.Series([[]] * len(d))
        si_ok = np.array([("DPQ090" not in (v if isinstance(v, list) else [])) for v in fi])
        g = d.severity.to_numpy() if cond == "talkdep" else None
        r = dict(source=f"{cond}: {tag}", rater=rater, **prof(X, si_ok), diversity=diversity(X, g), **same_totals(X, g))
        if cond == "free":
            r["calib"] = 100*np.mean([BANDS[s][0] <= t <= BANDS[s][1] for t, s in zip(X.sum(1), d.severity)])
        elif cond == "talkdep":
            r["calib"] = 100*np.mean([BANDS5[TD[n][1]][0] <= t <= BANDS5[TD[n][1]][1] for t, n in zip(X.sum(1), d.severity)])
            r["bdi_rho"] = spearmanr(X.sum(1), [TD[n][0] for n in d.severity]).statistic
            # personas whose label is moderate or above: all eight endorsed, against real respondents at their totals
            hi = np.array([TD[n][1] in ("moderate", "moderately_severe", "severe") for n in d.severity])
            r["all8_modplus"] = 100*(A.breadth(X[hi, :8]) == 8).mean()
            r["all8_modplus_h"] = A.at_same_totals(X[hi, :8], H8, lambda H: 100 * (A.breadth(H) == 8).mean())
            r["zero_lowlabel"] = 100*(X[~hi, :8] == 0).mean()
        else:
            T = np.array(d.true_profile.tolist(), int)
            r["item_r"] = pearsonr(T.ravel(), X.ravel()).statistic
            r["total_r"] = pearsonr(T.sum(1), X.sum(1)).statistic
            T8 = T[:, :8]
            r["true_breadth8"] = A.breadth(T8).mean(); r["true_uneven"] = A.spikiness(T8).mean()
            r["true_zero"] = 100*(T8==0).mean(); r["true_all8"] = 100*(A.breadth(T8)==8).mean()
            r["true_si"] = 100*(T[:, 8] > 0).mean(); r["true_div"] = diversity(T)
            # a symptom given at "several days" (1): share rated one level higher (2)
            r["given1_rated2"] = 100*(X[:, :8][T8 == 1] == 2).mean()
            g9, r9 = T[si_ok, 8] > 0, X[si_ok, 8] > 0
            if si_ok.sum() >= 20 and g9.any() and (~g9).any():
                po = (g9 == r9).mean(); pe = g9.mean() * r9.mean() + (1 - g9.mean()) * (1 - r9.mean())
                r["si_agree"] = 100 * po; r["si_kappa"] = (po - pe) / (1 - pe)
        rows.append(r)
    if all(f"rated__{x}" in d for x in RATERS):
        a, b = np.array(d[f"rated__{RATERS[0]}"].tolist()), np.array(d[f"rated__{RATERS[1]}"].tolist())
        agree.append(dict(file=f"{cond}:{tag}", exact=100*(a==b).mean(), within1=100*(np.abs(a-b)<=1).mean(),
                          total_r=pearsonr(a.sum(1), b.sum(1)).statistic))
T = pd.DataFrame(rows)
# exploratory GRM-profile interviews (patient given a profile sampled from the
# fitted GRM) go to their own table so the main tables keep free/oracle only
grm = T.source.str.startswith("grm:"); T[grm].to_csv(TABLES/"interview_grm.csv", index=False); T = T[~grm]
td = T.source.str.startswith("talkdep:"); T[td].to_csv(TABLES/"interview_talkdep.csv", index=False); T = T[~td]
agree = [a for a in agree if not a["file"].startswith(("grm:", "talkdep:"))]; sens = [s for s in sens if not s["source"].startswith(("grm:", "talkdep:"))]
T.to_csv(TABLES/"interview.csv", index=False)
pd.DataFrame(sens).to_csv(TABLES/"interview_sensitivity.csv", index=False)
pd.DataFrame(agree).to_csv(TABLES/"interview_agreement.csv", index=False)
print(T.to_string(index=False, float_format=lambda v: f"{v:6.2f}", na_rep="-"))
print("\n=== inter-rater agreement (GPT-4.1 vs Haiku) ===")
print(pd.DataFrame(agree).to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
print("\n=== sensitivity: including interviews with a filtered answer to items 1-8 ===")
print(pd.DataFrame(sens).to_string(index=False, float_format=lambda v: f"{v:6.2f}"))

# ---- given vs rated, pooled over patient models, per rater (oracle and GRM-sampled profiles) ----
# Separates presence (0 vs >0) from frequency (the exact level). A patient model is never rated by itself.
from sklearn.metrics import cohen_kappa_score
gr = []
for cond in ("oracle", "grm"):
    for rater in RATERS:
        Ts, Rs, tags = [], [], []
        for f in sorted(glob.glob(str(RUNS / f"interview__{cond}__*.jsonl"))):
            tag = f[:-6].split("__")[2]
            if PATIENT_ID.get(tag) == rater: continue
            d = A.drop_filtered_interviews(pd.DataFrame([json.loads(l) for l in open(f)]).query("status == 'ok'").drop_duplicates("trial_id"))
            if len(d) < 20 or f"rated__{rater}" not in d: continue
            Ts.append(np.array(d.true_profile.tolist(), int)[:, :8]); Rs.append(np.array(d[f"rated__{rater}"].tolist(), int)[:, :8]); tags.append(tag)
        T_, R_ = np.vstack(Ts).ravel(), np.vstack(Rs).ravel()
        gr.append(dict(cond=cond, rater=rater, patients=len(tags), n_items=len(T_), exact=100 * (T_ == R_).mean(),
                       within1=100 * (np.abs(T_ - R_) <= 1).mean(), qwk=cohen_kappa_score(T_, R_, weights="quadratic"),
                       kappa=cohen_kappa_score(T_, R_), shift=(R_ - T_).mean(),
                       presence_sens=100 * (R_[T_ > 0] > 0).mean(), presence_spec=100 * (R_[T_ == 0] == 0).mean(),
                       exact_present=100 * (R_[T_ > 0] == T_[T_ > 0]).mean(),
                       **{f"exact_given{k}": 100 * (R_[T_ == k] == k).mean() for k in range(4)},
                       **{f"up_given{k}": 100 * (R_[T_ == k] > k).mean() for k in range(3)}))
GR = pd.DataFrame(gr); GR.to_csv(TABLES / "interview_given_rated.csv", index=False)
print("\n=== given vs rated profile, items 1-8, pooled over patient models ===")
print(GR.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))

# ---- pipeline control: the interviewer and rater also act on real profiles ----
# Severity-persona interviews against real-profile interviews that went through the same
# interviewer and rater (items 1-8; totals pooled in bins of 3 because each condition has
# at most 80 interviews), next to the questionnaire ratio of the same models. Separability
# of the rated profiles from real respondents with the same totals, for every condition.
from mental_act.figlong import TAG2LABEL
def div_bins(X, w=3, min_n=3, g=None):
    """Diversity per bin of w total points; with g (TalkDep persona), pairs from different personas only."""
    t = X[:, :8].sum(1) // w; out = {}
    for b in np.unique(t):
        P = X[t == b][:, :8]
        if len(P) >= min_n:
            dm = np.abs(P[:, None] - P[None]).sum(-1); iu = np.triu_indices(len(P), 1); v = dm[iu]
            if g is not None:
                gb = np.asarray(g)[t == b]; v = v[gb[iu[0]] != gb[iu[1]]]
            if len(v): out[b] = v.mean()
    return out
AMq = pd.read_csv(TABLES / "all_models.csv").query("instrument == 'phq9'").set_index("model")
pl = []
for f in sorted(glob.glob(str(RUNS / "interview__free__*.jsonl"))):
    tag = f[:-6].split("__")[2]
    rater = "claude-haiku-4-5" if tag == "gpt41" else "gpt-4.1"
    load = lambda cond: A.drop_filtered_interviews(pd.DataFrame([json.loads(l) for l in open(RUNS / f"interview__{cond}__{tag}.jsonl")]).query("status == 'ok'").drop_duplicates("trial_id"))
    Xf = np.array(load("free")[f"rated__{rater}"].tolist(), int)
    r = dict(model=TAG2LABEL[tag], rater=rater, q_ratio=AMq.loc[TAG2LABEL[tag]].diversity_h / AMq.loc[TAG2LABEL[tag]].diversity,
             auc_free=A.separability_rep(Xf[:, :8], H8)[0])
    if (RUNS / f"interview__oracle__{tag}.jsonl").exists():
        do = load("oracle"); Xo = np.array(do[f"rated__{rater}"].tolist(), int)
        bf, bo = div_bins(Xf), div_bins(Xo); common = sorted(set(bf) & set(bo))
        a8 = lambda X, b: 100 * (A.breadth(X[(X[:, :8].sum(1) // 3) == b][:, :8]) == 8).mean()
        r.update(all8_free_b=np.mean([a8(Xf, b) for b in common]), all8_oracle_b=np.mean([a8(Xo, b) for b in common]))
        r.update(bins=len(common), div_free_b=np.mean([bf[b] for b in common]), div_oracle_b=np.mean([bo[b] for b in common]),
                 auc_oracle=A.separability_rep(Xo[:, :8], H8)[0])
        r["pipe_ratio"] = r["div_oracle_b"] / r["div_free_b"]
        if (RUNS / f"interview__grm__{tag}.jsonl").exists():
            Xg = np.array(load("grm")[f"rated__{rater}"].tolist(), int); r["auc_grm"] = A.separability_rep(Xg[:, :8], H8)[0]
    if (RUNS / f"interview__talkdep__{tag}.jsonl").exists():
        # TalkDep personas through the same interviewer and rater; against real-profile
        # interviews in the bins both conditions reach, as for the severity persona
        dt_ = load("talkdep"); Xt = np.array(dt_[f"rated__{rater}"].tolist(), int); gt = dt_.severity.to_numpy()
        r["auc_talkdep"] = A.separability_rep(Xt[:, :8], H8, mgroups=gt)[0]
        if (RUNS / f"interview__oracle__{tag}.jsonl").exists():
            bt = div_bins(Xt, g=gt); ct = sorted(set(bt) & set(bo))
            r.update(bins_talkdep=len(ct), div_talkdep_b=np.mean([bt[b] for b in ct]), div_oracle_tb=np.mean([bo[b] for b in ct]),
                     all8_talkdep_b=np.mean([a8(Xt, b) for b in ct]), all8_oracle_tb=np.mean([a8(Xo, b) for b in ct]))
            r["pipe_ratio_talkdep"] = r["div_oracle_tb"] / r["div_talkdep_b"]
    pl.append(r)
PL = pd.DataFrame(pl); PL.to_csv(TABLES / "interview_pipeline.csv", index=False)
print("\n=== pipeline control: severity-persona vs real-profile interviews (same interviewer and rater) ===")
print(PL.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
