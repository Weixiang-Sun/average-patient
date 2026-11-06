"""The interviewer and rater as an assessment agent, evaluated on simulated patients.

Tasks scored from each interview (the agent's PHQ-9 ratings, items 1-9):
  severity    band of the rated total (none 0-4, mild 5-9, moderate 10-14, moderately severe
              15-19, severe 20-27) against the patient's severity: the instructed severity for
              severity-persona patients, the band of the given profile for real- and
              sampled-profile patients; exact and within one band, above and below
  presence   for items 1-8: % rated absent; for real and sampled profiles, % of absent given
              symptoms rated absent (absence detection), % of symptoms whose presence the agent
              matches, and that % for an agent reporting every symptom present
  top symptom whether items 1-8 have a single most prominent symptom (one item at the
              highest rated level), for the interviews, the given profiles, and real NHANES
              respondents with the same item 1-8 totals; for real and sampled profiles,
              whether the rated and the given most prominent symptoms overlap
TalkDep patients (scripts/77) are scored against the PHQ-9 band named like the persona's
severity label (minimal -> none, mild, moderate, moderate to severe -> moderately severe,
severe) and, as in TalkDep's own evaluation, by the rank correlation with the BDI-II score.
Rater: the one used in the paper (GPT-4.1; Claude Haiku 4.5 for GPT-4.1 patients); the other
rater is reported as a second agent where it rated the same transcripts.
Human check: the human ratings of 64 transcripts (scripts/73) scored the same way.
-> outputs/tables/agent_eval.csv, agent_eval_confusion.csv, agent_eval_human.csv
"""
import sys, json, glob
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A
from mental_act.figlong import TAG2LABEL
from mental_act.config import RUNS, TABLES, ROOT
from mental_act.published_personas import talkdep_severity
from scipy.stats import spearmanr
TD = talkdep_severity()

BANDS = ["none", "mild", "moderate", "moderately_severe", "severe"]
CUT = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 27)]
band_of = lambda t: next(b for b, (lo, hi) in zip(BANDS, CUT) if lo <= t <= hi)
unique_top = lambda X: (X == X.max(1, keepdims=True)).sum(1) == 1
H8 = A.human_X("phq9")[:, :8]
RATERS = ["gpt-4.1", "claude-haiku-4-5"]
PRIMARY = lambda tag: "claude-haiku-4-5" if tag == "gpt41" else "gpt-4.1"


def score(truth, rated_total):
    pred = [band_of(int(t)) for t in rated_total]
    k = np.array([BANDS.index(p) - BANDS.index(t) for p, t in zip(pred, truth)])
    return dict(exact=100 * np.mean(k == 0), within1=100 * np.mean(np.abs(k) <= 1), above=100 * np.mean(k > 0), below=100 * np.mean(k < 0)), pred


def score_balanced(truth, rated_total):
    """The same rates averaged with equal weight over the patients' severities, as in the full
    design (20 interviews per severity); the human-rated sample has 2-5 per severity."""
    k = pd.Series([BANDS.index(band_of(int(t))) - BANDS.index(s) for s, t in zip(truth, rated_total)])
    g = pd.DataFrame(dict(sev=list(truth), exact=k == 0, within1=k.abs() <= 1, above=k > 0)).groupby("sev").mean()
    return {f"{c}_bal": 100 * g[c].mean() for c in ("exact", "within1", "above")}


rows, conf = [], []
for f in sorted(glob.glob(str(RUNS / "interview__*__*.jsonl"))):
    _, cond, tag = f[:-6].split("/")[-1].split("__")
    if tag.startswith("eff") or cond not in ("free", "oracle", "grm", "talkdep"): continue
    d = A.drop_filtered_interviews(pd.DataFrame([json.loads(l) for l in open(f)]).query("status == 'ok'").drop_duplicates("trial_id"))
    if cond == "free": truth = list(d.severity)
    elif cond == "talkdep": truth = [TD[n][1] for n in d.severity]
    else:
        G = np.array(d.true_profile.tolist(), int); truth = [band_of(int(t)) for t in G.sum(1)]
        assert all(t == s for t, s in zip(truth, d.severity)), "given profile outside its band"
    for rater in RATERS:
        if f"rated__{rater}" not in d or (tag in ("gpt41", "claudehaiku45") and rater == {"gpt41": "gpt-4.1", "claudehaiku45": "claude-haiku-4-5"}[tag]): continue
        R = np.array(d[f"rated__{rater}"].tolist(), int)
        s, pred = score(truth, R.sum(1))
        r = dict(cond=cond, model=TAG2LABEL[tag], rater=rater, primary=rater == PRIMARY(tag), n=len(d), **s,
                 rated_total=R.sum(1).mean(), unique_top=100 * unique_top(R[:, :8]).mean(),
                 unique_top_real=100 * A.at_same_totals(R[:, :8], H8, lambda Q: unique_top(Q).mean()))
        r["absent"] = 100 * (R[:, :8] == 0).mean()
        if cond == "talkdep":
            bdi = np.array([TD[n][0] for n in d.severity])
            r["bdi_rho"] = spearmanr(R.sum(1), bdi).statistic
            pm = pd.DataFrame(dict(p=list(d.severity), t=R.sum(1), b=bdi)).groupby("p").mean()
            r["bdi_rho_persona"] = spearmanr(pm.t, pm.b).statistic
            r.update(score_balanced(truth, R.sum(1)))
        if cond not in ("free", "talkdep"):
            g8, r8 = G[:, :8] > 0, R[:, :8] > 0
            r.update(given_absent=100 * (~g8).mean(), absent_detect=100 * (~r8[~g8]).mean(), agree=100 * (g8 == r8).mean(), agree_allpresent=100 * g8.mean())
            r["unique_top_given"] = 100 * unique_top(G[:, :8]).mean()
            r["top_hit"] = 100 * np.mean([bool(set(np.flatnonzero(R[i, :8] == R[i, :8].max())) & set(np.flatnonzero(G[i, :8] == G[i, :8].max()))) for i in range(len(G))])
        rows.append(r)
        if rater == PRIMARY(tag):
            conf += [dict(cond=cond, model=TAG2LABEL[tag], truth=t, rated=p) for t, p in zip(truth, pred)]
E = pd.DataFrame(rows); E.to_csv(TABLES / "agent_eval.csv", index=False)
C = pd.DataFrame(conf).groupby(["cond", "truth", "rated"]).size().rename("n").reset_index(); C.to_csv(TABLES / "agent_eval_confusion.csv", index=False)
pd.set_option("display.width", 220)
P = E[E.primary]
print(P.groupby("cond")[["exact", "within1", "above", "below", "unique_top", "unique_top_real"]].agg(["min", "median", "max"]).round(1).to_string())
print(P[P.cond != "free"][["cond", "model", "unique_top_given", "top_hit"]].round(1).to_string(index=False))

# the same scoring on the human ratings of 64 transcripts (the key gives condition and severity)
HL = pd.read_csv(TABLES / "human_rating_long.csv")
key = pd.read_csv(ROOT / "outputs" / "human_rating" / "key" / "key.csv").set_index("id")
hrows = []
for (src, cond), q in [((r, c), g) for (r, c), g in HL.groupby(["rater", "cond"])] + [(("LLM rater", c), g.drop_duplicates(["transcript", "item"]).assign(rating=lambda x: x.llm)) for c, g in HL.groupby("cond")]:
    tot = q.groupby("transcript").rating.sum()
    truth = [key.loc[t].severity if cond == "free" else band_of(int(sum(key.loc[t, f"given_{j}"] for j in range(1, 10)))) for t in tot.index]
    s, _ = score(truth, tot.values); s.update(score_balanced(truth, tot.values))
    q8 = q[q.item <= 8]; pres = dict(absent=100 * (q8.rating == 0).mean())
    if cond != "free":
        g8, r8 = q8.given > 0, q8.rating > 0
        pres.update(absent_detect=100 * (~r8[~g8]).mean(), agree=100 * (g8 == r8).mean())
    hrows.append(dict(source=src, cond=cond, n=len(tot), **s, **pres))
HE = pd.DataFrame(hrows); HE.to_csv(TABLES / "agent_eval_human.csv", index=False)
print(HE.round(1).to_string(index=False))
P8 = P.groupby("cond")[["absent", "given_absent", "absent_detect", "agree", "agree_allpresent"]].agg(["min", "max"]).round(1)
print(P8.to_string())
