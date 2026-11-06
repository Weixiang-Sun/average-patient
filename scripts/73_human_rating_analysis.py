"""Analyse the blind human ratings of interview transcripts (package from scripts/72).

Reads every CSV in outputs/human_rating/returns/ (or the folder given as the first argument)
and outputs/human_rating/key/key.csv. Items 1-8 unless stated.

  agreement   human-human: exact, within one, quadratically weighted kappa
  given       real- and sampled-profile interviews: each human rater, both humans pooled, and the
              LLM rater used in the paper (GPT-4.1, or Claude Haiku 4.5 for GPT-4.1 patients) on the
              same transcripts: exact, within one, kappa, mean shift (rated - given), present->present,
              absent->absent, exact if present, "several days" rated higher
  llm         humans against the LLM rater on all transcripts
  free        severity-persona interviews: share rated as endorsing all eight items, mean breadth
  intervals   95% bootstrap over transcripts (2,000 draws) for the pooled human and LLM values
              and their difference
-> outputs/tables/human_rating_summary.csv, human_rating_long.csv (all raters, by anonymous label only)
"""
import sys, glob
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
from mental_act.config import ROOT, TABLES

HR = ROOT / "outputs" / "human_rating"
RET = sys.argv[1] if len(sys.argv) > 1 else str(HR / "returns")
key = pd.read_csv(HR / "key" / "key.csv")
files = sorted(glob.glob(f"{RET}/*.csv"))
if not files: sys.exit(f"no returned ratings in {RET}")
R_ALL = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)[["rater", "transcript", "position", "item", "rating", "unsure"]]
# outputs name raters only by anonymous labels R1, R2, ... (sorted rater ids)
LABEL = {r: f"R{i + 1}" for i, r in enumerate(sorted(R_ALL.rater.unique()))}
R_ALL["rater"] = R_ALL.rater.map(LABEL)


def analyse(R, TAG):
    R = R.dropna(subset=["rating"]); R["rating"] = R.rating.astype(int)
    done = R.groupby(["rater", "transcript"]).item.nunique().reset_index()
    complete = done[done.item == 9][["rater", "transcript"]]
    R = R.merge(complete, on=["rater", "transcript"])
    raters = sorted(R.rater.unique())
    for r in raters: print(f"rater {r}: {R[R.rater == r].transcript.nunique()} complete transcripts")

    # long table: one row per rater x transcript x item, with given and LLM ratings
    llm_name = lambda m: "claude-haiku-4-5" if m == "gpt41" else "gpt-4.1"
    L = []
    for _, k in key.iterrows():
        for j in range(1, 10):
            L.append(dict(transcript=k.id, cond=k.cond, model=k.model, item=j, given=k[f"given_{j}"], llm=k[f"{llm_name(k.model)}_{j}"]))
    L = R.merge(pd.DataFrame(L), on=["transcript", "item"])
    L.to_csv(TABLES / f"human_rating_long{TAG}.csv", index=False)
    I8 = L[L.item <= 8]


    def scores(rated, given):
        rated, given = np.asarray(rated, int), np.asarray(given, int)
        p1 = given == 1
        return dict(exact=100 * np.mean(rated == given), within1=100 * np.mean(np.abs(rated - given) <= 1),
                    qwk=cohen_kappa_score(given, rated, weights="quadratic"), shift=np.mean(rated - given),
                    present_present=100 * np.mean(rated[given > 0] > 0), absent_absent=100 * np.mean(rated[given == 0] == 0),
                    exact_if_present=100 * np.mean(rated[given > 0] == given[given > 0]), several_up=100 * np.mean(rated[p1] > 1) if p1.any() else np.nan)


    rows = []
    def add(block, source, d):
        for k, v in d.items(): rows.append(dict(block=block, source=source, metric=k, value=v))

    # human-human agreement
    for a in range(len(raters)):
        for b in range(a + 1, len(raters)):
            x = I8[I8.rater == raters[a]].set_index(["transcript", "item"]).rating
            y = I8[I8.rater == raters[b]].set_index(["transcript", "item"]).rating
            x, y = x.align(y, join="inner")
            add("agreement", f"{raters[a]} vs {raters[b]}", dict(n=len(x), exact=100 * np.mean(x.values == y.values),
                within1=100 * np.mean(np.abs(x.values - y.values) <= 1), qwk=cohen_kappa_score(x.values, y.values, weights="quadratic")))

    # against the given profile
    G = I8[I8.cond.isin(["oracle", "grm"])].copy(); G["given"] = G.given.astype(int)
    for r in raters:
        q = G[G.rater == r]; add("given", f"human {r}", scores(q.rating, q.given))
    add("given", "humans pooled", scores(G.rating, G.given))
    Gc = G.groupby(["transcript", "item"]).agg(rating=("rating", lambda v: int(np.floor(np.median(v) + 0.5))), given=("given", "first")).reset_index()
    add("given", "human median", scores(Gc.rating, Gc.given))
    Gl = G.drop_duplicates(["transcript", "item"]); add("given", "LLM rater", scores(Gl.llm.astype(int), Gl.given))

    # humans against the LLM rater (all transcripts)
    for r in raters:
        q = I8[I8.rater == r]
        add("llm", f"human {r}", dict(n=len(q), exact=100 * np.mean(q.rating == q.llm), qwk=cohen_kappa_score(q.llm.astype(int), q.rating, weights="quadratic"),
                                      llm_minus_human=np.mean(q.llm - q.rating)))

    # severity-persona interviews: all eight endorsed, breadth
    F = I8[I8.cond == "free"]
    def free_stats(rated_col, q):
        w = q.pivot_table(index="transcript", columns="item", values=rated_col, aggfunc="first")
        return dict(n=len(w), all8=100 * np.mean((w > 0).all(1)), breadth=float((w > 0).sum(1).mean()))
    for r in raters: add("free", f"human {r}", free_stats("rating", F[F.rater == r]))
    add("free", "LLM rater", free_stats("llm", F.drop_duplicates(["transcript", "item"])))

    # unsure
    add("unsure", "humans pooled", dict(all_items=100 * L.unsure.mean(), given_several_days=100 * G[G.given == 1].unsure.mean()))

    # bootstrap over transcripts: several days rated higher, exact, all eight (free)
    rng = np.random.default_rng(0); tg = G.transcript.unique(); tf = F.transcript.unique(); bs = []
    for _ in range(2000):
        s = rng.choice(tg, len(tg)); qg = pd.concat([G[G.transcript == t] for t in s])
        ql = pd.concat([Gl[Gl.transcript == t] for t in s])
        sh, sl = scores(qg.rating, qg.given), scores(ql.llm.astype(int), ql.given)
        f = rng.choice(tf, len(tf)); qf = pd.concat([F[F.transcript == t].assign(transcript=f"{t}_{i}") for i, t in enumerate(f)])
        bs.append(dict(several_up_h=sh["several_up"], several_up_l=sl["several_up"], exact_h=sh["exact"], exact_l=sl["exact"],
                       all8_free_h=free_stats("rating", qf)["all8"], all8_free_l=free_stats("llm", qf.drop_duplicates(["transcript", "item"]))["all8"]))
    B = pd.DataFrame(bs)
    for c in ("several_up", "exact", "all8_free"):
        for s, lab in (("h", "humans pooled"), ("l", "LLM rater")):
            lo, hi = np.nanpercentile(B[f"{c}_{s}"], [2.5, 97.5]); rows.append(dict(block="interval", source=lab, metric=c, lo=lo, hi=hi))
        d = B[f"{c}_h"] - B[f"{c}_l"]; lo, hi = np.nanpercentile(d, [2.5, 97.5])
        rows.append(dict(block="interval", source="humans minus LLM", metric=c, lo=lo, hi=hi))

    ag = [r for r in rows if r["block"] == "agreement" and r["metric"] in ("exact", "qwk")]
    if ag:
        A_ = pd.DataFrame(ag)
        for mtr in ("exact", "qwk"):
            v = A_[A_.metric == mtr].value
            rows.append(dict(block="agreement", source=f"all {len(raters)} raters, pairwise", metric=f"{mtr}_mean", value=v.mean()))
            rows.append(dict(block="agreement", source=f"all {len(raters)} raters, pairwise", metric=f"{mtr}_min", value=v.min()))
            rows.append(dict(block="agreement", source=f"all {len(raters)} raters, pairwise", metric=f"{mtr}_max", value=v.max()))
    S = pd.DataFrame(rows); S.to_csv(TABLES / f"human_rating_summary{TAG}.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.round(3).to_string(index=False))


analyse(R_ALL.copy(), "")
