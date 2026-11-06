"""Deep analysis: mechanism, generality, separability."""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A
from mental_act import scoring
from mental_act.config import DATA_PROC, TABLES

pd.set_option("display.width", 200)
rows = []

for instr, models in [("phq9", A.PHQ9_ALL), ("gad7", A.GAD7_ALL)]:
    spec = A.INSTR[instr]
    Xh = A.human_X(instr); Hb = A.in_band(Xh, instr)
    print(f"\n{'='*88}\n{instr.upper()}  ({spec['survey']}, {spec['condition']}, "
          f"J={spec['J']})  human N={len(Xh):,}, in-band n={len(Hb):,}\n{'='*88}")
    for lab, run, vend in models:
        d = A.load_run(run)
        if d is None or len(d) < 300:
            continue
        X = A.run_X(run); Mb = A.in_band(X, instr)
        rows.append(dict(
            instrument=instr, model=lab, vendor=vend, n=len(X), n_band=len(Mb),
            breadth=A.breadth(Mb).mean(),
            breadth_h=A.breadth(Hb).mean(),
            all_items_pct=100*(A.breadth(Mb) == spec["J"]).mean(),
            all_items_pct_h=100*(A.breadth(Hb) == spec["J"]).mean(),
            depth=np.nanmean(A.depth(Mb)), depth_h=np.nanmean(A.depth(Hb)),
            spikiness=A.spikiness(Mb).mean(), spikiness_h=A.spikiness(Hb).mean(),
            top_share=A.top_share(Mb).mean(), top_share_h=A.top_share(Hb).mean(),
            order_rho=A.order_agreement(X, Xh),
            corr_dist=A.corr_distance(Mb, Hb),
            separability=A.separability_rep(Mb, Xh)[0],
        ))

t = pd.DataFrame(rows)
t.to_csv(TABLES/"deep_analysis.csv", index=False)

for instr in ("phq9", "gad7"):
    s = t[t.instrument == instr]
    print(f"\n--- {instr.upper()} ---")
    print(s[["model", "vendor", "breadth", "all_items_pct", "depth", "spikiness",
             "top_share", "order_rho", "corr_dist", "separability"]]
          .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    h = s.iloc[0]
    print(f"{'HUMAN':>22s} {h.breadth_h:7.3f} {h.all_items_pct_h:13.3f} "
          f"{h.depth_h:7.3f} {h.spikiness_h:7.3f} {h.top_share_h:7.3f} "
          f"{1.0:7.3f} {0.0:7.3f} {0.5:7.3f}")

# ---------------------------------------------------- item-level profiles
print(f"\n{'='*88}\nITEM-LEVEL: endorsement rate, humans vs models (matched band)\n{'='*88}")
prof = []
for instr, models in [("phq9", A.PHQ9_ALL), ("gad7", A.GAD7_ALL)]:
    spec = A.INSTR[instr]; items, labels = spec["items"], spec["labels"]
    Hb = A.in_band(A.human_X(instr), instr)
    hm = (Hb > 0).mean(0)*100
    Ms = [A.in_band(A.run_X(r), instr) for _, r, _ in models if A.load_run(r) is not None]
    mm = np.mean([(M > 0).mean(0)*100 for M in Ms], axis=0)
    print(f"\n{instr.upper()}  ({spec['survey']})")
    print(f"{'item':52s} {'human':>7s} {'models':>7s} {'gap':>7s}")
    ordr = np.argsort(hm)
    for j in ordr:
        nm = (labels or {}).get(items[j], items[j])[:50]
        print(f"{nm:52s} {hm[j]:6.1f}% {mm[j]:6.1f}% {mm[j]-hm[j]:+6.1f}")
        prof.append(dict(instrument=instr, item=items[j], label=nm,
                         human=hm[j], models=mm[j], gap=mm[j]-hm[j], n_models=len(Ms)))
pd.DataFrame(prof).to_csv(TABLES/"item_profiles.csv", index=False)

# -------------------------------------------------- coaching + heterogeneity
print(f"\n{'='*88}\nCOACHING EFFECT and BETWEEN-RESPONSE HETEROGENEITY (PHQ-9)\n{'='*88}")
Hb = A.in_band(A.human_X("phq9"), "phq9")
print(f"{'model':20s} {'naive brd':>10s} {'coach brd':>10s} {'d':>7s} "
      f"{'SD(total) LLM':>14s} {'SD(total) human':>16s}")
ch = []
for lab, run, _ in A.PHQ9_ALL:
    d = A.load_run(run)
    if d is None: continue
    X = np.array(d["ordered"].tolist(), int)
    nb = A.breadth(X[(d.arm == "naive").values])
    cb = A.breadth(X[(d.arm == "coached").values])
    # within-severity difference: the platform filter removed coached-mild
    # prompts for some OpenAI models, which would bias a pooled comparison
    sev = ["mild", "moderate", "moderately_severe", "severe"]
    diff = np.mean([A.breadth(X[((d.arm == "naive") & (d.severity == s)).values]).mean()
                    - A.breadth(X[((d.arm == "coached") & (d.severity == s)).values]).mean() for s in sev])
    sd_llm = np.mean([X[(d.severity == s).values].sum(1).std()
                      for s in ["mild", "moderate", "moderately_severe", "severe"]])
    print(f"{lab:20s} {nb.mean():10.2f} {cb.mean():10.2f} "
          f"{scoring.cohens_d(nb, cb):7.2f} {sd_llm:14.2f} {Hb.sum(1).std():16.2f}")
    ch.append(dict(model=lab, naive_breadth=nb.mean(), coached_breadth=cb.mean(),
                   drop_within_severity=diff, d=scoring.cohens_d(nb, cb), sd_total_llm=sd_llm))
pd.DataFrame(ch).to_csv(TABLES/"coaching.csv", index=False)
print(f"\nwrote {TABLES}/deep_analysis.csv, item_profiles.csv, coaching.csv")
