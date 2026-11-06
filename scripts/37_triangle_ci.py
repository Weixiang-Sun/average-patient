"""Uncertainty for the PCL-5 feigning comparison (OSF 6b8jf); the tests are in
src/mental_act/feign.py and are shared with scripts/63.

Bootstrap-with-replacement is NOT used for separability, and a genuine patient matched
to several responses is kept in one cross-validation fold: duplicated rows in both
training and test folds inflate cross-validated AUC.

  pcl5_triangle_ci.csv   separability of each group from the genuine patients: AUC
                         (mean of 50 matchings and fold splits), 95% interval over 80%
                         subsamples, exact permutation p (1000 permutations of the same
                         statistic, Holm over the models), direct test against the
                         feigners, real-patient baseline at the group's totals
  pcl5_profile_ci.csv    "not at all" ratio and symptoms endorsed relative to matched
                         genuine patients with bootstrap intervals and the paired
                         difference from the feigners; absolute "not at all" gap overall and
                         within the feigners' range of totals, with a paired test against the
                         feigners; feigning shift with intervals
  pcl5_permutation_nulls.npz  the permutation null of each group (Fig. 2b critical values)
"""
import sys, glob, json, zlib
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from joblib import Parallel
from statsmodels.stats.multitest import multipletests
from mental_act import feign as FG
from mental_act.config import RUNS, TABLES

groups, feigners = {"human feigner": FG.FEIGNER, "human honest": FG.HONEST}, {}
for f in sorted(glob.glob(str(RUNS / "pcl5__*.jsonl"))):
    tag = f.split("pcl5__")[1][:-6]
    d = pd.DataFrame([json.loads(l) for l in open(f)]).query("status=='ok'").drop_duplicates("trial_id")
    p, fe = d[d.arm == "patient"], d[d.arm == "feigner"]
    if len(p) >= 30: groups[f"LLM: {tag}"] = np.array(p["ordered"].tolist(), int)
    if len(fe) >= 30: feigners[f"LLM: {tag}"] = np.array(fe["ordered"].tolist(), int)

rows, prof, nulls = [], [], {}
print(f"{'group':30s} {'n':>3s} {'AUC':>6s} {'95% int.':>16s} {'p':>7s} {'direct p':>9s} {'base':>6s} {'ratio [95%]':>18s}")
with Parallel(n_jobs=FG.JOBS) as par:
    for name, X in groups.items():
        # zlib.crc32 is stable across processes; built-in hash() of a str is not
        seed = zlib.crc32(name.encode()); human = name.startswith("human")
        sep = FG.separability(X, seed, n_sub=200 if human else 60, par=par); nulls[name] = sep.pop("null")
        r = dict(group=name, n=len(X), **sep, **FG.real_baseline(X, seed, par=par))
        if name != "human feigner": r.update(FG.direct_test(X, seed, par=par))
        rows.append(r)
        q = dict(group=name, **FG.profile_ci(X, seed))
        Xf = FG.FEIGNER if name == "human feigner" else feigners.get(name)
        if name == "human feigner": q.update(FG.shift_ci(FG.FEIGNER, FG.GEN, seed))
        elif Xf is not None: q.update(FG.shift_ci(Xf, X, seed))
        prof.append(q)
        print(f"{name:30s} {len(X):3d} {r['auc']:6.3f} [{r['sub_lo']:.3f}, {r['sub_hi']:.3f}] {r['p']:7.4f} "
              f"{r.get('direct_p', np.nan):9.4f} {r['base_real']:6.3f} {q['naa_ratio']:6.2f} [{q['naa_lo']:.2f}, {q['naa_hi']:.2f}]", flush=True)

T = pd.DataFrame(rows); llm = T.group.str.startswith("LLM")
T.loc[llm, "p_holm"] = multipletests(T.loc[llm, "p"], method="holm")[1]
T.loc[llm, "direct_p_holm"] = multipletests(T.loc[llm, "direct_p"], method="holm")[1]
T.to_csv(TABLES / "pcl5_triangle_ci.csv", index=False)
Q = pd.DataFrame(prof); ql = Q.group.str.startswith("LLM")
Q.loc[ql, "abs_minus_feigner_p_holm"] = multipletests(Q.loc[ql, "abs_minus_feigner_p"], method="holm")[1]
Q.to_csv(TABLES / "pcl5_profile_ci.csv", index=False)
np.savez_compressed(TABLES / "pcl5_permutation_nulls.npz", **{k.replace(": ", "_"): v for k, v in nulls.items()})
L = T[llm]
print(f"\nmodels p<0.05: {(L.p < 0.05).sum()}, Holm: {(L.p_holm < 0.05).sum()} of {len(L)}; "
      f"direct p<0.05: {(L.direct_p < 0.05).sum()}, Holm: {(L.direct_p_holm < 0.05).sum()}")
