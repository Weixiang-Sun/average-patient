"""Step 6: construct and FREEZE the DSM criteria-generated pole.

Run BEFORE any LLM data exists. The freeze record is timestamped and hashed so
the ordering is auditable.
"""
import hashlib, json, sys, time
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import prototype, scoring
from mental_act.config import DATA_PROC, RUNS, TABLES

runs = list(RUNS.glob("*.jsonl"))
assert not runs, f"LLM runs already exist ({runs}); this pole must be frozen first"
print("verified: no LLM run data exists yet\n")

norms = scoring.load_norms()
human = pd.read_csv(DATA_PROC / "human_indices.csv")
rng = np.random.default_rng(11)

out = {}
for strict in (True, False):
    X, meta = prototype.build_pole(strict=strict)
    md = pd.DataFrame(meta)
    s = scoring.score(X, norms)
    s = pd.concat([md.drop(columns=["total"]), s], axis=1)
    tag = "strict" if strict else "relaxed"
    out[tag] = (X, s)
    print(f"--- pole: {tag} --- {len(X)} distinct criteria-consistent vectors")
    print(s.groupby("specifier").phq9_total.agg(["count", "mean", "min", "max"])
           .reindex(["mild", "moderate", "severe"]).to_string())
    print()

print("=== where the textbook pole sits vs real humans at the SAME total ===")
print(f"{'pole':9s} {'metric':26s} {'textbook':>9s} {'human':>8s} {'d':>7s} {'AUC':>6s}")
rows = []
for tag, (X, s) in out.items():
    m = scoring.matched_human_reference(human, s.phq9_total, rng=rng)
    for met in scoring.METRICS:
        d = scoring.cohens_d(s[met], m[met]); a = scoring.auc(s[met], m[met])
        rows.append(dict(pole=tag, metric=met, textbook=s[met].mean(),
                         human=m[met].mean(), d=d, auc=a))
        print(f"{tag:9s} {met:26s} {s[met].mean():9.2f} {m[met].mean():8.2f} {d:7.2f} {a:6.3f}")
    print()

pd.DataFrame(rows).to_csv(TABLES / "prototype_pole_vs_human.csv", index=False)

# ---- the structural prediction: DSM varies COUNT, humans vary INTENSITY ----
print("=== structural signature: how is severity encoded? ===")
Xs, ss = out["strict"]
hraw = pd.read_csv(DATA_PROC / "nhanes_phq9.csv")
from mental_act.config import PHQ9_ITEMS
Xh = hraw[PHQ9_ITEMS].to_numpy(int)
print(f"{'total':>6s} {'n_endorsed':>22s} {'mean_intensity_if_endorsed':>28s}")
for t in [15, 18, 21, 24, 27]:
    hp = Xh[Xh.sum(1) == t]
    pp = Xs[Xs.sum(1) == t]
    if len(hp) == 0 or len(pp) == 0:
        continue
    hn, pn = (hp > 0).sum(1).mean(), (pp > 0).sum(1).mean()
    hi = np.array([r[r > 0].mean() for r in hp]).mean()
    pi = np.array([r[r > 0].mean() for r in pp]).mean()
    print(f"{t:6d}   human {hn:5.2f} / textbook {pn:5.2f}      "
          f"human {hi:5.2f} / textbook {pi:5.2f}")

# --------------------------------------------------------------- FREEZE
payload = {t: out[t][0].tolist() for t in out}
blob = json.dumps(payload, sort_keys=True).encode()
digest = hashlib.sha256(blob).hexdigest()
(DATA_PROC / "prototype_frozen.json").write_text(json.dumps({
    "sha256": digest,
    "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "llm_runs_existing_at_freeze": [],
    "construction": "DSM-5 MDD Criterion A + PHQ-9 scoring wording only",
    "n_vectors": {t: len(out[t][0]) for t in out},
    "vectors": payload,
}, indent=2))
print(f"\nFROZEN  sha256={digest[:16]}...  -> data/processed/prototype_frozen.json")
