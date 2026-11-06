"""Step 2: fit the GRM on NHANES, compute every index for real humans, FREEZE.

Everything this script writes is the reference against which LLM simulators
are later scored. Once frozen, index definitions must not change -- that is
what makes the LLM comparison a genuine out-of-sample test.
"""
import hashlib, json, logging, pickle, sys, time
sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

import numpy as np, pandas as pd
from mental_act import indices, irt
from mental_act.config import DATA_PROC, PHQ9_ITEMS, PHQ9_LABELS, TABLES

src = DATA_PROC / "nhanes_phq9.csv"
df = pd.read_csv(src)
X = df[PHQ9_ITEMS].to_numpy(int)
print(f"human reference sample: N={len(df):,}")

# ---------------------------------------------------------------- 1. GRM
t0 = time.time()
print("\nfitting GRM (marginal ML / EM) ...")
fit = irt.fit_grm(X, n_cat=4, max_iter=100, verbose=True)
print(f"  converged in {fit['n_iter']} iters, loglik={fit['loglik']:,.1f}, {time.time()-t0:.0f}s")

par = pd.DataFrame({
    "item": PHQ9_ITEMS,
    "symptom": [PHQ9_LABELS[i] for i in PHQ9_ITEMS],
    "a": fit["a"], "b1": fit["b"][:, 0], "b2": fit["b"][:, 1], "b3": fit["b"][:, 2],
    "mean_score": X.mean(axis=0),
    "pct_endorsed": (X > 0).mean(axis=0) * 100,
})
print("\nGRM item parameters:")
print(par.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
par.to_csv(TABLES / "grm_item_parameters.csv", index=False)

# ------------------------------------------------- 2. reference + indices
print("\nbuilding empirical conditional reference ...")
reference = indices.build_reference(df)
print(f"  {len(reference['ref'])} conditioning cells "
      f"({len(reference['use_exact'])} exact totals, rest pooled to bands)")

theta = irt.eap_theta(X, fit)
item_order = np.argsort(-X.mean(axis=0))          # most-endorsed first

human = indices.compute(X, reference)
human["lz"] = irt.lz(X, fit, theta)
human["guttman"] = irt.guttman_errors(X, item_order)
human["guttman_normed"] = irt.guttman_normed(X, item_order)
human["theta"] = theta
human["severity"] = df["severity"].values
human["cycle"] = df["cycle"].values

human.to_csv(DATA_PROC / "human_indices.csv", index=False)

METRICS = ["lz", "guttman_normed", "rare_symptom_surprisal",
           "symptom_combo_surprisal", "extreme_severity"]
print("\n=== human reference distributions ===")
print(human[METRICS].describe().T.to_string(
    float_format=lambda v: f"{v:8.3f}"))

print("\n=== by severity band ===")
by = human.groupby("severity")[METRICS].mean().reindex(
    ["none_minimal", "mild", "moderate", "moderately_severe", "severe"])
print(by.to_string(float_format=lambda v: f"{v:8.3f}"))
by.to_csv(TABLES / "human_norms_by_severity.csv")

# ------------------------------------------------------------- 3. FREEZE
norms = {
    "grm": fit, "reference": reference, "item_order": item_order,
    "items": PHQ9_ITEMS, "n_human": len(df),
    "human_quantiles": {
        m: {str(q): float(np.quantile(human[m], q))
            for q in (.001, .01, .05, .25, .5, .75, .95, .99, .999)}
        for m in METRICS},
}
blob = pickle.dumps(norms)
(DATA_PROC / "norms_frozen.pkl").write_bytes(blob)
digest = hashlib.sha256(blob).hexdigest()

(DATA_PROC / "norms_frozen.json").write_text(json.dumps({
    "sha256": digest,
    "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "n_human": len(df),
    "metrics": METRICS,
    "human_quantiles": norms["human_quantiles"],
    "grm_a": fit["a"].tolist(),
    "grm_b": fit["b"].tolist(),
}, indent=2))

print(f"\nFROZEN  sha256={digest[:16]}...  -> data/processed/norms_frozen.pkl")
print("Index definitions are now fixed. Do not modify them after LLM data is collected.")
