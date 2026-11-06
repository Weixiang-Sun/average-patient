"""Fit GRM + freeze norms for any instrument (GAD-7, PHQ-8)."""
import hashlib, json, pickle, sys, time
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import indices, irt
from mental_act.config import (DATA_PROC, GAD7_BANDS, GAD7_ITEMS, GAD7_LABELS,
                               PHQ8_BANDS, PHQ8_ITEMS, PHQ8_LABELS, TABLES)

SPECS = {"gad7": (GAD7_ITEMS, GAD7_LABELS, GAD7_BANDS),
         "phq8": (PHQ8_ITEMS, PHQ8_LABELS, PHQ8_BANDS)}
METRICS = ["lz", "guttman_normed", "rare_symptom_surprisal",
           "symptom_combo_surprisal", "extreme_severity"]

for tag, (items, labels, bands) in SPECS.items():
    df = pd.read_csv(DATA_PROC / f"nhis_{tag}.csv")
    X = df[items].to_numpy(int)
    print(f"\n########## {tag.upper()}  N={len(df):,}  J={len(items)} ##########")

    fit = irt.fit_grm(X, n_cat=4, max_iter=100)
    par = pd.DataFrame({"item": items, "symptom": [labels[i] for i in items],
                        "a": fit["a"], "b1": fit["b"][:, 0], "b2": fit["b"][:, 1],
                        "b3": fit["b"][:, 2], "pct_endorsed": (X > 0).mean(0)*100})
    print(par.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    par.to_csv(TABLES / f"grm_items_{tag}.csv", index=False)

    ref = indices.build_reference(df.assign(**{c: df[c] for c in items}),
                                  items=items, bands=bands)
    theta = irt.eap_theta(X, fit)
    order = np.argsort(-X.mean(0))
    h = indices.compute(X, ref)
    h["lz"] = irt.lz(X, fit, theta)
    h["guttman_normed"] = irt.guttman_normed(X, order)
    h["theta"] = theta
    h.to_csv(DATA_PROC / f"human_indices_{tag}.csv", index=False)

    norms = {"grm": fit, "reference": ref, "item_order": order, "items": items,
             "bands": bands, "n_human": len(df),
             "human_quantiles": {m: {str(q): float(np.quantile(h[m], q))
                                     for q in (.05, .5, .95)} for m in METRICS}}
    blob = pickle.dumps(norms)
    (DATA_PROC / f"norms_{tag}_frozen.pkl").write_bytes(blob)
    d = hashlib.sha256(blob).hexdigest()
    (DATA_PROC / f"norms_{tag}_frozen.json").write_text(json.dumps(
        {"sha256": d, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "n_human": len(df), "items": items,
         "grm_a": fit["a"].tolist(), "grm_b": fit["b"].tolist()}, indent=2))
    print(f"FROZEN {tag}  sha256={d[:16]}...")
