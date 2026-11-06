"""Tokens per PHQ-9 form in the main runs (reasoning off; Gemini 2.5 Pro at its minimum thinking
budget) and, for the
models switched by scripts/65, in their API-default runs (eff__phq9__<tag>__default).

  generated   every token the model produced for one form: completion tokens (OpenAI,
              Azure-hosted models, Kimi, whose reasoning_content is counted there),
              total minus prompt (Gemini, whose thinking is not in the completion count),
              output tokens (Anthropic)
  reasoning   reasoning tokens as the API itemises them (OpenAI reasoning_tokens; Gemini
              thinking = total - prompt - completion); NaN where the API does not itemise
              them (Kimi, Anthropic)
-> outputs/tables/default_tokens.csv
"""
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A
from mental_act.config import RUNS, TABLES

def counts(u, model):
    u = u if isinstance(u, dict) else {}
    if "output_tokens" in u:                                    # Anthropic
        return u["output_tokens"], np.nan
    det = u.get("completion_tokens_details") or {}
    if model.startswith("gemini"):
        think = u["total_tokens"] - u["prompt_tokens"] - u["completion_tokens"]
        return u["total_tokens"] - u["prompt_tokens"], think
    r = det.get("reasoning_tokens")
    return u.get("completion_tokens", np.nan), (np.nan if r is None and "kimi" in model.lower() else (r or 0))

from mental_act.figlong import MODELS as FM
TAG = {m[0]: m[3] for m in FM}
def stats(path, lab):
    rs = [json.loads(l) for l in open(path) if l.strip()]
    rs = [r for r in rs if r.get("status") == "ok"]
    c = np.array([counts(r.get("usage"), str(r.get("model", lab)).lower()) for r in rs], float)
    return len(rs), np.nanmean(c[:, 0]), (np.nanmean(c[:, 1]) if np.isfinite(c[:, 1]).any() else np.nan)
rows = []
for lab, run, *_ in A.ALL_MODELS:
    n, g, r = stats(RUNS / f"{run}.jsonl", lab)
    d = RUNS / f"eff__phq9__{TAG[lab]}__default.jsonl"
    _, gd, rd = stats(d, lab) if d.exists() else (0, np.nan, np.nan)
    rows.append(dict(model=lab, n=n, generated=g, reasoning=r, generated_default=gd, reasoning_default=rd))
T = pd.DataFrame(rows); T.to_csv(TABLES / "default_tokens.csv", index=False)
print(T.round(1).to_string(index=False))
