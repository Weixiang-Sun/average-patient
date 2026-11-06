"""Coverage audit: does every analysis include every model it should?

Checks each result table against the model registry and exits non-zero if any
model is missing where it is expected. Subsets are allowed only where they are
by design (interventions, interviews, robustness) or forced by the API (token
log-probabilities), and those subsets are listed explicitly here.
Run last in scripts/99_rerun_analyses.sh.
"""
import sys, json
sys.path.insert(0, "src")
import pandas as pd
from mental_act import analysis as A
from mental_act.config import RUNS, TABLES
from mental_act.figlong import MODELS

ALL = [m[0] for m in MODELS]
TAG = {m[0]: m[3] for m in MODELS}
assert ALL == [m[0] for m in A.ALL_MODELS], "figlong.MODELS and analysis.ALL_MODELS disagree"
# No usable token log-probabilities (tested 2026-09-28/29): GPT-5-mini, Claude,
# Mistral, Phi-4 and every Gemini model return none; Kimi-K2.5 returns them but
# the top-logprob lists are misaligned with the sampled tokens (a sampled token
# with logprob 0 appears at -9 to -18 in its own list), so they are not used.
NO_LOGPROB = {"GPT-5-mini", "Claude Haiku 4.5", "Mistral-Large-3", "Phi-4",
              "Gemini-2.5-Flash-Lite", "Gemini-2.5-Flash", "Gemini-2.5-Pro"}
LOGPROB = [m for m in ALL if m not in NO_LOGPROB]
# designed subsets: interventions and interviews use the four original patient
# models plus the three Gemini models; the real-profile interview condition uses
# GPT-4o-mini, Llama-3.3 and the Gemini models
GEM = ["Gemini-2.5-Flash-Lite", "Gemini-2.5-Flash", "Gemini-2.5-Pro"]
FIX_MODELS = ["GPT-4o-mini", "GPT-4.1", "Claude Haiku 4.5", "Llama-3.3-70B"] + GEM
ORACLE_MODELS = ["GPT-4o-mini", "Llama-3.3-70B"] + GEM
LADDER = ["GPT-4.1-nano", "GPT-4.1-mini", "GPT-4.1", "GPT-4o-mini", "GPT-4o", "GPT-5-mini", "GPT-5.2",
          "Gemini-2.5-Flash-Lite", "Gemini-2.5-Flash", "Gemini-2.5-Pro"]
T = lambda f: pd.read_csv(TABLES / f"{f}.csv")
problems, lines = [], []

def check(name, have, want):
    have, want = set(have), set(want)
    miss, extra = sorted(want - have), sorted(have - want)
    ok = not miss
    lines.append(f"{'ok ' if ok else 'MISSING'} {name:52s} {len(have & want):2d}/{len(want):2d}" + (f"  missing: {miss}" if miss else "") + (f"  unexpected: {extra}" if extra else ""))
    if not ok: problems.append(name)

# ---- raw runs: every model, every instrument, enough answered trials
MIN_OK = {"phq9": 400, "gad7": 400, "pcl5": 170, "k6": 400, "ocir": 400, "phq15": 400, "itq": 400, "edeq": 380, "pcl5c": 400}
def n_ok(run):
    p = RUNS / f"{run}.jsonl"
    if not p.exists(): return 0
    return len({r["trial_id"] for r in map(json.loads, filter(str.strip, p.read_text().splitlines())) if r.get("status") == "ok"})
def n_ok_lp(label):
    p = RUNS / f"logprobs__{TAG[label]}.jsonl"
    return sum(json.loads(x).get("status") == "ok" for x in open(p) if x.strip()) if p.exists() else 0
PH = {m[0]: m[1] for m in A.ALL_MODELS}; GA = {m[0]: m[2] for m in A.ALL_MODELS}
for inst in MIN_OK:
    run = {"phq9": lambda l: PH[l], "gad7": lambda l: GA[l], "pcl5": lambda l: f"pcl5__{TAG[l]}"}.get(inst, lambda l, i=inst: f"x_{i}__{TAG[l]}")
    check(f"raw runs: {inst} (>= {MIN_OK[inst]} ok trials)", [l for l in ALL if n_ok(run(l)) >= MIN_OK[inst]], ALL)
check("raw runs: log-prob probes (90 ok calls)", [l for l in ALL if n_ok_lp(l) >= 90], LOGPROB)

# ---- analysis tables
for f in ("all_models", "main_ci"):
    d = T(f)
    for ins in ("phq9", "gad7"): check(f"{f}.csv [{ins}]", d[d.instrument == ins].model, ALL)
d = T("deep_analysis")
for ins in ("phq9", "gad7"): check(f"deep_analysis.csv [{ins}]", d[d.instrument == ins].model, ALL)
d = T("person_fit_probes")
for ins in ("phq9", "gad7"): check(f"person_fit_probes.csv [{ins}]", d[d.instrument == ins].model, ALL)
check("coaching.csv", T("coaching").model, ALL)
ip = T("item_profiles")
for ins in ("phq9", "gad7"):
    n = int(ip[ip.instrument == ins].n_models.iloc[0]); check(f"item_profiles.csv [{ins}] (models averaged)", ALL[:n], ALL)
# severity bands group responses by actual total: a model belongs in a band when it
# has at least 20 responses there, and then it must be in the table
bsm = T("by_severity_models")
for lab_, lo, hi in (("0-4 (none)", 0, 4), ("5-9 (mild)", 5, 9), ("10-14 (moderate)", 10, 14), ("15-19 (mod.-severe)", 15, 19), ("20-27 (severe)", 20, 27)):
    want = [l for l in ALL if ((lambda t: ((t >= lo) & (t <= hi)).sum())(A.run_X(PH[l]).sum(1))) >= 20]
    check(f"by_severity_models.csv [{lab_}] (models with >= 20 responses there)", bsm[bsm.band == lab_].model, want)
check("position_effects.csv", T("position_effects").model, ALL)
check("anchoring.csv", T("anchoring").model, ALL)
check("scale_ci.csv (family ladders)", T("scale_ci").model, LADDER)
tr = T("pcl5_triangle"); inv = {v: k for k, v in TAG.items()}
for arm in ("patient", "feigner", "honest"): check(f"pcl5_triangle.csv [{arm}]", [inv.get(t) for t in tr[tr.arm == arm].model], ALL)
ci = T("pcl5_triangle_ci"); check("pcl5_triangle_ci.csv", [inv.get(g.replace("LLM: ", "")) for g in ci.group if g.startswith("LLM")], ALL)
check("default_tokens.csv", T("default_tokens").model, ALL)
check("wholeform_items.csv", T("wholeform_items").model, ALL)
pc = T("pcl5_profile_ci"); check("pcl5_profile_ci.csv", [inv.get(g.replace("LLM: ", "")) for g in pc.group if g.startswith("LLM")], ALL)
check("pcl5_profile_ci.csv [feigning shift]", [inv.get(g.replace("LLM: ", "")) for g in pc[pc.dtotal.notna()].group if g.startswith("LLM")], ALL)
g = T("extra_generality")
for ref in [r for r in g.ref.unique() if r != "coach_vs_nhanes"]:
    check(f"extra_generality.csv [{ref}]", g[(g.ref == ref) & g.endorsed_m.notna()].model, ALL)
ei = T("extra_items_by_model")
for ins in ei.instrument.unique(): check(f"extra_items_by_model.csv [{ins}]", [inv.get(t) for t in ei[ei.instrument == ins].tag.unique()], ALL)
check("downstream_si.csv", T("downstream_si").train, ALL)
check("downstream_controls.csv", T("downstream_controls").train, ALL)
check("downstream_calibration.csv (columns)", [c[5:] for c in T("downstream_calibration").columns if c.startswith("pred_")], ALL + ["real", "LLM pooled"])
lp = T("logprob_mechanism"); check("logprob_mechanism.csv", [inv.get(t) for t in lp.model.unique()], LOGPROB)
sp = T("logprob_sampled_profiles"); check("logprob_sampled_profiles.csv", [inv.get(t) for t in sp.model.unique()], LOGPROB)
fa = T("filter_audit_runs")
want_runs = [PH[l] for l in ALL] + [GA[l] for l in ALL] + [f"pcl5__{TAG[l]}" for l in ALL] + \
            [f"x_{i}__{TAG[l]}" for i in ("k6", "ocir", "phq15", "itq", "edeq", "pcl5c") for l in ALL]
check("filter_audit_runs.csv (questionnaire runs)", [r for r in fa.run if r in set(want_runs)], want_runs)

fx = T("fixes"); fx = fx[fx.cond != "-"]
for c in ("label", "hetero", "biography", "numeric", "fewshot", "talkdep", "cactus", "cactus_cm"):
    check(f"fixes.csv [{c}]", [inv.get(t) for t in fx[fx.cond == c].model], FIX_MODELS)
iv = T("interview")
for cond, want in (("free", FIX_MODELS), ("oracle", ORACLE_MODELS)):
    q = iv[iv.source.str.startswith(cond + ":")]
    check(f"interview.csv [{cond}]", {inv.get(x.split(": ")[1]) for x in q.source}, want)
for f_ in ("reasoning_effort_refs", "reasoning_effort_gemini"): check(f"{f_}.csv [efforts]", set(T(f_).effort), set(T("reasoning_effort").effort) if f_.endswith("refs") else {e for m_, e in T("reasoning_effort")[["model", "effort"]].values if m_.startswith("Gemini")})
re_ = T("reasoning_effort"); check("reasoning_effort.csv [models x 4 efforts]", [m for m in re_.model if (re_.model == m).sum() == 4], ["GPT-5-mini"] * 4 + ["Gemini 2.5 Flash"] * 4)
check("reasoning_effort.csv [models x 2 efforts]", [m for m in re_.model if (re_.model == m).sum() == 2], ["Kimi-K2.5"] * 2 + ["Gemini 2.5 Pro"] * 2)
check("interview_grm.csv [sampled profiles]", {inv.get(x.split(": ")[1]) for x in T("interview_grm").source}, ORACLE_MODELS)
check("interview_talkdep.csv [TalkDep personas]", {inv.get(x.split(": ")[1]) for x in T("interview_talkdep").source}, FIX_MODELS)
pt = T("published_personas_talkdep"); check("published_personas_talkdep.csv", set(pt.model), ALL)
pp = T("published_personas"); check("published_personas.csv [PHQ-9 and GAD-7 x one-line, TalkDep, CACTUS, CACTUS + formulation]", {m for m, g in pp.groupby("model") if len(g) == 8}, ALL)

print("\n".join(lines))
print(f"\nBy design (not expected to cover every model): interventions (incl. the published personas in that protocol) and free interviews ({len(FIX_MODELS)} models), real- and sampled-profile interviews ({len(ORACLE_MODELS)}),")
print("robustness (GPT-4o-mini, GPT-5.2). No token log-probabilities from:", ", ".join(sorted(NO_LOGPROB)))
if problems:
    print(f"\n{len(problems)} coverage problem(s)"); sys.exit(1)
print("\nall analyses cover every expected model")
