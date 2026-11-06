"""How many data points did the platform content filter remove?

Per trial (unique trial_id): answered, blocked by the prompt filter, blocked by
the response filter, or failed for another reason (rate limit, malformed
output) after all retries. Interviews are counted per patient turn, since the
filter removes single answers inside an otherwise completed interview.
"""
import sys, json, glob, re
sys.path.insert(0, "src")
import pandas as pd
from mental_act import analysis as A
from mental_act.config import RUNS, TABLES

def kind(err):
    e = err or ""
    if "filtered due to the prompt" in e or ("content_filter" in e and "'param': 'prompt'" in e): return "prompt_filter"
    if "content_filter" in e or "Response content blocked" in e or "ContentFilter" in e: return "response_filter"
    # Early runs retried a blocked prompt and logged only the wrapper. Resending
    # these prompts (PHQ-9 coached-mild, PCL-5 feigner-020) returns the Azure
    # prompt filter (self_harm, medium), so a wrapped BadRequestError is a block.
    if "raised BadRequestError" in e: return "prompt_filter"
    return "other"

def audit(path):
    rs = [json.loads(l) for l in open(path) if l.strip()]
    last = {}
    for r in rs:
        t = r["trial_id"]
        if r.get("status") == "ok": last[t] = "ok"
        elif last.get(t) != "ok": last[t] = kind(r.get("error"))
    s = pd.Series(last).value_counts()
    return {k: int(s.get(k, 0)) for k in ("ok", "prompt_filter", "response_filter", "other")}

GROUP = {}
for m in A.ALL_MODELS:
    GROUP[m[1]] = "Main: PHQ-9"
    if m[2]: GROUP[m[2]] = "Main: GAD-7"
for f in glob.glob(str(RUNS / "*.jsonl")):
    n = f.split("/")[-1][:-6]
    if n.startswith("pcl5__"): GROUP[n] = "PCL-5 feigning roles"
    elif n.startswith("x_"): GROUP[n] = "Extension: " + {"k6": "K6", "ocir": "OCI-R", "phq15": "PHQ-15", "itq": "ITQ", "edeq": "EDE-Q", "pcl5c": "PCL-5 (severity)"}[n[2:].split("__")[0]]
    elif n.startswith("fixes__"): GROUP[n] = "Interventions"
    elif n.startswith("persona_"): GROUP[n] = "Published personas" + (" (GAD-7)" if "_gad7__" in n else " (PHQ-9)")
    elif n in ("temp00", "temp05", "4om_chart", "4om_impairment", "g52_chart", "g52_impairment", "peritem"): GROUP[n] = "Robustness"
rows = []
for n, g in GROUP.items():
    p = RUNS / f"{n}.jsonl"
    if p.exists(): rows.append(dict(group=g, run=n, **audit(p)))
# log-probability probes: one call per item
for f in sorted(glob.glob(str(RUNS / "logprobs__*.jsonl"))):
    rs = [json.loads(l) for l in open(f) if l.strip()]
    ok = sum(r.get("status") == "ok" for r in rs)
    rows.append(dict(group="Log-prob probes", run=f.split("/")[-1][:-6], ok=ok,
                     prompt_filter=sum(kind(r.get("error")) == "prompt_filter" for r in rs if r.get("status") != "ok"),
                     response_filter=sum(kind(r.get("error")) == "response_filter" for r in rs if r.get("status") != "ok"),
                     other=sum(kind(r.get("error")) == "other" for r in rs if r.get("status") != "ok")))
T = pd.DataFrame(rows)
T["total"] = T[["ok", "prompt_filter", "response_filter", "other"]].sum(1)
T.to_csv(TABLES / "filter_audit_runs.csv", index=False)

# interviews: per patient answer, and per suicidality question
IV = []
for f in sorted(glob.glob(str(RUNS / "interview__*.jsonl"))):
    rs = [json.loads(l) for l in open(f) if l.strip()]
    ok = {}
    for r in rs:
        if r.get("status") == "ok": ok[r["trial_id"]] = r
    fail = len({r["trial_id"] for r in rs} - set(ok))
    turns = blocked = si_blocked = 0
    for r in ok.values():
        pts = [l for l in r["transcript"].split("\n") if l.startswith("Patient:")]
        turns += len(pts); b = [("not available" in l) for l in pts]; blocked += sum(b)
        si_blocked += bool(b and b[-1])
    IV.append(dict(run=f.split("/")[-1][:-6], interviews=len(ok), failed_interviews=fail,
                   patient_answers=turns, answers_blocked=blocked, si_answer_blocked=si_blocked))
IV = pd.DataFrame(IV); IV.to_csv(TABLES / "filter_audit_interviews.csv", index=False)

pd.set_option("display.width", 200)
G = T.groupby("group", sort=False)[["total", "ok", "prompt_filter", "response_filter", "other"]].sum()
G.loc["ALL questionnaire-type"] = G.sum()
G["filtered_%"] = 100 * (G.prompt_filter + G.response_filter) / G.total
G["other_%"] = 100 * G.other / G.total
print(G.round(2).to_string())
print("\nfiltered by run (non-zero):")
F = T[(T.prompt_filter + T.response_filter) > 0].copy()
F["filtered_%"] = 100 * (F.prompt_filter + F.response_filter) / F.total
print(F[["group", "run", "total", "prompt_filter", "response_filter", "filtered_%"]].round(1).to_string(index=False))
print("\nother failures (non-zero):")
print(T[T.other > 0][["group", "run", "total", "other"]].to_string(index=False))
print("\ninterviews:")
print(IV.to_string(index=False))
