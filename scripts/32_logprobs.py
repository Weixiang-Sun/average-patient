"""Mechanism: is the flat profile in the per-item response distributions?

One item per call, no conversation history, so no composition across items.
The model answers with a single digit; we read P(0..3) from top_logprobs.
For each (model, framing, severity, item) this gives the model's distribution
over response categories, which we compare with the human conditional
distribution P(item = k | PHQ-9 band) from NHANES.
"""
import sys, json, math, os, time, argparse
sys.path.insert(0, "src")
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv; load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from mental_act.config import PHQ9_ITEMS, PHQ9_LABELS, PHQ9_RESPONSE_LABELS, RUNS
from mental_act.personas import persona_prompt, SYSTEM

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", required=True)
p.add_argument("--workers", type=int, default=6)
a = p.parse_args()
cl = OpenAI(base_url=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_ANTHROPIC_KEY"], timeout=180)

SEV = ["none", "mild", "moderate", "moderately_severe", "severe"]
FRAMES = ["label", "impairment"]

def prompt(sev, frame, item):
    arm = "honest" if sev == "none" else "naive"
    opts = "\n".join(f"  {k} = {v}" for k, v in PHQ9_RESPONSE_LABELS.items())
    return (persona_prompt(sev, arm, frame) + "\n\nOver the last 2 weeks, how often have you "
            f"been bothered by the following problem?\n\n  {PHQ9_LABELS[item]}\n\nResponse options:\n{opts}\n\n"
            "Reply with only a single digit: 0, 1, 2 or 3.")

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=40))
def ask(model, text):
    # Kimi-K2.5 runs with reasoning off (as in the questionnaire runs, scripts/65): with
    # its reasoning block on, the Azure top-logprob lists after </think> were
    # misaligned with the sampled tokens; with it off the digit is the first token.
    kw = ({"max_completion_tokens": 3000} if model.startswith("gpt-5")
          else {"max_tokens": 4, "temperature": 1.0, "reasoning_effort": "none"} if "kimi" in model.lower()
          else {"max_tokens": 1, "temperature": 1.0})
    # grok caps top_logprobs at 8; four digit candidates fit comfortably
    # per-provider caps on top_logprobs; four digit candidates always fit
    k = 5 if model.startswith("gpt-5") else 8 if "grok" in model.lower() else 20
    return cl.chat.completions.create(model=model, logprobs=True, top_logprobs=k,
        messages=[{"role": "system", "content": SYSTEM.replace(" Respond only with the requested JSON.", "")},
                  {"role": "user", "content": text}], **kw)

def one(job):
    model, frame, sev, item = job
    rec = dict(model=model, frame=frame, severity=sev, item=item)
    try:
        r = ask(model, prompt(sev, frame, item))
        toks = r.choices[0].logprobs.content or []
        # the answer starts after any reasoning block
        ends = [i for i, t in enumerate(toks) if t.token.strip() == "</think>"]
        if ends: toks = toks[ends[-1] + 1:]
        # first content token that is (or starts with) a digit 0-3
        tok = next((t for t in toks if t.token.strip() and t.token.strip()[0] in "0123"), None)
        if tok is None:
            raise ValueError(f"no digit token in {r.choices[0].message.content!r}")
        pr = {k: 0.0 for k in "0123"}
        for c in tok.top_logprobs:
            s = c.token.strip()
            if s in pr: pr[s] += math.exp(c.logprob)
        mass = sum(pr.values())
        rec.update(status="ok", p=[pr[k] / mass for k in "0123"], mass=mass,
                   answer=tok.token.strip()[:1])
    except Exception as e:
        rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:150]}")
    return rec

for model in a.models:
    jobs = [(model, f, s, i) for f in FRAMES for s in SEV for i in PHQ9_ITEMS]
    out = RUNS / f"logprobs__{model.replace('.','').replace('-','').lower()}.jsonl"
    done = set()
    if out.exists():
        done = {(r["frame"], r["severity"], r["item"]) for r in map(json.loads, out.read_text().splitlines()) if r.get("status") == "ok"}
    jobs = [j for j in jobs if (j[1], j[2], j[3]) not in done]
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex, out.open("a") as f:
        for rec in ex.map(one, jobs):
            f.write(json.dumps(rec) + "\n"); f.flush()
    rs = [json.loads(l) for l in out.read_text().splitlines()]
    ok = sum(r["status"] == "ok" for r in rs)
    print(f"{model}: {ok}/{len(rs)} ok, {time.time()-t0:.0f}s", flush=True)
