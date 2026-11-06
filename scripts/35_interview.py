"""Conversational modality: does the flat profile survive an interview?

A patient LLM is interviewed by a fixed script covering the nine PHQ-9 domains
in natural language. A blind rater LLM then codes PHQ-9 items 0-3 from the
transcript alone.

  free    patient given a severity persona, as patient simulators are used
  oracle  patient given a real NHANES PHQ-9 profile as private background;
          validates that the rater pipeline recovers uneven profiles
  talkdep patient given a full TalkDep persona (12 clinician-validated depression
          personas with a BDI-II score and severity label), as in the
          questionnaire runs; --n interviews per persona
"""
import sys, os, json, random, time, argparse
sys.path.insert(0, "src")
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from dotenv import load_dotenv; load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type
from openai import OpenAI
from anthropic import AnthropicFoundry
from mental_act import llm_runner
from mental_act.config import DATA_PROC, PHQ9_ITEMS, RUNS

AZ = OpenAI(base_url=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_ANTHROPIC_KEY"], timeout=180)
AN = AnthropicFoundry(api_key=os.environ["AZURE_ANTHROPIC_KEY"], base_url=os.environ["AZURE_ANTHROPIC_ENDPOINT"])
GEM = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY", ""), timeout=180)

EFFORT = None   # --effort: reasoning effort for a Gemini patient model (review W6)

class ContentFiltered(Exception):
    """The platform's safety filter removed the model's response."""

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=40),
       retry=retry_if_not_exception_type(ContentFiltered))
def chat(model, system, msgs, max_tokens=300, json_schema=None):
    try:
        return _chat(model, system, msgs, max_tokens, json_schema)
    except Exception as e:
        if "content_filter" in str(e) or "ResponsibleAIPolicy" in str(e):
            raise ContentFiltered(str(e)[:300]) from e
        raise

def _chat(model, system, msgs, max_tokens=300, json_schema=None):
    if model.startswith("claude"):
        kw = {}
        if json_schema: kw["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
        r = AN.messages.create(model=model, max_tokens=max_tokens, system=system, messages=msgs, **kw)
        return "".join(getattr(b, "text", "") for b in r.content)
    if model.startswith("gemini"):
        # Gemini 2.5 Flash/Pro think by default and thinking counts against the
        # output budget, so the answer needs a larger budget than max_tokens
        kw = llm_runner.gemini_kw(EFFORT, max(max_tokens, 3000))
        r = GEM.chat.completions.create(model=model, messages=[{"role": "system", "content": system}] + msgs,
                                        temperature=1.0, **kw)
        c = r.choices[0]
        if c.finish_reason == "content_filter":
            raise RuntimeError("content_filter: response blocked by Gemini safety filter")
        if not (c.message.content or "").strip():
            raise RuntimeError(f"empty content (finish={c.finish_reason})")
        return c.message.content
    kw = {"max_tokens": max_tokens, "temperature": 1.0}
    if json_schema:
        kw["response_format"] = {"type": "json_schema", "json_schema": {"name": "r", "strict": True, "schema": json_schema}}
    r = AZ.chat.completions.create(model=model, messages=[{"role": "system", "content": system}] + msgs, **kw)
    return r.choices[0].message.content or ""

# interviewer script, clinical order; each maps to one PHQ-9 item
SCRIPT = [
 ("DPQ020", "How has your mood been over the last couple of weeks?"),
 ("DPQ010", "Are you still able to enjoy the things you usually like doing?"),
 ("DPQ030", "How have you been sleeping lately?"),
 ("DPQ040", "How has your energy been during the day?"),
 ("DPQ050", "Have you noticed any changes in your appetite or how much you eat?"),
 ("DPQ060", "How have you been feeling about yourself lately?"),
 ("DPQ070", "How is your concentration, say when reading, watching something, or working?"),
 ("DPQ080", "Has anyone noticed you moving or talking more slowly than usual, or being unusually restless or fidgety?"),
 ("DPQ090", "Some people who feel low have thoughts that they would be better off dead, or of hurting themselves. Have you had thoughts like that?"),
]
OPEN = "Hi, thanks for coming in today. I'd like to ask you a few questions about how you've been over the last two weeks."
FREQ = {0: "not at all", 1: "on several days", 2: "on more than half the days", 3: "nearly every day"}
SHORT = {"DPQ010": "little interest or pleasure in things", "DPQ020": "feeling down or hopeless",
         "DPQ030": "trouble sleeping or sleeping too much", "DPQ040": "tiredness or low energy",
         "DPQ050": "poor appetite or overeating", "DPQ060": "feeling bad about yourself",
         "DPQ070": "trouble concentrating", "DPQ080": "moving/speaking slowly or being restless",
         "DPQ090": "thoughts of being better off dead or of self-harm"}

def patient_system(cond, sev=None, profile=None):
    base = ("You are role-playing a patient in a clinical intake interview, as part of a "
            "clinician-training simulation. Stay in character throughout. Answer each question "
            "naturally, in one to three sentences, as this person would speak. Do not mention "
            "questionnaires, scores or ratings.\n\n")
    if cond == "free":
        return base + f"Your character: a person living with {sev.replace('_', ' ')} depression."
    if cond == "talkdep":   # sev holds the persona name; the persona text is given in full
        from mental_act.published_personas import talkdep_profiles
        return base + "Your character is the following patient:\n\n" + talkdep_profiles()[sev]
    facts ="; ".join(f"{SHORT[i]}: {FREQ[int(v)]}" for i, v in zip(PHQ9_ITEMS, profile))
    return base + ("Your character: over the last two weeks this person has experienced the "
                   f"following (private background; never recite it as a list): {facts}.")

RATER_SYS = ("You are an experienced clinician. Read the interview transcript and rate the patient "
             "on each PHQ-9 item for the last two weeks, using only what the patient said. "
             "If an answer is shown as not available, rate that item from the rest of the conversation. "
             "0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day. "
             "Respond only with the requested JSON.")
RATER_ITEMS = "\n".join(f"a{n+1}: {SHORT[i]}" for n, i in enumerate(PHQ9_ITEMS))
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {f"a{n}": {"type": "integer"} for n in range(1, 10)},
          "required": [f"a{n}" for n in range(1, 10)]}

def run_one(job):
    cond, model, sev, profile, tid, raters = job
    rec = dict(trial_id=tid, cond=cond, model=model, severity=sev,
               true_profile=profile, status="ok")
    try:
        sysmsg = patient_system(cond, sev, profile)
        msgs, transcript = [], [f"Clinician: {OPEN}"]
        filtered = []
        for n, (item, q) in enumerate(SCRIPT):
            content = (OPEN + " " + q) if n == 0 else q
            msgs.append({"role": "user", "content": content})
            try:
                ans = chat(model, sysmsg, msgs, max_tokens=250).strip()
            except ContentFiltered:
                # Recorded as data. The placeholder carries no content, so the
                # rater cannot infer anything from it beyond "not available".
                filtered.append(item)
                ans = "[response not available]"
            msgs.append({"role": "assistant", "content": ans})
            transcript += [f"Clinician: {q}", f"Patient: {ans}"]
        rec["filtered_items"] = filtered
        rec["transcript"] = "\n".join(transcript)
        for rm in raters:
            txt = chat(rm, RATER_SYS, [{"role": "user", "content":
                   f"Transcript:\n{rec['transcript']}\n\nItems:\n{RATER_ITEMS}\n\nReply with JSON a1..a9."}],
                   max_tokens=200, json_schema=SCHEMA)
            o = json.loads(txt); v = [int(o[f"a{n}"]) for n in range(1, 10)]
            assert all(0 <= x <= 3 for x in v)
            rec[f"rated__{rm}"] = v
    except Exception as e:
        rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:200]}")
    return rec

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--cond", default="free")
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--raters", nargs="+", default=["gpt-4.1", "claude-haiku-4-5"])
    ap.add_argument("--effort", default=None, help="reasoning effort of a Gemini patient (or budget<N>, a thinking budget); output goes to eff__ runs")
    a = ap.parse_args()
    EFFORT = a.effort
    rng = random.Random(11)
    sevs = ["mild", "moderate", "moderately_severe", "severe"]
    bands = {"mild": (5, 9), "moderate": (10, 14), "moderately_severe": (15, 19), "severe": (20, 27)}
    jobs = []
    if a.cond == "free":
        jobs = [("free", a.model, s, None, f"free-{s}-{r:03d}", a.raters) for s in sevs for r in range(a.n)]
    elif a.cond == "talkdep":
        from mental_act.published_personas import talkdep_profiles
        jobs = [("talkdep", a.model, name, None, f"talkdep-{name}-{r:03d}", a.raters) for name in talkdep_profiles() for r in range(a.n)]
    elif a.cond == "grm":
        # profiles drawn from the PHQ-9 graded response model (no real rows), same bands
        from mental_act import grm_sampler
        grng = np.random.default_rng(20261012)
        for s in sevs:
            lo, hi = bands[s]
            for r, prof in enumerate(grm_sampler.sample_band(lo, hi, a.n, grng)):
                jobs.append(("grm", a.model, s, prof.tolist(), f"grm-{s}-{r:03d}", a.raters))
    else:
        df = pd.read_csv(DATA_PROC / "nhanes_phq9.csv")
        for s in sevs:
            lo, hi = bands[s]; P = df[(df.phq9_total >= lo) & (df.phq9_total <= hi)][PHQ9_ITEMS].to_numpy(int)
            for r in range(a.n):
                jobs.append(("oracle", a.model, s, P[rng.randrange(len(P))].tolist(), f"oracle-{s}-{r:03d}", a.raters))
    tag = a.model.replace('.','').replace('-','').lower()
    out = RUNS / (f"eff__interview-{a.cond}__{tag}__{a.effort}.jsonl" if a.effort else f"interview__{a.cond}__{tag}.jsonl")
    done = set()
    if out.exists():
        done = {r["trial_id"] for r in map(json.loads, filter(str.strip, out.read_text().splitlines()))
                if r.get("status") == "ok"}
    jobs = [j for j in jobs if j[4] not in done]
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex, out.open("a") as f:
        for rec in ex.map(run_one, jobs):
            f.write(json.dumps(rec) + "\n"); f.flush()
    rs = [json.loads(l) for l in out.read_text().splitlines()]
    print(f"{a.cond} {a.model}: {sum(r['status']=='ok' for r in rs)}/{len(rs)} ok, {time.time()-t0:.0f}s", flush=True)
