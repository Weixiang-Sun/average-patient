"""Build the blind human-rating package for the interview transcripts (sixth review, rater check).

Sample (seeded, interviews with any filtered answer left out):
  oracle  real NHANES profile as background: 5 patient models x 4 severity bands x 2 = 40
  grm     GRM-sampled profile as background: 5 patient models x 2 (different bands) = 10
  free    severity persona only:             7 patient models x 2 (different bands) = 14
Raters see only the transcript, under a neutral id (T01-T64), in an order seeded by their rater id.

Writes outputs/human_rating/
  package/rating_form.html   self-contained offline form (send this file to each rater)
  package/codebook.md        the same instructions as in the form
  key/key.csv                id -> condition, model, trial, given profile, LLM ratings (do not send)
  returns/                   put the raters' exported CSV files here (scripts/73 reads them)
"""
import sys, json, glob, random
sys.path.insert(0, "src")
import pandas as pd
from mental_act import analysis as A
from mental_act.config import RUNS, ROOT

OUT = ROOT / "outputs" / "human_rating"
for sub in ("package", "key", "returns"): (OUT / sub).mkdir(parents=True, exist_ok=True)
rng = random.Random(20261010)
SEV = ["mild", "moderate", "moderately_severe", "severe"]
RATERS = ["gpt-4.1", "claude-haiku-4-5"]
SELF = {"gpt41": "gpt-4.1", "claudehaiku45": "claude-haiku-4-5"}


def load(cond, tag):
    d = pd.DataFrame([json.loads(l) for l in open(RUNS / f"interview__{cond}__{tag}.jsonl")]).query("status == 'ok'").drop_duplicates("trial_id")
    fi = d["filtered_items"] if "filtered_items" in d else pd.Series([[]] * len(d), index=d.index)
    d = d[[not (isinstance(v, list) and len(v)) for v in fi]]          # no filtered answer at all
    return d[~d.transcript.str.contains("[response not available]", regex=False)]


rows = []
def take(cond, tag, sev, k):
    d = load(cond, tag); d = d[d.severity == sev]
    for _, r in d.sample(n=k, random_state=rng.randrange(10**9)).iterrows():
        rows.append(dict(cond=cond, model=tag, trial_id=r.trial_id, severity=r.severity, transcript=r.transcript,
                         given=r.get("true_profile") if cond != "free" else None,
                         **{f"llm_{x}": (r[f"rated__{x}"] if f"rated__{x}" in r and SELF.get(tag) != x else None) for x in RATERS}))

tags = lambda cond: sorted(t.split("__")[2][:-6] for t in map(lambda p: p.split("/")[-1], glob.glob(str(RUNS / f"interview__{cond}__*.jsonl"))) if not t.split("__")[2].startswith("eff"))
for tag in tags("oracle"):
    for s in SEV: take("oracle", tag, s, 2)
for tag in tags("grm"):
    for s in rng.sample(SEV, 2): take("grm", tag, s, 1)
for tag in tags("free"):
    for s in rng.sample(SEV, 2): take("free", tag, s, 1)

rng.shuffle(rows)
for i, r in enumerate(rows): r["id"] = f"T{i + 1:02d}"
K = pd.DataFrame(rows)
key = K[["id", "cond", "model", "trial_id", "severity"]].copy()
for j in range(9):
    key[f"given_{j + 1}"] = [g[j] if isinstance(g, list) else None for g in K.given]
    for x in RATERS:
        key[f"{x}_{j + 1}"] = [v[j] if isinstance(v, list) else None for v in K[f"llm_{x}"]]
key.to_csv(OUT / "key" / "key.csv", index=False)
print(K.groupby(["cond", "model"]).size().to_string())
print("total", len(K))

ITEMS = ["Little interest or pleasure in doing things",
         "Feeling down, depressed, or hopeless",
         "Trouble falling or staying asleep, or sleeping too much",
         "Feeling tired or having little energy",
         "Poor appetite or overeating",
         "Feeling bad about yourself, or that you are a failure or have let yourself or your family down",
         "Trouble concentrating on things, such as reading the newspaper or watching television",
         "Moving or speaking so slowly that other people could have noticed, or the opposite: being so fidgety or restless that you have been moving around a lot more than usual",
         "Thoughts that you would be better off dead, or of hurting yourself in some way"]
CODEBOOK = """# Rating instructions

You will read %d short transcripts of the same scripted clinical interview. A clinician asks about the nine areas of the PHQ-9 depression questionnaire and a patient answers. The patients are played by AI models; some transcripts mention thoughts of death or self-harm. You can take breaks and stop at any time.

For each transcript, rate the nine PHQ-9 items below **for the last two weeks, using only what the patient says**.

| Rating | Meaning |
|---|---|
| 0 | Not at all |
| 1 | Several days |
| 2 | More than half the days |
| 3 | Nearly every day |

Items:
%s

Rules
1. Use only what the patient says about that item. Do not fill in an item from the overall picture of the person, and do not assume a problem is present because others are.
2. Rate how often the problem occurred. Use the patient's own description of frequency: "several days", "some days", "now and then", "a couple of times" usually mean 1; "more than half the days", "most days", "more often than not" usually mean 2; "nearly every day", "every day", "all the time", "constantly" usually mean 3. When the patient gives no frequency, choose the option that best fits what they describe.
3. If the patient denies the problem, rate 0.
4. Tick "unsure" when the transcript does not let you decide between two ratings. Still choose the closer one.
5. Work alone and do not discuss the transcripts with the other rater until both of you have finished. Do not use AI tools.

A transcript takes about 3 to 5 minutes. The form saves your progress in this browser automatically. When you have rated every transcript, press "Export CSV" and send the file back.
""" % (len(K), "\n".join(f"{j + 1}. {t}" for j, t in enumerate(ITEMS)))
(OUT / "package" / "codebook.md").write_text(CODEBOOK)

data = json.dumps({"items": ITEMS, "transcripts": [{"id": r["id"], "text": r["transcript"]} for r in rows]}).replace("</", "<\\/")
html = open(ROOT / "scripts" / "human_rating_form.html").read()
html = html.replace("__DATA__", data).replace("__CODEBOOK__", json.dumps(CODEBOOK).replace("</", "<\\/"))
(OUT / "package" / "rating_form.html").write_text(html)
assert not any(t in html for t in ("gpt", "gemini", "llama", "claude", "haiku", "oracle", "true_profile")), "identifier leaked into the form"
print("wrote", OUT / "package" / "rating_form.html")
