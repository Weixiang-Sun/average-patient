"""Main runs with reasoning off (2026-10-08), or at the minimum for Gemini 2.5 Pro, which cannot
turn it off (the API rejects a zero thinking budget; added 2026-10-09). Four models reasoned at
their API default and are switched:

  GPT-5-mini        reasoning_effort "minimal" (0 reasoning tokens on every call)
  Gemini 2.5 Flash  reasoning_effort "none"    (thinking budget 0)
  Kimi-K2.5         reasoning_effort "none"    (Azure; no reasoning_content)
  Gemini 2.5 Pro    thinking budget 128 tokens (its minimum; "budget128", scripts/68)

For each main run file of these models, the API-default run is kept as
eff__<exp>__<tag>__default.jsonl and the reasoning-off run (scripts/61, 35 --effort)
(scripts/61, 35 --effort, 68) is copied to the main name, so every analysis that reads the main runs uses it.
Idempotent; writes outputs/runs/reasoning_off_manifest.json with checksums.
"""
import sys, json, shutil, hashlib
sys.path.insert(0, "src")
from mental_act.config import RUNS

X = ["x_k6", "x_ocir", "x_phq15", "x_itq", "x_edeq", "x_pcl5c"]
SWAP = {   # tag: (effort of the off run, {main run name: experiment})
    "gpt5mini": ("minimal", {"phq9__gpt5mini": "phq9", "gad7__gpt5mini": "gad7", "pcl5__gpt5mini": "pcl5",
                             **{f"{x}__gpt5mini": x for x in X}}),
    "gemini25flash": ("none", {"phq9__gemini25flash": "phq9", "gad7__gemini25flash": "gad7", "pcl5__gemini25flash": "pcl5",
                               **{f"{x}__gemini25flash": x for x in X},
                               "fixes__gemini25flash": "fixes", "persona_talkdep__gemini25flash": "persona_talkdep",
                               "persona_cactus__gemini25flash": "persona_cactus",
                               **{f"interview__{c}__gemini25flash": f"interview-{c}" for c in ("free", "oracle", "grm", "talkdep")}}),
    "kimik25": ("none", {"kimi_k25": "phq9", "gad_kimi": "gad7", "pcl5__kimik25": "pcl5", **{f"{x}__kimik25": x for x in X}}),
    # 2026-10-09: Gemini 2.5 Pro at its minimum thinking budget (scripts/68)
    "gemini25pro": ("budget128", {"phq9__gemini25pro": "phq9", "gad7__gemini25pro": "gad7", "pcl5__gemini25pro": "pcl5",
                                  **{f"{x}__gemini25pro": x for x in X}, "fixes__gemini25pro": "fixes",
                                  **{f"persona_{p}__gemini25pro": f"persona_{p}" for p in
                                     ("talkdep", "cactus", "cactus_cm", "talkdep_gad7", "cactus_gad7", "cactus_cm_gad7")},
                                  **{f"interview__{c}__gemini25pro": f"interview-{c}" for c in ("free", "oracle", "grm", "talkdep")}}),
}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
manifest = []
for tag, (eff, runs) in SWAP.items():
    for main, exp in runs.items():
        for ext in (".jsonl", ".meta.json"):
            m, off, dflt = RUNS / f"{main}{ext}", RUNS / f"eff__{exp}__{tag}__{eff}{ext}", RUNS / f"eff__{exp}__{tag}__default{ext}"
            if ext == ".meta.json" and not off.exists(): continue
            assert off.exists(), f"missing reasoning-off run {off.name}"
            if not dflt.exists() and m.exists():      # first pass: keep the API-default run
                shutil.move(m, dflt)                  # (experiments added later, e.g. TalkDep interviews, have none)
            if not m.exists() or sha(m) != sha(off):
                shutil.copyfile(off, m)
            manifest.append(dict(main=m.name, reasoning_off=off.name, api_default=dflt.name if dflt.exists() else None,
                                 sha_main=sha(m), sha_default=sha(dflt) if dflt.exists() else None))
            print(f"{m.name:40s} <- {off.name}   (API default kept as {dflt.name})")
(RUNS / "reasoning_off_manifest.json").write_text(json.dumps(manifest, indent=1))
print(f"{len(manifest)} files")
