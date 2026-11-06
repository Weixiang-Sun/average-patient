"""Personas from published patient simulators, in place of the one-line severity persona.

talkdep  12 clinician-validated depression personas (TalkDep; Anxo06/TalkDep on
         GitHub): demographics, BDI-II score, up to four key symptoms with
         severities, life history, social context, communication style,
         behavioural constraints. Each is administered 30 times.
cactus   client intake forms from CACTUS (LangAGI-Lab/cactus, GPL), used there to
         drive LLM-simulated counselling clients: demographics, presenting
         problem, history, functioning, relationships. 400 distinct clients,
         one administration each.
cactus_cm the same 400 clients with the elements of their CBT case formulation that
         CACTUS records with each client: an automatic thought and the cognitive
         distortions it shows (the closest public analogue of the cognitive models
         of Patient-Psi, whose data are released only on request).
Any of them can be given with the PHQ-9 (main) or the GAD-7 questionnaire.
The system message and the questionnaire block are the same as in the main
experiment; only the persona text changes.
"""
from __future__ import annotations
import json, random
from pathlib import Path
from .config import PHQ9_ITEMS, PHQ9_LABELS, GAD7_ITEMS, GAD7_LABELS
from .personas import SYSTEM, questionnaire_block

EXT = Path(__file__).resolve().parents[2] / "data" / "external"

def talkdep_profiles() -> dict[str, str]:
    return {p.stem: p.read_text().strip() for p in sorted((EXT / "talkdep").glob("*.md"))}

TALKDEP_BAND = {"minimal": "none", "mild": "mild", "moderate": "moderate",
                "moderate to severe": "moderately_severe", "severe": "severe"}

def talkdep_severity() -> dict[str, tuple[int, str]]:
    """Persona name -> (published BDI-II score, PHQ-9 band named like the persona's severity
    label: minimal -> none 0-4, mild, moderate, moderate to severe -> moderately severe, severe)."""
    import re
    out = {}
    for name, text in talkdep_profiles().items():
        m = re.search(r"BDI-II Score\W*(\d+)\s*\(([^)]*)\)", text)
        out[name] = (int(m.group(1)), TALKDEP_BAND[m.group(2).lower().replace(" depression", "").strip()])
    return out

def cactus_forms(n: int = 400, seed: int = 20261012) -> list[str]:
    d = json.load(open(EXT / "cactus" / "cactus.json"))
    forms = sorted({x["intake_form"].strip() for x in d})
    return random.Random(seed).sample(forms, n)

def cactus_formulations(n: int = 400, seed: int = 20261012) -> list[str]:
    """The same 400 clients as cactus_forms, each with the automatic thought and the
    cognitive distortions of its first CACTUS record (file order)."""
    d = json.load(open(EXT / "cactus" / "cactus.json")); first = {}
    for x in d: first.setdefault(x["intake_form"].strip(), x)
    out = []
    for form in cactus_forms(n, seed):
        x = first[form]; pats = x["patterns"] if isinstance(x["patterns"], list) else [x["patterns"]]
        out.append(f"{form}\n\nAutomatic thought: {x['thought'].strip()}\nCognitive distortions: {', '.join(pats)}")
    return out

INSTR = {"phq9": (PHQ9_ITEMS, PHQ9_LABELS), "gad7": (GAD7_ITEMS, GAD7_LABELS)}

def _trial(tid, source, persona_id, persona_text, rng, instrument="phq9"):
    items, labels = INSTR[instrument]
    order = items[:]; rng.shuffle(order)
    lead = {"talkdep": "Your character is the following patient:",
            "cactus": "Your character is the client described in this counselling intake form:",
            "cactus_cm": "Your character is the client described in this counselling intake form, with the automatic thought and cognitive distortions from their case formulation:"}[source]
    user = (f"{lead}\n\n{persona_text}\n\nAnswer the following questionnaire as this person would answer it "
            f"about themselves.\n\n" + questionnaire_block(order, labels))
    return {"trial_id": tid, "arm": source, "severity": persona_id, "rep": 0, "item_order": order,
            "system": SYSTEM, "user": user, "mode": "block", "instrument": instrument}

def build_trials(source: str, reps: int = 30, n_cactus: int = 400, seed: int = 20261012, instrument: str = "phq9") -> list[dict]:
    rng = random.Random(seed); out = []
    if source == "talkdep":
        for name, text in talkdep_profiles().items():
            for r in range(reps):
                t = _trial(f"talkdep-{name}-{r:03d}", source, name, text, rng, instrument); t["rep"] = r; out.append(t)
    elif source in ("cactus", "cactus_cm"):
        texts = cactus_forms(n_cactus, seed) if source == "cactus" else cactus_formulations(n_cactus, seed)
        for i, text in enumerate(texts):
            out.append(_trial(f"{source}-{i:04d}", source, f"client{i:04d}", text, rng, instrument))
    else:
        raise ValueError(source)
    return out

def run_name(source: str, tag: str, instrument: str = "phq9") -> str:
    """persona_<source>__<tag> for the PHQ-9 (the original runs), persona_<source>_gad7__<tag> for the GAD-7."""
    return f"persona_{source}__{tag}" if instrument == "phq9" else f"persona_{source}_{instrument}__{tag}"
