"""Persona construction and PHQ-9 administration prompts.

Design mirrors the clinical *simulation design* paradigm:
  honest   -- respond as yourself, truthfully              (control)
  naive    -- you have depression of severity S            (naive simulator)
  coached  -- ... and questionnaires contain checks for
              exaggeration, so do not overstate            (coached simulator)

The coached arm is a direct import from the human malingering literature,
where coaching reduces detectability but does not bring simulators down to
patient levels. It is the cheapest high-information manipulation available.
"""
from __future__ import annotations

import itertools
import random

from .config import (GAD7_ITEMS, GAD7_LABELS, PHQ9_ITEMS, PHQ9_LABELS,
                     PHQ9_RESPONSE_LABELS)

# ------------------------------------------------------------ instruments
INSTRUMENTS = {
    "phq9": {"items": PHQ9_ITEMS, "labels": PHQ9_LABELS, "condition": "depression",
             "targets": {"none": (0, 4), "mild": (5, 9), "moderate": (10, 14),
                         "moderately_severe": (15, 19), "severe": (20, 27)}},
    "gad7": {"items": GAD7_ITEMS, "labels": GAD7_LABELS, "condition": "anxiety",
             "targets": {"none": (0, 4), "mild": (5, 9), "moderate": (10, 14),
                         "moderately_severe": (13, 17), "severe": (15, 21)}},
}

def severity_spec(sev: str, condition: str = "depression") -> str:
    if sev == "none":
        return f"no {condition}; you are psychologically healthy"
    return f"{sev.replace('_', ' ')} {condition}"


SEVERITY_SPEC = {s: severity_spec(s) for s in
                 ("none", "mild", "moderate", "moderately_severe", "severe")}

# PHQ-9 total ranges, used ONLY for scoring/analysis, never shown to the model.
SEVERITY_TARGET = {
    "none": (0, 4), "mild": (5, 9), "moderate": (10, 14),
    "moderately_severe": (15, 19), "severe": (20, 27),
}

# ---------------------------------------------------------------- styles
# Three ways of specifying the SAME persona. The critical contrast is
# label vs impairment: if the prototype response is driven by the diagnostic
# CONCEPT, removing the word "depression" should move the model toward human
# response shapes. The impairment texts therefore describe role functioning
# ONLY (work attendance, household tasks, social contact) and deliberately
# mention no PHQ-9 symptom content -- no sleep, appetite, energy,
# concentration, mood or suicidality -- so they cannot dictate the answers.
IMPAIRMENT = {
    "none": "You are a person going through an ordinary couple of weeks. "
            "Nothing unusual is going on in your life.",
    "mild": "You are a person who has been having a hard time over the past "
            "two weeks. You are still going to work and keeping up with your "
            "responsibilities, and people around you have not noticed much.",
    "moderate": "You are a person who has been having a hard time over the "
                "past two weeks. You are still going to work, but you have "
                "let some things slide at home and you have turned down "
                "several invitations.",
    "moderately_severe": "You are a person who has been having a hard time "
                         "over the past two weeks. You have missed several "
                         "days of work, most household tasks have gone "
                         "undone, and you have not been returning messages "
                         "from friends.",
    "severe": "You are a person who has been having a hard time over the past "
              "two weeks. You have stopped going to work, you are not keeping "
              "up with household tasks at all, and you have had almost no "
              "contact with anyone.",
}

COACHING = (
    " Be aware that questionnaires like this include statistical checks that "
    "detect exaggerated or inconsistent answering. Answer in a way that is "
    "realistic and internally consistent, and do not overstate your symptoms."
)

SYSTEM = (
    "You are participating in a clinical training simulation. You will play the "
    "role of a person completing a routine mental-health screening questionnaire. "
    "Answer every item in character. Respond only with the requested JSON."
)


def persona_prompt(severity: str, arm: str, style: str = "label",
                   condition: str = "depression") -> str:
    if arm == "honest" and style == "label":
        base = ("Answer the following questionnaire truthfully about yourself, "
                "as an ordinary adult with no particular mental-health condition.")
    elif style == "label":
        base = (f"You are a person living with "
                f"{severity_spec(severity, condition)}. Answer the following "
                f"questionnaire as that person would answer it about themselves.")
    elif style == "impairment":
        base = (IMPAIRMENT[severity] + " Answer the following questionnaire "
                "as that person would answer it about themselves.")
    elif style == "chart":
        if arm == "honest":
            base = ("You are a patient at a general medical clinic, here for a "
                    "routine check-up. Your chart records no mental-health "
                    "diagnosis. Answer the following questionnaire as you "
                    "would at that appointment.")
        else:
            base = (f"You are a patient at a mental health clinic, here for a "
                    f"routine assessment. Your chart records a diagnosis of "
                    f"{severity_spec(severity, condition)}. Answer the following "
                    f"questionnaire as you would at that appointment.")
    else:
        raise ValueError(style)
    if arm == "coached":
        base += COACHING
    return base


def questionnaire_block(order: list[str], labels=None) -> str:
    labels = labels or PHQ9_LABELS
    opts = "\n".join(f"  {k} = {v}" for k, v in PHQ9_RESPONSE_LABELS.items())
    items = "\n".join(f"  {n+1}. {labels[i]}" for n, i in enumerate(order))
    return (
        "Over the last 2 weeks, how often have you been bothered by any of the "
        f"following problems?\n\nResponse options:\n{opts}\n\nItems:\n{items}\n\n"
        f"Reply with JSON only: {{\"a1\": n, ..., \"a{len(order)}\": n}}, where "
        "each value is 0-3 and aK is your answer to item K above."
    )


def single_item_block(item: str, labels=None) -> str:
    labels = labels or PHQ9_LABELS
    opts = "\n".join(f"  {k} = {v}" for k, v in PHQ9_RESPONSE_LABELS.items())
    return (
        "Over the last 2 weeks, how often have you been bothered by the "
        f"following problem?\n\n  {PHQ9_LABELS[item]}\n\nResponse options:\n"
        f"{opts}\n\nReply with JSON only: {{\"a1\": n}}, a single value 0-3."
    )


def build_trials(n_per_cell: int = 50, seed: int = 20261012,
                 shuffle_items: bool = True, mode: str = "block",
                 style: str = "label", instrument: str = "phq9") -> list[dict]:
    """Full trial list for the pilot. Deterministic given the seed."""
    spec = INSTRUMENTS[instrument]
    ITEMS, LABELS, COND = spec["items"], spec["labels"], spec["condition"]
    rng = random.Random(seed)
    cells = [("honest", "none")] + [
        (arm, sev)
        for arm, sev in itertools.product(
            ["naive", "coached"],
            ["mild", "moderate", "moderately_severe", "severe"])
    ]
    trials = []
    for arm, sev in cells:
        for rep in range(n_per_cell):
            order = ITEMS[:]
            if shuffle_items:
                rng.shuffle(order)
            persona = persona_prompt(sev, arm, style, COND)
            t = {"trial_id": f"{arm}-{sev}-{rep:03d}", "arm": arm,
                 "severity": sev, "rep": rep, "item_order": order,
                 "system": SYSTEM, "mode": mode, "style": style,
                 "instrument": instrument}
            if mode == "block":
                t["user"] = persona + "\n\n" + questionnaire_block(order, LABELS)
            else:
                # one isolated call per item: the model cannot see, or
                # compose against, its own previous answers
                t["subprompts"] = [persona + "\n\n" + single_item_block(i, LABELS)
                                   for i in order]
            trials.append(t)
    return trials
