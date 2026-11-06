"""Interventions aimed at the mechanism: severity word -> uniform response level.

  label        baseline (as in the main experiments)
  hetero       + explicit statement that people at one severity differ, most
                 having some symptoms absent and a few severe
  numeric      no severity word; target PHQ-9 total drawn inside the band
  biography    individuated persona (name, age, occupation, life stressor),
                 no symptom content, plus the severity label
  fewshot      five real NHANES response vectors from the target band, drawn
                 from the exemplar half of NHANES (even SEQN); evaluation uses
                 the other half (odd SEQN)
"""
from __future__ import annotations
import random
import numpy as np, pandas as pd
from .config import DATA_PROC, PHQ9_ITEMS, PHQ9_LABELS
from .personas import SYSTEM, questionnaire_block

BANDS = {"mild": (5, 9), "moderate": (10, 14), "moderately_severe": (15, 19), "severe": (20, 27)}
SEVS = list(BANDS)

FIRST = ["Maria","James","Aisha","Daniel","Mei","Carlos","Grace","Tom","Priya","Samuel",
         "Elena","Marcus","Yuki","David","Fatima","Liam","Rosa","Kevin","Hannah","Omar"]
JOBS = ["warehouse supervisor","nurse","high-school teacher","software developer","retail cashier",
        "accountant","bus driver","graduate student","restaurant cook","paralegal","electrician",
        "stay-at-home parent","sales representative","retired machinist","hotel receptionist"]
STRESS = ["whose father died six months ago","going through a divorce","who lost their job this year",
          "caring for a parent with dementia","who moved to a new city alone last year",
          "struggling with debt after a medical bill","whose long-term relationship recently ended",
          "dealing with chronic back pain","under heavy pressure at work for months",
          "whose child has been seriously ill","with no single obvious trigger"]

HETERO = (" Keep in mind that people with the same level of depression differ a lot from "
          "one another. Most have some of the problems below not at all, a few quite "
          "strongly, and the rest somewhere in between. Answer as one specific person, "
          "not as an average.")

def _fewshot_pool():
    df = pd.read_csv(DATA_PROC / "nhanes_phq9.csv")
    ex = df[df.SEQN % 2 == 0]
    return {s: ex[(ex.phq9_total >= lo) & (ex.phq9_total <= hi)][PHQ9_ITEMS].to_numpy(int)
            for s, (lo, hi) in BANDS.items()}

def _fewshot_block(rng, pool, sev, k=5):
    P = pool[sev]; pick = P[rng.sample(range(len(P)), k)]
    lines = []
    for n, v in enumerate(pick, 1):
        ans = "; ".join(f"{PHQ9_LABELS[i]}: {int(x)}" for i, x in zip(PHQ9_ITEMS, v))
        lines.append(f"  Person {n}: {ans}")
    return ("Here are the questionnaire answers of five real people with "
            f"{sev.replace('_',' ')} depression, to show how answers vary between people "
            "(0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day):\n"
            + "\n".join(lines) + "\n\nNow answer as a different person with "
            f"{sev.replace('_',' ')} depression."), pick

def persona(cond, sev, rng, pool=None):
    lab = f"{sev.replace('_',' ')} depression"
    if cond == "label":
        return f"You are a person living with {lab}. Answer the following questionnaire as that person would answer it about themselves.", None
    if cond == "hetero":
        return f"You are a person living with {lab}.{HETERO} Answer the following questionnaire as that person would answer it about themselves.", None
    if cond == "numeric":
        lo, hi = BANDS[sev]; t = rng.randint(lo, hi)
        return (f"You are a person whose total score on the following questionnaire would be "
                f"about {t} out of 27. Answer it as that person would answer it about themselves."), t
    if cond == "biography":
        g = rng.choice(FIRST); age = rng.randint(22, 68)
        j = rng.choice(JOBS); art = "an" if j[0] in "aeiou" else "a"
        return (f"You are {g}, {age}, {art} {j} {rng.choice(STRESS)}. "
                f"You are living with {lab}. Answer the following questionnaire as "
                f"{g} would answer it about themselves."), None
    if cond == "fewshot":
        txt, pick = _fewshot_block(rng, pool, sev)
        return txt, pick.tolist()
    raise ValueError(cond)

def build_trials(conds, n_per_cell=40, seed=7):
    rng = random.Random(seed); pool = _fewshot_pool() if "fewshot" in conds else None
    trials = []
    for c in conds:
        for s in SEVS:
            for rep in range(n_per_cell):
                order = PHQ9_ITEMS[:]; rng.shuffle(order)
                txt, meta = persona(c, s, rng, pool)
                trials.append({"trial_id": f"{c}-{s}-{rep:03d}", "arm": c, "severity": s,
                               "rep": rep, "item_order": order, "system": SYSTEM, "mode": "block",
                               "instrument": "phq9", "meta": meta,
                               "user": txt + "\n\n" + questionnaire_block(order)})
    return trials
