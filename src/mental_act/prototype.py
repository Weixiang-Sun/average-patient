"""The criteria-generated ("textbook") response pole.

Constructed ONLY from DSM-5 MDD Criterion A and the PHQ-9 scoring
documentation. It answers: what would a PHQ-9 protocol look like if it were
generated from the diagnostic criteria rather than from lived experience?

Three textual facts from DSM-5 do all the work -- none of them is an
assumption about how LLMs behave:

  1. CARDINAL SYMPTOMS.  Criterion A requires >=5 of 9 symptoms, at least one
     of which must be depressed mood or loss of interest/pleasure. A generator
     reading the criteria endorses both cardinal symptoms.

  2. FREQUENCY WORDING.  DSM-5 qualifies most Criterion A symptoms with
     "nearly every day", which is verbatim the top PHQ-9 category (3).
     A generator reading the criteria answers 3, not 1 or 2.

  3. SEVERITY = SYMPTOM COUNT.  DSM-5 severity specifiers are defined by the
     NUMBER of symptoms present. PHQ-9 severity is defined by the SUM of
     count x intensity. So a criteria-driven generator varies how MANY
     symptoms it endorses while pinning intensity at the ceiling; a real
     respondent varies intensity too.

Fact 3 is the falsifiable structural prediction this pole exists to test.

Frozen before any LLM data was collected. See prototype_frozen.json.
"""
from __future__ import annotations

import itertools

import numpy as np

from .config import PHQ9_ITEMS

# PHQ-9 item index for each DSM-5 Criterion A symptom
CARDINAL = [1, 0]          # DPQ020 depressed mood, DPQ010 anhedonia
NON_CARDINAL = [2, 3, 4, 5, 6, 7, 8]   # sleep, fatigue, appetite,
                                       # worthlessness, concentration,
                                       # psychomotor, suicidality

# DSM-5 severity specifiers, expressed as number of Criterion A symptoms
SPECIFIER_COUNTS = {
    "mild": [5],                 # "few, if any, symptoms in excess of those
                                 #  required to make the diagnosis"
    "moderate": [6, 7],
    "severe": [8, 9],            # "the number of symptoms is substantially
                                 #  in excess of that required"
}


def build_pole(strict: bool = True) -> tuple[np.ndarray, list[dict]]:
    """Enumerate every criteria-consistent PHQ-9 vector.

    strict=True  : endorsed = 3 ("nearly every day"), absent = 0.
                   The literal DSM reading.
    strict=False : endorsed in {2,3}, absent in {0,1}. Same COUNT structure,
                   relaxed intensity, so results do not hinge on the ceiling.
    """
    rows, meta = [], []
    for spec, counts in SPECIFIER_COUNTS.items():
        for k in counts:
            for extra in itertools.combinations(NON_CARDINAL, k - 2):
                endorsed = set(CARDINAL) | set(extra)
                if strict:
                    variants = [[3 if j in endorsed else 0 for j in range(9)]]
                else:
                    variants = []
                    for hi in itertools.product([2, 3], repeat=len(endorsed)):
                        v = [0] * 9
                        for j, h in zip(sorted(endorsed), hi):
                            v[j] = h
                        variants.append(v)
                for v in variants:
                    rows.append(v)
                    meta.append({"specifier": spec, "n_symptoms": k,
                                 "strict": strict, "total": sum(v)})
    return np.array(rows, int), meta
