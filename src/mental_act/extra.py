"""Additional instruments and human references (generality extension).

Six instruments beyond the PHQ-9 / GAD-7 / OSF PCL-5 set, each with a public
item-level human reference:

  k6     Kessler-6 psychological distress         NHIS 2016-2018 (US national)     6 items, 0-4
  ocir   Obsessive-Compulsive Inventory-Revised   C19PRC-UK wave 2 (UK quota)     18 items, 0-4
  phq15  PHQ-15 somatic symptoms (14 items; the   C19PRC-UK wave 1                14 items, 0-2
         menstrual item was not administered)
  itq    ITQ PTSD symptoms, anchored to the       C19PRC-UK wave 1                 6 items, 0-4
         COVID-19 pandemic as in the survey
  edeq   EDE-Q 6.0, the 22 attitudinal items      Mexican general population      22 items, 0-6
         (the 6 behaviour-count items excluded)   (Contreras-Valdez et al. 2022)
  pcl5c  PCL-5 with severity personas             Orovou et al. 2021, postpartum  20 items, 0-4
                                                  women after caesarean (clinical)

and one clinical reference for an instrument already administered:

  phq9 vs COACH: rural Chinese older adults screened into a depression trial
  (baseline PHQ-9 >= 10, administered by primary-care physicians).

All raw files are in data/external/extra (see README there for sources).
"""
from __future__ import annotations

import random
import zipfile

import numpy as np
import pandas as pd

from .config import DATA_PROC, DATA_RAW, ROOT
from .personas import SYSTEM, persona_prompt

EXT = ROOT / "data" / "external" / "extra"

# ------------------------------------------------------------------ items
K6 = {
    "K6_1": "So sad that nothing could cheer you up",
    "K6_2": "Nervous",
    "K6_3": "Restless or fidgety",
    "K6_4": "Hopeless",
    "K6_5": "That everything was an effort",
    "K6_6": "Worthless",
}
OCIR = {
    "OCI_01": "I have saved up so many things that they get in the way.",
    "OCI_02": "I check things more often than necessary.",
    "OCI_03": "I get upset if objects are not arranged properly.",
    "OCI_04": "I feel compelled to count while I am doing things.",
    "OCI_05": "I find it difficult to touch an object when I know it has been touched by strangers or certain people.",
    "OCI_06": "I find it difficult to control my own thoughts.",
    "OCI_07": "I collect things I don't need.",
    "OCI_08": "I repeatedly check doors, windows, drawers, etc.",
    "OCI_09": "I get upset if others change the way I have arranged things.",
    "OCI_10": "I feel I have to repeat certain numbers.",
    "OCI_11": "I sometimes have to wash or clean myself simply because I feel contaminated.",
    "OCI_12": "I am upset by unpleasant thoughts that come into my mind against my will.",
    "OCI_13": "I avoid throwing things away because I am afraid I might need them later.",
    "OCI_14": "I repeatedly check gas and water taps and light switches after turning them off.",
    "OCI_15": "I need things to be arranged in a particular way.",
    "OCI_16": "I feel that there are good and bad numbers.",
    "OCI_17": "I wash my hands more often and longer than necessary.",
    "OCI_18": "I frequently get nasty thoughts and have difficulty in getting rid of them.",
}
PHQ15 = {
    "SOM_01": "Stomach pain",
    "SOM_02": "Back pain",
    "SOM_03": "Pain in your arms, legs, or joints (knees, hips, etc.)",
    "SOM_04": "Headaches",
    "SOM_05": "Chest pain",
    "SOM_06": "Dizziness",
    "SOM_07": "Fainting spells",
    "SOM_08": "Feeling your heart pound or race",
    "SOM_09": "Shortness of breath",
    "SOM_10": "Pain or problems during sexual intercourse",
    "SOM_11": "Constipation, loose bowels, or diarrhoea",
    "SOM_12": "Nausea, gas, or indigestion",
    "SOM_13": "Feeling tired or having low energy",
    "SOM_14": "Trouble sleeping",
}
ITQ = {
    "ITQ_1": "Having upsetting dreams that replay part of the experience or are clearly related to the experience?",
    "ITQ_2": "Having powerful images or memories that sometimes come into your mind in which you feel the experience is happening again in the here and now?",
    "ITQ_3": "Avoiding internal reminders of the experience (for example, thoughts, feelings, or physical sensations)?",
    "ITQ_4": "Avoiding external reminders of the experience (for example, people, places, conversations, objects, activities, or situations)?",
    "ITQ_5": "Being \"super-alert\", watchful, or on guard?",
    "ITQ_6": "Feeling jumpy or easily startled?",
}
EDEQ_DAYS = {   # items 1-12: number of days out of the past 28
    "EDEQ_01": "Have you been deliberately trying to limit the amount of food you eat to influence your shape or weight (whether or not you have succeeded)?",
    "EDEQ_02": "Have you gone for long periods of time (8 waking hours or more) without eating anything at all in order to influence your shape or weight?",
    "EDEQ_03": "Have you tried to exclude from your diet any foods that you like in order to influence your shape or weight (whether or not you have succeeded)?",
    "EDEQ_04": "Have you tried to follow definite rules regarding your eating (for example, a calorie limit) in order to influence your shape or weight (whether or not you have succeeded)?",
    "EDEQ_05": "Have you had a definite desire to have an empty stomach with the aim of influencing your shape or weight?",
    "EDEQ_06": "Have you had a definite desire to have a totally flat stomach?",
    "EDEQ_07": "Has thinking about food, eating or calories made it very difficult to concentrate on things you are interested in (for example, working, following a conversation, or reading)?",
    "EDEQ_08": "Has thinking about shape or weight made it very difficult to concentrate on things you are interested in (for example, working, following a conversation, or reading)?",
    "EDEQ_09": "Have you had a definite fear of losing control over eating?",
    "EDEQ_10": "Have you had a definite fear that you might gain weight?",
    "EDEQ_11": "Have you felt fat?",
    "EDEQ_12": "Have you had a strong desire to lose weight?",
}
EDEQ_19 = {"EDEQ_19": "Over the past 28 days, on how many days have you eaten in secret (i.e., furtively)? Do not count episodes of binge eating."}
EDEQ_20 = {"EDEQ_20": "On what proportion of the times that you have eaten have you felt guilty (felt that you've done wrong) because of its effect on your shape or weight? Do not count episodes of binge eating."}
EDEQ_SEV = {    # items 21-28: 0 not at all ... 6 markedly
    "EDEQ_21": "Over the past 28 days, how concerned have you been about other people seeing you eat? Do not count episodes of binge eating.",
    "EDEQ_22": "Has your weight influenced how you think about (judge) yourself as a person?",
    "EDEQ_23": "Has your shape influenced how you think about (judge) yourself as a person?",
    "EDEQ_24": "How much would it have upset you if you had been asked to weigh yourself once a week (no more, or less, often) for the next four weeks?",
    "EDEQ_25": "How dissatisfied have you been with your weight?",
    "EDEQ_26": "How dissatisfied have you been with your shape?",
    "EDEQ_27": "How uncomfortable have you felt seeing your body (for example, seeing your shape in the mirror, in a shop window reflection, while undressing or taking a bath or shower)?",
    "EDEQ_28": "How uncomfortable have you felt about others seeing your shape or figure (for example, in communal changing rooms, when swimming, or wearing tight clothes)?",
}
from .pcl5 import PCL5_LABELS, PCL5_RESPONSES  # noqa: E402

DAYS = {0: "No days", 1: "1-5 days", 2: "6-12 days", 3: "13-15 days",
        4: "16-22 days", 5: "23-27 days", 6: "Every day"}
PROPORTION = {0: "None of the times", 1: "A few of the times", 2: "Less than half",
              3: "Half of the times", 4: "More than half", 5: "Most of the time",
              6: "Every time"}
MARKEDLY = {0: "Not at all", 1: "1 (slightly)", 2: "2 (slightly)", 3: "3 (moderately)",
            4: "4 (moderately)", 5: "5 (markedly)", 6: "6 (markedly)"}

# A section = (preface, response options, items). Items are shuffled within a
# section; sections keep their order because their response scales differ.
SPEC = {
    "k6": dict(
        name="K6", condition="psychological distress", max_cat=4,
        sections=[("During the past 30 days, how often did you feel ...",
                   {0: "None of the time", 1: "A little of the time",
                    2: "Some of the time", 3: "Most of the time", 4: "All of the time"},
                   K6)]),
    "ocir": dict(
        name="OCI-R", condition="obsessive-compulsive disorder", max_cat=4,
        sections=[("The following statements refer to experiences that many people "
                   "have in their everyday lives. For each one, give the number that "
                   "best describes HOW MUCH that experience has DISTRESSED or BOTHERED "
                   "you during the PAST MONTH.",
                   {0: "Not at all", 1: "A little", 2: "Moderately", 3: "A lot",
                    4: "Extremely"}, OCIR)]),
    "phq15": dict(
        name="PHQ-15", condition="somatic symptom disorder", max_cat=2,
        sections=[("During the past 4 weeks, how much have you been bothered by any "
                   "of the following problems?",
                   {0: "Not bothered at all", 1: "Bothered a little",
                    2: "Bothered a lot"}, PHQ15)]),
    "itq": dict(
        name="ITQ", condition="post-traumatic stress related to your experience "
                              "of the COVID-19 pandemic", max_cat=4,
        sections=[("Below are problems that people sometimes have following a "
                   "traumatic or stressful life event. Answer in relation to your "
                   "experience of the COVID-19 pandemic. Indicate how much you have "
                   "been bothered by each problem in the past month.",
                   {0: "Not at all", 1: "A little bit", 2: "Moderately",
                    3: "Quite a bit", 4: "Extremely"}, ITQ)]),
    "edeq": dict(
        name="EDE-Q", condition="eating disorder symptoms", max_cat=6,
        sections=[
            ("The following questions refer to the past four weeks (28 days) only. "
             "On how many of the past 28 days ...", DAYS, EDEQ_DAYS),
            ("Over the past 28 days:", DAYS, EDEQ_19),
            ("Over the past 28 days:", PROPORTION, EDEQ_20),
            ("Over the past 28 days:", MARKEDLY, EDEQ_SEV)]),
    "pcl5c": dict(
        name="PCL-5", condition="post-traumatic stress disorder (PTSD)", max_cat=4,
        sections=[("Below is a list of problems that people sometimes have in "
                   "response to a very stressful experience. Keeping your worst "
                   "stressful experience in mind, indicate how much you have been "
                   "bothered by each problem in the past month.",
                   PCL5_RESPONSES, PCL5_LABELS)]),
}
for _k, _s in SPEC.items():
    _s["items"] = [i for _, _, d in _s["sections"] for i in d]
    _s["J"] = len(_s["items"])


def questionnaire(instr: str, rng: random.Random) -> tuple[str, list[str]]:
    """Render the form with items shuffled within sections; return (text, order)."""
    spec, parts, order, n = SPEC[instr], [], [], 0
    for pre, opts, items in spec["sections"]:
        ids = list(items); rng.shuffle(ids)
        o = "\n".join(f"  {k} = {v}" for k, v in opts.items())
        lines = []
        for i in ids:
            n += 1; order.append(i); lines.append(f"  {n}. {items[i]}")
        parts.append(f"{pre}\nResponse options:\n{o}\nItems:\n" + "\n".join(lines))
    text = "\n\n".join(parts) + (
        f"\n\nReply with JSON only: {{\"a1\": n, ..., \"a{n}\": n}}, where aK is your "
        "answer to item K above, using that item's response options.")
    return text, order


def build_trials(instr: str, n_per_cell: int = 50, seed: int = 20261012) -> list[dict]:
    """Same cell design as the PHQ-9 / GAD-7 runs: honest + naive/coached x 4 severities."""
    spec, rng, trials = SPEC[instr], random.Random(seed), []
    cells = [("honest", "none")] + [(a, s) for a in ("naive", "coached")
                                    for s in ("mild", "moderate", "moderately_severe", "severe")]
    for arm, sev in cells:
        for rep in range(n_per_cell):
            form, order = questionnaire(instr, rng)
            persona = persona_prompt(sev, arm, "label", spec["condition"])
            trials.append({"trial_id": f"{arm}-{sev}-{rep:03d}", "arm": arm,
                           "severity": sev, "rep": rep, "item_order": order,
                           "system": SYSTEM, "mode": "block", "style": "label",
                           "instrument": instr, "max_cat": spec["max_cat"],
                           "user": persona + "\n\n" + form})
    return trials


# ------------------------------------------------------------ human data
def _complete(df: pd.DataFrame, items: list[str], lo: int, hi: int) -> pd.DataFrame:
    x = df[items].apply(pd.to_numeric, errors="coerce")
    ok = x.notna().all(1) & ((x >= lo) & (x <= hi)).all(1)
    out = x[ok].astype(int).reset_index(drop=True)
    out["total"] = out[items].sum(1)
    return out


def load_k6() -> pd.DataFrame:
    """NHIS 2016-2018 Sample Adult, ASI K6 items, 1=all ... 5=none of the time."""
    cols = ["ASISAD", "ASINERV", "ASIRSTLS", "ASIHOPLS", "ASIEFFRT", "ASIWTHLS"]
    frames = []
    for y in (2016, 2017, 2018):
        z = zipfile.ZipFile(DATA_RAW / "nhis" / f"samadult{y}.zip")
        d = pd.read_csv(z.open(z.namelist()[0]), usecols=lambda c: c.upper() in cols,
                        low_memory=False)
        d.columns = [c.upper() for c in d.columns]
        d = d[cols].where(d[cols].isin([1, 2, 3, 4, 5]))
        d = 5 - d                                   # 0 = none ... 4 = all of the time
        d.columns = list(K6); d["year"] = y
        frames.append(d)
    return _complete(pd.concat(frames, ignore_index=True), list(K6), 0, 4)


def _c19prc() -> pd.DataFrame:
    import pyreadstat
    d, _ = pyreadstat.read_sav(EXT / "c19prc_w1w2" / "C19PRC_UKW1W2_archive_final.sav")
    return d


def load_c19prc() -> dict[str, pd.DataFrame]:
    d = _c19prc()
    oc = d[[f"W2_OCI_{i}" for i in range(1, 19)]].copy(); oc.columns = list(OCIR)
    so = d[[f"W1_som_{i}" for i in range(1, 15)]].copy(); so.columns = list(PHQ15)
    it = d[[f"W1_PTSD{i}" for i in range(1, 7)]].copy(); it.columns = list(ITQ)
    return {"ocir": _complete(oc, list(OCIR), 0, 4),
            "phq15": _complete(so, list(PHQ15), 0, 2),
            "itq": _complete(it, list(ITQ), 0, 4)}


def load_edeq() -> pd.DataFrame:
    d = pd.read_excel(EXT / "edeq_contreras2022.xlsx", sheet_name="Data")
    src = [f"edeq_i{i:02d}" for i in list(range(1, 13)) + list(range(19, 29))]
    x = d[src].copy(); x.columns = SPEC["edeq"]["items"]
    return _complete(x, SPEC["edeq"]["items"], 0, 6)


def load_pcl5_clinical() -> pd.DataFrame:
    import pyreadstat
    d, _ = pyreadstat.read_sav(EXT / "pcl5_orovou2021.bin")
    x = d[[f"PCL{i}" for i in range(1, 21)]].copy(); x.columns = list(PCL5_LABELS)
    return _complete(x, list(PCL5_LABELS), 0, 4)


def load_coach_phq9() -> pd.DataFrame:
    from .config import PHQ9_ITEMS
    d = pd.read_excel(EXT / "coach_raw.xlsx")
    x = d[[f"PCP_Baseline_PHQ9_Q{i}" for i in range(1, 10)]].copy(); x.columns = PHQ9_ITEMS
    return _complete(x, PHQ9_ITEMS, 0, 3)


def build_human() -> dict[str, pd.DataFrame]:
    out = {"k6": load_k6(), **load_c19prc(), "edeq": load_edeq(),
           "pcl5c": load_pcl5_clinical(), "phq9_coach": load_coach_phq9()}
    for k, v in out.items():
        v.to_csv(DATA_PROC / f"extra_{k}.csv", index=False)
    return out


def human_X(instr: str) -> np.ndarray:
    items = (SPEC[instr]["items"] if instr in SPEC else
             __import__("mental_act.config", fromlist=["PHQ9_ITEMS"]).PHQ9_ITEMS)
    return pd.read_csv(DATA_PROC / f"extra_{instr}.csv")[items].to_numpy(int)
