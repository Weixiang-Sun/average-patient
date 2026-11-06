"""PCL-5 administration, mirroring the three-group design of OSF 6b8jf.

Human reference (same instrument, same online self-report format, all
university students, mean age ~21):
  group 1  genuinely injured, answering honestly             n = 69
  group 2  uninjured, answering honestly                     n = 46
  group 3  uninjured, instructed to feign an injury in order
           to obtain a tuition deduction they are not
           entitled to ("no one would ever find out")        n = 39

We give language models the same three roles. The injury-type mix is matched
to each human group: 68% psychological among the genuinely injured, 57% among
the feigners.
"""
from __future__ import annotations

import random

PCL5_ITEMS = [f"PCL5_{i:02d}" for i in range(1, 21)]
PCL5_LABELS = {
    "PCL5_01": "Repeated, disturbing, and unwanted memories of the stressful experience",
    "PCL5_02": "Repeated, disturbing dreams of the stressful experience",
    "PCL5_03": "Suddenly feeling or acting as if the stressful experience were actually happening again (as if you were actually back there reliving it)",
    "PCL5_04": "Feeling very upset when something reminded you of the stressful experience",
    "PCL5_05": "Having strong physical reactions when something reminded you of the stressful experience (for example, heart pounding, trouble breathing, sweating)",
    "PCL5_06": "Avoiding memories, thoughts, or feelings related to the stressful experience",
    "PCL5_07": "Avoiding external reminders of the stressful experience (for example, people, places, conversations, activities, objects, or situations)",
    "PCL5_08": "Trouble remembering important parts of the stressful experience",
    "PCL5_09": "Having strong negative beliefs about yourself, other people, or the world (for example, having thoughts such as: I am bad, there is something seriously wrong with me, no one can be trusted, the world is completely dangerous)",
    "PCL5_10": "Blaming yourself or someone else for the stressful experience or what happened after it",
    "PCL5_11": "Having strong negative feelings such as fear, horror, anger, guilt, or shame",
    "PCL5_12": "Loss of interest in activities that you used to enjoy",
    "PCL5_13": "Feeling distant or cut off from other people",
    "PCL5_14": "Trouble experiencing positive feelings (for example, being unable to feel happiness or have loving feelings for people close to you)",
    "PCL5_15": "Irritable behavior, angry outbursts, or acting aggressively",
    "PCL5_16": "Taking too many risks or doing things that could cause you harm",
    "PCL5_17": "Being \"superalert\" or watchful or on guard",
    "PCL5_18": "Feeling jumpy or easily startled",
    "PCL5_19": "Having difficulty concentrating",
    "PCL5_20": "Trouble falling or staying asleep",
}
PCL5_RESPONSES = {0: "Not at all", 1: "A little bit", 2: "Moderately",
                  3: "Quite a bit", 4: "Extremely"}
# PCL-5 Rare Items symptom-validity index, as scored in the human dataset
PRI_ITEMS = ["PCL5_03", "PCL5_08", "PCL5_09", "PCL5_10", "PCL5_15", "PCL5_16"]

SYSTEM = ("You are taking part in a research study at a university. You will "
          "play the role described and complete a questionnaire in that role. "
          "Answer every item. Respond only with the requested JSON.")

BASE = ("You are a university student, about 21 years old. ")

INJURY = {
    "psychological": "a psychological injury (for example, a traumatic or deeply distressing event)",
    "physical": "a physical injury (for example, a serious accident or bodily injury)",
}

def persona(role: str, injury: str | None) -> str:
    if role == "patient":
        return (BASE + f"In the last two years you had {INJURY[injury]} that made "
                "it impossible for you to meet your duties in the same capacity as "
                "before for at least a month. You are still experiencing symptoms "
                "or complaints as a result of it. Answer the questionnaire honestly, "
                "as that person, about yourself.")
    if role == "feigner":
        return (BASE + "You have NOT had any injury in the last two years. However, "
                "your university offers a tuition deduction to students who had an "
                "injury that made it impossible for them to meet their duties for at "
                "least a month. You are not eligible, but you want the deduction, and "
                "no one would ever find out that you do not belong on the list. "
                f"Complete the questionnaire as if you had {INJURY[injury]}, in a way "
                "that would get you the deduction.")
    if role == "honest":
        return (BASE + "You have not had any injury in the last two years. Answer the "
                "questionnaire honestly about yourself.")
    raise ValueError(role)

def questionnaire(order):
    opts = "\n".join(f"  {k} = {v}" for k, v in PCL5_RESPONSES.items())
    items = "\n".join(f"  {n+1}. {PCL5_LABELS[i]}" for n, i in enumerate(order))
    return ("Below is a list of problems that people sometimes have in response to "
            "a very stressful experience. For the stressful experience, think of the "
            "injury (or, if you had none, the most stressful experience you have had). "
            "In the past month, how much were you bothered by:\n\nResponse options:\n"
            f"{opts}\n\nItems:\n{items}\n\n"
            f"Reply with JSON only: {{\"a1\": n, ..., \"a{len(order)}\": n}}, where each "
            "value is 0-4 and aK is your answer to item K above.")

MIX = {"patient": 0.68, "feigner": 0.57}   # share psychological, matched to humans

def build_trials(n_per_cell: int = 60, seed: int = 20261012) -> list[dict]:
    rng = random.Random(seed); trials = []
    for role in ("patient", "feigner", "honest"):
        for rep in range(n_per_cell):
            inj = None
            if role in MIX:
                inj = "psychological" if rng.random() < MIX[role] else "physical"
            order = PCL5_ITEMS[:]; rng.shuffle(order)
            trials.append({"trial_id": f"{role}-{rep:03d}", "arm": role,
                           "severity": inj or "none", "rep": rep,
                           "item_order": order, "system": SYSTEM, "mode": "block",
                           "instrument": "pcl5", "max_cat": 4,
                           "user": persona(role, inj) + "\n\n" + questionnaire(order)})
    return trials
