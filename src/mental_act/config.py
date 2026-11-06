"""Paths and shared constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_NHANES = DATA_RAW / "nhanes"
DATA_PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
RUNS = OUTPUTS / "runs"

for _p in (DATA_NHANES, DATA_PROC, TABLES, FIGURES, RUNS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- PHQ-9
PHQ9_ITEMS = [f"DPQ{n:03d}" for n in range(10, 100, 10)]  # DPQ010..DPQ090

PHQ9_LABELS = {
    "DPQ010": "little interest or pleasure in doing things",
    "DPQ020": "feeling down, depressed, or hopeless",
    "DPQ030": "trouble sleeping or sleeping too much",
    "DPQ040": "feeling tired or having little energy",
    "DPQ050": "poor appetite or overeating",
    "DPQ060": "feeling bad about yourself, or that you are a failure",
    "DPQ070": "trouble concentrating on things",
    "DPQ080": "moving or speaking slowly, or being fidgety/restless",
    "DPQ090": "thoughts that you would be better off dead or of hurting yourself",
}

# DSM-5 symptom construct each PHQ-9 item maps to (used for probe design)
PHQ9_CONSTRUCT = {
    "DPQ010": "anhedonia",
    "DPQ020": "depressed_mood",
    "DPQ030": "sleep",
    "DPQ040": "fatigue",
    "DPQ050": "appetite",
    "DPQ060": "worthlessness",
    "DPQ070": "concentration",
    "DPQ080": "psychomotor",
    "DPQ090": "suicidality",
}

PHQ9_RESPONSE_LABELS = {
    0: "Not at all",
    1: "Several days",
    2: "More than half the days",
    3: "Nearly every day",
}

# NHANES missing-data codes for the DPQ module
DPQ_MISSING = {7, 9, 77, 99}

# PHQ-9 severity bands (Kroenke et al. 2001)
SEVERITY_BANDS = [
    (0, 4, "none_minimal"),
    (5, 9, "mild"),
    (10, 14, "moderate"),
    (15, 19, "moderately_severe"),
    (20, 27, "severe"),
]

# ---------------------------------------------------------------- GAD-7
# NHIS Sample Adult carries item-level GAD-7 and PHQ-8 for the same
# respondents, which gives a second instrument and an independent survey.
GAD7_ITEMS = [f"GAD7{n}_A" for n in range(1, 8)]
PHQ8_ITEMS = [f"PHQ8{n}_A" for n in range(1, 9)]

GAD7_LABELS = {
    "GAD71_A": "feeling nervous, anxious, or on edge",
    "GAD72_A": "not being able to stop or control worrying",
    "GAD73_A": "worrying too much about different things",
    "GAD74_A": "trouble relaxing",
    "GAD75_A": "being so restless that it is hard to sit still",
    "GAD76_A": "becoming easily annoyed or irritable",
    "GAD77_A": "feeling afraid, as if something awful might happen",
}
PHQ8_LABELS = {
    "PHQ81_A": "little interest or pleasure in doing things",
    "PHQ82_A": "feeling down, depressed, or hopeless",
    "PHQ83_A": "trouble falling or staying asleep, or sleeping too much",
    "PHQ84_A": "feeling tired or having little energy",
    "PHQ85_A": "poor appetite or overeating",
    "PHQ86_A": "feeling bad about yourself, or that you are a failure",
    "PHQ87_A": "trouble concentrating on things",
    "PHQ88_A": "moving or speaking slowly, or being fidgety/restless",
}

# GAD-7 severity bands (Spitzer et al. 2006)
GAD7_BANDS = [(0, 4, "none_minimal"), (5, 9, "mild"),
              (10, 14, "moderate"), (15, 21, "severe")]
# PHQ-8 bands (PHQ-9 bands minus the suicidality item; same cut points)
PHQ8_BANDS = [(0, 4, "none_minimal"), (5, 9, "mild"), (10, 14, "moderate"),
              (15, 19, "moderately_severe"), (20, 24, "severe")]
