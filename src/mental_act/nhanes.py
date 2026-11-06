"""Download and assemble NHANES PHQ-9 (DPQ) + demographics (DEMO).

NHANES public data carries no access restrictions ("Access Constraints: None"),
so this runs without any application or licence.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging

import pandas as pd
import requests

from .config import DATA_NHANES, DATA_PROC, DPQ_MISSING, PHQ9_ITEMS, SEVERITY_BANDS

log = logging.getLogger(__name__)

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{dir}/DataFiles/{stem}.xpt"

# cycle -> (url directory, file stem suffix, calendar years)
# NOTE: cycle "P" (2017-Mar 2020 pre-pandemic) OVERLAPS cycle "J" (2017-2018).
#       Including both double-counts respondents, so P is excluded by default.
CYCLES: dict[str, tuple[str, str, str]] = {
    "D": ("2005", "_D", "2005-2006"),
    "E": ("2007", "_E", "2007-2008"),
    "F": ("2009", "_F", "2009-2010"),
    "G": ("2011", "_G", "2011-2012"),
    "H": ("2013", "_H", "2013-2014"),
    "I": ("2015", "_I", "2015-2016"),
    "J": ("2017", "_J", "2017-2018"),
    "L": ("2021", "_L", "2021-2023"),
}
OVERLAPPING = {"P": ("2017", "P_", "2017-Mar2020 pre-pandemic")}

DEMO_VARS = [
    "SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "RIDRETH3",
    "DMDEDUC2", "INDFMPIR", "SDMVPSU", "SDMVSTRA", "WTINT2YR", "WTMEC2YR",
]


def _url(kind: str, cycle: str) -> str:
    if cycle in CYCLES:
        d, suf, _ = CYCLES[cycle]
        stem = f"{kind}{suf}"
    else:
        d, pre, _ = OVERLAPPING[cycle]
        stem = f"{pre}{kind}"
    return BASE.format(dir=d, stem=stem)


def fetch(kind: str, cycle: str, force: bool = False) -> pd.DataFrame:
    """Download one NHANES .xpt (cached on disk) and return it as a DataFrame."""
    path = DATA_NHANES / f"{kind}_{cycle}.xpt"
    if force or not path.exists():
        url = _url(kind, cycle)
        log.info("GET %s", url)
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        if len(r.content) < 1000:
            raise RuntimeError(f"suspiciously small payload for {url}: {len(r.content)}B")
        path.write_bytes(r.content)
    df = pd.read_sas(io.BytesIO(path.read_bytes()), format="xport")
    df.columns = [c.upper() for c in df.columns]
    return _fix_xport_zeros(df)


def _fix_xport_zeros(df: pd.DataFrame, tol: float = 1e-30) -> pd.DataFrame:
    """pandas' XPORT reader decodes SAS numeric 0 as a denormal (~5.4e-79).

    Left alone this silently corrupts every downstream count, so snap any
    finite value with |x| < tol to exactly 0.0.
    """
    num = df.select_dtypes("number").columns
    for c in num:
        col = df[c]
        df[c] = col.mask(col.abs() < tol, 0.0)
    return df


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_phq9(cycles: list[str] | None = None, adults_only: bool = True,
              force: bool = False) -> pd.DataFrame:
    """Pooled, cleaned, item-level PHQ-9 with demographics.

    Returns one row per respondent with columns:
      SEQN, cycle, years, DPQ010..DPQ090, phq9_total, severity, n_missing,
      age, sex, race, plus survey-design columns.
    Respondents with ANY missing/refused/don't-know item are dropped, because
    every downstream index (person-fit, Guttman errors, symptom combinations)
    requires a complete response vector.
    """
    cycles = cycles or list(CYCLES)
    frames = []
    for cyc in cycles:
        dpq = fetch("DPQ", cyc, force=force)
        demo = fetch("DEMO", cyc, force=force)
        keep = [c for c in DEMO_VARS if c in demo.columns]
        df = dpq.merge(demo[keep], on="SEQN", how="left")
        df["cycle"] = cyc
        df["years"] = CYCLES[cyc][2] if cyc in CYCLES else OVERLAPPING[cyc][2]
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    n_raw = len(raw)

    # ---- clean the 9 scored items -------------------------------------
    items = raw[PHQ9_ITEMS].copy()
    items = items.mask(items.isin(DPQ_MISSING))          # 7=Refused, 9=Don't know
    items = items.mask(~items.isin([0, 1, 2, 3]))        # anything else -> NaN
    raw[PHQ9_ITEMS] = items
    raw["n_missing"] = items.isna().sum(axis=1)

    drop_rate = (raw["n_missing"] > 0).mean()
    if drop_rate > 0.35:
        raise RuntimeError(
            f"complete-case filter would drop {drop_rate:.1%} of respondents; "
            "expected <35%. Suspect a value-decoding problem, not real missingness."
        )

    df = raw[raw["n_missing"] == 0].copy()
    df[PHQ9_ITEMS] = df[PHQ9_ITEMS].astype(int)
    df["phq9_total"] = df[PHQ9_ITEMS].sum(axis=1)

    def band(t):
        for lo, hi, name in SEVERITY_BANDS:
            if lo <= t <= hi:
                return name
        return "unknown"

    df["severity"] = df["phq9_total"].map(band)
    df["age"] = df.get("RIDAGEYR")
    df["sex"] = df.get("RIAGENDR", pd.Series(dtype=float)).map({1.0: "male", 2.0: "female"})

    if adults_only and "age" in df:
        df = df[df["age"] >= 18]

    df = df.reset_index(drop=True)
    log.info("PHQ-9: %d raw -> %d complete-case%s", n_raw, len(df),
             " adults" if adults_only else "")
    return df


def build(cycles: list[str] | None = None, force: bool = False) -> pd.DataFrame:
    """Build and persist the analysis-ready human reference sample."""
    df = load_phq9(cycles=cycles, force=force)
    out = DATA_PROC / "nhanes_phq9.parquet"
    try:
        df.to_parquet(out, index=False)
    except Exception:                                    # no pyarrow -> csv
        out = DATA_PROC / "nhanes_phq9.csv"
        df.to_csv(out, index=False)

    manifest = {
        "cycles": cycles or list(CYCLES),
        "n_respondents": int(len(df)),
        "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                  for p in sorted(DATA_NHANES.glob("*.xpt"))},
        "note": "cycle P (2017-Mar2020) excluded by default: overlaps cycle J",
    }
    (DATA_PROC / "nhanes_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote %s (%d rows)", out, len(df))
    return df
