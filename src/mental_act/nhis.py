"""NHIS Sample Adult: item-level GAD-7 and PHQ-8.

NHIS is a separate CDC survey from NHANES, public and unrestricted. It gives
(a) a second instrument (GAD-7, anxiety) and (b) an independent replication
sample for depression (PHQ-8), from the same respondents.

Response coding differs from NHANES: NHIS uses 1-4, NHANES DPQ uses 0-3.
Both are recoded here to 0-3.
"""
from __future__ import annotations

import io
import logging
import zipfile

import pandas as pd
import requests

from .config import DATA_PROC, DATA_RAW, GAD7_ITEMS, PHQ8_ITEMS

log = logging.getLogger(__name__)

URL = ("https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/"
       "{year}/adult{yy}csv.zip")
YEARS = [2019, 2021, 2022, 2023]
MISSING = {7, 8, 9, 97, 98, 99}      # Refused / Not ascertained / Don't know


def fetch(year: int, force: bool = False) -> pd.DataFrame:
    d = DATA_RAW / "nhis"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"adult{year}.zip"
    if force or not p.exists():
        url = URL.format(year=year, yy=str(year)[-2:])
        log.info("GET %s", url)
        last = None
        for attempt in range(3):                 # ftp.cdc.gov drops large reads
            try:
                with requests.get(url, timeout=(30, 120), stream=True) as r:
                    r.raise_for_status()
                    with p.open("wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                break
            except Exception as e:               # noqa: BLE001
                last = e
                p.unlink(missing_ok=True)
                log.warning("attempt %d failed for %s: %s", attempt + 1, url, e)
        else:
            raise RuntimeError(f"could not fetch {url}: {last}")
    with zipfile.ZipFile(io.BytesIO(p.read_bytes())) as z:
        name = max(z.namelist(), key=lambda n: z.getinfo(n).file_size)
        with z.open(name) as f:
            df = pd.read_csv(f, low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    return df


def load(items: list[str], years: list[int] | None = None,
         force: bool = False) -> pd.DataFrame:
    """Pooled, complete-case, 0-3 recoded responses for one instrument."""
    frames = []
    for y in (years or YEARS):
        df = fetch(y, force=force)
        miss = [c for c in items if c not in df.columns]
        if miss:
            log.warning("%d: missing %s -- skipping", y, miss)
            continue
        keep = ["AGEP_A", "SEX_A"] if "AGEP_A" in df.columns else []
        sub = df[items + keep].copy()
        sub["year"] = y
        frames.append(sub)

    raw = pd.concat(frames, ignore_index=True)
    n0 = len(raw)
    x = raw[items].mask(raw[items].isin(MISSING))
    x = x.mask(~x.isin([1, 2, 3, 4]))
    raw[items] = x - 1                       # NHIS 1-4 -> 0-3
    raw["n_missing"] = x.isna().sum(axis=1)

    drop = (raw["n_missing"] > 0).mean()
    if drop > 0.75:
        raise RuntimeError(f"filter would drop {drop:.1%}; suspect a coding problem")

    df = raw[raw["n_missing"] == 0].copy()
    df[items] = df[items].astype(int)
    df["total"] = df[items].sum(axis=1)
    log.info("%s: %d raw -> %d complete-case (%.1f%% retained)",
             items[0][:4], n0, len(df), 100 * len(df) / n0)
    return df.reset_index(drop=True)


def build(force: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    for tag, items in [("gad7", GAD7_ITEMS), ("phq8", PHQ8_ITEMS)]:
        df = load(items, force=force)
        df.to_csv(DATA_PROC / f"nhis_{tag}.csv", index=False)
        out[tag] = df
    return out
