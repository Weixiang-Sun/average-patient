"""Step 1: download NHANES DPQ (item-level PHQ-9) + DEMO, build the human reference sample."""
import logging, sys
import pandas as pd
sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from mental_act import nhanes
from mental_act.config import PHQ9_ITEMS

df = nhanes.build()

print("\n=== NHANES pooled PHQ-9 reference sample ===")
print(f"respondents (complete-case adults): {len(df):,}")
print("\nby cycle:")
print(df.groupby(["cycle", "years"]).size().to_string())
print("\nPHQ-9 total: mean=%.2f  sd=%.2f  median=%d  max=%d"
      % (df.phq9_total.mean(), df.phq9_total.std(), df.phq9_total.median(), df.phq9_total.max()))
print("\nseverity bands:")
sv = df.severity.value_counts(normalize=True).reindex(
    ["none_minimal", "mild", "moderate", "moderately_severe", "severe"])
for k, v in sv.items():
    print(f"  {k:20s} {v*100:5.2f}%") if pd.notna(v) else None
print(f"\nPHQ-9 >= 10 (moderate+): {(df.phq9_total>=10).mean()*100:.2f}%")
