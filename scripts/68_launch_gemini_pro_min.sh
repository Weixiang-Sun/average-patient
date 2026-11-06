#!/usr/bin/env bash
# Gemini 2.5 Pro cannot turn thinking off; rerun every main experiment it takes part in at its
# minimum thinking budget (128 tokens), as eff__<exp>__gemini25pro__budget128 runs. scripts/65
# then keeps the API-default runs as eff__<exp>__gemini25pro__default and swaps these in.
cd "$(dirname "$0")/.."
L=outputs/runs/reasoning_logs; mkdir -p "$L"
R="uv run --quiet python scripts/61_run_reasoning.py gemini gemini-2.5-pro budget128"
$R phq9 gad7 pcl5 x_k6 x_ocir > "$L/gpro_min_a.log" 2>&1 &
$R x_phq15 x_itq x_edeq x_pcl5c fixes > "$L/gpro_min_b.log" 2>&1 &
$R persona_talkdep persona_cactus persona_cactus_cm persona_talkdep_gad7 persona_cactus_gad7 persona_cactus_cm_gad7 > "$L/gpro_min_c.log" 2>&1 &
( for c in free oracle grm; do
    uv run --quiet python scripts/35_interview.py --model gemini-2.5-pro --cond $c --n 20 --workers 8 --effort budget128
  done ) > "$L/gpro_min_interviews.log" 2>&1 &
wait
grep -h "ok" "$L"/gpro_min_*.log
