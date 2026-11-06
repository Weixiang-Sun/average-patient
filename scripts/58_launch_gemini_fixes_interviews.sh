#!/usr/bin/env bash
# Interventions and interviews for the three Gemini 2.5 models, same design as
# the four original patient models. Resumable.
cd "$(dirname "$0")/.."
L=outputs/runs/gemini_logs; mkdir -p "$L"
for m in gemini-2.5-flash-lite gemini-2.5-flash gemini-2.5-pro; do
  uv run --quiet python scripts/33_run_fixes.py gemini $m > "$L/fixes_$m.log" 2>&1 &
  for c in free oracle; do
    uv run --quiet python scripts/35_interview.py --model $m --cond $c --n 20 --workers 12 > "$L/interview_${c}_$m.log" 2>&1 &
  done
done
wait
tail -n 1 "$L"/fixes_*.log "$L"/interview_*.log
