#!/usr/bin/env bash
# All instruments for the three Gemini 2.5 models (Google AI Studio), two
# processes per model: main instruments + PCL-5 roles, and the six extension
# instruments. Resumable: rerunning retries only failed trials.
cd "$(dirname "$0")/.."
L=outputs/runs/gemini_logs; mkdir -p "$L"
for j in GEMFL GEMF GEMPRO; do uv run --quiet python scripts/21_batch.py $j > "$L/main_$j.log" 2>&1 & done
for m in gemini-2.5-flash-lite gemini-2.5-flash gemini-2.5-pro; do
  uv run --quiet python scripts/50_run_extra.py gemini $m --workers 12 > "$L/extra_$m.log" 2>&1 &
done
wait
grep -h "DONE\|FAIL\|ok trials" "$L"/*.log
