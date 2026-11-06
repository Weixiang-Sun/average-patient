#!/usr/bin/env bash
# Interviews with GRM-sampled profiles as private background, for the five
# models that also ran the real-profile condition.
cd "$(dirname "$0")/.."
L=outputs/runs/persona_logs; mkdir -p "$L"
for m in gpt-4o-mini Llama-3.3-70B-Instruct gemini-2.5-flash-lite gemini-2.5-flash gemini-2.5-pro; do
  uv run --quiet python scripts/35_interview.py --model "$m" --cond grm --n 20 --workers 10 > "$L/grm_interview_$m.log" 2>&1 &
done
wait
tail -qn 1 "$L"/grm_interview_*.log
