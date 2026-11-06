#!/usr/bin/env bash
# Interviews with the 12 TalkDep personas (seven per persona, 84 per model) for the seven
# patient models of the severity-persona interviews, same interviewer, raters and settings
# (reasoning off: Gemini 2.5 Flash at none, Gemini 2.5 Pro at its 128-token minimum; the
# eff__ runs are copied to the main names by scripts/65). Resumable.
cd "$(dirname "$0")/.."
L=outputs/runs/talkdep_interview_logs; mkdir -p "$L"
I="uv run --quiet python scripts/35_interview.py --cond talkdep --n 7 --workers 8"
for m in gpt-4o-mini gpt-4.1 claude-haiku-4-5 Llama-3.3-70B-Instruct gemini-2.5-flash-lite; do
  $I --model $m > "$L/$m.log" 2>&1 &
done
$I --model gemini-2.5-flash --effort none > "$L/gemini-2.5-flash.log" 2>&1 &
$I --model gemini-2.5-pro --effort budget128 > "$L/gemini-2.5-pro.log" 2>&1 &
wait
tail -n 1 "$L"/*.log
