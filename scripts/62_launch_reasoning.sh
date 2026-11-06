#!/usr/bin/env bash
# Reasoning-effort runs (review W6) for every experiment GPT-5-mini (Azure) and
# Gemini 2.5 Flash took part in. The API default of each model is the main run.
cd "$(dirname "$0")/.."
L=outputs/runs/reasoning_logs; mkdir -p "$L"
Q="phq9 gad7 pcl5 x_k6 x_ocir x_phq15 x_itq x_edeq x_pcl5c"
for e in minimal low high; do
  uv run --quiet python scripts/61_run_reasoning.py azure gpt-5-mini $e $Q > "$L/gpt5mini_$e.log" 2>&1 &
done
for e in none low high; do
  uv run --quiet python scripts/61_run_reasoning.py gemini gemini-2.5-flash $e $Q fixes persona_talkdep persona_cactus > "$L/gemini25flash_$e.log" 2>&1 &
  ( for c in free oracle grm; do
      uv run --quiet python scripts/35_interview.py --model gemini-2.5-flash --cond $c --n 20 --workers 6 --effort $e
    done ) > "$L/interview_gemini25flash_$e.log" 2>&1 &
done
wait
cat "$L"/*.log | grep -h "ok" | sort
