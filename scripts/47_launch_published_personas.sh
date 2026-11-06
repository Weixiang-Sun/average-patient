#!/usr/bin/env bash
# Published-persona PHQ-9 runs (TalkDep, CACTUS) for the seven patient models.
cd "$(dirname "$0")/.."
L=outputs/runs/persona_logs; mkdir -p "$L"
run() { uv run --quiet python scripts/46_run_published_personas.py "$1" "$2" > "$L/$2.log" 2>&1 & }
run azure gpt-4o-mini
run azure gpt-4.1
run anthropic claude-haiku-4-5
run azure Llama-3.3-70B-Instruct
run gemini gemini-2.5-flash-lite
run gemini gemini-2.5-flash
run gemini gemini-2.5-pro
wait
grep -h "ok," "$L"/*.log
