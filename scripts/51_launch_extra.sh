#!/usr/bin/env bash
# Launch the extension instruments for all 13 Azure-available models in parallel
# (one process per model; each deployment has its own rate limit), then wait.
cd "$(dirname "$0")/.."
L=outputs/runs/extra_logs; mkdir -p "$L"
run() { uv run --quiet python scripts/50_run_extra.py "$@" > "$L/$(echo "$2" | tr -d '.-' | tr A-Z a-z).log" 2>&1 & }
run azure gpt-4o-mini
run azure gpt-4o
run azure gpt-4.1-nano
run azure gpt-4.1-mini
run azure gpt-4.1
run azure gpt-5-mini
run azure gpt-5.2
run azure grok-4-1-fast-non-reasoning
run azure Llama-3.3-70B-Instruct
run azure Kimi-K2.5 --workers 16
run azure Mistral-Large-3
run azure Phi-4
run anthropic claude-haiku-4-5
wait
grep -h "x_\|DONE\|FAIL" "$L"/*.log
