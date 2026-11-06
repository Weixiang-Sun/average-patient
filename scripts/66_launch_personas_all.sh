#!/usr/bin/env bash
# Published personas on all 17 models, PHQ-9 and GAD-7 (TalkDep, CACTUS, CACTUS with its
# case formulation). The PHQ-9 TalkDep/CACTUS runs of the seven intervention models already
# exist and are skipped by the resumable runner. Reasoning off as in the main runs. GPT-4o-mini,
# GPT-5.2 and GPT-5.4-mini use their Azure deployments here (the OpenAI API key was revoked).
cd "$(dirname "$0")/.."
L=outputs/runs/persona_logs; mkdir -p "$L"
run() {   # provider model
  t=$(echo "$2" | tr -d '.-' | tr 'A-Z' 'a-z')
  ( uv run --quiet python scripts/46_run_published_personas.py "$1" "$2" talkdep cactus cactus_cm
    uv run --quiet python scripts/46_run_published_personas.py "$1" "$2" --gad7 talkdep cactus cactus_cm ) > "$L/all_$t.log" 2>&1 &
}
run azure gpt-4o-mini; run azure gpt-4o; run azure gpt-4.1-nano; run azure gpt-4.1-mini; run azure gpt-4.1
run azure gpt-5-mini; run azure gpt-5.2; run azure gpt-5.4-mini; run anthropic claude-haiku-4-5
run azure grok-4-1-fast-non-reasoning; run azure Llama-3.3-70B-Instruct; run azure Kimi-K2.5
run azure Mistral-Large-3; run azure Phi-4
run gemini gemini-2.5-flash-lite; run gemini gemini-2.5-flash; run gemini gemini-2.5-pro
wait
grep -h "ok," "$L"/all_*.log
