"""Administer the PHQ-9 to an LLM and record item-level responses.

Two providers behind one interface:
  openai    -- json_schema structured output
  anthropic -- forced tool use (Anthropic has no json_schema response_format)
Both constrain the model to emit exactly n integers; the administration is
otherwise identical. Refusals are recorded as data, not treated as failures.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import PHQ9_ITEMS, RUNS

load_dotenv()


def _schema(n: int) -> dict:
    """One required integer field per item.

    Used for BOTH providers. An array with minItems/maxItems would be the
    natural encoding, but Anthropic's json_schema output format rejects
    minItems > 1, so a fixed-arity object is the only shape that enforces
    exact length identically on both sides.
    """
    props = {f"a{i+1}": {"type": "integer"} for i in range(n)}
    return {"type": "object", "additionalProperties": False,
            "properties": props, "required": list(props)}


def _unpack(obj: dict, n: int) -> list[int]:
    return [obj[f"a{i+1}"] for i in range(n)]


def client(provider: str = "openai"):
    if provider == "openai":
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAI(api_key=key)
    if provider == "azure":
        # Azure AI Foundry exposes third-party models behind an
        # OpenAI-compatible surface (xAI, Meta, Moonshot, ...).
        from openai import OpenAI
        key = os.environ.get("AZURE_ANTHROPIC_KEY")
        ep = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not (key and ep):
            raise RuntimeError("AZURE_ANTHROPIC_KEY / AZURE_OPENAI_ENDPOINT not set")
        return OpenAI(base_url=ep, api_key=key)
    if provider == "gemini":
        # Google AI Studio, OpenAI-compatible surface (json_schema supported;
        # token logprobs are not enabled for any Gemini model on this API)
        from openai import OpenAI
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        return OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                      api_key=key, timeout=180)
    if provider == "anthropic":
        from anthropic import AnthropicFoundry
        key = os.environ.get("AZURE_ANTHROPIC_KEY")
        ep = os.environ.get("AZURE_ANTHROPIC_ENDPOINT")
        if not (key and ep):
            raise RuntimeError("AZURE_ANTHROPIC_KEY / _ENDPOINT not set")
        return AnthropicFoundry(api_key=key, base_url=ep)
    raise ValueError(provider)


def _retryable(e: BaseException) -> bool:
    # a content-filter block is deterministic for a given prompt: retrying wastes time
    return "content_filter" not in str(e)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30),
       retry=retry_if_exception(_retryable))
def gemini_kw(reasoning_effort, max_tokens):
    """Request arguments for Gemini 2.5 on its OpenAI endpoint. reasoning_effort None: the
    model's default (dynamic thinking); "budget<N>": a thinking budget of N tokens, set through
    the endpoint's Google extension (Gemini 2.5 Pro cannot turn thinking off; 128 is its
    minimum); otherwise an OpenAI reasoning_effort, with a budget large enough for "high"."""
    if not reasoning_effort:
        return {"max_tokens": max_tokens}
    if reasoning_effort.startswith("budget"):
        return {"max_tokens": max_tokens, "extra_body": {"extra_body": {"google": {"thinking_config": {
            "thinking_budget": int(reasoning_effort[len("budget"):])}}}}}
    return {"max_tokens": 32000 if reasoning_effort == "high" else 8000, "reasoning_effort": reasoning_effort}


def _ask(cl, provider, model, system, user, temperature, n_items, reasoning_effort=None):
    """Return (answers, usage_dict). Raises on refusal or malformed output.
    reasoning_effort (optional): passed to reasoning models (GPT-5 series on Azure,
    Gemini 2.5 via its OpenAI endpoint) with a budget large enough for "high"."""
    if provider == "azure":
        # Reasoning models on this endpoint (e.g. Kimi) emit a long
        # `reasoning_content` before the answer and silently truncate to an
        # empty content field if the budget is too small.
        m = model.lower()
        if m.startswith(("gpt-5", "o1", "o3", "o4")):
            kw = {"max_completion_tokens": 4000}
            if reasoning_effort:
                kw = {"max_completion_tokens": 32000 if reasoning_effort == "high" else 8000, "reasoning_effort": reasoning_effort}
        elif any(k in m for k in ("kimi", "reason", "think")):
            kw = {"max_tokens": 4000, "temperature": temperature}
            if reasoning_effort:   # Kimi-K2.5 on Azure: "none" turns thinking off
                kw["reasoning_effort"] = reasoning_effort
        else:
            kw = {"max_tokens": 600, "temperature": temperature}
        # Some Azure deployments (Phi-4) reject json_schema with a 422; they get
        # JSON mode instead, and arity/range are validated after parsing.
        if m.startswith("phi-"):
            rf = {"type": "json_object"}
        else:
            rf = {"type": "json_schema", "json_schema": {
                "name": "phq", "strict": True, "schema": _schema(n_items)}}
        r = cl.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format=rf, **kw)
        c = r.choices[0]
        if not (c.message.content or "").strip():
            raise RuntimeError(f"empty content (finish={c.finish_reason})")
        return _unpack(json.loads(c.message.content), n_items), (
            r.usage.model_dump() if r.usage else None)

    if provider == "gemini":
        # Gemini 2.5 Flash and Pro think by default and the thinking tokens
        # count against the output budget, so they get the reasoning budget.
        r = cl.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "phq", "strict": True, "schema": _schema(n_items)}},
            temperature=temperature,
            **gemini_kw(reasoning_effort, 4000))
        c = r.choices[0]
        if c.finish_reason == "content_filter":
            raise RuntimeError("content_filter: response blocked by Gemini safety filter")
        if not (c.message.content or "").strip():
            raise RuntimeError(f"empty content (finish={c.finish_reason})")
        return _unpack(json.loads(c.message.content), n_items), (
            r.usage.model_dump() if r.usage else None)

    if provider == "openai":
        # gpt-5.x rejects `temperature` and `max_tokens`, like the Anthropic
        # arm. Those models therefore run at the API default and cannot be
        # temperature-swept; gpt-4o-mini's temperature-invariance is what
        # licenses comparing them against it at t=1.0.
        if model.startswith(("gpt-5", "o1", "o3", "o4")):
            kw = {"max_completion_tokens": 2000}
        else:
            kw = {"temperature": temperature, "max_tokens": 200}
        r = cl.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "phq", "strict": True, "schema": _schema(n_items)}},
            **kw)
        ans = _unpack(json.loads(r.choices[0].message.content), n_items)
        return ans, (r.usage.model_dump() if r.usage else None)

    # NOTE: anthropic>=1.3 removed `temperature` from messages.create, so the
    # Anthropic arm runs at the API default and cannot be swept. GPT-4o-mini
    # was shown to be temperature-invariant here, which licenses comparing it
    # at t=1.0 against Anthropic's default.
    r = cl.messages.create(
        model=model, max_tokens=400, system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema",
                                  "schema": _schema(n_items)}})
    txt = "".join(getattr(b, "text", "") for b in r.content)
    if not txt.strip():
        raise RuntimeError(f"empty response (stop={r.stop_reason})")
    return _unpack(json.loads(txt), n_items), {
        "input_tokens": r.usage.input_tokens,
        "output_tokens": r.usage.output_tokens}


def _one(cl, provider, model, trial, temperature, reasoning_effort=None):
    rec = {"trial_id": trial["trial_id"], "arm": trial["arm"],
           "severity": trial["severity"], "rep": trial["rep"],
           "provider": provider, "model": model, "temperature": temperature,
           "mode": trial.get("mode", "block"),
           "item_order": trial["item_order"], "ts": time.time(),
           **({"reasoning_effort": reasoning_effort} if reasoning_effort else {})}
    try:
        t0 = time.time()
        n_items = len(trial["item_order"])
        if rec["mode"] == "block":
            ans, usage = _ask(cl, provider, model, trial["system"],
                              trial["user"], temperature, n_items, reasoning_effort)
        else:
            ans, tok = [], 0
            for sp in trial["subprompts"]:
                a, u = _ask(cl, provider, model, trial["system"], sp,
                            temperature, 1)
                ans.append(a[0])
                tok += sum(v for v in (u or {}).values() if isinstance(v, int))
            usage = {"total_tokens": tok}
        rec["latency_s"] = round(time.time() - t0, 3)
        rec["usage"] = usage
        top = trial.get("max_cat", 3)
        if len(ans) != n_items or not all(isinstance(a, int) and 0 <= a <= top for a in ans):
            raise ValueError(f"bad answer vector: {ans}")
        rec["responses"] = {it: int(v) for it, v in zip(trial["item_order"], ans)}
        rec["ordered"] = [rec["responses"][i] for i in sorted(rec["responses"])]
        rec["instrument"] = trial.get("instrument", "phq9")
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def run(trials, model="gpt-4o-mini", provider="openai", temperature=1.0,
        out_name="run", limit=None, resume=True, workers=12, reasoning_effort=None):
    cl = client(provider)
    path = Path(RUNS) / f"{out_name}.jsonl"
    done = set()
    if resume and path.exists():
        with path.open() as f:
            done = {r["trial_id"] for r in map(json.loads, filter(str.strip, f))
                    if r.get("status") == "ok"}
    todo = [t for t in trials if t["trial_id"] not in done]
    if limit:
        todo = todo[:limit]

    lock, n = threading.Lock(), [0]
    with path.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        for rec in ex.map(lambda t: _one(cl, provider, model, t, temperature, reasoning_effort), todo):
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
                n[0] += 1
                if n[0] % 50 == 0:
                    print(f"  {n[0]}/{len(todo)}")
    return path
