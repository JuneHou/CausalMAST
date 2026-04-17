#!/usr/bin/env python3
"""
Minimal litellm wrapper — shared by patch_generator.py, judge_a_resolved.py,
judge_b_effect.py.

Adapted from TRAIL causal/patch/patch_generator_llm.py: stripped to the
_call_llm() function only (no trail_io dependencies).
"""
from __future__ import annotations

import time

try:
    from litellm import completion, RateLimitError
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    RateLimitError = Exception  # fallback type


def _call_llm(
    system: str,
    user_content: str,
    model: str = "openai/gpt-4o",
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> str:
    """Call litellm with a system + user message pair. Returns response string."""
    if not LITELLM_AVAILABLE:
        raise RuntimeError("litellm not installed. Run: pip install litellm")

    params: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "drop_params": True,
    }
    if any(x in model for x in ("o1", "o3", "o4", "anthropic", "gemini-2.5")):
        params["reasoning_effort"] = "medium"
        params.pop("temperature", None)

    def _do() -> str:
        resp = completion(**params)
        return (resp.choices[0].message.content or "").strip()

    try:
        return _do()
    except RateLimitError:
        time.sleep(30)
        return _do()
