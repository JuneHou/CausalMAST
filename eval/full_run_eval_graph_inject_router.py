"""
eval/full_run_eval_graph_inject_router.py

OpenRouter wrapper for the MAST two-pass graph-inject eval
(eval/full_run_eval_graph_inject_api.py).

This script intentionally does NOT modify the original runner. It
monkey-patches only the underlying LLM call so the existing two-pass
graph logic, prompts, parsing, and output layout remain unchanged.

Typical usage (run from MAST/):
  export OPENROUTER_API_KEY=...
  python eval/full_run_eval_graph_inject_router.py \
    --model "openrouter/google/gemini-2.5-flash" \
    --corr_threshold 0.5 \
    --output_dir outputs_thres/t0.5
"""

import argparse
import math
import os
import re
import sys
import time
from typing import Dict, Optional

import litellm
from litellm import RateLimitError, completion

import full_run_eval_graph_inject_api as base


def _build_router_call(
    *,
    api_base: str,
    api_key: str,
    request_interval_sec: float,
    max_retries: int,
    extra_headers: Dict[str, str],
    default_max_completion_tokens: int,
    pass1_reasoning_effort: str,
    max_prompt_tokens: int,
):
    def _safe_token_count(model: str, text: str) -> int:
        messages = [{"role": "user", "content": text}]
        try:
            return int(litellm.token_counter(model=model, messages=messages))
        except Exception:  # noqa: BLE001
            if model.startswith("openrouter/"):
                try:
                    bare = model.split("/", 1)[1]
                    return int(litellm.token_counter(model=bare, messages=messages))
                except Exception:  # noqa: BLE001
                    pass
            return max(1, math.ceil(len(text) / 4))

    def _trim_text_to_limit(model: str, text: str, limit: int) -> str:
        if _safe_token_count(model, text) <= limit:
            return text
        lo, hi = 0, len(text)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = text[:mid]
            if _safe_token_count(model, cand) <= limit:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _truncate_prompt_if_needed(prompt: str, model: str) -> str:
        if max_prompt_tokens <= 0:
            return prompt
        original_tokens = _safe_token_count(model, prompt)
        if original_tokens <= max_prompt_tokens:
            return prompt

        markers = [
            "Here is the trace:\n",
            "The data to analyze:\n\n",
            "The trace to analyze:\n\n",
        ]
        marker_pos = -1
        marker_len = 0
        for m in markers:
            pos = prompt.rfind(m)
            if pos > marker_pos:
                marker_pos = pos
                marker_len = len(m)

        if marker_pos >= 0:
            prefix = prompt[: marker_pos + marker_len]
            body = prompt[marker_pos + marker_len:]
            prefix_tokens = _safe_token_count(model, prefix)
            if prefix_tokens >= max_prompt_tokens:
                trimmed = _trim_text_to_limit(model, prefix, max_prompt_tokens)
                print(
                    f"  [router-truncate] prompt header alone exceeded limit: "
                    f"{original_tokens} -> {_safe_token_count(model, trimmed)} tokens"
                )
                return trimmed
            remaining = max_prompt_tokens - prefix_tokens
            body_trim = _trim_text_to_limit(model, body, remaining)
            truncated = prefix + body_trim
        else:
            truncated = _trim_text_to_limit(model, prompt, max_prompt_tokens)

        final_tokens = _safe_token_count(model, truncated)
        print(
            f"  [router-truncate] prompt truncated: "
            f"{original_tokens} -> {final_tokens} (limit={max_prompt_tokens})"
        )
        return truncated

    def _router_call_llm(prompt: str, model: str, max_tokens: int = 4000) -> str:
        prompt = _truncate_prompt_if_needed(prompt, model)
        messages = [{"role": "user", "content": prompt}]
        is_reasoning = any(x in model for x in ("o1", "o3", "o4", "anthropic", "gemini-2.5"))

        params = {
            "messages": messages,
            "model": model,
            "drop_params": True,
            "api_base": api_base,
            "api_key": api_key,
        }
        if not is_reasoning:
            params["temperature"] = 0.0
            params["top_p"] = 1

        # Reasoning control via OpenRouter's normalized `reasoning` block.
        # See run_eval_yesno_router.py for the full schema notes.
        extra_body: Dict[str, object] = {}
        if pass1_reasoning_effort == "none":
            extra_body["reasoning"] = {"enabled": False, "exclude": True}
        elif is_reasoning and pass1_reasoning_effort in {"low", "medium", "high"}:
            extra_body["reasoning"] = {"effort": pass1_reasoning_effort, "exclude": True}
        if extra_body:
            params["extra_body"] = extra_body

        if extra_headers:
            params["extra_headers"] = extra_headers
        # Honor caller's max_tokens but cap at default to keep OpenRouter
        # credit checks predictable.
        effective_max = max(max_tokens, default_max_completion_tokens)
        params["max_completion_tokens"] = effective_max

        for attempt in range(max_retries):
            try:
                response = completion(**params)
                if request_interval_sec > 0:
                    time.sleep(request_interval_sec)
                content = response.choices[0].message["content"]
                if content is None:
                    finish_reason = response.choices[0].finish_reason
                    raise RuntimeError(
                        f"Model returned content=None (finish_reason={finish_reason!r}). "
                        f"Check model/provider compatibility for '{model}'."
                    )
                return content
            except RateLimitError:
                wait_sec = min(120, 5 * (2 ** attempt))
                print(
                    f"  Rate limit (attempt {attempt+1}/{max_retries}): "
                    f"sleeping {wait_sec}s..."
                )
                time.sleep(wait_sec)
            except Exception as exc:  # noqa: BLE001
                err = str(exc).lower()
                is_rl = ("429" in err) or ("rate limit" in err)
                m = re.search(
                    r"requested up to\s+(\d+)\s+tokens,\s+but can only afford\s+(\d+)",
                    err,
                )
                if m:
                    requested = int(m.group(1))
                    affordable = int(m.group(2))
                    new_cap = max(256, min(params["max_completion_tokens"], affordable - 32))
                    if new_cap < params["max_completion_tokens"]:
                        print(
                            "  [router-token-cap] OpenRouter affordability limit hit: "
                            f"requested={requested}, affordable={affordable}. "
                            f"Retrying with max_completion_tokens={new_cap}."
                        )
                        params["max_completion_tokens"] = new_cap
                        continue
                if is_rl and attempt < max_retries - 1:
                    wait_sec = min(120, 5 * (2 ** attempt))
                    print(
                        f"  Provider rate issue (attempt {attempt+1}/{max_retries}): "
                        f"sleeping {wait_sec}s..."
                    )
                    time.sleep(wait_sec)
                    continue
                raise
        raise RuntimeError(f"Exceeded {max_retries} retries for model {model}")

    return _router_call_llm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenRouter wrapper for full_run_eval_graph_inject_api.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--openrouter_api_key_env",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Environment variable holding the OpenRouter API key.",
    )
    parser.add_argument(
        "--openrouter_base_url",
        type=str,
        default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        help="OpenRouter base URL.",
    )
    parser.add_argument(
        "--http_referer",
        type=str,
        default=os.getenv("OR_SITE_URL", ""),
        help="Optional HTTP-Referer header for OpenRouter analytics.",
    )
    parser.add_argument(
        "--x_title",
        type=str,
        default=os.getenv("OR_APP_NAME", "mast-benchmark"),
        help="Optional X-Title header for OpenRouter analytics.",
    )
    parser.add_argument(
        "--request_interval_sec",
        type=float,
        default=1.0,
        help="Delay between requests to avoid provider throttling.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Retry budget for rate-limit/transient failures.",
    )
    parser.add_argument(
        "--default_max_completion_tokens",
        type=int,
        default=15000,
        help=(
            "Lower bound on max_completion_tokens for OpenRouter calls "
            "(raises the base runner's default 4000 to keep enough headroom)."
        ),
    )
    parser.add_argument(
        "--pass1_reasoning_effort",
        type=str,
        choices=["none", "low", "medium", "high"],
        default="low",
        help="Reasoning effort for Pass-1 on reasoning models.",
    )
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=130000,
        help="Hard cap for prompt tokens before sending.",
    )
    # Router consumes only its own flags; everything else is forwarded
    # verbatim to full_run_eval_graph_inject_api.py via sys.argv.
    args, forwarded = parser.parse_known_args()

    api_key = os.getenv(args.openrouter_api_key_env) or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenRouter API key missing. Set OPENROUTER_API_KEY (or override "
            "--openrouter_api_key_env)."
        )

    headers: Dict[str, str] = {}
    if args.http_referer:
        headers["HTTP-Referer"] = args.http_referer
    if args.x_title:
        headers["X-Title"] = args.x_title

    base.call_llm = _build_router_call(
        api_base=args.openrouter_base_url,
        api_key=api_key,
        request_interval_sec=args.request_interval_sec,
        max_retries=args.max_retries,
        extra_headers=headers,
        default_max_completion_tokens=args.default_max_completion_tokens,
        pass1_reasoning_effort=args.pass1_reasoning_effort,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    model_idx = None
    for i, tok in enumerate(forwarded):
        if tok == "--model" and i + 1 < len(forwarded):
            model_idx = i + 1
            break
    if model_idx is not None:
        model_val = forwarded[model_idx]
        if not re.search(r"^(openrouter/|google/|gemini/)", model_val):
            print(
                "[warn] Model name may be invalid for OpenRouter. "
                "Expected something like 'openrouter/google/gemini-2.5-flash'."
            )

    litellm.drop_params = True
    sys.argv = [sys.argv[0], *forwarded]
    base.main()


if __name__ == "__main__":
    main()
