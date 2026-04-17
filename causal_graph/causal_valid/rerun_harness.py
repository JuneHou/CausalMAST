#!/usr/bin/env python3
"""
Step 5: Counterfactual rerun harness for MAST-AG2.

Adapted from TRAIL causal/patch/rerun_harness.py.
Key changes vs TRAIL:
  - No trail_io / OpenInference spans. Reads from annotation_ag2_filtered.jsonl.
  - Step roles parsed from [agent_name] prefix in step content.
  - Rerun strategy: apply patch at step_A, then simulate subsequent ASSISTANT turns
    with an LLM; non-assistant (orchestrator) turns reuse original trace content.
    This maintains the AG2 back-and-forth structure while testing the counterfactual.
  - No smolagents → OpenAI role conversion needed.
  - No tool result queue (MAST steps contain plain text, not live tool calls).
  - rerun_status: "live_rerun_success" | "rerun_missing_suffix"

Input:  patch_results.jsonl  (one per unique A-instance)
Output: rerun_results.jsonl  (one per unique A-instance)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

try:
    from litellm import completion, RateLimitError
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    RateLimitError = Exception


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIVE_RERUN_SUCCESS = "live_rerun_success"
RERUN_MISSING_SUFFIX = "rerun_missing_suffix"

# Agents that are treated as the "assistant" (LLM-driven) in AG2 traces
ASSISTANT_AGENTS = {"assistant", "ai", "agent", "solver", "coder"}


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class RerunResult:
    trace_id: str
    error_id: str
    a_location: str           # step_XX of intervention point t_A
    patch_side: str
    patch_payload: str
    rerun_status: str         # LIVE_RERUN_SUCCESS | RERUN_MISSING_SUFFIX
    rerun_success: bool
    rerun_error: str
    original_suffix_spans: list  # step contents from t_A+1 onward in original trace
    rerun_suffix_spans: list     # simulated contents from t_A+1 onward


# ---------------------------------------------------------------------------
# Step parsing helpers
# ---------------------------------------------------------------------------

def _parse_step_content(content: str) -> Tuple[str, str]:
    """
    Parse '[agent_name]\\ncontent...' → (agent_name_lower, content_text).
    Falls back to ("assistant", content) if no role prefix found.
    """
    lines = content.strip().split("\n", 1)
    first = lines[0].strip()
    if first.startswith("[") and first.endswith("]"):
        agent = first[1:-1].strip().lower()
        text = lines[1].strip() if len(lines) > 1 else ""
        return agent, text
    return "assistant", content.strip()


def _is_assistant_role(agent_name: str) -> bool:
    """Return True if this agent is the LLM-driven assistant (not the orchestrator)."""
    return agent_name.lower() in ASSISTANT_AGENTS


def _step_to_openai_message(step: dict) -> dict:
    """Convert a MAST step to an OpenAI-format message dict."""
    agent, text = _parse_step_content(step["content"])
    role = "assistant" if _is_assistant_role(agent) else "user"
    return {"role": role, "content": text}


def _step_num(step_id: str) -> int:
    try:
        return int(step_id.split("_")[1])
    except (IndexError, ValueError):
        return -1


# ---------------------------------------------------------------------------
# LLM continuation call
# ---------------------------------------------------------------------------

def _call_llm_messages(
    model: str,
    messages: List[dict],
    system: str = "",
    max_tokens: int = 1024,
) -> str:
    """Call litellm with a full message list. Returns response content string."""
    if not LITELLM_AVAILABLE:
        raise RuntimeError("litellm not installed. Run: pip install litellm")

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    params: dict = {
        "model": model,
        "messages": full_messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "drop_params": True,
    }
    if any(x in model for x in ("o1", "o3", "o4")):
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


# ---------------------------------------------------------------------------
# Trace loader
# ---------------------------------------------------------------------------

def _load_trace(input_jsonl: str, trace_id: str) -> Optional[dict]:
    """Load a single trace record from the JSONL by trace_id."""
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if str(rec.get("trace_id")) == str(trace_id):
                return rec
    return None


# ---------------------------------------------------------------------------
# Real rerun
# ---------------------------------------------------------------------------

def _real_rerun(
    patch_result: dict,
    trace_record: dict,
    model: str,
    max_steps_after: int,
) -> RerunResult:
    """
    Apply patch at step_A then simulate agent continuation for up to max_steps_after steps.

    Strategy:
      - All steps before step_A are kept as-is.
      - step_A content is replaced with patch_payload.
      - For each subsequent step in the original trace (up to max_steps_after):
          * If it is an "assistant" step → call LLM to simulate the counterfactual response.
          * If it is an "orchestrator" step → use the original content (code results, feedback).
      This preserves the AG2 back-and-forth structure while testing the counterfactual.
    """
    a_location = patch_result.get("location", "")
    patch_payload = patch_result.get("patch_payload", "")
    steps = trace_record.get("steps", [])

    a_idx = next((i for i, s in enumerate(steps) if s["id"] == a_location), -1)
    if a_idx == -1:
        raise ValueError(f"Step {a_location} not found in trace {patch_result['trace_id']}")

    # Task description for system prompt context
    task_desc = steps[0]["content"][:800] if steps else ""

    system_prompt = (
        "You are an AI assistant in a multi-agent system (AG2/AutoGen). "
        "Continue the conversation given the message history. "
        "Respond only as the assistant agent, producing the next natural step. "
        f"Task context: {task_desc}"
    )

    # Build conversation history up to (but not including) step_A
    history: List[dict] = []
    for step in steps[:a_idx]:
        history.append(_step_to_openai_message(step))

    # Add patched step_A as assistant message
    history.append({"role": "assistant", "content": patch_payload})

    # Original suffix (for baseline)
    original_suffix = [s["content"] for s in steps[a_idx + 1:]]

    # Simulate continuation
    new_spans: List[str] = []
    sim_history = list(history)
    suffix_steps = steps[a_idx + 1:]

    for i, next_step in enumerate(suffix_steps[:max_steps_after]):
        agent, orig_text = _parse_step_content(next_step["content"])

        if _is_assistant_role(agent):
            # Simulate this assistant turn with LLM
            sim_response = _call_llm_messages(
                model, sim_history, system=system_prompt, max_tokens=1024
            )
            new_spans.append(f"[{agent}]\n{sim_response}")
            sim_history.append({"role": "assistant", "content": sim_response})
        else:
            # Use original orchestrator content (tool results, code output, feedback)
            new_spans.append(next_step["content"])
            sim_history.append({"role": "user", "content": orig_text})

        # Early stop: if simulated text indicates task completion
        combined = new_spans[-1].lower()
        if any(t in combined for t in ["terminate", "exitcode: 0", "task is complete", "final answer"]):
            break

    return RerunResult(
        trace_id=patch_result["trace_id"],
        error_id=patch_result.get("error_id", ""),
        a_location=a_location,
        patch_side=patch_result.get("patch_side", "replace_step_content"),
        patch_payload=patch_payload,
        rerun_status=LIVE_RERUN_SUCCESS,
        rerun_success=True,
        rerun_error="",
        original_suffix_spans=original_suffix,
        rerun_suffix_spans=new_spans,
    )


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def run_rerun(
    patch_result: dict,
    input_jsonl: str,
    model: str = "openai/gpt-4o",
    max_steps_after: int = 8,
) -> RerunResult:
    """
    Load the trace and run the counterfactual rerun.
    Returns a RerunResult (rerun_success=False on any failure).
    """
    trace_id = patch_result["trace_id"]
    a_location = patch_result.get("location", "")

    trace_record = _load_trace(input_jsonl, trace_id)
    if trace_record is None:
        return RerunResult(
            trace_id=trace_id,
            error_id=patch_result.get("error_id", ""),
            a_location=a_location,
            patch_side=patch_result.get("patch_side", ""),
            patch_payload=patch_result.get("patch_payload", ""),
            rerun_status=RERUN_MISSING_SUFFIX,
            rerun_success=False,
            rerun_error=f"trace {trace_id} not found in {input_jsonl}",
            original_suffix_spans=[],
            rerun_suffix_spans=[],
        )

    try:
        return _real_rerun(patch_result, trace_record, model=model,
                           max_steps_after=max_steps_after)
    except Exception as e:
        # Extract original suffix even on failure for partial baseline reference
        steps = trace_record.get("steps", [])
        a_idx = next((i for i, s in enumerate(steps) if s["id"] == a_location), -1)
        original_suffix = [s["content"] for s in steps[a_idx + 1:]] if a_idx >= 0 else []
        return RerunResult(
            trace_id=trace_id,
            error_id=patch_result.get("error_id", ""),
            a_location=a_location,
            patch_side=patch_result.get("patch_side", ""),
            patch_payload=patch_result.get("patch_payload", ""),
            rerun_status=RERUN_MISSING_SUFFIX,
            rerun_success=False,
            rerun_error=str(e),
            original_suffix_spans=original_suffix,
            rerun_suffix_spans=[],
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply patches and simulate MAST-AG2 agent continuation."
    )
    parser.add_argument("--patch_results",
                        default="outputs/interventions/patch_results.jsonl")
    parser.add_argument("--input", default="../data/annotation/annotation_ag2_filtered.jsonl",
                        help="Path to annotation_ag2_filtered.jsonl")
    parser.add_argument("--out_dir", default="outputs/interventions")
    parser.add_argument("--model", default="openai/gpt-4o",
                        help="LLM for simulating assistant continuation")
    parser.add_argument("--max_steps_after", type=int, default=8,
                        help="Max steps to simulate after t_A (default: 8)")
    args = parser.parse_args()

    with open(args.patch_results, "r", encoding="utf-8") as f:
        patch_results = [json.loads(l) for l in f if l.strip()]

    to_rerun = [p for p in patch_results if p.get("postcheck_passed")]
    print(f"Rerunning {len(to_rerun)} / {len(patch_results)} patches (postcheck passed)")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "rerun_results.jsonl")

    from collections import Counter
    counts: Counter = Counter()
    with open(out_path, "w", encoding="utf-8") as f:
        for pr in to_rerun:
            result = run_rerun(
                pr, args.input,
                model=args.model, max_steps_after=args.max_steps_after,
            )
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            counts[result.rerun_status] += 1
            n_new = len(result.rerun_suffix_spans)
            print(f"  [{result.rerun_status}] {str(result.trace_id)[:8]} "
                  f"err={result.error_id[-24:]} new_spans={n_new}")

    print(f"\nWrote {out_path}.")
    for status, n in sorted(counts.items()):
        print(f"  {status}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
