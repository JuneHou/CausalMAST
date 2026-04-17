"""
Pass 1 (single scan over steps) and Pass 2 (per-type onset + adjudication).
Evidence must be verbatim substring of step t.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from . import config
from .load_definitions import load_definitions
from .prompts import build_pass1_prompt, build_pass2_adjudication_prompt
from .vllm_client import complete as vllm_complete

# Max prompt size for Pass 1 (truncate step content if exceeded)
MAX_PASS1_PROMPT_TOKENS = 32768
CHARS_PER_TOKEN_ESTIMATE = 4


def _parse_pass1_response(response: str, step_content: str, failure_type_ids: List[str]) -> List[Tuple[str, str]]:
    """
    Parse Pass 1 model output. Return list of (type_id, evidence) for which evidence is substring of step_content.
    Uses only the label block (from "Types present" onward); strips <think> and other leading text.
    """
    # Strip <think>...</think> so we only parse the label block
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()
    # Keep only from "Types present" onward
    types_present_start = re.search(r"types?\s*present\s*[:\[]", response, re.IGNORECASE)
    if types_present_start:
        response = response[types_present_start.start() :]
    if not response:
        return []

    results = []
    response_lower = response.lower()
    # "Types present: 1.2, 2.1" or "Types present: none"
    types_match = re.search(r"types?\s*present\s*[:\[]\s*([^\n\]]+)", response_lower, re.IGNORECASE)
    if not types_match:
        return results
    raw = types_match.group(1).strip()
    if "none" in raw and len(raw) < 20:
        return results
    # Evidence for 1.2: "quote"
    evidence_pattern = re.compile(r"evidence\s+for\s+([\d.]+\s*)[:\"]\s*[\"']?([^\n\"]+)[\"']?", re.IGNORECASE)
    for m in evidence_pattern.finditer(response):
        tid = m.group(1).strip().strip(".")
        if tid not in failure_type_ids:
            continue
        evidence = m.group(2).strip().strip("'\"").strip()
        if not evidence:
            continue
        # Verbatim check: evidence must be substring of step_content
        if evidence in step_content:
            results.append((tid, evidence))
    # Fallback: collect type IDs from "Types present: 1.2, 2.1" and look for "Evidence for 1.2: ..."
    if not results and types_match:
        for tid in failure_type_ids:
            if tid not in raw:
                continue
            pat = re.compile(rf"evidence\s+for\s+{re.escape(tid)}\s*[:\"]\s*[\"']?([^\n\"]+)[\"']?", re.IGNORECASE)
            for m in pat.finditer(response):
                ev = m.group(1).strip().strip("'\"").strip()
                if ev and ev in step_content:
                    results.append((tid, ev))
                    break
    return results


def _evidence_is_verbatim(evidence: str, step_content: str) -> bool:
    return evidence.strip() in step_content


def _format_model_input(prompt: str) -> str:
    """Exact string sent to the model (same as vllm_client.complete_offline)."""
    return f"User: {prompt}\n\nAssistant:"


def _debug_log_pass1_step(
    debug_log_path: Path,
    trace_id: str,
    step_index: int,
    n_steps: int,
    prompt: str,
    response: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Write one JSONL line per Pass 1 step: exactly what the model sees (model_input) and returns (model_response)."""
    model_input = _format_model_input(prompt)
    payload = {
        "trace_id": trace_id,
        "step_index": step_index,
        "n_steps": n_steps,
        "model_input": model_input,
        "model_response": response if response is not None else None,
    }
    if error is not None:
        payload["error"] = error
    with open(debug_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_pass1(
    steps: List[str],
    failure_type_ids: List[str],
    definitions_text: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    local_window_prev: bool = True,
    use_offline: bool = False,
    llm: Optional[Any] = None,
    debug_log_path: Optional[Path] = None,
    trace_id: str = "",
) -> Dict[int, List[Tuple[str, str]]]:
    """
    Pass 1: one LLM call per step. Returns step_index -> [(type_id, evidence), ...] for valid evidence only.
    """
    step_flags = {}
    n_steps = len(steps)
    for t, step_content in enumerate(steps):
        if not step_content.strip():
            step_flags[t] = []
            continue
        if (t + 1) % 5 == 0 or t == 0 or t == n_steps - 1:
            print(f"  Pass 1 step {t + 1}/{n_steps} ...", flush=True)
        # Use only the current step (no previous step) to control length and focus
        window = step_content
        prompt = build_pass1_prompt(
            definitions_text, failure_type_ids, window, t, local_window_prev=False
        )
        max_prompt_chars = 30000 * CHARS_PER_TOKEN_ESTIMATE
        if len(prompt) > max_prompt_chars:
            max_step_chars = max_prompt_chars - (len(prompt) - len(window))
            if max_step_chars > 0:
                prompt = build_pass1_prompt(
                    definitions_text, failure_type_ids, window, t,
                    local_window_prev=False, max_step_chars=max_step_chars,
                )
            # else keep prompt as-is (frame already too large; vLLM may still error)
        response = None
        try:
            response = vllm_complete(
                prompt, model=model, base_url=base_url, api_key=api_key,
                use_offline=use_offline, llm=llm,
            )
        except Exception as e:
            if debug_log_path is not None:
                _debug_log_pass1_step(
                    debug_log_path, trace_id, t, n_steps, prompt, response=None, error=str(e)
                )
            raise
        if debug_log_path is not None:
            _debug_log_pass1_step(
                debug_log_path, trace_id, t, n_steps, prompt, response=response, error=None
            )
        parsed = _parse_pass1_response(response, step_content, failure_type_ids)
        step_flags[t] = parsed
    return step_flags


def run_pass2(
    step_flags: Dict[int, List[Tuple[str, str]]],
    steps: List[str],
    failure_type_ids: List[str],
    definitions_text: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    adjudicate_when_multiple: bool = True,
    use_offline: bool = False,
    llm: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Pass 2: for each type A in failure_type_ids, set T_A = earliest step with valid evidence.
    If multiple candidates and adjudicate_when_multiple, run one adjudication call.
    Returns { type_id: { first_step, evidence, confidence } }.
    """
    failure_type_to_first_step = {}
    for i, A in enumerate(failure_type_ids):
        if (i + 1) % 3 == 0 or i == 0 or i == len(failure_type_ids) - 1:
            print(f"  Pass 2 type {A} ({i + 1}/{len(failure_type_ids)}) ...", flush=True)
        candidates = []
        for t in sorted(step_flags.keys()):
            for tid, ev in step_flags[t]:
                if tid == A and _evidence_is_verbatim(ev, steps[t]):
                    candidates.append((t, ev))
                    break
        if not candidates:
            failure_type_to_first_step[A] = {"first_step": None, "evidence": None, "confidence": None}
            continue
        # Earliest
        first_t, first_ev = candidates[0]
        if len(candidates) == 1 or not adjudicate_when_multiple:
            failure_type_to_first_step[A] = {
                "first_step": first_t,
                "evidence": first_ev,
                "confidence": "high" if len(candidates) == 1 else "medium",
            }
            continue
        # Adjudication: pick first occurrence among first few candidates
        candidate_steps = [steps[t] for t, _ in candidates[:5]]
        candidate_indices = [t for t, _ in candidates[:5]]
        adj_prompt = build_pass2_adjudication_prompt(definitions_text, A, candidate_steps, candidate_indices)
        adj_response = vllm_complete(
            adj_prompt, model=model, base_url=base_url, api_key=api_key,
            use_offline=use_offline, llm=llm,
        )
        idx_match = re.search(r"first\s+step\s+index\s*[:\s]*(\d+)", adj_response, re.IGNORECASE)
        if idx_match:
            chosen = int(idx_match.group(1))
            if chosen in candidate_indices:
                pos = candidate_indices.index(chosen)
                first_t, first_ev = candidates[pos]
        failure_type_to_first_step[A] = {
            "first_step": first_t,
            "evidence": first_ev,
            "confidence": "medium",
        }
    return failure_type_to_first_step


def run_labeling_for_trace(
    steps: List[str],
    failure_type_ids: List[str],
    definitions_text: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    adjudicate: bool = True,
    use_offline: bool = False,
    llm: Optional[Any] = None,
    debug_log_path: Optional[Path] = None,
    trace_id: str = "",
) -> Dict[str, Dict[str, Any]]:
    """
    Full pipeline for one trace: Pass 1 then Pass 2. Returns failure_type_to_first_step.
    """
    if not definitions_text:
        definitions_text = load_definitions()
    if not steps or not failure_type_ids:
        return {}
    step_flags = run_pass1(
        steps, failure_type_ids, definitions_text,
        model=model, base_url=base_url, api_key=api_key,
        use_offline=use_offline, llm=llm,
        debug_log_path=debug_log_path,
        trace_id=trace_id,
    )
    return run_pass2(
        step_flags, steps, failure_type_ids, definitions_text,
        model=model, base_url=base_url, api_key=api_key,
        adjudicate_when_multiple=adjudicate,
        use_offline=use_offline, llm=llm,
    )
