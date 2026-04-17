#!/usr/bin/env python3
"""
Steps 2-4: Patch generation for MAST-AG2 A-instances.

Adapted from TRAIL causal/patch/patch_generator.py.
Key changes vs TRAIL:
  - patch_library.json uses MAST 13 categories (1.1–3.3) instead of TRAIL operator families.
  - _run_postcheck(): removed TRAIL-specific category checks (Formatting Errors,
    Tool Selection Errors, Resource Abuse). Only universal checks remain:
    (1) patch_payload must differ from local_snippet, (2) patch_payload must be non-empty.
  - All other logic (LLM call, retry loop, PatchResult) is identical to TRAIL.

Input:  a_instances.jsonl  (one record per unique A-instance)
Output: patch_results.jsonl, postcheck_failures.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from patch_generator_llm import _call_llm


# ---------------------------------------------------------------------------
# Shared scaffold (identical to TRAIL — generic do(A=0) language)
# ---------------------------------------------------------------------------

PATCH_SYSTEM = """You are generating a localized intervention corresponding to do(A=0) for one annotated source error instance.

Goal:
Remove ONLY the annotated source error instance of type A by minimally replacing the exact labeled step content.
This is a causal intervention for testing whether fixing A changes downstream error B.
You must not directly repair B.

Hard constraints:
1. Modify ONLY the provided LOCAL_SNIPPET.
2. Target ONLY the annotated source error type A.
3. Do NOT directly fix, mention, or optimize for the downstream error type B.
4. Preserve as much original meaning and content as possible.
5. Do NOT invent new facts, results, or tool outputs unless they appear verbatim in ERROR_DESCRIPTION, ERROR_EVIDENCE, USER_REQUIREMENTS, or LOCAL_SNIPPET.
6. Make the smallest possible edit that eliminates error A.
7. Return ONLY the JSON object in the required schema.

Required output schema:
{
  "source_error_type": "string",
  "downstream_error_type": "string",
  "patch_side": "replace_step_content",
  "slot_values": { "key": "value or null" },
  "patch_payload": "string",
  "postcheck": {
    "passed": true,
    "checks": ["string"],
    "notes": "string"
  }
}"""


PATCH_USER_TEMPLATE = """\
ERROR_TYPE_SPEC:
<<<
{ERROR_TYPE_SPEC}
>>>

SOURCE_ERROR_TYPE: {A}
DOWNSTREAM_ERROR_TYPES (do NOT directly fix any of these): {B_LIST}
ERROR_DESCRIPTION:
{ERROR_DESCRIPTION}
ERROR_EVIDENCE:
{ERROR_EVIDENCE}

LOCAL_SNIPPET (exact step content to replace):
<<<
{LOCAL_SNIPPET}
>>>

USER_REQUIREMENTS:
<<<
{USER_REQUIREMENTS}
>>>

Task:
Generate a minimal patch that implements do(A=0): remove the source error instance only.\
"""


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class PatchResult:
    trace_id: str
    error_id: str
    location: str
    patch_side: str
    template_used: str        # A category key from patch_library
    slot_values: dict
    patch_payload: str
    postcheck_passed: bool
    postcheck_failures: list
    attempts: int
    patch_reason: str
    llm_postcheck: dict


# ---------------------------------------------------------------------------
# Postcheck (universal only — no TRAIL-specific category rules)
# ---------------------------------------------------------------------------

def _run_postcheck(
    patch_payload: str,
    local_snippet: str,
) -> tuple:
    """
    Universal rule-based postcheck.
    Returns (passed, list_of_failure_messages).

    TRAIL had category-specific checks (Formatting Errors, Tool Selection Errors,
    Resource Abuse). For MAST those do not apply, so only the two universal rules
    are enforced here.
    """
    failures = []

    if not patch_payload.strip():
        failures.append("patch_payload is empty.")

    if patch_payload.strip() == local_snippet.strip():
        failures.append("patch_payload is identical to local_snippet — no change made.")

    return (len(failures) == 0), failures


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_patch(
    case: dict,
    patch_library: dict,
    model: str = "openai/gpt-4o",
    max_retries: int = 3,
) -> PatchResult:
    """Generate a patch for one AInstanceRecord dict. Returns PatchResult."""
    a_cat = case["a_instance"]["category"]
    b_list = ", ".join(case.get("b_types") or []) or "unknown"
    lib_entry = patch_library.get(a_cat, {})
    error_type_spec = lib_entry.get("error_type_spec_text", f"error_type: {a_cat}")

    user_msg = PATCH_USER_TEMPLATE.format(
        ERROR_TYPE_SPEC=error_type_spec,
        A=a_cat,
        B_LIST=b_list,
        ERROR_DESCRIPTION=(case["a_instance"].get("description") or "")[:1200],
        ERROR_EVIDENCE=(case["a_instance"].get("evidence") or "")[:1200],
        LOCAL_SNIPPET=(case["local_snippet"] or "")[:4000],
        USER_REQUIREMENTS=(case.get("user_requirements") or "")[:1500],
    )

    patch_payload = ""
    slot_values: dict = {}
    postcheck_passed = False
    postcheck_failures: list = []
    patch_reason = ""
    llm_postcheck: dict = {}
    attempts = 0

    for attempt in range(1, max_retries + 1):
        attempts = attempt
        retry_note = ""
        if attempt > 1 and postcheck_failures:
            retry_note = (
                f"\n\nPREVIOUS ATTEMPT FAILED POSTCHECK:\n"
                + "\n".join(f"- {f}" for f in postcheck_failures)
                + "\nPlease fix these issues in your new response."
            )

        try:
            raw = _call_llm(
                PATCH_SYSTEM,
                user_msg + retry_note,
                model=model,
                max_tokens=2048,
            )
        except Exception as e:
            postcheck_failures = [f"LLM call failed: {e}"]
            continue

        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)
            raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    postcheck_failures = ["LLM response is not valid JSON."]
                    continue
            else:
                postcheck_failures = ["LLM response contains no JSON object."]
                continue

        if not isinstance(parsed, dict):
            postcheck_failures = ["LLM JSON is not a dict."]
            continue

        patch_payload = (parsed.get("patch_payload") or "").strip()
        slot_values = parsed.get("slot_values") or {}
        llm_postcheck = parsed.get("postcheck") or {}
        patch_reason = llm_postcheck.get("notes") or ""

        postcheck_passed, postcheck_failures = _run_postcheck(
            patch_payload, case["local_snippet"],
        )
        if postcheck_passed:
            break

    return PatchResult(
        trace_id=case["trace_id"],
        error_id=case["a_instance"].get("error_id", ""),
        location=case.get("intervention_location") or case["a_instance"].get("location", ""),
        patch_side=case["patch_side"],
        template_used=a_cat,
        slot_values=slot_values,
        patch_payload=patch_payload,
        postcheck_passed=postcheck_passed,
        postcheck_failures=postcheck_failures,
        attempts=attempts,
        patch_reason=patch_reason,
        llm_postcheck=llm_postcheck,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate do(A=0) patches for all MAST-AG2 AInstanceRecords."
    )
    parser.add_argument("--cases", default="outputs/interventions/a_instances.jsonl")
    parser.add_argument("--patch_library", default="patch_library.json")
    parser.add_argument("--out_dir", default="outputs/interventions")
    parser.add_argument("--model", default="openai/gpt-4o")
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    with open(args.patch_library, "r", encoding="utf-8") as f:
        patch_library = json.load(f)

    cases = []
    with open(args.cases, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "patch_results.jsonl")
    failures_path = os.path.join(args.out_dir, "postcheck_failures.jsonl")

    n_ok = n_fail = 0
    with open(results_path, "w", encoding="utf-8") as rf, \
         open(failures_path, "w", encoding="utf-8") as ff:
        for case in cases:
            result = generate_patch(case, patch_library, model=args.model,
                                    max_retries=args.max_retries)
            rec = asdict(result)
            rf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if result.postcheck_passed:
                status = "OK"
                n_ok += 1
            else:
                err_detail = result.postcheck_failures[0][:120] if result.postcheck_failures else "unknown"
                status = f"FAIL: {err_detail}"
                ff.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_fail += 1
            print(f"  [{status}] {result.trace_id} {result.error_id[-24:]} attempts={result.attempts}")

    print(f"\nWrote {results_path}. OK={n_ok} FAIL={n_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
