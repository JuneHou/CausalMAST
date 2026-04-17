"""
Pass 1 and Pass 2 prompts for MAST step-level labeling.
Uses short failure-type definitions only (no evidence or long examples).
"""
from typing import List, Optional

# Short definitions for ground-truth failure types only; no evidence or examples.
FAILURE_DEFS = {
    "1.1": "Disobey Task Specification: violates stated constraints/requirements; produces incorrect/irrelevant/constraint-breaking output.",
    "1.2": "Disobey Role Specification: violates responsibilities/constraints of assigned role; behaves like another role.",
    "1.3": "Step Repetition: unnecessarily repeats an already-completed phase/step due to poor state/context tracking.",
    "1.4": "Loss of Conversation History: forgets/disregards recent context; reverts to earlier conversational state.",
    "1.5": "Unaware of Termination Conditions: fails to monitor stopping criteria; continues aimlessly or ends without meeting stop conditions.",
    "2.1": "Conversation Reset: unexpectedly restarts dialogue, losing context/progress.",
    "2.2": "Fail to Ask for Clarification: does not request missing info when inputs are unclear/incomplete.",
    "2.3": "Task Derailment: deviates from intended objective; becomes irrelevant/unproductive.",
    "2.4": "Information Withholding: has critical info but fails to share it with other agents/components that need it.",
    "2.5": "Ignored Other Agent's Input: does not properly consider other agents' suggestions; causes bad decisions/stalls.",
    "2.6": "Action-Reasoning Mismatch: stated reasoning/conclusion does not match actual action/output produced.",
    "3.1": "Premature Termination: ends too early before required steps/verification/deliverables are complete.",
    "3.2": "Weak Verification: verification exists but is superficial/incomplete; misses essential checks.",
    "3.3": "No or Incorrect Verification: verification is omitted or done incorrectly; errors propagate undetected.",
}


def _definitions_for_types(type_ids: List[str]) -> str:
    """
    Return short definitions for the given failure type IDs only (from FAILURE_DEFS).
    No evidence or long examples.
    """
    if not type_ids:
        return ""
    parts = [f"{tid}: {FAILURE_DEFS[tid]}" for tid in type_ids if tid in FAILURE_DEFS]
    return "\n\n".join(parts) if parts else ""


def build_pass1_prompt(
    definitions_text: str,
    failure_type_ids: List[str],
    step_content: str,
    step_index: int,
    local_window_prev: bool = False,
    max_step_chars: Optional[int] = None,
) -> str:
    """
    Pass 1: one call per step. Ask which of the failure types in F appear in this step,
    with verbatim evidence from this step only. Uses only the current step (no previous step).
    """
    types_list = ", ".join(failure_type_ids)
    definitions_only = _definitions_for_types(failure_type_ids)
    if max_step_chars is not None and len(step_content) > max_step_chars:
        step_content = step_content[:max_step_chars] + "\n\n[Step content truncated for context length.]"
    return f"""You are a classifier for multi-agent failure types (MAST taxonomy).

## MAST failure taxonomy (definitions for types in this trace only)
{definitions_only}

## Task
For the following step from a multi-agent trace, determine which of the failure types **from this list ONLY** appear in this step: [{types_list}].
Step content is shown below.
**Evidence rule:** For each type you select, you must provide a **exact substring from the step content below**, otherwise do NOT select that type.

## Step index: {step_index}

## Step content
------------
{step_content}
------------

## Output format (STRICT)
Output exactly 1–3 lines.

1. Types present: <comma-separated type IDs from [{types_list}], or "none">
2. Evidence for <ID>: <exact verbatim quote from step content> (only if any types present)

Example (types present):
Types present: 2.2
Evidence for 2.2: "<exact quote from step content>"

Your response:"""


def build_pass2_adjudication_prompt(
    definitions_text: str,
    failure_type_id: str,
    candidate_steps: List[str],
    step_indices: List[int],
) -> str:
    """
    Pass 2 adjudication: when multiple steps were flagged for type A, ask which is the first occurrence.
    """
    steps_blob = ""
    for i, (idx, content) in enumerate(zip(step_indices, candidate_steps)):
        excerpt = content[:1500] + "..." if len(content) > 1500 else content
        steps_blob += f"\n--- Step index {idx} ---\n{excerpt}\n"
    definition_for_type = _definitions_for_types([failure_type_id])
    return f"""You are determining the **earliest** step at which failure type **{failure_type_id}** first appears.

## MAST definition for {failure_type_id}
{definition_for_type}

## Candidate steps (in order)
These steps were flagged as possibly containing failure type {failure_type_id}. Which one is the **first** (earliest) occurrence?
{steps_blob}

## Output format (STRICT)
Do not include reasoning or explanation. Output exactly one line:
First step index: <number>

Your response:"""
