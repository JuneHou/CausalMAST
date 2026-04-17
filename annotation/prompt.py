"""
Prompt builders for MAST step-level annotation — two-pass pipeline.

Pass 1  (onset localization)
    Ask only: which step is the earliest candidate onset for each error?
    Output: {"candidates": [{"category", "location", "short_reason"}, ...]}

Pass 2  (verification + full annotation)
    Given the candidate locations from Pass 1, verify and render:
    evidence, description, why_not_earlier, impact.
    Output: {"errors": [{"category", "location", "evidence", "description",
                          "why_not_earlier", "impact"}, ...]}

Variants:
    zero_shot  — instructions only, no examples
    few_shot   — same instructions + one in-context example (2.6 Action-Reasoning Mismatch)
"""
from typing import List, Dict

from .definitions import format_definitions_for_prompt, category_label


# ---------------------------------------------------------------------------
# Pass 1: onset localization instructions
# ---------------------------------------------------------------------------

_PASS1_INSTRUCTIONS = """\
You will be given:
1. An ordered multi-agent execution trace segmented into steps.
2. A list of error categories that are ALREADY KNOWN to appear somewhere in the trace.

Your only job in this pass is localization — NOT detection or explanation.
For each given error category, find the single earliest step where the error
first becomes observable from the trace text alone.

Rules:
- Work through the given error categoriesand steps in the trace one at a time.
- For each category output exactly one candidate entry, then move on.
- If two adjacent steps are both plausible, choose the earlier one.
- short_reason must be grounded in the trace (quote or close paraphrase).
- Do not invent content not present in the trace.
- Do not output any category that is not in the provided list.

Output format:
Return a JSON object with one field:
{
  "candidates": [
    {
      "category": "...",
      "location": "...",
      "short_reason": "..."
    }
  ]
}

Field requirements:
- category: exactly one of the provided categories (include numeric ID and name,
  e.g. "2.6 Action-Reasoning Mismatch")
- location: exactly one valid step ID from the trace (e.g. "step_03")
- short_reason: 1-2 sentences quoting or closely paraphrasing the trace text
  that makes the error first observable at this step

CRITICAL: You MUST produce exactly one entry for EVERY category in the provided
list. Skipping a category is not acceptable.

Before finalising, silently check:
1. Does the output contain one entry for EVERY provided category?
2. Does each category appear exactly once?
3. Is each location a valid step ID?
4. Is each chosen location the EARLIEST defensible one — not a later consequence?"""


# ---------------------------------------------------------------------------
# Pass 2: verification + full annotation instructions
# ---------------------------------------------------------------------------

_PASS2_INSTRUCTIONS = """\
You will be given:
1. An ordered multi-agent execution trace segmented into steps.
2. A list of candidate onset locations identified in Pass 1
   (one per error category, with a short reason).
3. Official taxonomy definitions for each error category.

Your job is to verify each candidate location and produce the full annotation.

Rules:
- For each candidate: review the trace and verify the proposed location.
  - If the candidate is correct, accept it and document it fully.
  - If you find an earlier step that is also defensible, use that instead
    and explain the correction in why_not_earlier.
- Output exactly one entry per error category — never two for the same category.
- Evidence must be a direct quote or close paraphrase from the trace.
- Description must connect the evidence to the official taxonomy definition.
- why_not_earlier must explicitly explain why each step before the chosen
  location does NOT qualify as an onset (e.g. "step_00 shows the correct
  behaviour; the error is not yet observable until step_02 when...").
- Do not invent content not present in the trace.
- Impact scale:
  - LOW    = limited local issue, little downstream effect
  - MEDIUM = meaningful disruption or wasted work, partial recovery possible
  - HIGH   = major task derailment, fabricated progress, invalid execution,
             or strong downstream harm

Output format:
Return a JSON object with one field:
{
  "errors": [
    {
      "category": "...",
      "location": "...",
      "evidence": "...",
      "description": "...",
      "why_not_earlier": "...",
      "impact": "LOW | MEDIUM | HIGH"
    }
  ]
}

Field requirements:
- category: exactly one of the provided categories
- location: exactly one valid step ID (accept or correct the Pass 1 candidate)
- evidence: 1-3 sentences, trace-grounded
- description: 1-3 sentences connecting evidence to the taxonomy definition
- why_not_earlier: 1-3 sentences ruling out earlier steps
- impact: LOW, MEDIUM, or HIGH

CRITICAL: You MUST produce exactly one entry for EVERY category in the provided
list. Skipping a category is not acceptable — if the evidence is weak or
ambiguous, still choose the most defensible step and note the ambiguity in
description or why_not_earlier.

Before finalising, silently check:
1. Does the output contain one entry for EVERY provided category?
2. Is each category exactly one of the given labels?
3. Does each category appear exactly once?
4. Is the chosen location the earliest supported one?
5. Does evidence come directly from the trace?
6. Does why_not_earlier explicitly address the steps before the chosen location?"""


# ---------------------------------------------------------------------------
# Few-shot examples — 2.6 Action-Reasoning Mismatch
#
# Chosen because it requires cross-step comparison (harder than Step Repetition):
# the agent makes a claim in step_02 that contradicts its own output in step_00,
# so the annotator must compare across steps rather than spotting a local pattern.
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLE_TRACE = """\
[step_00]
Navigator: The ColumnTransformer class is in sklearn/compose/_column_transformer.py. The relevant methods are `_iter` and `_add_prefix_for_feature_names_out`. Here is the full implementation of `_add_prefix_for_feature_names_out`:

def _add_prefix_for_feature_names_out(self, feature_names_out):
    if self.verbose_feature_names_out:
        return [f"{name}__{fname}" for name, fname in feature_names_out]
    return [fname for _, fname in feature_names_out]

[step_01]
Planner: Good. Now trace how feature names flow from `_iter` to the final output so we can plan the fix.

[step_02]
Navigator: The `_iter` method is the main orchestrator for feature name output. Note that the `_add_prefix_for_feature_names_out` method is not explicitly shown in the code snippet — it is only mentioned in the context of the `_iter` method and its implementation is unavailable to us."""


_FEW_SHOT_EXAMPLE_PASS1 = """\
Pass 1 Example

Given error categories:
["2.6 Action-Reasoning Mismatch"]

Trace:
""" + _FEW_SHOT_EXAMPLE_TRACE + """

Output:
{
  "candidates": [
    {
      "category": "2.6 Action-Reasoning Mismatch",
      "location": "step_02",
      "short_reason": "step_02 claims '_add_prefix_for_feature_names_out is not explicitly shown' and its 'implementation is unavailable,' directly contradicting step_00 which provided the method's complete implementation."
    }
  ]
}"""


_FEW_SHOT_EXAMPLE_PASS2 = """\
Pass 2 Example

Error categories with Pass 1 candidate locations:
[
  {
    "category": "2.6 Action-Reasoning Mismatch",
    "location": "step_02",
    "short_reason": "step_02 claims _add_prefix_for_feature_names_out is not shown, contradicting step_00 which showed its full implementation."
  }
]

Definition:
2.6 Action-Reasoning Mismatch:
This error occurs when there is a discrepancy or mismatch between agents' logical discussion conclusion or a single agent's internal decision-making processes and the actual actions or outputs the system produces.

Trace:
""" + _FEW_SHOT_EXAMPLE_TRACE + """

Output:
{
  "errors": [
    {
      "category": "2.6 Action-Reasoning Mismatch",
      "location": "step_02",
      "evidence": "In step_02, Navigator states '_add_prefix_for_feature_names_out is not explicitly shown' and its 'implementation is unavailable,' directly contradicting step_00 which provided the method's complete implementation.",
      "description": "This matches Action-Reasoning Mismatch because the agent's stated claim in step_02 ('method unavailable') is inconsistent with its own prior output in step_00 (the full implementation was shown). The reasoning in step_02 does not follow from the information the agent itself produced.",
      "why_not_earlier": "step_00 correctly reports the method location and shows its full implementation — no mismatch occurs there. step_01 is a planning message from the Planner with no factual claim about code. The mismatch first becomes observable in step_02 when the Navigator makes a claim that contradicts step_00.",
      "impact": "MEDIUM"
    }
  ]
}"""


# ---------------------------------------------------------------------------
# Task-specific instructions
# ---------------------------------------------------------------------------

# Injected after the shared instructions when a known task is given.
# Addresses systematic annotation pitfalls specific to each MAS type.

_TASK_INSTRUCTIONS: Dict[str, str] = {
    "metagpt": """\
Task-specific note — MetaGPT traces:
This trace follows a fixed four-role pipeline:
  step_00  Human           — task specification
  step_01  SimpleCoder     — initial implementation
  step_02  SimpleTester    — initial tests
  step_03  SimpleReviewer  — first review
  step_04  SimpleTester    — revised tests
  step_05  SimpleReviewer  — final review
(Shorter traces may be missing the last one or two steps.)

Critical localization rule for this pipeline:
When a SimpleReviewer step comments on a missing, incorrect, or incomplete aspect
of the code or tests, that comment is NOT the onset of the error — it is merely
the step where the error was surfaced. The error was introduced earlier:
- If SimpleCoder's implementation is wrong or incomplete → onset is step_01.
- If SimpleTester's tests are wrong, missing, or incomplete → onset is step_02.
- If the reviewer's REVISED test still repeats an earlier mistake → onset is
  the step where that mistake first appeared, not the reviewer step.
Always choose the step where the error was INTRODUCED, not where it was POINTED OUT.""",

    "ag2": """\
Task-specific note — AG2 traces:
This trace is a two-agent math problem-solving dialogue:
  [mathproxyagent]  — task orchestrator / code executor (provides problem, runs code, gives results)
  [assistant]       — problem solver (provides reasoning, writes code, gives final answer)

The pipeline typically follows:
  step_00  mathproxyagent: problem specification
  step_01  assistant: key idea and/or initial code
  step_02  mathproxyagent: execution result (numeric output or error)
  step_03  assistant: interpretation and/or correction
  ... (may repeat for multi-round traces)
  last     assistant: final boxed answer

Critical localization rules for this pipeline:
- Reasoning errors (wrong approach, ignoring problem constraints) originate in the
  FIRST assistant turn where the flawed reasoning appears — not in a later turn that
  merely propagates or repeats the same mistake.
- Verification failures (accepting a wrong answer without checking) belong to the
  FINAL assistant turn where the answer is accepted and boxed, if no earlier turn
  shows verification behaviour.
- If the mathproxyagent execution result reveals an error (e.g. wrong number), but
  the assistant already reasoned incorrectly in a prior step, the onset is the
  prior reasoning step, not the result step.
- "Continue" prompts from mathproxyagent carry no information and are never the
  onset of any error.
- Translating a reasoning plan into code is the NORMAL next action, not Step
  Repetition (1.3). Step Repetition requires the agent to redo a task or phase
  that was already completed and produced a result — e.g. re-deriving the same
  reasoning, rewriting functionally identical code, or re-running the same
  calculation. If step_01 outlines a plan and step_02 implements it in code,
  that is NOT repetition. Only annotate 1.3 if the trace shows a genuinely
  redundant redo of already-completed work.

What each error type looks like in AG2 math traces:
- 1.1 (Disobey Task Specification) in this context means the assistant's
  reasoning or code ignores or misreads a specific constraint stated in the
  problem — e.g. using the wrong base quantity, omitting a given condition
  from the calculation, or misinterpreting what the question is asking. It is
  NOT triggered by arithmetic mistakes alone (a correct approach with a
  numerical error is not 1.1).
- 2.6 (Action-Reasoning Mismatch) means the assistant explicitly states a
  reasoning plan in one step but the code or conclusion in the same or next
  step does something different from what was stated.
- 3.3 (No or Incorrect Verification) means the assistant accepts a final
  answer that visibly contradicts information already present in the trace
  (e.g. the execution result contradicts the expected output) without any
  acknowledgement or correction.

Negative rules — do NOT annotate these unless the evidence is unambiguous:
- Do NOT label 2.2 (Fail to Ask for Clarification) if the missing quantity or
  constraint is directly inferable from the problem statement. Clarification is
  only needed when the problem is genuinely ambiguous or under-specified; a
  solver that reads the problem and derives values from it is behaving correctly.
- Do NOT label 2.3 (Task Derailment) for ordinary wrong arithmetic or flawed
  but on-task reasoning. Derailment means the agent shifts focus away from the
  asked question entirely. A wrong calculation that still targets the correct
  question is a reasoning error (1.1 or 2.6), not a derailment.
- Do NOT label 1.1 (Disobey Task Specification) when the assistant introduces a
  reasonable intermediate variable or helper calculation to solve the asked
  question. 1.1 requires violating an explicit constraint or requirement, not
  merely choosing an approach the problem did not prescribe.
- Do NOT label 3.3 (No or Incorrect Verification) merely because the assistant
  did not explicitly print or state a verification step. In this pipeline,
  verification failures are only annotatable when the system has a dedicated
  verifier role or when the final answer contradicts a result that was already
  visible in the trace and the assistant accepted it anyway without comment.""",

    "openmanus": """\
Task-specific note — OpenManus traces:
This trace is a single-agent browser-use log. Each step contains the agent's
thoughts, tool selection, and tool result. Steps are segmented on
'Executing step N/M' boundaries.

Critical localization rule for this pipeline:
The agent operates sequentially — an error in one step often propagates silently
for several steps before becoming visible (e.g., the agent navigates to the wrong
page in step_02 but the consequence only appears in step_08). Always locate the
step where the CAUSAL ACTION that led to the error first occurs, not the step
where its consequence becomes visible.""",
}


# ---------------------------------------------------------------------------
# Prompt assembly helpers
# ---------------------------------------------------------------------------

def _task_instruction_block(task: str) -> str:
    """Return a formatted task-specific instruction block, or empty string if none."""
    if not task:
        return ""
    note = _TASK_INSTRUCTIONS.get(task.lower().strip())
    return f"\n\n{note}" if note else ""


def _format_steps(steps: List[Dict]) -> str:
    parts = []
    for s in steps:
        parts.append(f"[{s['id']}]\n{s['content']}")
    return "\n\n".join(parts)


def _format_candidates(candidates: List[Dict]) -> str:
    """Format Pass 1 candidates as a JSON block for the Pass 2 prompt."""
    import json
    return json.dumps(candidates, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pass1_prompt(
    steps: List[Dict],
    error_ids: List[str],
    definitions: Dict[str, dict],
    variant: str = "few_shot",
    task: str = None,
) -> str:
    """
    Build the Pass 1 (onset localization) prompt.

    Args:
        steps:       list of {"id": "step_00", "content": "..."} from the extractor
        error_ids:   list of MAST error IDs present in this trace, e.g. ["1.3", "2.6"]
        definitions: parsed definitions dict from definitions.parse_definitions()
        variant:     "zero_shot" or "few_shot"
        task:        task name (e.g. "metagpt", "openmanus") for task-specific instructions
    """
    if variant not in ("zero_shot", "few_shot"):
        raise ValueError(f"Unknown prompt variant: {variant!r}. Choose 'zero_shot' or 'few_shot'.")

    labels = [category_label(eid, definitions) for eid in error_ids]
    given_categories = str(labels)
    trace_block = _format_steps(steps)
    few_shot_block = (_FEW_SHOT_EXAMPLE_PASS1 + "\n\n---\n\n") if variant == "few_shot" else ""
    task_block = _task_instruction_block(task)

    return (
        f"You are an expert annotator for multi-agent system failure traces.\n\n"
        f"{_PASS1_INSTRUCTIONS}{task_block}\n\n"
        f"{few_shot_block}"
        f"Now annotate the trace.\n\n"
        f"Given error categories:\n{given_categories}\n\n"
        f"Ordered steps/spans:\n{trace_block}"
    )


def build_pass2_prompt(
    steps: List[Dict],
    candidates: List[Dict],
    definitions: Dict[str, dict],
    variant: str = "few_shot",
    task: str = None,
) -> str:
    """
    Build the Pass 2 (verification + full annotation) prompt.

    Args:
        steps:       same step list used in Pass 1
        candidates:  validated Pass 1 output — list of
                     {"category": "...", "location": "...", "short_reason": "..."}
        definitions: parsed definitions dict
        variant:     "zero_shot" or "few_shot"
        task:        task name (e.g. "metagpt", "openmanus") for task-specific instructions
    """
    if variant not in ("zero_shot", "few_shot"):
        raise ValueError(f"Unknown prompt variant: {variant!r}. Choose 'zero_shot' or 'few_shot'.")

    # Extract error IDs from candidate categories (e.g. "2.6 Action-Reasoning Mismatch" → "2.6")
    error_ids = []
    for c in candidates:
        cat = c.get("category", "")
        eid = cat.split()[0] if cat else ""
        if eid:
            error_ids.append(eid)

    defs_block = format_definitions_for_prompt(error_ids, definitions)
    candidates_block = _format_candidates(candidates)
    trace_block = _format_steps(steps)
    few_shot_block = (_FEW_SHOT_EXAMPLE_PASS2 + "\n\n---\n\n") if variant == "few_shot" else ""
    task_block = _task_instruction_block(task)

    return (
        f"You are an expert annotator for multi-agent system failure traces.\n\n"
        f"{_PASS2_INSTRUCTIONS}{task_block}\n\n"
        f"{few_shot_block}"
        f"Now complete the annotation.\n\n"
        f"Error categories with Pass 1 candidate locations:\n{candidates_block}\n\n"
        f"Official taxonomy definitions:\n{defs_block}\n\n"
        f"Ordered steps/spans:\n{trace_block}"
    )


def build_pass1_retry_prompt(
    steps: List[Dict],
    missing_error_ids: List[str],
    definitions: Dict[str, dict],
    task: str = None,
) -> str:
    """
    Build a targeted retry prompt for Pass 1 when some categories were skipped.

    Args:
        steps:             same step list from the original Pass 1 call
        missing_error_ids: error IDs that were not annotated in the first attempt
        definitions:       parsed definitions dict
        task:              task name for task-specific instructions
    """
    labels = [category_label(eid, definitions) for eid in missing_error_ids]
    defs_block = format_definitions_for_prompt(missing_error_ids, definitions)
    trace_block = _format_steps(steps)
    task_block = _task_instruction_block(task)

    return (
        f"You are an expert annotator for multi-agent system failure traces.\n\n"
        f"In your previous response you missed the following error categories.\n"
        f"You MUST now annotate ONLY these missing categories — do not repeat ones "
        f"you already annotated.\n\n"
        f"Missing categories:\n{labels}\n\n"
        f"Official taxonomy definitions:\n{defs_block}\n\n"
        f"Rules: for each missing category find the earliest step where it first "
        f"becomes observable and provide a short_reason grounded in the trace.{task_block}\n\n"
        f"Return a JSON object:\n"
        f'{{"candidates": [{{"category": "...", "location": "...", "short_reason": "..."}}]}}\n\n'
        f"Ordered steps/spans:\n{trace_block}"
    )


def build_pass2_retry_prompt(
    steps: List[Dict],
    missing_candidates: List[Dict],
    definitions: Dict[str, dict],
    task: str = None,
) -> str:
    """
    Build a targeted retry prompt for Pass 2 when some categories were skipped.

    Args:
        steps:              same step list from the original Pass 2 call
        missing_candidates: Pass 1 candidate entries that were not annotated
        definitions:        parsed definitions dict
        task:               task name for task-specific instructions
    """
    missing_error_ids = [c.get("category", "").split()[0] for c in missing_candidates
                         if c.get("category", "").split()]
    defs_block = format_definitions_for_prompt(missing_error_ids, definitions)
    candidates_block = _format_candidates(missing_candidates)
    trace_block = _format_steps(steps)
    task_block = _task_instruction_block(task)

    return (
        f"You are an expert annotator for multi-agent system failure traces.\n\n"
        f"In your previous response you missed the following error categories.\n"
        f"You MUST now annotate ONLY these missing categories — do not repeat ones "
        f"you already annotated.\n\n"
        f"Missing categories with their Pass 1 candidate locations:\n{candidates_block}\n\n"
        f"Official taxonomy definitions:\n{defs_block}\n\n"
        f"Rules: verify each candidate location and produce the full annotation "
        f"(evidence, description, why_not_earlier, impact).{task_block}\n\n"
        f"Return a JSON object:\n"
        f'{{"errors": [{{"category": "...", "location": "...", "evidence": "...", '
        f'"description": "...", "why_not_earlier": "...", "impact": "LOW|MEDIUM|HIGH"}}]}}\n\n'
        f"Ordered steps/spans:\n{trace_block}"
    )
