#!/usr/bin/env python3
"""
Step 7: Judge 2 — B-effect label (outcome evaluation) for MAST-AG2.

Adapted from TRAIL causal/patch/judge_b_effect.py.
Key changes vs TRAIL:
  - B_DEFINITIONS replaced with MAST 13-category definitions (1.1–3.3).
  - Removed trail_io import; uses patch_generator_llm._call_llm.
  - Prompt templates and BEffectVerdict structure identical to TRAIL.
  - Context labels updated: "ORIGINAL_TRACE_SUFFIX" = original steps after t_A;
    "RERUN_TRACE_SUFFIX" = simulated steps from the counterfactual run.

Fan-out: one LLM call per EdgePair where Judge 1 confirmed resolved=True.
All EdgePairs sharing the same error_id reuse the same RerunResult.

Input:  rerun_results.jsonl, a_resolved.jsonl, edge_pairs.jsonl
Output: b_effect.jsonl  (one record per EdgePair, resolved A-instances only)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

from patch_generator_llm import _call_llm


# ---------------------------------------------------------------------------
# B taxonomy definitions — MAST 13 categories
# ---------------------------------------------------------------------------

B_DEFINITIONS = {
    "1.1": (
        "Disobey Task Specification — Violates constraints or requirements explicitly stated in the task. "
        "The agent performs actions or produces outputs that contradict explicit task requirements, "
        "constraints, or specifications provided in the problem statement."
    ),
    "1.2": (
        "Disobey Role Specification — Ignores or violates the assigned agent role. "
        "The agent acts outside its designated role, takes actions not permitted by its role, "
        "or behaves inconsistently with the role defined for it."
    ),
    "1.3": (
        "Step Repetition — Repeats a task or phase already completed with a result. "
        "The agent re-executes work that was already done and had a result, producing redundant "
        "computation without forward progress."
    ),
    "1.4": (
        "Loss of Conversation History — Fails to retain or use prior conversation context. "
        "The agent ignores or forgets important prior information, context, or decisions "
        "that were established earlier in the conversation."
    ),
    "1.5": (
        "Unaware of Termination Conditions — Continues past a valid stopping point or stops too early. "
        "The agent either keeps running after the task is complete (continues past a valid stop) "
        "or terminates before the required output is produced (stops too early)."
    ),
    "2.1": (
        "Conversation Reset — Resets the conversation, losing prior context and progress. "
        "The agent restarts the conversation from the beginning, discarding previously "
        "established context, decisions, and intermediate results."
    ),
    "2.2": (
        "Fail to Ask for Clarification — Proceeds without resolving ambiguity that required clarification. "
        "The agent makes an assumption about an ambiguous requirement and proceeds, "
        "rather than requesting clarification from the user or orchestrator."
    ),
    "2.3": (
        "Task Derailment — Shifts focus away from the intended objective. "
        "The agent abandons or deprioritizes the primary task objective and begins "
        "pursuing a different or tangential goal."
    ),
    "2.4": (
        "Information Withholding — Fails to share information needed by other agents. "
        "The agent retains or omits information that another agent in the system needs "
        "to complete its part of the task."
    ),
    "2.6": (
        "Action-Reasoning Mismatch — Executes an action inconsistent with the stated reasoning. "
        "The agent describes one plan or rationale but performs a different action, "
        "creating a contradiction between its stated reasoning and its actual behavior."
    ),
    "3.1": (
        "Premature Termination — Stops before the task is complete or a required output is produced. "
        "The agent declares the task done or stops responding before all required outputs "
        "have been produced or all task requirements have been satisfied."
    ),
    "3.2": (
        "Weak Verification — Performs only superficial or incomplete verification. "
        "The agent checks the result in a cursory way that does not adequately validate "
        "whether the output actually meets the task requirements."
    ),
    "3.3": (
        "No or Incorrect Verification — Skips verification entirely or verifies against wrong criteria. "
        "The agent either performs no verification of its output, or checks the output "
        "against criteria that do not match the actual task requirements."
    ),
}


# ---------------------------------------------------------------------------
# Judge prompt (identical structure to TRAIL)
# ---------------------------------------------------------------------------

JUDGE_B_SYSTEM = """You are evaluating the downstream effect of a do(A=0) intervention on error type B.

The source error A was locally patched at one labeled step in a multi-agent conversation.
The rerun trace suffix shows the counterfactual execution after the intervention.

You must judge ONLY the downstream error type B.

Effect labels:
- disappeared    : B was present in baseline, absent in rerun (intervention removed B)
- delayed        : B was present in baseline, appears later in rerun
- unchanged      : B was present in baseline, appears at similar position in rerun
- earlier        : B was present in baseline, appears earlier in rerun
- weakened       : B was present in baseline, present in rerun but less severe
- strengthened   : B was present in baseline, present in rerun and more severe
- emerged        : B was ABSENT in baseline, but NOW PRESENT in rerun (intervention introduced B)
- not_observable : B was absent in baseline and absent in rerun; effect cannot be assessed

Return ONLY JSON."""

JUDGE_B_USER_TEMPLATE = """\
SOURCE_ERROR_TYPE: {A}
TARGET_ERROR_TYPE: {B}

TARGET_ERROR_DEFINITION:
{B_TAXONOMY_DEFINITION_OR_INSTANCE_DESCRIPTION}

ORIGINAL_TRACE_SUFFIX (steps after t_A in original trace):
<<<
{ORIGINAL_SUFFIX}
>>>

ORIGINAL_ONSET_REF:
{ORIGINAL_B_ONSET}

RERUN_TRACE_SUFFIX_AFTER_DO_A_0 (simulated steps after patching t_A):
<<<
{RERUN_SUFFIX}
>>>

Task:
Judge how B changed after the do(A=0) intervention.

Required output schema:
{{
  "source_error_type": "string",
  "target_error_type": "string",
  "effect_label": "disappeared|delayed|unchanged|earlier|weakened|strengthened|emerged|not_observable",
  "target_present_after": true,
  "original_onset_ref": "string|null",
  "rerun_onset_ref": "string|null",
  "confidence": "high|medium|low",
  "evidence": "string"
}}\
"""


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class BEffectVerdict:
    trace_id: str
    error_id: str
    edge: dict
    effect_label: str
    target_present_after: bool
    original_onset_ref: Optional[str]
    rerun_onset_ref: Optional[str]
    confidence: str
    evidence: str
    b_present_baseline: bool
    rerun_status: str
    model_used: str


VALID_EFFECT_LABELS = {
    "disappeared", "delayed", "unchanged", "earlier",
    "weakened", "strengthened", "emerged", "not_observable",
}


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def judge_b_effect(
    rerun_result: dict,
    edge_pair: dict,
    model: str = "openai/gpt-4o",
) -> BEffectVerdict:
    """Run Judge 2 for one EdgePair. Returns BEffectVerdict."""
    a_cat = edge_pair["edge"]["a"]
    b_cat = edge_pair["edge"]["b"]
    b_definition = B_DEFINITIONS.get(b_cat, f"MAST category: {b_cat}")

    original_suffix_spans = rerun_result.get("original_suffix_spans") or []
    original_suffix = "\n---\n".join(str(s)[:600] for s in original_suffix_spans[:8])
    if not original_suffix:
        original_suffix = "(no suffix steps extracted)"

    b_onset_baseline = edge_pair.get("b_onset_baseline", -1)
    b_present_baseline = edge_pair.get("b_present_baseline", False)
    if b_present_baseline and b_onset_baseline >= 0:
        original_onset_ref = f"step number {b_onset_baseline}"
    else:
        original_onset_ref = "not present in baseline"

    rerun_status = rerun_result.get("rerun_status", "rerun_missing_suffix")
    rerun_suffix_spans = rerun_result.get("rerun_suffix_spans") or []

    if rerun_status == "live_rerun_success" and rerun_suffix_spans:
        rerun_suffix = "\n---\n".join(str(s)[:600] for s in rerun_suffix_spans[:8])
    else:
        rerun_suffix = (
            "(rerun_missing_suffix: LLM simulation did not produce a counterfactual trace. "
            "Effect on B cannot be fully assessed from trace evidence.)"
        )

    user_msg = JUDGE_B_USER_TEMPLATE.format(
        A=a_cat,
        B=b_cat,
        B_TAXONOMY_DEFINITION_OR_INSTANCE_DESCRIPTION=b_definition,
        ORIGINAL_SUFFIX=original_suffix[:3000],
        ORIGINAL_B_ONSET=original_onset_ref,
        RERUN_SUFFIX=rerun_suffix[:3000],
    )

    effect_label = "not_observable"
    target_present_after = False
    rerun_onset_ref = None
    confidence = "low"
    evidence = ""

    try:
        raw = _call_llm(JUDGE_B_SYSTEM, user_msg, model=model, max_tokens=512)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)
            raw = raw.strip()
        parsed = json.loads(raw)
        label = str(parsed.get("effect_label", "not_observable")).lower().strip()
        if label not in VALID_EFFECT_LABELS:
            label = "not_observable"
        effect_label = label
        target_present_after = bool(parsed.get("target_present_after", False))
        rerun_onset_ref = parsed.get("rerun_onset_ref") or None
        confidence = str(parsed.get("confidence", "low"))
        evidence = str(parsed.get("evidence", ""))[:600]
    except Exception as e:
        evidence = f"judge_error: {e}"

    return BEffectVerdict(
        trace_id=rerun_result["trace_id"],
        error_id=rerun_result.get("error_id", ""),
        edge=edge_pair.get("edge", {}),
        effect_label=effect_label,
        target_present_after=target_present_after,
        original_onset_ref=original_onset_ref,
        rerun_onset_ref=rerun_onset_ref,
        confidence=confidence,
        evidence=evidence,
        b_present_baseline=b_present_baseline,
        rerun_status=rerun_status,
        model_used=model,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Judge 2: evaluate downstream B-effect after do(A=0) for MAST-AG2."
    )
    parser.add_argument("--rerun_results",
                        default="outputs/interventions/rerun_results.jsonl")
    parser.add_argument("--a_resolved",
                        default="outputs/interventions/a_resolved.jsonl")
    parser.add_argument("--edge_pairs",
                        default="outputs/interventions/edge_pairs.jsonl")
    parser.add_argument("--out_dir", default="outputs/interventions")
    parser.add_argument("--model", default="openai/gpt-4o")
    args = parser.parse_args()

    def _load_jsonl(path: str) -> List[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    rerun_results = _load_jsonl(args.rerun_results)
    a_resolved = _load_jsonl(args.a_resolved)
    edge_pairs = _load_jsonl(args.edge_pairs)

    resolved_keys = {
        (v["trace_id"], v.get("error_id", ""))
        for v in a_resolved if v.get("resolved")
    }

    rerun_idx = {
        (rr["trace_id"], rr.get("error_id", "")): rr
        for rr in rerun_results if rr.get("rerun_success")
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "b_effect.jsonl")

    from collections import Counter
    label_counts: Counter = Counter()
    n_skipped = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for ep in edge_pairs:
            key = (ep["trace_id"], ep.get("error_id", ""))
            if key not in resolved_keys:
                n_skipped += 1
                continue
            rr = rerun_idx.get(key)
            if not rr:
                n_skipped += 1
                continue

            verdict = judge_b_effect(rr, ep, model=args.model)
            f.write(json.dumps(asdict(verdict), ensure_ascii=False) + "\n")
            label_counts[verdict.effect_label] += 1
            print(f"  {str(verdict.trace_id)[:8]} {verdict.edge} "
                  f"→ {verdict.effect_label} [{verdict.confidence}]")

    print(f"\nWrote {out_path}. Skipped (unresolved A)={n_skipped}")
    print("Effect label distribution:", dict(label_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
