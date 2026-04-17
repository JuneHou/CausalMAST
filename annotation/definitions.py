"""MAST taxonomy definitions — compact one-line format."""
from typing import Dict


# ---------------------------------------------------------------------------
# Canonical short definitions (one line each: "Name: description")
# ---------------------------------------------------------------------------

_FAILURE_DEFS = {
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_definitions(path=None) -> Dict[str, dict]:
    """
    Return the MAST taxonomy as {id: {"name": str, "text": str}}.

    The path argument is accepted for backward compatibility but ignored —
    definitions are now hardcoded from the compact one-line format.
    """
    entries = {}
    for eid, line in _FAILURE_DEFS.items():
        # Each line is "Name: description text"
        colon = line.index(":")
        name = line[:colon].strip()
        text = line[colon + 1:].strip()
        entries[eid] = {"name": name, "text": text}
    return entries


def format_definitions_for_prompt(error_ids: list, definitions: Dict[str, dict]) -> str:
    """
    Format definitions for a subset of error IDs as a prompt block.

    Returns a string like:
        1.3 Step Repetition:
        unnecessarily repeats an already-completed phase/step...
    """
    lines = []
    for eid in error_ids:
        entry = definitions.get(eid)
        if not entry:
            continue
        lines.append(f"{eid} {entry['name']}:\n{entry['text']}")
    return "\n\n".join(lines)


def category_label(error_id: str, definitions: Dict[str, dict]) -> str:
    """Return the full category label, e.g. '1.3 Step Repetition'."""
    entry = definitions.get(error_id)
    if not entry:
        return error_id
    return f"{error_id} {entry['name']}"
