"""
AG2 (AutoGen2) step extraction and record loading.

Trajectory formats
------------------
AG2 trajectories appear in two formats within MAD_full_dataset.json:

Format A — dict_repr (384 / 394 error traces):
    A string of adjacent Python dicts with no separator, e.g.:
        {'content': ['...'], 'role': 'assistant', 'name': 'mathproxyagent'}
        {'content': ['...'], 'role': 'user',      'name': 'assistant'}
        ...
    Parsed by walking brace depth and calling ast.literal_eval on each block.

Format B — yaml_text (10 / 394 error traces):
    A YAML-like indented log:
        [header block]

              content:
                    <12-space-indented lines>
              role: user
              name: assistant
    Parsed by splitting on "      content:" boundaries.

Turn roles
----------
Two agents alternate:
  mathproxyagent (role=assistant)  — task orchestrator / code executor
  assistant      (role=user)       — problem solver

Turn filtering
--------------
mathproxyagent emits two kinds of turns:
  • Task turn (turn 0):      the problem specification — KEEP, strip boilerplate preamble
  • Execution result turns:  code output, error messages, numeric results — KEEP
  • "Continue" turns:        "Continue. Please keep solving..." — SKIP (no information)

assistant turns are always kept.

After filtering, step counts range from 2 to 22, median 4.

Step IDs: step_00, step_01, ... (zero-padded).
"""

import ast
import json
import re
from pathlib import Path
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# "Continue" turn detection
# ---------------------------------------------------------------------------

_CONTINUE_PREFIX = "Continue. Please keep solving"


def _is_continue_turn(content: str) -> bool:
    return content.strip().startswith(_CONTINUE_PREFIX)


# ---------------------------------------------------------------------------
# First-turn preamble stripping
# ---------------------------------------------------------------------------

def _clean_task_content(content: str) -> str:
    """
    Strip the boilerplate preamble from the first mathproxyagent turn.
    The preamble is the generic "Let's use Python to solve a math problem.
    Query requirements: ..." block that precedes the actual problem statement.
    Keeps only the "Problem:" section onward.
    """
    m = re.search(r"Problem:\n(.+)", content, re.DOTALL)
    if m:
        return f"Problem:\n{m.group(1).strip()}"
    return content.strip()


# ---------------------------------------------------------------------------
# Content normalisation
# ---------------------------------------------------------------------------

def _normalise_content(raw) -> str:
    """Convert content field (str or list[str]) to a clean string."""
    if isinstance(raw, list):
        return "\n".join(raw).strip()
    return str(raw).strip()


# ---------------------------------------------------------------------------
# Format A: dict_repr parser
# ---------------------------------------------------------------------------

def _parse_dict_repr(traj: str) -> List[Dict]:
    """
    Parse a string of adjacent Python dicts into a list of turn dicts.
    Each turn dict has keys: name, role, content (str after normalisation).
    """
    turns = []
    depth = 0
    start = None
    for i, ch in enumerate(traj):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = ast.literal_eval(traj[start : i + 1])
                    turns.append({
                        "name": obj.get("name", ""),
                        "role": obj.get("role", ""),
                        "content": _normalise_content(obj.get("content", "")),
                    })
                except Exception:
                    pass
                start = None
    return turns


# ---------------------------------------------------------------------------
# Format B: yaml_text parser
# ---------------------------------------------------------------------------

def _parse_yaml_text(traj: str) -> List[Dict]:
    """
    Parse the YAML-like text format into a list of turn dicts.
    Blocks are separated by lines starting with "      content:" (6-space indent).
    Content lines are indented 12 spaces.
    """
    blocks = re.split(r"\n(?=      content:)", traj)
    turns = []
    for block in blocks:
        name_m = re.search(r"name:\s*(\S+)", block)
        role_m = re.search(r"role:\s*(\S+)", block)
        lines = re.findall(r"^ {12}(.+)", block, re.MULTILINE)
        content = "\n".join(lines).strip()
        if name_m and content:
            turns.append({
                "name": name_m.group(1),
                "role": role_m.group(1) if role_m else "",
                "content": content,
            })
    return turns


# ---------------------------------------------------------------------------
# Turn → step content
# ---------------------------------------------------------------------------

def _turn_to_content(turn: Dict, is_first: bool) -> Optional[str]:
    """
    Convert a parsed turn to step content string, or None if it should be skipped.

    Skipped turns:
      - mathproxyagent "Continue" prompts (no information content)

    First mathproxyagent turn: strip boilerplate, keep Problem section.
    All other turns: label with agent name and include full content.
    """
    name = turn["name"]
    content = turn["content"]

    if not content:
        return None

    if name == "mathproxyagent":
        if is_first:
            cleaned = _clean_task_content(content)
            return f"[mathproxyagent]\n{cleaned}" if cleaned else None
        if _is_continue_turn(content):
            return None
        return f"[mathproxyagent]\n{content}"

    # assistant (solver) turn — always keep
    return f"[assistant]\n{content}"


# ---------------------------------------------------------------------------
# Step extraction
# ---------------------------------------------------------------------------

def extract_steps(trajectory: str) -> List[Dict]:
    """
    Split an AG2 trajectory into ordered steps.

    Returns a list of {"id": "step_00", "content": "<cleaned content>"}.
    Returns [] if no turns can be parsed or all turns are filtered out.
    """
    trajectory = trajectory.replace("\r\n", "\n")
    if not trajectory.strip():
        return []

    # Detect format
    if trajectory.strip().startswith("{'content'"):
        turns = _parse_dict_repr(trajectory)
    else:
        turns = _parse_yaml_text(trajectory)

    if not turns:
        return []

    blocks = []
    first_proxy = True
    for turn in turns:
        is_first = (turn["name"] == "mathproxyagent" and first_proxy)
        if is_first:
            first_proxy = False
        content = _turn_to_content(turn, is_first)
        if content and len(content) >= 20:
            blocks.append(content)

    if not blocks:
        return []

    width = max(2, len(str(len(blocks) - 1)))
    return [{"id": f"step_{i:0{width}d}", "content": c} for i, c in enumerate(blocks)]


# ---------------------------------------------------------------------------
# MAD record loading
# ---------------------------------------------------------------------------

def load_records(mad_path: Path, min_errors: int = 2) -> List[Dict]:
    """
    Load AG2 records from MAD_full_dataset.json.

    Each returned record has:
        trace_id, mas_name, mast_annotation, error_ids, trajectory

    Only records with >= min_errors error labels are kept.
    """
    mad_path = Path(mad_path)
    with open(mad_path, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for r in data:
        if r.get("mas_name") != "AG2":
            continue
        ann = r.get("mast_annotation") or {}
        error_ids = [k for k, v in ann.items() if v == 1]
        if len(error_ids) < min_errors:
            continue
        traj = (r.get("trace") or {}).get("trajectory") or ""
        if not isinstance(traj, str) or not traj.strip():
            continue
        records.append({
            "trace_id": r.get("trace_id", ""),
            "mas_name": "AG2",
            "mast_annotation": ann,
            "error_ids": error_ids,
            "trajectory": traj,
        })
    return records
