"""
MetaGPT step extraction and record loading.

Log structure
-------------
Each MetaGPT trace is a plain-text agent communication log with this structure:

    === MetaGPT Agent Communication Log - Started at TIMESTAMP ===

    [TIMESTAMP] FROM: Human TO: {'<all>'}
    ACTION: metagpt.actions.add_requirement.UserRequirement
    CONTENT:
    <task description>
    ---------------...

    [TIMESTAMP] NEW MESSAGES:

    SimpleCoder:
    <code>
    ---------------...

    [TIMESTAMP] NEW MESSAGES:

    SimpleTester:
    <tests>
    ---------------...

    [TIMESTAMP] NEW MESSAGES:

    SimpleReviewer:
    <review>
    ---------------...

    (two more NEW MESSAGES blocks: revised SimpleTester, final SimpleReviewer)

    === Communication Log Ended at TIMESTAMP ===

Every trace has exactly 6 message blocks:
  step_00  Human task specification      (FROM: Human block)
  step_01  SimpleCoder initial code      (NEW MESSAGES)
  step_02  SimpleTester initial tests    (NEW MESSAGES)
  step_03  SimpleReviewer review         (NEW MESSAGES)
  step_04  SimpleTester revised tests    (NEW MESSAGES)
  step_05  SimpleReviewer final review   (NEW MESSAGES)

Step boundaries are the [TIMESTAMP] lines. The prologue and epilogue
(=== headers) are discarded. Each block's content is the agent message
body after stripping structural noise (--- separators, ACTION:/CONTENT:
labels, === section markers).

Step IDs: step_00 ... step_05 (zero-padded).
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Step extraction
# ---------------------------------------------------------------------------

# Matches a full [TIMESTAMP] header line, e.g.:
#   "[2025-03-31 12:59:36] FROM: Human TO: {'<all>'}"
#   "[2025-03-31 12:59:40] NEW MESSAGES:"
_TIMESTAMP_RE = re.compile(
    r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\].*$",
    re.MULTILINE,
)

# Structural noise to strip from block bodies
_SEP_RE = re.compile(r"^-{10,}\s*$", re.MULTILINE)          # --- separator lines
_SECTION_RE = re.compile(r"^={3,}.*$", re.MULTILINE)         # === header/footer lines


def _parse_block(raw: str) -> Optional[str]:
    """
    Parse a single log block (from one [TIMESTAMP] line to the next).

    Returns cleaned step content, or None if the block is structural-only.
    """
    # Split header line from body
    nl = raw.find("\n")
    header = raw[:nl] if nl != -1 else raw
    body = raw[nl:].strip() if nl != -1 else ""

    # Strip structural markers from body
    body = _SEP_RE.sub("", body)
    body = _SECTION_RE.sub("", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    if "FROM:" in header:
        # Human task block — strip ACTION: and CONTENT: labels, keep task text
        from_m = re.search(r"FROM:\s*(\S+)", header)
        agent = from_m.group(1) if from_m else "Human"
        body = re.sub(r"^ACTION:.*\n?", "", body, flags=re.MULTILINE)
        body = re.sub(r"^CONTENT:\s*\n?", "", body, flags=re.MULTILINE)
        body = body.strip()
        return f"{agent} (task):\n{body}" if body else None

    if "NEW MESSAGES:" in header:
        # Agent message block — body already starts with "AgentName:\n<content>"
        return body if body else None

    return None


def extract_steps(trajectory: str) -> List[Dict]:
    """
    Split a MetaGPT log trajectory into ordered steps.

    Returns a list of {"id": "step_00", "content": "<cleaned content>"}.
    Returns [] if no [TIMESTAMP] markers are found.
    """
    trajectory = trajectory.replace("\r\n", "\n")
    hits = list(_TIMESTAMP_RE.finditer(trajectory))
    if not hits:
        return []

    blocks = []
    for i, m in enumerate(hits):
        start = m.start()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(trajectory)
        raw = trajectory[start:end]
        content = _parse_block(raw)
        if content and len(content) >= 20:
            blocks.append(content)

    if not blocks:
        return []

    width = max(2, len(str(len(blocks) - 1)))
    return [{"id": f"step_{i:0{width}d}", "content": c} for i, c in enumerate(blocks)]


# ---------------------------------------------------------------------------
# MAD record loading
# ---------------------------------------------------------------------------

def load_records(mad_path: Path, min_errors: int = 1) -> List[Dict]:
    """
    Load MetaGPT records from MAD_full_dataset.json.

    Each returned record has:
        trace_id, mas_name, mast_annotation, error_ids, trajectory

    Only records with >= min_errors error labels are kept.
    """
    mad_path = Path(mad_path)
    with open(mad_path, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for r in data:
        if r.get("mas_name") != "MetaGPT":
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
            "mas_name": "MetaGPT",
            "mast_annotation": ann,
            "error_ids": error_ids,
            "trajectory": traj,
        })
    return records
