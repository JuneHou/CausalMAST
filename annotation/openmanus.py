"""
OpenManus step extraction and record loading.

Log structure
-------------
Each OpenManus trace is a plain-text log. The key structural marker is:

    ... | app.agent.base:run:140 - Executing step N/M

Every "Executing step N/M" line starts a new agent action cycle consisting of:
  - Manus's thoughts (may be on subsequent lines after the ✨ marker, or empty)
  - Tool selection  (🛠️ / 🧰 / 🔧 lines)
  - Tool execution result (🎯 line + result text)

We split the log at each "Executing step" boundary. The prologue (before
step 1) is discarded. Each resulting block becomes one step in the annotation.

Step IDs: step_00, step_01, ... (zero-padded to the number of steps in the trace).
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Log cleaning
# ---------------------------------------------------------------------------

# Matches the full prefix of an OpenManus log line:
#   "2025-04-01 02:17:03.592 | INFO     | app.agent.base:run:140 - "
_LOG_PREFIX_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\s*\|\s*\w+\s*\|\s*[^\|]+-\s*"
)

# Lines whose message content we drop entirely
_SKIP_PATTERNS = (
    "Token usage:",          # token accounting — never relevant to error annotation
    "Activating tool:",      # redundant: tool name already in 🧰 and 🎯 lines
)


def _clean_step_content(raw: str) -> str:
    """
    Strip log infrastructure from a raw OpenManus step block.

    Keeps the semantic content only:
      - Step header:   "Executing step N/M"
      - Thoughts:      "✨ Manus's thoughts: ..." (+ continuation lines)
      - Tool selected: "🛠️ Manus selected N tools to use"
      - Tools list:    "🧰 Tools being prepared: [...]"
      - Tool args:     "🔧 Tool arguments: {...}"
      - Tool result:   "🎯 Tool '...' completed ... Result: ..." (+ result content)

    Discards:
      - Timestamps and module:function:line prefixes
      - Token usage accounting lines
      - "Activating tool" lines (redundant)
    """
    out = []
    for line in raw.split("\n"):
        m = _LOG_PREFIX_RE.match(line)
        if m:
            # It's a log line — extract just the message after the prefix
            message = line[m.end():]
            if any(skip in message for skip in _SKIP_PATTERNS):
                continue
            out.append(message)
        else:
            # Non-log line: thought continuation text, result content, blank lines
            out.append(line)

    # Collapse runs of blank lines to at most one
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(out))
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Step extraction
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r"(?m)^.*\bExecuting step \d+/\d+\b.*$")


def extract_steps(trajectory: str) -> List[Dict]:
    """
    Split an OpenManus log trajectory into ordered steps and clean each block.

    Returns a list of {"id": "step_00", "content": "<cleaned content>"}.
    Returns [] if no "Executing step" markers are found.
    """
    trajectory = trajectory.replace("\r\n", "\n")
    hits = list(_STEP_RE.finditer(trajectory))
    if not hits:
        return []

    blocks = []
    for i, m in enumerate(hits):
        start = m.start()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(trajectory)
        raw = trajectory[start:end]
        content = _clean_step_content(raw)
        if len(content) >= 20:
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
    Load OpenManus records from MAD_full_dataset.json.

    Each returned record has:
        trace_id, mas_name, mast_annotation, error_ids, trajectory

    Only records with >= min_errors error labels (mast_annotation value == 1) are kept.
    Default min_errors=1 means error-free traces (16 out of 30 OpenManus traces) are excluded.
    """
    mad_path = Path(mad_path)
    with open(mad_path, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for r in data:
        if r.get("mas_name") != "OpenManus":
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
            "mas_name": "OpenManus",
            "mast_annotation": ann,
            "error_ids": error_ids,
            "trajectory": traj,
        })
    return records
