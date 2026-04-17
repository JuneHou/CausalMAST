"""
Load MAST failure taxonomy from definitions.txt for use in prompts.
"""
from pathlib import Path
from typing import Optional

from . import config


def load_definitions(path: Optional[Path] = None) -> str:
    """Load full definitions file content. Uses config DEFINITIONS_PATH if path not given."""
    p = Path(path or config.DEFINITIONS_PATH)
    if not p.is_file():
        return ""
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
