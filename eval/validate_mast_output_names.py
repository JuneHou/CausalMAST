#!/usr/bin/env python3
"""
Validate MAST output directory naming conventions.

This checker flags accidental *_span_index artifacts in MAST outputs, where
span_index is not a valid concept (MAST has no location labels).
"""

from pathlib import Path
import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Flag non-canonical MAST output names (e.g., *_span_index)."
    )
    ap.add_argument(
        "--root",
        default="baselines/who&when/causal/outputs",
        help="Directory containing MAST who&when output folders and metrics files.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] root does not exist: {root}", file=sys.stderr)
        return 2

    bad = sorted(
        p.name
        for p in root.iterdir()
        if "_span_index" in p.name
    )

    if not bad:
        print("[OK] No *_span_index artifacts found in MAST outputs.")
        return 0

    print("[FAIL] Found non-canonical *_span_index artifacts:")
    for name in bad:
        print(f"  - {name}")
    print("\nAction: rename completed artifacts to canonical names without '_span_index'.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
