"""
One-time correction for missed Pass-2 upgrades caused by a regex bug.

`parse_pass2_response` in run_eval_graph_inject.py (and its arc/deepinfra
clones) matches `:\\s*(yes|no)`, so bracketed answers like
"3.1 Premature Termination: <yes>" were silently dropped — the merged
`predictions` retained Pass-1's 0 instead of being flipped to 1.

This script:
  1. Re-parses every record's `pass2_raw` with a bracket-aware regex.
  2. Only ADDS upgrades (Pass-1 0 -> 1 for a re-parsed `yes`). Never removes
     existing upgrades; conflicting `no` reads are reported, not applied.

Usage (from MAST/):
    python eval/fix_pass2_brackets.py                  # dry-run
    python eval/fix_pass2_brackets.py --apply          # write changes
    python eval/fix_pass2_brackets.py --apply \\
        --outputs_dir /data/wang/junh/githubs/MAST/outputs_full

After --apply, re-run `eval/calculate_scores_yesno.py` to refresh
`-metrics.json` files.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

MAST_MODES = [
    "1.1", "1.2", "1.3", "1.4", "1.5",
    "2.1", "2.2", "2.3", "2.4", "2.6",
    "3.1", "3.2", "3.3",
]

# Matches "<mode> <anything-up-to-colon>: [<]yes|no[>]" (case-insensitive).
ANSWER_PATTERNS = {
    mode: re.compile(
        rf"{re.escape(mode)}\s*[^:\n]*:\s*<?\s*(yes|no)\s*>?",
        re.IGNORECASE,
    )
    for mode in MAST_MODES
}


def parse_pass2_robust(raw: str) -> Dict[str, int]:
    if not raw:
        return {}
    cleaned = raw.strip()
    if cleaned.startswith("@@"):
        cleaned = cleaned[2:]
    if cleaned.endswith("@@"):
        cleaned = cleaned[:-2]
    out: Dict[str, int] = {}
    for mode, pat in ANSWER_PATTERNS.items():
        m = pat.search(cleaned)
        if m:
            out[mode] = 1 if m.group(1).lower() == "yes" else 0
    return out


def correct_record(rec_path: Path, apply: bool) -> Tuple[List[str], List[str]]:
    """Return (new_upgrades, conflict_no_signals) for this record."""
    with rec_path.open() as f:
        rec = json.load(f)
    if not rec.get("pass2_triggered"):
        return [], []
    raw = rec.get("pass2_raw") or ""
    if not raw.strip():
        return [], []

    parsed = parse_pass2_robust(raw)
    if not parsed:
        return [], []

    preds = dict(rec["predictions"])
    cur_upgrades = dict(rec.get("pass2_upgrades") or {})

    # Reconstruct Pass-1: subtract currently applied upgrades.
    p1 = dict(preds)
    for k in cur_upgrades:
        p1[k] = 0

    new_ups: List[str] = []
    conflicts: List[str] = []
    for cat, val in parsed.items():
        if val == 1:
            if p1.get(cat, 0) == 0 and preds.get(cat, 0) == 0:
                preds[cat] = 1
                cur_upgrades[cat] = 1
                new_ups.append(cat)
        else:
            if cur_upgrades.get(cat) == 1:
                conflicts.append(cat)

    if new_ups and apply:
        rec["predictions"] = preds
        rec["pass2_upgrades"] = cur_upgrades
        with rec_path.open("w") as f:
            json.dump(rec, f, indent=2)
    return new_ups, conflicts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outputs_dir",
        default="/data/wang/junh/githubs/MAST/outputs_full",
    )
    ap.add_argument("--pattern", default="*graph-inject*")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    root = Path(args.outputs_dir)
    if not root.is_dir():
        print(f"[ERROR] not a directory: {root}", file=sys.stderr)
        return 2

    total_recs = 0
    total_fixed_recs = 0
    total_new_ups = 0
    total_conflicts = 0

    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name.startswith("old_"):
            continue
        if not sub.match(args.pattern):
            continue
        json_files = sorted(sub.glob("*.json"))
        if not json_files:
            continue

        per_fixed = 0
        per_new = 0
        per_conf = 0
        for jf in json_files:
            try:
                ups, confs = correct_record(jf, apply=args.apply)
            except Exception as e:
                print(f"[WARN] {jf}: {e}", file=sys.stderr)
                continue
            if ups:
                per_fixed += 1
                per_new += len(ups)
            if confs:
                per_conf += len(confs)
        total_recs += len(json_files)
        total_fixed_recs += per_fixed
        total_new_ups += per_new
        total_conflicts += per_conf

        print(
            f"{sub.name}: scanned={len(json_files)} "
            f"records_fixed={per_fixed} new_yes_upgrades={per_new} "
            f"conflict_no={per_conf}"
        )

    print()
    print(f"TOTAL records scanned     : {total_recs}")
    print(f"TOTAL records fixed       : {total_fixed_recs}")
    print(f"TOTAL new yes-upgrades    : {total_new_ups}")
    print(f"TOTAL conflict-no signals : {total_conflicts}")
    if not args.apply:
        print("\n[dry-run] re-run with --apply to write changes.")
    else:
        print("\nDone. Re-run eval/calculate_scores_yesno.py to refresh metrics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
