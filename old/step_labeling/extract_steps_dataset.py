#!/usr/bin/env python3
"""
Extract and save the step-split dataset before labeling.

Loads MAD, extracts steps per trace via get_steps_from_trace(), and writes:
- steps_dataset.jsonl: one JSON object per trace (trace_id, mas_name, n_steps, steps, failure_types, mast_annotation)
- steps_summary.json: aggregate stats (n_steps per mas_name, min/max/median)

Run this first to inspect step counts and fix extraction if needed.
Then run run_labeling with --mad (or use the saved dataset).
"""
import argparse
import json
from pathlib import Path

from . import config
from .preprocessing import load_mad_records
from .step_extraction import get_steps_from_trace


def main():
    ap = argparse.ArgumentParser(description="Extract steps from MAD and save locally")
    ap.add_argument(
        "--mad",
        type=str,
        default=str(config.DEFAULT_MAD_PATH),
        help=f"MAD path (default: {config.DEFAULT_MAD_PATH})",
    )
    ap.add_argument(
        "--min_failures",
        type=int,
        default=2,
        help="Keep traces with >= N failure types",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(config.DATA_DIR.parent / "processed"),
        help="Output directory for steps_dataset.jsonl and steps_summary.json",
    )
    ap.add_argument("--limit", type=int, default=None, help="Max traces (for debugging)")
    args = ap.parse_args()

    mad_path = Path(args.mad)
    if not mad_path.is_file():
        print(f"Error: MAD not found at {mad_path}. Run: python scripts/0_download_mad.py")
        return 1

    records = load_mad_records(mad_path, min_failures=args.min_failures, exclude_magentic=True)
    print(f"Loaded {len(records)} MAD records (>= {args.min_failures} failure types, Magentic excluded).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_path = out_dir / "steps_dataset.jsonl"
    summary_path = out_dir / "steps_summary.json"

    n_steps_by_mas = {}
    limit = args.limit or len(records)
    with open(steps_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(records):
            if i >= limit:
                break
            trajectory = r.get("trajectory")
            mas_name = r.get("mas_name", "")
            steps, _ = get_steps_from_trace(data={"trajectory": trajectory}, task_type=mas_name)
            n = len(steps)
            n_steps_by_mas.setdefault(mas_name, []).append(n)
            obj = {
                "trace_id": r.get("trace_id", ""),
                "mas_name": mas_name,
                "n_steps": n,
                "steps": steps,
                "failure_types": r.get("failure_types", []),
                "mast_annotation": r.get("mast_annotation", {}),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  Extracted {i + 1}/{limit} traces.")

    # Summary stats
    summary = {}
    for mas, counts in sorted(n_steps_by_mas.items()):
        summary[mas] = {
            "n_traces": len(counts),
            "min_steps": min(counts),
            "max_steps": max(counts),
            "mean_steps": round(sum(counts) / len(counts), 1),
            "median_steps": sorted(counts)[len(counts) // 2],
        }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {steps_path} ({limit} records)")
    print(f"Wrote {summary_path}")
    for mas, s in summary.items():
        print(f"  {mas}: {s['n_traces']} traces, steps min={s['min_steps']} max={s['max_steps']} mean={s['mean_steps']:.1f}")
    return 0


if __name__ == "__main__":
    exit(main())
