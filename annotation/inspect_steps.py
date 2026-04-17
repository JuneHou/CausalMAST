"""
Phase 0 validation helper: inspect extracted steps on sampled traces.

Writes a human-readable text file showing each step's ID, length, and content,
so you can verify that step boundaries are correct before running annotation.

Usage:
    python -m annotation.inspect_steps --task openmanus --n 3 --output inspect_openmanus.txt
    python -m annotation.inspect_steps --task openmanus --n 5  # prints to stdout
"""
import argparse
import sys
from pathlib import Path

from . import config
from . import openmanus as _openmanus_mod
from . import metagpt as _metagpt_mod
from . import ag2 as _ag2_mod

# Map lowercase task name -> (loader, extractor)
_TASK_REGISTRY = {
    "openmanus": (_openmanus_mod.load_records, _openmanus_mod.extract_steps),
    "metagpt": (_metagpt_mod.load_records, _metagpt_mod.extract_steps),
    "ag2": (_ag2_mod.load_records, _ag2_mod.extract_steps),
}


def _get_task(name: str):
    key = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    match = _TASK_REGISTRY.get(key)
    if not match:
        available = list(_TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task {name!r}. Available: {available}")
    return match


def inspect(task: str, n: int, mad_path: Path, output: str = None):
    load_fn, extract_fn = _get_task(task)
    records = load_fn(mad_path, min_errors=0)  # min_errors=0: inspect all traces, including error-free ones
    n_with_errors = sum(1 for r in records if r.get("error_ids"))
    print(f"Found {len(records)} {task} traces total ({n_with_errors} with errors, {len(records)-n_with_errors} error-free).", file=sys.stderr)
    print(f"Note: run_annotation.py only processes traces with >= 2 errors (min_errors=2).", file=sys.stderr)

    sample = records[:n]
    lines = []

    for rec_idx, rec in enumerate(sample):
        trace_id = rec.get("trace_id", f"trace_{rec_idx}")
        error_ids = rec.get("error_ids", [])
        steps = extract_fn(rec["trajectory"])

        header = "=" * 72
        lines.append(f"{header}")
        lines.append(f"TRACE {rec_idx + 1}/{len(sample)}: {trace_id}")
        lines.append(f"Known errors: {error_ids}")
        lines.append(f"Extracted steps: {len(steps)}")
        lines.append(f"{header}")

        if not steps:
            lines.append("  [WARNING] No steps extracted — check extraction logic.")
            lines.append("")
            continue

        for step in steps:
            sid = step["id"]
            content = step["content"]
            char_count = len(content)
            lines.append(f"\n--- {sid} ({char_count} chars) ---")
            # Show up to 600 chars of each step; mark truncation
            if char_count > 600:
                lines.append(content[:600] + f"\n  ... [truncated, {char_count - 600} more chars]")
            else:
                lines.append(content)

        lines.append("")

    out_text = "\n".join(lines)

    if output:
        Path(output).write_text(out_text, encoding="utf-8")
        print(f"Written to {output}", file=sys.stderr)
    else:
        print(out_text)


def main():
    ap = argparse.ArgumentParser(description="Inspect extracted steps for Phase 0 validation.")
    ap.add_argument("--task", required=True, help="Task name (e.g. openmanus)")
    ap.add_argument("--n", type=int, default=3, help="Number of traces to sample (default: 3)")
    ap.add_argument("--mad", default=str(config.DEFAULT_MAD_PATH), help="Path to MAD_full_dataset.json")
    ap.add_argument("--output", default=None, help="Output file path (default: stdout)")
    args = ap.parse_args()

    inspect(
        task=args.task,
        n=args.n,
        mad_path=Path(args.mad),
        output=args.output,
    )


if __name__ == "__main__":
    main()
