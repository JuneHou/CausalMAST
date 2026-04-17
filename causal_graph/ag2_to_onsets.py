"""
Step 0: Convert annotation_ag2_filtered.jsonl → onsets.jsonl for the CAPRI pipeline.

Replaces TRAIL steps 1-3 (filter_split + span_order + build_onsets).
MAST steps are already sequential (step_00, step_01, ...), so rank = int index.

Only uses errors where mast_annotation[cat] = 1 (human GT).
Run on annotation_ag2_filtered.jsonl (does-not-match entries already removed).

Output format per record:
  {
    "trace_id": 3,
    "present": {"1.1": 0, "1.2": 0, ...},   # 1 if category in onset
    "onset":   {"2.2": 1, "3.1": 2, ...},    # earliest step index per category
    "count":   {"2.2": 1, "3.1": 1, ...}     # n error entries per category
  }

Usage:
    python ag2_to_onsets.py
    python ag2_to_onsets.py --input ../data/annotation/annotation_ag2_filtered.jsonl --out_path data/onsets.jsonl
"""

import argparse
import json
import os
from collections import defaultdict


MAST_CATEGORIES = [
    "1.1", "1.2", "1.3", "1.4", "1.5",
    "2.1", "2.2", "2.3", "2.4", "2.6",
    "3.1", "3.2", "3.3",
]


def step_rank(step_id: str) -> int:
    """Convert 'step_04' → 4."""
    try:
        return int(step_id.split("_")[1])
    except (IndexError, ValueError):
        return -1


def main():
    ap = argparse.ArgumentParser(description="Convert AG2 annotations to onsets.jsonl")
    ap.add_argument("--input", default="../data/annotation/annotation_ag2_filtered.jsonl",
                    help="Path to annotation_ag2_filtered.jsonl")
    ap.add_argument("--out_path", default="data/onsets.jsonl",
                    help="Output onsets.jsonl path")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))

    written = 0
    with open(args.out_path, "w") as out:
        for r in records:
            trace_id = r["trace_id"]
            gt_cats = {k for k, v in r["mast_annotation"].items() if v == 1}

            # Map each GT category to its earliest step rank
            category_ranks = defaultdict(list)
            category_count = defaultdict(int)
            for e in r["errors"]:
                cat_key = e["category"].split()[0]   # "2.2 Fail..." → "2.2"
                if cat_key not in gt_cats:
                    continue
                loc = e.get("location", "")
                rank = step_rank(loc)
                if rank >= 0:
                    category_ranks[cat_key].append(rank)
                    category_count[cat_key] += 1

            onset = {cat: min(ranks) for cat, ranks in category_ranks.items() if ranks}
            present = {cat: (1 if cat in onset else 0) for cat in MAST_CATEGORIES}
            count = {cat: category_count.get(cat, 0) for cat in MAST_CATEGORIES}

            out.write(json.dumps({
                "trace_id": trace_id,
                "present": present,
                "onset": onset,
                "count": count,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"✓ Wrote {written} onset records → {args.out_path}")

    # Quick summary
    cat_counts = defaultdict(int)
    with open(args.out_path) as f:
        for line in f:
            r = json.loads(line)
            for cat, v in r["present"].items():
                if v == 1:
                    cat_counts[cat] += 1
    print("\nCategory presence counts:")
    for cat in MAST_CATEGORIES:
        print(f"  {cat}: {cat_counts[cat]}")


if __name__ == "__main__":
    main()
