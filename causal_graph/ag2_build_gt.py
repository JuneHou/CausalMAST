"""
Build per-trace GT JSON files from annotation_ag2_filtered.jsonl.

Output: data/gt/{idx:04d}.json per record, in the format expected by calculate_scores.py:
  {
    "errors": [
      {"category": "2.2 Fail to Ask for Clarification", "location": "step_01"},
      ...
    ]
  }

Files are keyed by row index (0000, 0001, ...) because trace_id is not unique in the
JSONL — multiple distinct traces share the same trace_id number.

Only includes errors where mast_annotation[cat] = 1.

Usage:
    python ag2_build_gt.py
    python ag2_build_gt.py --input ../data/annotation/annotation_ag2_filtered.jsonl --out_dir data/gt
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description="Build per-trace GT files from AG2 annotations")
    ap.add_argument("--input", default="../data/annotation/annotation_ag2_filtered.jsonl",
                    help="Path to annotation_ag2_filtered.jsonl")
    ap.add_argument("--out_dir", default="data/gt",
                    help="Output directory for per-trace GT JSON files")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))

    written = 0
    for idx, r in enumerate(records):
        gt_cats = {k for k, v in r["mast_annotation"].items() if v == 1}

        gt_errors = []
        seen_cats = set()
        for e in r.get("errors", []):
            cat_key = e["category"].split()[0]
            if cat_key not in gt_cats:
                continue
            if cat_key in seen_cats:
                continue  # keep only first location per category
            seen_cats.add(cat_key)
            gt_errors.append({
                "category": e["category"],
                "location": e.get("location", ""),
            })

        out_path = os.path.join(args.out_dir, f"{idx:04d}.json")
        with open(out_path, "w") as f:
            json.dump({"errors": gt_errors}, f, indent=2, ensure_ascii=False)
        written += 1

    print(f"✓ Wrote {written} GT files → {args.out_dir}/")


if __name__ == "__main__":
    main()
