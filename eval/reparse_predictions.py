"""
eval/reparse_predictions.py — Re-parse raw_response strings in existing output JSONs.

Use this when parse_response() is improved (e.g., to handle **bold** yes/no) and you
want to update predictions without re-running inference.

Usage (run from MAST/):
    python eval/reparse_predictions.py --pred_dir outputs_v2/mistralai-Mistral-Small-3.1-24B-v2-yesno-baseline
    python eval/reparse_predictions.py --pred_dir outputs_v2/mistralai-Mistral-Small-3.1-24B-v2-yesno-with-graph-causal_only
    python eval/reparse_predictions.py --pred_dir outputs/   # re-parse all subdirs at once
"""

import re
import json
import argparse
from pathlib import Path

MAST_MODES = ["1.1", "1.2", "1.3", "1.4", "1.5",
              "2.1", "2.2", "2.3", "2.4", "2.6",
              "3.1", "3.2", "3.3"]


def parse_response(response: str) -> dict:
    cleaned = response.strip()
    if cleaned.startswith("@@"):
        cleaned = cleaned[2:]
    if cleaned.endswith("@@"):
        cleaned = cleaned[:-2]
    cleaned = re.sub(r'\*\*(yes|no)\*\*', r'\1', cleaned, flags=re.IGNORECASE)
    result = {}
    for mode in MAST_MODES:
        patterns = [
            rf"{mode}\s*[^:\n]*:\s*(yes|no)",
            rf"{mode}\s+(yes|no)",
            rf"{mode}\s*\n\s*(yes|no)",
        ]
        found = False
        for pattern in patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                result[mode] = 1 if match.group(1).lower() == "yes" else 0
                found = True
                break
        if not found:
            result[mode] = 0
    return result


def reparse_dir(pred_dir: Path) -> int:
    json_files = sorted(pred_dir.glob("*.json"))
    if not json_files:
        return 0
    updated = 0
    for fpath in json_files:
        with open(fpath) as f:
            rec = json.load(f)
        if "raw_response" not in rec:
            continue
        new_preds = parse_response(rec["raw_response"])
        if new_preds != rec.get("predictions"):
            rec["predictions"] = new_preds
            with open(fpath, "w") as f:
                json.dump(rec, f, indent=2, ensure_ascii=False)
            updated += 1
    return updated


def main():
    ap = argparse.ArgumentParser(description="Re-parse raw_response strings in existing output JSONs")
    ap.add_argument("--pred_dir", required=True,
                    help="Path to a prediction directory (or parent dir to recurse one level)")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        print(f"Error: {pred_dir} does not exist")
        return

    # Check if this dir directly contains *.json files
    json_files = list(pred_dir.glob("*.json"))
    if json_files:
        updated = reparse_dir(pred_dir)
        print(f"{pred_dir.name}: {updated}/{len(json_files)} files updated")
    else:
        # Recurse one level into subdirectories
        subdirs = [d for d in sorted(pred_dir.iterdir()) if d.is_dir()]
        for subdir in subdirs:
            n = len(list(subdir.glob("*.json")))
            if n == 0:
                continue
            updated = reparse_dir(subdir)
            print(f"{subdir.name}: {updated}/{n} files updated")


if __name__ == "__main__":
    main()
