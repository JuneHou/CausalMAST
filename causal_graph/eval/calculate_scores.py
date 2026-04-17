"""
eval/calculate_scores.py — Score LLM predictions against MAST-AG2 GT annotations.

Metrics (per trace, then averaged):
  - location_accuracy:  |set(gt step_ids) ∩ set(pred step_ids)| / |set(gt step_ids)|
  - joint_accuracy:     |set(gt (step,cat) pairs) ∩ set(pred pairs)| / |set(gt pairs)|
  - weighted_F1:        sklearn weighted F1 on binary category presence vectors

GT directory:   data/gt/{trace_id}.json  (built by ag2_build_gt.py)
Pred directory: outputs/{model_tag}/     (built by run_eval.py or run_eval_with_graph.py)

Usage:
    python eval/calculate_scores.py --gt_dir data/gt --pred_dir outputs/openai-gpt-4o-baseline
    python eval/calculate_scores.py --gt_dir data/gt --pred_dir outputs/  # scores all subdirs
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from sklearn.metrics import f1_score


MAST_CATEGORIES = [
    "1.1 Disobey Task Specification",
    "1.2 Disobey Role Specification",
    "1.3 Step Repetition",
    "1.4 Loss of Conversation History",
    "1.5 Unaware of Termination Conditions",
    "2.1 Conversation Reset",
    "2.2 Fail to Ask for Clarification",
    "2.3 Task Derailment",
    "2.4 Information Withholding",
    "2.6 Action-Reasoning Mismatch",
    "3.1 Premature Termination",
    "3.2 Weak Verification",
    "3.3 No or Incorrect Verification",
]


def normalize_category(cat: str) -> str:
    """Normalize predicted category to the closest MAST_CATEGORIES entry."""
    if not cat:
        return ""
    cat = cat.strip()
    # Exact match
    if cat in MAST_CATEGORIES:
        return cat
    cat_lower = cat.lower().replace(" ", "")
    for std in MAST_CATEGORIES:
        if cat_lower == std.lower().replace(" ", ""):
            return std
    # Prefix match on category code (e.g. "2.2" → "2.2 Fail to Ask for Clarification")
    prefix = cat.split()[0] if cat.split() else ""
    for std in MAST_CATEGORIES:
        if std.startswith(prefix + " ") or std == prefix:
            return std
    # Substring match
    cat_no_spaces = cat.lower().replace(" ", "")
    for std in MAST_CATEGORIES:
        if cat_no_spaces in std.lower().replace(" ", ""):
            return std
    return cat  # unrecognized, keep as-is


def extract_json(text: str) -> dict:
    """Extract JSON object from LLM response text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"errors": []}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        # Try trimming from end
        s = match.group(0)
        while len(s) > 2:
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                s = s[:-1]
    return {"errors": []}


def calculate_metrics(gt: dict, pred: dict) -> dict:
    gt_errors = gt.get("errors", [])
    pred_errors = pred.get("errors", [])

    gt_cats = [normalize_category(e.get("category", "")) for e in gt_errors]
    gt_locs = [e.get("location", "") for e in gt_errors]
    pred_cats = [normalize_category(e.get("category", "")) for e in pred_errors]
    pred_locs = [e.get("location", "") for e in pred_errors]

    gt_pairs = set(zip(gt_locs, gt_cats))
    pred_pairs = set(zip(pred_locs, pred_cats))
    joint_accuracy = len(gt_pairs & pred_pairs) / len(gt_pairs) if gt_pairs else 0.0

    gt_loc_set = set(gt_locs)
    pred_loc_set = set(pred_locs)
    location_accuracy = len(gt_loc_set & pred_loc_set) / len(gt_loc_set) if gt_loc_set else 0.0

    y_true = np.zeros(len(MAST_CATEGORIES))
    y_pred = np.zeros(len(MAST_CATEGORIES))
    for cat in gt_cats:
        if cat in MAST_CATEGORIES:
            y_true[MAST_CATEGORIES.index(cat)] = 1
    for cat in pred_cats:
        if cat in MAST_CATEGORIES:
            y_pred[MAST_CATEGORIES.index(cat)] = 1

    return {
        "location_accuracy": location_accuracy,
        "joint_accuracy": joint_accuracy,
        "y_true": y_true,
        "y_pred": y_pred,
        "gt_cats": gt_cats,
        "pred_cats": pred_cats,
    }


def score_directory(gt_dir: str, pred_dir: str) -> dict | None:
    gt_files = glob.glob(os.path.join(gt_dir, "*.json"))
    if not gt_files:
        print(f"  No GT files in {gt_dir}")
        return None

    loc_acc_sum = 0.0
    joint_acc_sum = 0.0
    all_y_true = []
    all_y_pred = []
    n = 0

    for gt_file in gt_files:
        fname = os.path.basename(gt_file)
        pred_file = os.path.join(pred_dir, fname)
        if not os.path.exists(pred_file):
            continue

        with open(gt_file) as f:
            gt = json.load(f)
        with open(pred_file) as f:
            raw = f.read()
        pred = extract_json(raw)

        m = calculate_metrics(gt, pred)
        loc_acc_sum += m["location_accuracy"]
        joint_acc_sum += m["joint_accuracy"]
        all_y_true.append(m["y_true"])
        all_y_pred.append(m["y_pred"])
        n += 1

    if n == 0:
        print(f"  No matching pred files found in {pred_dir}")
        return None

    y_true_arr = np.vstack(all_y_true)
    y_pred_arr = np.vstack(all_y_pred)
    weighted_f1 = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)

    # Per-category stats
    cat_metrics = {}
    for i, cat in enumerate(MAST_CATEGORIES):
        tp = np.sum((y_true_arr[:, i] == 1) & (y_pred_arr[:, i] == 1))
        fp = np.sum((y_true_arr[:, i] == 0) & (y_pred_arr[:, i] == 1))
        fn = np.sum((y_true_arr[:, i] == 1) & (y_pred_arr[:, i] == 0))
        support = int(np.sum(y_true_arr[:, i]))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        cat_metrics[cat] = {"precision": prec, "recall": rec, "f1": f1, "support": support}

    return {
        "n_traces": n,
        "location_accuracy": loc_acc_sum / n,
        "joint_accuracy": joint_acc_sum / n,
        "weighted_f1": weighted_f1,
        "category_metrics": cat_metrics,
    }


def print_results(label: str, results: dict) -> None:
    print(f"\n{'='*70}")
    print(f"Results: {label}")
    print(f"{'='*70}")
    print(f"Traces scored:                 {results['n_traces']}")
    print(f"Location Accuracy (avg):       {results['location_accuracy']:.4f}")
    print(f"Joint Accuracy (avg):          {results['joint_accuracy']:.4f}")
    print(f"Weighted F1 (category):        {results['weighted_f1']:.4f}")
    print(f"\nPer-Category Breakdown:")
    print(f"  {'Category':<45} {'P':>6} {'R':>6} {'F1':>6} {'Supp':>6}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for cat, m in results["category_metrics"].items():
        if m["support"] > 0:
            print(f"  {cat:<45} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>6}")

    # Save to file
    metrics_file = pred_dir_global + "-metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Metrics saved to {metrics_file}")


pred_dir_global = ""


def main():
    global pred_dir_global
    ap = argparse.ArgumentParser(description="Score MAST-AG2 LLM predictions")
    ap.add_argument("--gt_dir", default="data/gt", help="Directory of GT JSON files")
    ap.add_argument("--pred_dir", default=None,
                    help="Prediction directory (or parent dir to score all subdirs)")
    args = ap.parse_args()

    if args.pred_dir is None:
        print("Please specify --pred_dir")
        return

    # Check if pred_dir is a direct output dir or a parent containing multiple runs
    json_files = glob.glob(os.path.join(args.pred_dir, "*.json"))
    subdirs = [d for d in os.scandir(args.pred_dir) if d.is_dir()]

    if json_files:
        # Single run directory
        pred_dir_global = args.pred_dir
        results = score_directory(args.gt_dir, args.pred_dir)
        if results:
            print_results(os.path.basename(args.pred_dir), results)
    elif subdirs:
        # Parent directory — score each subdir
        for d in sorted(subdirs, key=lambda x: x.name):
            pred_dir_global = d.path
            results = score_directory(args.gt_dir, d.path)
            if results:
                print_results(d.name, results)
    else:
        print(f"No JSON files or subdirectories found in {args.pred_dir}")


if __name__ == "__main__":
    main()
