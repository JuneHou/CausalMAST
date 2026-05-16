"""
Per-trace category breakdown for a MAST yes/no prediction directory.

MAST predictions are multi-label yes/no flags over 13 categories (no
location field), so this is the category-only analog of trail-benchmark's
_dump_location_hits.py:

  - per-trace cat_tp / cat_fp / cat_fn over the 13 MAST modes
  - per-trace precision / recall / F1
  - weighted F1 across all traces (matches calculate_scores_yesno.py)
  - per-predicted-category status (TP / FP)

Ground truth comes from data/annotation/annotation_ag2_filtered.jsonl
(same source as eval/calculate_scores_yesno.py).

Usage:
    python eval/_dump_category_hits.py PRED_DIR
    python eval/_dump_category_hits.py PARENT_DIR   # writes one JSON per subdir

Output `{pred_dir_name}-category_hits.json` is written next to the prediction dir
(same convention as the existing `-metrics.json` files).
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


MAST_MODES = ["1.1", "1.2", "1.3", "1.4", "1.5",
              "2.1", "2.2", "2.3", "2.4", "2.6",
              "3.1", "3.2", "3.3"]

MAST_NAMES = {
    "1.1": "Disobey Task Specification",
    "1.2": "Disobey Role Specification",
    "1.3": "Step Repetition",
    "1.4": "Loss of Conversation History",
    "1.5": "Unaware of Termination Conditions",
    "2.1": "Conversation Reset",
    "2.2": "Fail to Ask for Clarification",
    "2.3": "Task Derailment",
    "2.4": "Information Withholding",
    "2.6": "Action-Reasoning Mismatch",
    "3.1": "Premature Termination",
    "3.2": "Weak Verification",
    "3.3": "No or Incorrect Verification",
}

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ANNOTATION = REPO / "data" / "annotation" / "annotation_ag2_filtered.jsonl"


def load_gt(annotation_path: Path) -> dict:
    gt = {}
    with open(annotation_path) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            rec_id = f"{idx:04d}"
            gt[rec_id] = {m: int(r["mast_annotation"].get(m, 0)) for m in MAST_MODES}
    return gt


def dump_for_dir(pred_dir: Path, gt: dict) -> Path | None:
    pred_files = sorted(glob.glob(str(pred_dir / "*.json")))
    if not pred_files:
        print(f"  No prediction files in {pred_dir}")
        return None

    rows = []
    all_y_true: list[list[int]] = []
    all_y_pred: list[list[int]] = []

    for pf in pred_files:
        rec_id = os.path.splitext(os.path.basename(pf))[0]
        if rec_id not in gt:
            print(f"  Warning: {rec_id} not in GT — skipping")
            continue

        with open(pf) as f:
            pred = json.load(f)

        predictions = pred.get("predictions", {})
        gt_vec = [gt[rec_id].get(m, 0) for m in MAST_MODES]
        pred_vec = [int(predictions.get(m, 0)) for m in MAST_MODES]
        all_y_true.append(gt_vec)
        all_y_pred.append(pred_vec)

        gt_cats = {m for m, v in zip(MAST_MODES, gt_vec) if v == 1}
        pred_cats = {m for m, v in zip(MAST_MODES, pred_vec) if v == 1}

        tp_set = gt_cats & pred_cats
        fp_set = pred_cats - gt_cats
        fn_set = gt_cats - pred_cats
        tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_pred_errors = []
        for m in sorted(pred_cats, key=lambda x: MAST_MODES.index(x)):
            per_pred_errors.append({
                "mode": m,
                "name": MAST_NAMES[m],
                "status": "TP" if m in gt_cats else "FP",
            })
        missed_errors = []
        for m in sorted(fn_set, key=lambda x: MAST_MODES.index(x)):
            missed_errors.append({
                "mode": m,
                "name": MAST_NAMES[m],
            })

        def _named(modes):
            return [f"{m} {MAST_NAMES[m]}" for m in sorted(modes, key=lambda x: MAST_MODES.index(x))]

        rows.append({
            "rec_id": rec_id,
            "trace_id": pred.get("trace_id"),
            "n_gt_categories": len(gt_cats),
            "n_pred_categories": len(pred_cats),
            "n_correct_categories": tp,
            "cat_tp": tp,
            "cat_fp": fp,
            "cat_fn": fn,
            "cat_precision": precision,
            "cat_recall": recall,
            "cat_f1": f1,
            "exact_match": gt_cats == pred_cats,
            "gt_categories": _named(gt_cats),
            "pred_categories": _named(pred_cats),
            "correct_categories": _named(tp_set),
            "false_positives": _named(fp_set),
            "missed_categories": _named(fn_set),
            "pred_errors": per_pred_errors,
            "missed_errors": missed_errors,
        })

    if not rows:
        return None

    rows.sort(key=lambda r: (-r["cat_f1"], -r["cat_tp"], r["rec_id"]))

    exact = [r for r in rows if r["exact_match"]]
    partial = [r for r in rows if 0 < r["cat_f1"] < 1.0]
    missed_all = [r for r in rows if r["cat_tp"] == 0 and r["n_gt_categories"] > 0]
    avg_p = sum(r["cat_precision"] for r in rows) / len(rows)
    avg_r = sum(r["cat_recall"] for r in rows) / len(rows)
    avg_f1 = sum(r["cat_f1"] for r in rows) / len(rows)

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    out_json = pred_dir.parent / f"{pred_dir.name}-category_hits.json"
    with open(out_json, "w") as f:
        json.dump({
            "pred_dir": str(pred_dir),
            "annotation_path": str(DEFAULT_ANNOTATION),
            "n_traces": len(rows),
            "average_cat_precision_per_trace": avg_p,
            "average_cat_recall_per_trace": avg_r,
            "average_cat_f1_per_trace": avg_f1,
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
            "n_exact_match": len(exact),
            "n_partial_match": len(partial),
            "n_missed_all": len(missed_all),
            "traces": rows,
        }, f, indent=2)

    print(f"wrote {out_json}")
    print(f"  traces={len(rows)}  exact={len(exact)}  partial={len(partial)}  missed_all={len(missed_all)}")
    print(f"  avg per-trace F1={avg_f1:.4f}   weighted F1={weighted_f1:.4f}   macro F1={macro_f1:.4f}")
    return out_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pred_dir", type=Path,
                    help="Prediction directory, or parent containing multiple subdirs.")
    ap.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION,
                    help="Ground-truth JSONL (default: data/annotation/annotation_ag2_filtered.jsonl)")
    args = ap.parse_args()

    gt = load_gt(args.annotation)
    print(f"Loaded GT for {len(gt)} records from {args.annotation}")

    pred_path = args.pred_dir.resolve()
    subdirs = [Path(d.path) for d in os.scandir(pred_path) if d.is_dir()]
    json_files = glob.glob(str(pred_path / "*.json"))

    if subdirs:
        for d in sorted(subdirs, key=lambda x: x.name):
            dump_for_dir(d, gt)
    elif json_files:
        dump_for_dir(pred_path, gt)
    else:
        print(f"No JSON files or subdirectories found in {pred_path}")


if __name__ == "__main__":
    main()
