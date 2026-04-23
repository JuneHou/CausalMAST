"""
eval/calculate_scores_yesno.py — Score yes/no predictions against MAST human GT.

Ground truth comes directly from mast_annotation in annotation_ag2_filtered.jsonl.
Predictions come from run_eval_yesno.py output (one JSON per trace with "predictions" dict).

Metrics (matching original MAST evaluation):
  - Per-category detection rate (fraction of traces where LLM predicts yes)
  - Per-category precision, recall, F1 vs. human GT
  - Macro F1, weighted F1 across all 13 categories

Usage (run from MAST/):
    python eval/calculate_scores_yesno.py \\
        --annotation data/annotation/annotation_ag2_filtered.jsonl \\
        --pred_dir outputs/openai-o1-yesno-baseline

    # Score all subdirs at once:
    python eval/calculate_scores_yesno.py \\
        --annotation data/annotation/annotation_ag2_filtered.jsonl \\
        --pred_dir outputs
"""

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, cohen_kappa_score


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


def load_gt(annotation_path: str) -> dict:
    """Load mast_annotation ground truth keyed by rec_id (0000, 0001, ...)."""
    gt = {}
    with open(annotation_path) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            rec_id = f"{idx:04d}"
            # mast_annotation may include "2.5" (always 0 for AG2); ignore it
            gt[rec_id] = {m: int(r["mast_annotation"].get(m, 0)) for m in MAST_MODES}
    return gt


def score_directory(pred_dir: str, gt: dict) -> dict | None:
    pred_files = glob.glob(os.path.join(pred_dir, "*.json"))
    if not pred_files:
        print(f"  No prediction files in {pred_dir}")
        return None

    y_true_rows = []
    y_pred_rows = []
    matched = 0

    for pf in sorted(pred_files):
        rec_id = os.path.splitext(os.path.basename(pf))[0]
        if rec_id not in gt:
            print(f"  Warning: {rec_id} not in GT — skipping")
            continue

        with open(pf) as f:
            pred = json.load(f)

        predictions = pred.get("predictions", {})
        gt_vec = [gt[rec_id].get(m, 0) for m in MAST_MODES]
        pred_vec = [int(predictions.get(m, 0)) for m in MAST_MODES]

        y_true_rows.append(gt_vec)
        y_pred_rows.append(pred_vec)
        matched += 1

    if matched == 0:
        print(f"  No matching GT/pred pairs in {pred_dir}")
        return None

    y_true = np.array(y_true_rows)  # (n_traces, 13)
    y_pred = np.array(y_pred_rows)

    weighted_f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_f1     = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    macro_prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec    = recall_score(y_true, y_pred, average="macro",    zero_division=0)

    # Per-category metrics (binary per label, matching MAST paper Table 2)
    cat_metrics = {}
    kappas = []
    accs   = []
    for i, mode in enumerate(MAST_MODES):
        tp      = int(np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1)))
        tn      = int(np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0)))
        fp      = int(np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1)))
        fn      = int(np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0)))
        support  = int(np.sum(y_true[:, i]))
        pred_pos = int(np.sum(y_pred[:, i]))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc  = (tp + tn) / matched

        try:
            kappa = cohen_kappa_score(y_true[:, i], y_pred[:, i])
        except Exception:
            kappa = 0.0

        kappas.append(kappa)
        accs.append(acc)

        cat_metrics[mode] = {
            "name":           MAST_NAMES[mode],
            "precision":      round(prec, 4),
            "recall":         round(rec, 4),
            "f1":             round(f1, 4),
            "accuracy":       round(acc, 4),
            "kappa":          round(kappa, 4),
            "support":        support,
            "pred_positives": pred_pos,
            "detection_rate": round(pred_pos / matched, 4),
        }

    macro_acc        = float(np.mean(accs))
    macro_kappa_A    = float(np.mean(kappas))          # per-label, then averaged
    kappa_B          = float(cohen_kappa_score(         # pooled over all labels×traces
                           y_true.flatten(), y_pred.flatten()))
    kappas_C = []
    for i in range(len(y_true)):
        try:
            kappas_C.append(cohen_kappa_score(y_true[i], y_pred[i]))
        except Exception:
            pass
    macro_kappa_C = float(np.mean(kappas_C)) if kappas_C else 0.0  # per-trace, then averaged

    return {
        "n_traces":        matched,
        "weighted_f1":     round(weighted_f1, 4),
        "macro_f1":        round(macro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall":    round(macro_rec, 4),
        "macro_accuracy":  round(macro_acc, 4),
        "kappa_per_label": round(macro_kappa_A, 4),   # per-label macro-avg (harsh; negative if miscalibrated)
        "kappa_pooled":    round(kappa_B, 4),          # pooled over all label×trace decisions
        "kappa_per_trace": round(macro_kappa_C, 4),   # per-trace macro-avg (likely matches paper)
        "category_metrics": cat_metrics,
    }


def print_results(label: str, results: dict, pred_dir: str) -> None:
    print(f"\n{'='*80}")
    print(f"Results: {label}")
    print(f"{'='*80}")
    print(f"Traces scored:    {results['n_traces']}")
    print(f"Weighted F1:      {results['weighted_f1']:.4f}")
    print(f"Macro F1:         {results['macro_f1']:.4f}   "
          f"Macro P: {results['macro_precision']:.4f}   "
          f"Macro R: {results['macro_recall']:.4f}")
    print(f"Macro Accuracy:   {results['macro_accuracy']:.4f}")
    print(f"Kappa (per-label avg): {results['kappa_per_label']:.4f}   "
          f"Kappa (pooled): {results['kappa_pooled']:.4f}   "
          f"Kappa (per-trace avg): {results['kappa_per_trace']:.4f}")
    print()
    print(f"  {'Mode':<6} {'Name':<38} {'P':>6} {'R':>6} {'F1':>6} {'Acc':>6} {'κ':>6} {'Supp':>5} {'Det%':>6}")
    print(f"  {'-'*6} {'-'*38} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*6}")
    for mode, m in results["category_metrics"].items():
        print(f"  {mode:<6} {m['name']:<38} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} "
              f"{m['accuracy']:>6.3f} {m['kappa']:>6.3f} "
              f"{m['support']:>5} {m['detection_rate']*100:>5.1f}%")

    metrics_file = pred_dir + "-metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Metrics saved to {metrics_file}")


def main():
    ap = argparse.ArgumentParser(description="Score MAST yes/no predictions against human GT")
    ap.add_argument("--annotation",
                    default="data/annotation/annotation_ag2_filtered.jsonl",
                    help="Path to annotation_ag2_filtered.jsonl (ground truth)")
    ap.add_argument("--pred_dir", required=True,
                    help="Prediction directory or parent dir to score all subdirs")
    args = ap.parse_args()

    print(f"Loading GT from {args.annotation} ...")
    gt = load_gt(args.annotation)
    print(f"  {len(gt)} records loaded.")

    pred_path = args.pred_dir.rstrip("/")
    json_files = glob.glob(os.path.join(pred_path, "*.json"))
    subdirs    = [d for d in os.scandir(pred_path) if d.is_dir()]

    if json_files:
        # Single run directory
        results = score_directory(pred_path, gt)
        if results:
            print_results(os.path.basename(pred_path), results, pred_path)
    elif subdirs:
        # Parent directory — score each subdir
        for d in sorted(subdirs, key=lambda x: x.name):
            results = score_directory(d.path, gt)
            if results:
                print_results(d.name, results, d.path)
    else:
        print(f"No JSON files or subdirectories found in {pred_path}")


if __name__ == "__main__":
    main()
