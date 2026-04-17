"""
run_causal_pipeline.py — Run the full MAST-AG2 causal graph pipeline end-to-end.

Executes in order:
  0. ag2_to_onsets.py         (annotation_ag2_filtered.jsonl → onsets.jsonl)
  1. CAPRI/1_build_order_pairs.py
  2. CAPRI/2_suppes_screen.py
  3. CAPRI/3_capri_prune.py
  4. CAPRI/4_bootstrap_stability.py
  5. CAPRI/5_shuffle_control.py
  6. CAPRI/6_export_hierarchy.py

Usage:
    python run_causal_pipeline.py
    python run_causal_pipeline.py --n_bootstrap 200 --stability_threshold 0.5
    python run_causal_pipeline.py --skip_shuffle   # skip shuffle control
    python run_causal_pipeline.py --start_step 2   # resume from step 2
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd: list, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n✗ Step failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser(description="Run the full MAST-AG2 causal graph pipeline")

    # I/O paths
    ap.add_argument("--input", default="../data/annotation/annotation_ag2_filtered.jsonl",
                    help="Input annotation JSONL (default: annotation_ag2_filtered.jsonl)")
    ap.add_argument("--data_dir", default="data",
                    help="Directory for intermediate data files (default: data/)")
    ap.add_argument("--out_dir", default="outputs",
                    help="Directory for output files (default: outputs/)")

    # Suppes parameters
    ap.add_argument("--min_precedence", type=float, default=0.55,
                    help="Suppes: min fraction of traces where A precedes B (default: 0.55)")
    ap.add_argument("--min_pr_delta", type=float, default=0.05,
                    help="Suppes: min probability-raising delta (default: 0.05)")
    ap.add_argument("--min_joint", type=int, default=3,
                    help="Suppes: min traces where both A and B co-occur (default: 3)")

    # CAPRI parameters
    ap.add_argument("--criterion", choices=["BIC", "AIC"], default="BIC",
                    help="CAPRI pruning criterion (default: BIC)")
    ap.add_argument("--max_parents", type=int, default=None,
                    help="CAPRI: max parents per node (default: no limit)")

    # Bootstrap parameters
    ap.add_argument("--n_bootstrap", type=int, default=100,
                    help="Number of bootstrap samples (default: 100)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")

    # Shuffle control
    ap.add_argument("--n_shuffles", type=int, default=50,
                    help="Number of shuffle iterations for negative control (default: 50)")
    ap.add_argument("--skip_shuffle", action="store_true",
                    help="Skip the shuffle control step")

    # Hierarchy
    ap.add_argument("--stability_threshold", type=float, default=0.3,
                    help="Min bootstrap stability for hierarchy export (default: 0.3)")

    # Resume
    ap.add_argument("--start_step", type=int, default=0,
                    help="Resume from this step (0=onset, 1=order_pairs, 2=suppes, "
                         "3=capri, 4=bootstrap, 5=shuffle, 6=hierarchy)")

    args = ap.parse_args()

    py = sys.executable
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Derived paths
    onsets       = os.path.join(args.data_dir, "onsets.jsonl")
    order_pairs  = os.path.join(args.data_dir, "order_pairs.jsonl")
    suppes_graph = os.path.join(args.out_dir, "suppes_graph.json")
    capri_graph  = os.path.join(args.out_dir, "capri_graph.json")
    stability_csv = os.path.join(args.out_dir, "edge_stability.csv")
    stability_json = os.path.join(args.out_dir, "edge_stability.json")
    shuffle_out  = os.path.join(args.out_dir, "controls_shuffle.json")
    hierarchy    = os.path.join(args.out_dir, "hierarchy_levels.json")

    steps = []

    # Step 0 — onsets
    steps.append((0, "Step 0: Build onsets from annotations", [
        py, str(HERE / "ag2_to_onsets.py"),
        "--input", args.input,
        "--out_path", onsets,
    ]))

    # Step 1 — order pairs
    steps.append((1, "Step 1: Build order pairs", [
        py, str(HERE / "CAPRI" / "1_build_order_pairs.py"),
        "--in_path", onsets,
        "--out_path", order_pairs,
    ]))

    # Step 2 — Suppes screen
    steps.append((2, "Step 2: Suppes probabilistic causation screen", [
        py, str(HERE / "CAPRI" / "2_suppes_screen.py"),
        "--in_path", onsets,
        "--out_path", suppes_graph,
        "--min_precedence", str(args.min_precedence),
        "--min_pr_delta", str(args.min_pr_delta),
        "--min_joint", str(args.min_joint),
    ]))

    # Step 3 — CAPRI prune
    capri_cmd = [
        py, str(HERE / "CAPRI" / "3_capri_prune.py"),
        "--onsets_path", onsets,
        "--suppes_path", suppes_graph,
        "--out_path", capri_graph,
        "--criterion", args.criterion,
    ]
    if args.max_parents is not None:
        capri_cmd += ["--max_parents", str(args.max_parents)]
    steps.append((3, "Step 3: CAPRI DAG pruning (BIC/AIC hill-climbing)", capri_cmd))

    # Step 4 — Bootstrap stability
    steps.append((4, f"Step 4: Bootstrap stability ({args.n_bootstrap} samples)", [
        py, str(HERE / "CAPRI" / "4_bootstrap_stability.py"),
        "--onsets_path", onsets,
        "--suppes_path", suppes_graph,
        "--capri_path", capri_graph,
        "--out_path", stability_csv,
        "--n_bootstrap", str(args.n_bootstrap),
        "--seed", str(args.seed),
    ]))

    # Step 5 — Shuffle control
    if not args.skip_shuffle:
        steps.append((5, f"Step 5: Shuffle control ({args.n_shuffles} iterations)", [
            py, str(HERE / "CAPRI" / "5_shuffle_control.py"),
            "--onsets_path", onsets,
            "--suppes_path", suppes_graph,
            "--out_path", shuffle_out,
            "--n_shuffles", str(args.n_shuffles),
            "--seed", str(args.seed),
        ]))

    # Step 6 — Export hierarchy
    steps.append((6, "Step 6: Export hierarchy levels", [
        py, str(HERE / "CAPRI" / "6_export_hierarchy.py"),
        "--capri_path", capri_graph,
        "--stability_path", stability_json,
        "--out_path", hierarchy,
        "--stability_threshold", str(args.stability_threshold),
    ]))

    # Run
    for step_num, label, cmd in steps:
        if step_num < args.start_step:
            print(f"  [skip] {label}")
            continue
        run(cmd, label)

    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"{'='*60}")
    print(f"  Suppes graph:       {suppes_graph}")
    print(f"  CAPRI graph:        {capri_graph}")
    print(f"  Edge stability:     {stability_json}")
    print(f"  Hierarchy levels:   {hierarchy}")
    print()
    print("  Next: run the LLM evaluation")
    print(f"    python eval/run_eval.py --model openai/gpt-4o")
    print(f"    python eval/run_eval_with_graph.py --model openai/gpt-4o "
          f"--stability_path {stability_json}")


if __name__ == "__main__":
    main()
