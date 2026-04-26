"""
eval/sample_for_o1.py — Greedy multi-label stratified sample for o1 evaluation.

Selects 100 traces from the full 393 that preserve per-category GT positive rates.
At each step picks the candidate minimizing squared deviation from target rates.

Output: data/o1_sample_indices.json
"""

import json
import random
import numpy as np
from pathlib import Path

GT_FILE = "data/annotation/annotation_ag2_filtered.jsonl"
OUT_FILE = "data/o1_sample_indices.json"
MODES = ["1.1", "1.2", "1.3", "1.4", "1.5", "2.1", "2.2", "2.3", "2.4", "2.6",
         "3.1", "3.2", "3.3"]
SEED = 42
TARGET_N = 100
CANDIDATE_POOL = 50  # candidates sampled per greedy step


def main():
    gt = []
    with open(GT_FILE) as f:
        for line in f:
            r = json.loads(line)
            gt.append({m: int(r["mast_annotation"].get(m, 0)) for m in MODES})

    n_total = len(gt)
    target_rates = {m: sum(r[m] for r in gt) / n_total for m in MODES}

    print(f"Total traces: {n_total}")
    print(f"Target rates:")
    for m in MODES:
        print(f"  {m}: {target_rates[m]:.3f}  ({target_rates[m]*TARGET_N:.1f}/{TARGET_N})")

    random.seed(SEED)
    np.random.seed(SEED)

    indices = list(range(n_total))
    random.shuffle(indices)

    selected = []
    selected_set = set()

    for step in range(TARGET_N):
        remaining = [i for i in indices if i not in selected_set]
        pool = random.sample(remaining, min(CANDIDATE_POOL, len(remaining)))

        def score(idx):
            tentative = selected + [idx]
            k = len(tentative)
            return sum(
                (sum(gt[j][m] for j in tentative) / k - target_rates[m]) ** 2
                for m in MODES
            )

        best = min(pool, key=score)
        selected.append(best)
        selected_set.add(best)

    final_rates = {m: sum(gt[i][m] for i in selected) / TARGET_N for m in MODES}
    max_dev = max(abs(final_rates[m] - target_rates[m]) for m in MODES)

    print(f"\nSampled {TARGET_N} traces. Max rate deviation: {max_dev:.4f}")
    print(f"\nPer-category rates (sample vs full):")
    print(f"  {'Cat':>4}  {'Target':>7}  {'Sample':>7}  {'Dev':>7}  {'Sample#':>8}  {'Full#':>6}")
    for m in MODES:
        dev = final_rates[m] - target_rates[m]
        print(f"  {m:>4}  {target_rates[m]:>7.3f}  {final_rates[m]:>7.3f}  {dev:>+7.3f}"
              f"  {int(final_rates[m]*TARGET_N):>8}  {int(target_rates[m]*n_total):>6}")

    out = {
        "seed": SEED,
        "n": TARGET_N,
        "indices": sorted(selected),
        "rates": final_rates,
        "target_rates": target_rates,
        "max_deviation": max_dev,
    }
    Path(OUT_FILE).write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
