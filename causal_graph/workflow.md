# MAST-AG2 Causal Graph Pipeline — Workflow

End-to-end pipeline for building a causal graph of MAST error types from AG2 traces,
then using the graph to guide LLM-based error detection.

All commands are run from `MAST/causal_graph/`.

---

## Prerequisites

```bash
pip install litellm scikit-learn tqdm python-dotenv numpy
```

Set your API key in a `.env` file (or environment variable):

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

---

## Input Files

| File | Description |
|------|-------------|
| `../annotation_ag2.jsonl` | Original AG2 annotations (with does-not-match entries) |
| `../data/annotation/annotation_ag2_filtered.jsonl` | Filtered version — does-not-match entries removed **(use this)** |

---

## Stage 1: Build Causal Graph

### Option A — Run the full pipeline with one command

```bash
python run_causal_pipeline.py
```

Common overrides:

```bash
# More bootstrap samples, stricter stability threshold
python run_causal_pipeline.py --n_bootstrap 200 --stability_threshold 0.5

# Skip the shuffle control (faster)
python run_causal_pipeline.py --skip_shuffle

# Resume from a specific step (e.g. redo from bootstrap onwards)
python run_causal_pipeline.py --start_step 4
```

---

### Option B — Run steps individually

### Step 0 — Convert annotations to onsets

Converts `annotation_ag2_filtered.jsonl` into `data/onsets.jsonl` (one row per trace
with the earliest step rank per error category).

```bash
python ag2_to_onsets.py \
  --input ../data/annotation/annotation_ag2_filtered.jsonl \
  --out_path data/onsets.jsonl
```

### Step 1 — Build order pairs

For each trace, generates all `(A, B)` pairs where error A's onset precedes B's onset.

```bash
python CAPRI/1_build_order_pairs.py \
  --in_path data/onsets.jsonl \
  --out_path data/order_pairs.jsonl
```

### Step 2 — Suppes probabilistic causation screen

Keeps edges `A → B` where A statistically precedes B and raises the probability of B.

```bash
python CAPRI/2_suppes_screen.py \
  --in_path data/onsets.jsonl \
  --out_path outputs/suppes_graph.json \
  --min_precedence 0.55 \
  --min_pr_delta 0.05 \
  --min_joint 3
```

> `--min_joint 3` is appropriate for 393 traces. Raise to 5 if you want stricter edges.

### Step 3 — CAPRI pruning (BIC-based DAG search)

Prunes the Suppes candidate edges via hill-climbing with BIC score.

```bash
python CAPRI/3_capri_prune.py \
  --onsets_path data/onsets.jsonl \
  --suppes_path outputs/suppes_graph.json \
  --out_path outputs/capri_graph.json \
  --criterion BIC
```

### Step 4 — Bootstrap stability

Resamples traces 100 times and re-runs Suppes + CAPRI each time.
Assigns each edge a stability score (fraction of bootstrap samples it survives).

```bash
python CAPRI/4_bootstrap_stability.py \
  --onsets_path data/onsets.jsonl \
  --suppes_path outputs/suppes_graph.json \
  --capri_path outputs/capri_graph.json \
  --out_path outputs/edge_stability.csv \
  --n_bootstrap 100 \
  --seed 42
```

Output: `outputs/edge_stability.csv` and `outputs/edge_stability.json`

### Step 5 — Shuffle control (negative control)

Randomly permutes onset ranks and re-runs Suppes. Edges should mostly disappear,
confirming the structure is not an artifact.

```bash
python CAPRI/5_shuffle_control.py \
  --onsets_path data/onsets.jsonl \
  --suppes_path outputs/suppes_graph.json \
  --out_path outputs/controls_shuffle.json \
  --n_shuffles 50
```

### Step 6 — Export hierarchy

Assigns topological levels to the final DAG (after applying stability threshold).

```bash
python CAPRI/6_export_hierarchy.py \
  --capri_path outputs/capri_graph.json \
  --stability_path outputs/edge_stability.json \
  --out_path outputs/hierarchy_levels.json \
  --stability_threshold 0.3
```

---

## Stage 2: Prepare GT Files

Build per-trace ground-truth JSON files for scoring:

```bash
python ag2_build_gt.py \
  --input ../data/annotation/annotation_ag2_filtered.jsonl \
  --out_dir data/gt
```

---

## Stage 3: LLM Evaluation

### Baseline (no graph)

```bash
python eval/run_eval.py \
  --model openai/gpt-4o \
  --input ../data/annotation/annotation_ag2_filtered.jsonl \
  --output_dir outputs \
  --max_workers 5
```

Output: `outputs/openai-gpt-4o-baseline/{trace_id}.json`

### Graph-guided

Uses edges with bootstrap stability ≥ 0.5 (adjust `--edge_threshold` as needed).

```bash
python eval/run_eval_with_graph.py \
  --model openai/gpt-4o \
  --input ../data/annotation/annotation_ag2_filtered.jsonl \
  --stability_path outputs/edge_stability.json \
  --edge_threshold 0.5 \
  --output_dir outputs \
  --max_workers 5
```

Output: `outputs/openai-gpt-4o-graph_t0.5/{trace_id}.json`

---

## Stage 4: Score Predictions

Score a single run:

```bash
python eval/calculate_scores.py \
  --gt_dir data/gt \
  --pred_dir outputs/openai-gpt-4o-baseline
```

Score all runs at once (compares baseline vs. graph-guided):

```bash
python eval/calculate_scores.py \
  --gt_dir data/gt \
  --pred_dir outputs
```

Metrics reported per run:
- **Location Accuracy** — fraction of GT step locations correctly predicted
- **Joint Accuracy** — fraction of exact (step, error_type) pairs correctly predicted
- **Weighted F1** — binary category presence, weighted by support

Per-run metrics are saved to `outputs/{run_name}-metrics.json`.

---

## Output Directory Structure

```
causal_graph/
├── data/
│   ├── onsets.jsonl              # Step 0 output
│   ├── order_pairs.jsonl         # Step 1 output
│   └── gt/                       # Per-trace GT files
│       ├── 3.json
│       ├── 5.json
│       └── ...
└── outputs/
    ├── suppes_graph.json          # Step 2
    ├── capri_graph.json           # Step 3
    ├── edge_stability.csv         # Step 4
    ├── edge_stability.json        # Step 4
    ├── controls_shuffle.json      # Step 5
    ├── hierarchy_levels.json      # Step 6
    ├── openai-gpt-4o-baseline/    # Stage 3 baseline predictions
    ├── openai-gpt-4o-graph_t0.5/  # Stage 3 graph-guided predictions
    ├── openai-gpt-4o-baseline-metrics.json
    └── openai-gpt-4o-graph_t0.5-metrics.json
```

---

## Notes

- **Filtered vs. original**: Always use `annotation_ag2_filtered.jsonl` (265 "does not match"
  entries removed). The original `annotation_ag2.jsonl` is kept for reference/comparison.
- **Category 2.5 missing**: MAST taxonomy has no 2.5 category; the 13 active categories
  are 1.1–1.5, 2.1–2.4, 2.6, 3.1–3.3.
- **Rare categories**: 1.2 (7 traces) and 2.4 (21 traces) have limited support.
  Use `--min_joint 3` (not higher) in the Suppes screen to avoid dropping their edges.

---

## Rationale: Why AG2?

The causal analysis targets the AG2 subset specifically, based on three criteria:

**1. Sample size** — AG2 has 393 annotated traces, the largest in the MAST corpus.
CAPRI (BIC-penalized structure learning) requires sufficient co-occurrence counts per
edge to survive pruning; other frameworks fall well below this threshold.

**2. Structural regularity** — AG2 traces consist of a small number of discrete,
named-agent turns (avg 5.5 steps/trace) with explicit role assignments and clean
message boundaries, enabling unambiguous step-level error localization.

**3. Step separability** — AG2 steps are self-contained, semantically complete agent
utterances with clear role headers, making manual step-level annotation feasible.
Other frameworks require nontrivial segmentation before annotation is even possible
(see table below).

### Framework Comparison

| Framework | Task Benchmark | Annotated Traces | Avg Steps/Trace | Step Format | Manually Separable | Suitable for Causal Analysis |
|-----------|---------------|:---:|:---:|---|:---:|:---:|
| **AG2** | Math reasoning (GSM8K) | **393** | **5.5** | Structured dict (role + content) | **Yes** | **Yes** |
| MetaGPT | Software dev | 132 | — | Structured role turns | Yes | Borderline (n=132) |
| ChatDev | Software dev | 71 | 13.7 | Structured role turns (`[RoleName]` headers) | Yes | Borderline (n=71) |
| OpenManus | GAIA web tasks | 10 | 31.6 | Mixed application log + tool calls | Partially | No (n too small) |
| AppWorld | API/app interaction | 10 | 14.2 | Multi-task interaction dump | Partially | No (n too small) |
| HyperAgent | SWE-bench (GitHub issues) | 10 | ~3,917* | Flat log strings (undifferentiated) | No | No (n too small, no step structure) |
| MagenticOne | GAIA (3 levels) | ~165 | — | — | Unknown | No (unannotated) |

\* HyperAgent's "steps" are raw log lines, not semantic agent turns; ~3,917 is the avg log-line count per trace.

---

## Label Distribution (AG2 Task)

393 traces total; 375 labeled (18 unlabeled), 341 multi-label.
Total label instances: 1,279 — avg 3.25 labels/trace.

| Category | Name | Traces |
|----------|------|-------:|
| 1.1 | Disobey Task Specification | 132 |
| 1.2 | Disobey Role Specification | 6 |
| 1.3 | Step Repetition | 207 |
| 1.4 | Loss of Conversation History | 35 |
| 1.5 | Unaware of Termination Conditions | 149 |
| 2.1 | Conversation Reset | 22 |
| 2.2 | Fail to Ask for Clarification | 134 |
| 2.3 | Task Derailment | 100 |
| 2.4 | Information Withholding | 12 |
| 2.6 | Action-Reasoning Mismatch | 195 |
| 3.1 | Premature Termination | 106 |
| 3.2 | Weak Verification | 81 |
| 3.3 | No or Incorrect Verification | 100 |

Label-count distribution per trace (# labels → # traces):
0→18, 1→34, 2→120, 3→77, 4→67, 5→27, 6→21, 7→10, 8→8, 9→5, 10→2, 11→2, 12→2.

---

## Causal Validation Report

Scope: 393 traces → 310 eligible → 230 A-instances → 654 edge-pairs
Patch generation: 219/230 postcheck-passed (11 failed)
Judge-1 (A-resolved): 172/188 resolved
Judge-2 labels (overall):
emerged = 51, disappeared = 47, unchanged = 29,
not_observable = 171, weakened = 4, delayed = 4
Validated edges: 7/14
(Using gpt-4o for evaluation)

Test Results (per CAPRI edge):

**1.1 → 2.3** (Disobey Task Spec → Task Derailment): n=36, Δ = +0.278 — not validated
Effect labels: {'not_observable': 12, 'disappeared': 7, 'unchanged': 4, 'weakened': 3, 'emerged': 9, 'delayed': 1}

**1.1 → 2.6** (Disobey Task Spec → Action-Reasoning Mismatch): n=63, Δ = -0.127 — **validated ✓**
Effect labels: {'not_observable': 39, 'disappeared': 13, 'emerged': 8, 'unchanged': 3}

**1.3 → 1.5** (Step Repetition → Unaware of Termination Conditions): n=86, Δ = -0.081 — **validated ✓**
Effect labels: {'not_observable': 66, 'unchanged': 8, 'emerged': 8, 'disappeared': 4}

**1.4 → 1.5** (Loss of Conversation History → Unaware of Termination Conditions): n=10, Δ = -0.100 — **validated ✓**
Effect labels: {'unchanged': 5, 'not_observable': 3, 'disappeared': 1, 'delayed': 1}

**2.1 → 1.3** (Conversation Reset → Step Repetition): n=1, Δ = 0.000 — not validated
Effect labels: {'not_observable': 1}

**2.1 → 1.4** (Conversation Reset → Loss of Conversation History): n=1, Δ = 0.000 — not validated
Effect labels: {'not_observable': 1}

**2.1 → 3.3** (Conversation Reset → No or Incorrect Verification): n=1, Δ = -1.000 — **validated ✓**
Effect labels: {'not_observable': 1}

**2.2 → 1.4** (Fail to Ask for Clarification → Loss of Conversation History): n=4, Δ = +0.250 — not validated
Effect labels: {'unchanged': 2, 'emerged': 1, 'delayed': 1}

**2.2 → 1.5** (Fail to Ask for Clarification → Unaware of Termination Conditions): n=30, Δ = -0.300 — **validated ✓**
Effect labels: {'not_observable': 10, 'disappeared': 8, 'unchanged': 5, 'emerged': 6, 'delayed': 1}

**2.2 → 2.3** (Fail to Ask for Clarification → Task Derailment): n=24, Δ = +0.083 — not validated
Effect labels: {'not_observable': 13, 'emerged': 8, 'disappeared': 3}

**2.2 → 2.4** (Fail to Ask for Clarification → Information Withholding): n=2, Δ = -0.500 — **validated ✓**
Effect labels: {'emerged': 1, 'disappeared': 1}

**2.2 → 2.6** (Fail to Ask for Clarification → Action-Reasoning Mismatch): n=41, Δ = 0.000 — not validated
Effect labels: {'not_observable': 24, 'emerged': 9, 'disappeared': 5, 'unchanged': 2, 'weakened': 1}

**2.3 → 2.4** (Task Derailment → Information Withholding): n=0, Δ = N/A — not validated
Effect labels: {}

**2.3 → 3.1** (Task Derailment → Premature Termination): n=7, Δ = -0.429 — **validated ✓**
Effect labels: {'disappeared': 5, 'not_observable': 1, 'emerged': 1}
