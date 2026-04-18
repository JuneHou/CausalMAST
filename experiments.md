# MAST Benchmark — Evaluation Plan & Experiment Log

All experiments use AG2 traces from `data/annotation/annotation_ag2_filtered.jsonl` (265 filtered traces removed).
All outputs save to `outputs/`.

---

## Task Definition

**Original MAST yes/no task**: Given a multi-agent trace, predict which of the 13 MAST error categories are present (binary yes/no per category).

- **Ground truth**: `mast_annotation` field — human-labeled binary annotations per trace
- **Evaluation unit**: per-trace binary vector of length 13
- **No step-level location prediction** — only category presence matters

---

## Evaluation Scripts

| Script | Purpose |
|---|---|
| `eval/run_eval_yesno.py` | LLM predicts yes/no per category (original MAST prompt format) |
| `eval/calculate_scores_yesno.py` | Scores predictions vs. human GT; reports P/R/F1 per category + weighted/macro F1 |

All scripts run from `MAST/` root.

---

## Experiment Versions

### E0 — MAST Paper Baseline (reference, not re-run)

- **Model**: o1
- **Prompt**: Original `llm_judge_pipeline.ipynb` format (yes/no per category, @@ delimiters)
- **GT**: Human `mast_annotation` labels (AG2 subset, 3 sources of 30 traces each = 90 traces in the paper)
- **Purpose**: Reference point for comparison; reproduced from the MAST paper

---

### E1 — Our Baseline (MAST prompt format, full filtered set)

- **Script**: `eval/run_eval_yesno.py`
- **Model**: `openai/o1` (same as paper for direct comparison)
- **Prompt**: Same @@ delimiter format as original MAST notebook
- **Data**: Full `annotation_ag2_filtered.jsonl` (393 AG2 traces)
- **GT**: `mast_annotation` field (human labels)
- **Output dir**: `outputs/openai-o1-yesno-baseline/`
- **Metrics**: Per-category P/R/F1, detection rate; weighted F1, macro F1

Run command:
```bash
cd /data/wang/junh/githubs/MAST
python eval/run_eval_yesno.py \
    --model openai/o1 \
    --input data/annotation/annotation_ag2_filtered.jsonl \
    --output_dir outputs \
    --max_workers 1
```

Score:
```bash
python eval/calculate_scores_yesno.py \
    --annotation data/annotation/annotation_ag2_filtered.jsonl \
    --pred_dir outputs/openai-o1-yesno-baseline
```

**Status**: [ ] pending

---

### E2 — Baseline with GPT-4o (cheaper, faster comparison)

- **Script**: `eval/run_eval_yesno.py`
- **Model**: `openai/gpt-4o`
- **Prompt**: Same as E1
- **Output dir**: `outputs/openai-gpt-4o-yesno-baseline/`

Run command:
```bash
python eval/run_eval_yesno.py \
    --model openai/gpt-4o \
    --input data/annotation/annotation_ag2_filtered.jsonl \
    --output_dir outputs \
    --max_workers 5
```

Score:
```bash
python eval/calculate_scores_yesno.py \
    --annotation data/annotation/annotation_ag2_filtered.jsonl \
    --pred_dir outputs/openai-gpt-4o-yesno-baseline
```

**Status**: [ ] pending

---

### E3 — Static Graph Context in Prompt (single-pass)

- **Script**: `eval/run_eval_with_graph.py` *(to be created)*
- **Model**: `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (vLLM)
- **Design**: Single-pass. The full validated causal graph (all edges + stability weights) is
  prepended to the prompt as **static background context** before the yes/no questions.
  The model reads the graph upfront and uses it during holistic error detection.
  - Uses all validated edges from `causal_graph/outputs/edge_stability.json`
  - Graph is presented as: `"1.3 → 1.5 [stability: 0.82]"` etc.
- **Purpose**: Does showing the model the global co-occurrence structure improve detection vs. no-graph baseline?
- **Output dir**: `outputs/{model}-yesno-with-graph/`

Run command:
```bash
CUDA_VISIBLE_DEVICES=4,5 python eval/run_eval_with_graph.py
```

**Status**: [ ] planned

---

### E4 — Dynamic Graph Injection (2-pass, full method)

- **Script**: `eval/run_eval_graph_inject.py` *(to be created)*
- **Model**: `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (vLLM)
- **Design**: 2-pass with dynamic subgraph propagation — fundamentally different from E3's
  static graph context.
- **Pass 1**: Same as E3/E4 — holistic detection + span index extraction.
- **Propagation**: `propagate_confidence()` takes Pass 1 detections as sources, walks the
  causal graph, and builds a **filtered subgraph**: only edges where
  `src ∈ detected` and `dst ∉ detected` and `boosted_score(dst) > threshold`.
  This identifies exactly which undetected categories are statistically likely given
  what was already found.
- **Pass 2**: Re-checks only the propagated target categories, with the filtered subgraph
  and span index injected. Only fires when Pass 1 detected at least one graph source.
  Results merged with Pass 1 output (deduplication).
- **Purpose**: Full proposed method — dynamic graph propagation surfaces missed errors
  that are causally downstream of already-detected ones.
- **Output dir**: `outputs/{model}-yesno-graph-inject/`

Run command:
```bash
CUDA_VISIBLE_DEVICES=4,5 python eval/run_eval_graph_inject.py
```

**Status**: [ ] planned — after E3

---

## Metrics Reference

| Metric | Description |
|---|---|
| Weighted F1 | sklearn weighted F1 across all 13 categories (weighted by GT support) |
| Macro F1 | Unweighted average F1 across all 13 categories |
| Per-category P/R/F1 | Binary classification metrics for each error type |
| Detection rate | % of traces where LLM predicts yes (regardless of GT) |

---

## Results Summary Table

*(to be filled in as experiments complete)*

| Experiment | Model | Weighted F1 | Macro F1 | Notes |
|---|---|---|---|---|
| E0 (MAST paper) | o1 | — | — | paper numbers TBD |
| E1 | o1 | — | — | pending |
| E2 (gpt-4o) | gpt-4o | 0.0946 | 0.0656 | baseline |
| E2 (mistral) | Mistral-Small-3.1-24B | 0.1190 | 0.0929 | baseline |
| E3 | Mistral-Small-3.1-24B | — | — | + static graph context in prompt (single-pass) |
| E4 | Mistral-Small-3.1-24B | — | — | + dynamic graph injection (2-pass) |

---

## Directory Structure

```
MAST/
├── data/
│   ├── annotation/
│   │   ├── annotation_ag2_filtered.jsonl   ← main input (filtered, 393 traces)
│   │   ├── annotation_ag2.jsonl            ← original (includes does-not-match)
│   │   └── annotation_openmanus.jsonl
│   └── raw/
│       └── MAD_full_dataset.json
├── eval/
│   ├── run_eval_yesno.py                   ← E1/E2 prediction script (OpenAI API)
│   ├── run_eval_yesno_vllm.py              ← E2 prediction script (vLLM in-process)
│   ├── run_eval_with_graph.py              ← E3/E4 (graph hint injection; span index)
│   ├── run_eval_graph_inject.py            ← E4 (2-pass dynamic graph injection)
│   ├── calculate_scores_yesno.py           ← scoring script (all experiments)
│   └── experiments.md                      ← this file (lives at MAST root)
├── causal_graph/                           ← causal graph pipeline (Stage 1)
│   ├── data/gt/                            ← step-level GT (for causal_graph eval only)
│   └── outputs/                            ← causal graph outputs
└── outputs/                                ← all downstream eval outputs
    ├── openai-o1-yesno-baseline/
    ├── openai-gpt-4o-yesno-baseline/
    └── ...
```
