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

### E3 — Graph-Guided Yes/No (causal graph injection)

- **Script**: `eval/run_eval_yesno_graph.py` *(to be created)*
- **Model**: `openai/gpt-4o` or `openai/o1`
- **Prompt**: Same @@ format + causal graph hints injected
  - Uses validated edges from `causal_graph/outputs/interventions/effect_edges.json`
  - Example hint: "When you detect 1.3 Step Repetition, also check for 1.5 Unaware of Termination Conditions (causal evidence, stability=0.82)"
- **Output dir**: `outputs/{model}-yesno-graph/`

**Status**: [ ] planned — create after E1/E2 establish baseline

---

### E4 — Alternative Prompt: JSON Format (Option B)

- **Script**: `eval/run_eval_yesno.py` with `--prompt_format json` flag *(to be added)*
- **Model**: `openai/gpt-4o`
- **Prompt**: Ask LLM to output `{"1.1": true, "1.2": false, ...}` directly (cleaner parsing)
- **Purpose**: Ablation — does prompt format affect prediction quality?
- **Output dir**: `outputs/openai-gpt-4o-yesno-json/`

**Status**: [ ] planned — after E1/E2

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
| E1 | o1 | — | — | |
| E2 | gpt-4o | — | — | |
| E3 | gpt-4o | — | — | + causal graph |
| E4 | gpt-4o | — | — | JSON prompt |

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
│   ├── run_eval_yesno.py                   ← E1/E2 prediction script
│   ├── calculate_scores_yesno.py           ← scoring script
│   └── experiments.md                      ← this file (lives at MAST root)
├── causal_graph/                           ← causal graph pipeline (Stage 1)
│   ├── data/gt/                            ← step-level GT (for causal_graph eval only)
│   └── outputs/                            ← causal graph outputs
└── outputs/                                ← all downstream eval outputs
    ├── openai-o1-yesno-baseline/
    ├── openai-gpt-4o-yesno-baseline/
    └── ...
```
