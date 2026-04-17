# MAST-AG2 Causal Intervention Validation

Validates the causal edges in the CAPRI-pruned graph using do(A=0) counterfactual interventions,
adapted from the TRAIL benchmark's `causal/patch/` pipeline.

---

## What this pipeline does

For each causal edge A → B in the CAPRI graph:

1. **Find** traces where error A is annotated.
2. **Generate a patch** that minimally fixes A in the step where it occurs (do(A=0)).
3. **Simulate** the agent continuation after the patched step using an LLM.
4. **Judge 1**: Verify the patch actually eliminated A.
5. **Judge 2**: Determine what happened to B in the counterfactual trace.
6. **Aggregate** Δ(A→B) across all valid interventions.

An edge is **validated** if fixing A significantly reduces B (Δ < −threshold).

---

## Prerequisites

```bash
pip install litellm scikit-learn tqdm python-dotenv
echo "OPENAI_API_KEY=sk-..." > .env
```

---

## Quick start (full pipeline)

Run from `MAST/causal_graph/`:

```bash
python causal_valid/run_pipeline.py \
    --input ../annotation_ag2_filtered.jsonl \
    --causal_graph outputs/capri_graph.json \
    --model openai/gpt-4o
```

Output is written to `outputs/interventions/`.

---

## Step-by-step

All commands run from `MAST/causal_graph/`.

### Step 0 — Filter eligible traces

```bash
python causal_valid/filter_traces.py \
    --input ../annotation_ag2_filtered.jsonl \
    --causal_graph outputs/capri_graph.json
```

Output: `outputs/eligible_traces.json`

### Step 1 — Build A-instances and edge pairs

```bash
python causal_valid/case_builder.py \
    --input ../annotation_ag2_filtered.jsonl \
    --causal_graph outputs/capri_graph.json \
    --eligible_traces outputs/eligible_traces.json \
    --out_dir outputs/interventions
```

Output: `outputs/interventions/a_instances.jsonl`, `edge_pairs.jsonl`

### Step 2 — Generate patches (do(A=0))

```bash
python causal_valid/patch_generator.py \
    --cases outputs/interventions/a_instances.jsonl \
    --patch_library causal_valid/patch_library.json \
    --out_dir outputs/interventions \
    --model openai/gpt-4o
```

Output: `outputs/interventions/patch_results.jsonl`

### Step 3 — Simulate counterfactual continuation

```bash
python causal_valid/rerun_harness.py \
    --patch_results outputs/interventions/patch_results.jsonl \
    --input ../annotation_ag2_filtered.jsonl \
    --out_dir outputs/interventions \
    --model openai/gpt-4o \
    --max_steps_after 8
```

Output: `outputs/interventions/rerun_results.jsonl`

**Note**: For MAST (static traces), the rerun simulates assistant turns using an LLM
while preserving original orchestrator turns (code results, feedback). This differs from
TRAIL where the original agent is actually re-run with live tool calls.

### Step 4 — Judge 1: A-resolved?

```bash
python causal_valid/judge_a_resolved.py \
    --rerun_results outputs/interventions/rerun_results.jsonl \
    --patch_results outputs/interventions/patch_results.jsonl \
    --cases outputs/interventions/a_instances.jsonl \
    --out_dir outputs/interventions \
    --model openai/gpt-4o
```

Output: `outputs/interventions/a_resolved.jsonl`

### Step 5 — Judge 2: B-effect?

```bash
python causal_valid/judge_b_effect.py \
    --rerun_results outputs/interventions/rerun_results.jsonl \
    --a_resolved outputs/interventions/a_resolved.jsonl \
    --edge_pairs outputs/interventions/edge_pairs.jsonl \
    --out_dir outputs/interventions \
    --model openai/gpt-4o
```

Output: `outputs/interventions/b_effect.jsonl`

Effect labels: `disappeared | delayed | unchanged | earlier | weakened | strengthened | emerged | not_observable`

### Step 6 — Aggregate Δ(A→B)

```bash
python causal_valid/effect_aggregator.py \
    --b_effect outputs/interventions/b_effect.jsonl \
    --a_resolved outputs/interventions/a_resolved.jsonl \
    --patch_results outputs/interventions/patch_results.jsonl \
    --causal_graph outputs/capri_graph.json \
    --out_dir outputs/interventions \
    --threshold 0.15 \
    --min_n 1
```

Output: `outputs/interventions/effect_edges.json`

---

## Output directory structure

```
outputs/interventions/
├── eligible_traces.json         # Step 0
├── a_instances.jsonl            # Step 1: one per unique (trace_id, error_type, step)
├── edge_pairs.jsonl             # Step 1: one per (A-instance × B-type)
├── intervention_location_conflicts.jsonl  # Step 1: dedup log
├── patch_results.jsonl          # Step 2: LLM-generated patches
├── postcheck_failures.jsonl     # Step 2: failed patches
├── rerun_results.jsonl          # Step 3: counterfactual simulations
├── a_resolved.jsonl             # Step 4: Judge 1 verdicts
├── b_effect.jsonl               # Step 5: Judge 2 verdicts
└── effect_edges.json            # Step 6: Δ(A→B) per edge, validated flag
```

---

## Key differences from TRAIL

| Aspect | TRAIL | MAST |
|--------|-------|------|
| Trace format | OpenInference spans (span_id, LLM/TOOL/AGENT kinds) | `steps[{id, content}]` plain text |
| Annotation storage | Per-file `annotations_dir/{trace_id}.json` | Single `annotation_ag2_filtered.jsonl` |
| Error location | `span_id` (hash) | `step_XX` (sequential) |
| patch_side | `replace_span_output` or `replace_span_input` | `replace_step_content` (single mode) |
| Rerun | Live agent re-run with original tool results | LLM simulates assistant turns; orchestrator turns reuse original |
| Error categories | TRAIL taxonomy (20 types) | MAST taxonomy (13 categories: 1.1–3.3) |
| `patch_library.json` | 20 TRAIL operator families | 13 MAST categories |
| `B_DEFINITIONS` | TRAIL taxonomy | MAST taxonomy |

---

## Notes

- **min_n=1** (default): Edges are validated even with a single intervention.
  Raise to 3+ for stricter evidence (matches TRAIL's default of 3).
- **threshold=0.15**: An edge is validated if fixing A reduces B by ≥15 percentage points.
- **max_steps_after=8**: The rerun simulates up to 8 steps after t_A.
  Increase to 12 for longer traces if needed.
- **Category 2.5 missing**: MAST has no 2.5 category; 13 active categories used.
- **Rare categories**: 1.2 (7 traces) is not an A-type in the CAPRI graph (low support).
