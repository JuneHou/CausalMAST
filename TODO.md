# MAST Evaluation — Experiment TODO

All experiments use `data/annotation/annotation_ag2_filtered.jsonl` (393 traces) unless noted as sampled.
Best current setup: **code+name edge format, E3 with-graph, causal_only** (Mistral W-F1=0.5295).

---

## Model Alignment with TRAIL Benchmark

TRAIL uses Gemini-2.5-Flash (closed-source) and Mistral-Small-3.1-24B (open-source).
MAST replaces Gemini with GPT-4o and o1. Mistral is shared across both benchmarks.

| Role | TRAIL model | MAST model | Status |
|---|---|---|---|
| Open-source | Mistral-Small-3.1-24B | Mistral-Small-3.1-24B | ✓ done — best W-F1=0.5295 |
| Open-source (thinking) | — | QwQ-32B | ✓ done — best W-F1=0.1717 (E4) |
| Closed-source (standard) | Gemini-2.5-Flash | GPT-4o | partial — with-graph done (0.141), graph-inject pending |
| Closed-source (thinking) | — | o1 | partial — baseline only (0.0908 on 100 traces) |

---

## Experiment Status Matrix

All graph experiments use causal_only edges (7 intervention-validated edges).
"code+name" = `1.1(Disobey Task Specification) -> 3.3(No or Incorrect Verification)  (strength: X.XX)`

| Model | Baseline | +with-graph code+name | +graph-inject code+name | Notes |
|---|---|---|---|---|
| Mistral-Small-24B | ✓ 0.3773 | ✓ **0.5295** | ✓ 0.3701 | vLLM; all done |
| Gemma-3-27B | ✓ 0.1419 | ✓ 0.1323 | ✓ 0.1412 | vLLM; done |
| GPT-oss 20B | ✓ 0.1766 | ✓ 0.1872 | ✓ 0.1884 | vLLM; done |
| QwQ-32B | ✓ 0.1608 | ✓ 0.1513 | ✗ codename not run (T9); non-codename=0.1717 | vLLM |
| GPT-4o | ✓ 0.0894 | ✓ 0.1410 | ✗ not run (T2) | API |
| o1 | ✓ 0.0908† | ✗ | ✗ | API; omitted from paper |

† Scored on 100-trace stratified sample only.

---

## Direction Analysis

### What's confirmed
- **Mistral + E3 code+name is the best setup** (0.5295). The causal graph drives large recall gains on under-detected categories.
- **GPT-4o and o1 have fundamental RLHF conservatism** — causal graph helps GPT-4o slightly (+0.047) but doesn't fix the root issue. o1 is even worse (0.091), ruling it out as a viable judge without structural prompt changes.
- **Thinking models don't outperform instruction-tuned models here** — QwQ-32B (0.1717) is well below Mistral (0.5295). Reasoning budget increases conservatism and reduces recall.
- **QwQ-32B is the only model where E4 > E3 > baseline** — targeted 2nd-pass injection fits thinking model reasoning flow better than upfront context flooding.

### Highest-value next experiments
1. **o1 with-graph codename** (T4 completion) — expensive; GPT-4o with-graph showed graph can help slightly (+0.047); o1 might respond similarly. Worth running if cost is acceptable.
2. **GPT-4o graph-inject codename** — low priority; with-graph only got to 0.141; unlikely to improve much.

### Dead ends
- Gemma: under-detection not fixable by edge format; deprioritized.
- GPT-oss 120B: incomplete run, outperformed by smaller models; deprioritized.
- Graph-inject for non-thinking models: consistently below E3; no point running more.
- Full stability graph (t0.5) for Mistral: **done, worse** (0.5237 vs 0.5295). Extra edges add noise to 3.3 (-0.084 F1). causal_only edges are already optimal.
- E4 + full stability graph: not worth running — E3 t0.5 < E3 causal_only, and E4 already loses ~0.16 to E3 for Mistral.
- Detection-oriented prompt (eval_detect): **done, worse**. Baseline -0.007, E3 -0.157. Evidence requirement (section C: step IDs + quotes) collapses recall for subtle categories (2.6, 3.1, 2.3, 1.1) and eliminates graph benefit. Original prompt is better.

---

## Task List (Priority Order)

### T9 — QwQ-32B: graph-inject codename + thinking (new run)

Existing QwQ +GI used `eval/thinking/run_eval_graph_inject.py` (non-codename).
Need codename version via `full_run_eval_graph_inject.py` with `--enable_thinking`:

```bash
CUDA_VISIBLE_DEVICES=<gpus> python eval/full_run_eval_graph_inject.py \
    --model <qwq_model_path> \
    --model_tag QwQ-32B \
    --causal_only \
    --enable_thinking \
    --output_dir outputs_think
```

Expected output: `outputs_think/QwQ-32B-yesno-graph-inject-codename-causal_only-thinking/`

Score after run:

```bash
python eval/calculate_scores_yesno.py \
    --pred_dir outputs_think/QwQ-32B-yesno-graph-inject-codename-causal_only-thinking
```

---

### T10 — Mistral: +CG and +GI with t≥0.4 stability graph (new geomean scoring)

Previous t0.5 runs used raw stability frequency as strength — now replaced with geomean(P(B|A), PR_delta).
t≥0.4 covers 11/13 error types (adds 3.3 vs t≥0.5; 1.2 and 3.2 uncoverable at any threshold).
14 edges at t≥0.4 vs 11 at t≥0.5.

```bash
# E3 +CG
CUDA_VISIBLE_DEVICES=<gpus> python eval/full_run_eval_with_graph.py \
    --edge_threshold 0.4 \
    --output_dir outputs_corr

# E4 +GI
CUDA_VISIBLE_DEVICES=<gpus> python eval/full_run_eval_graph_inject.py \
    --edge_threshold 0.4 \
    --output_dir outputs_corr
```

Expected outputs:
- `outputs_corr/mistralai-Mistral-Small-3.1-24B-v2-yesno-with-graph-codename-t0.4/`
- `outputs_corr/mistralai-Mistral-Small-3.1-24B-v2-yesno-graph-inject-codename-t0.4/`

Score after each run:
```bash
python eval/calculate_scores_yesno.py --pred_dir outputs_corr/<subdir>
```

Note: old t0.5 runs in `outputs_corr/` used raw frequency as strength — not comparable to new geomean scoring.

---

### T2 (partial) — GPT-4o: graph-inject code+name (low priority)

```bash
python eval/full_run_eval_graph_inject_api.py \
    --model openai/gpt-4o \
    --causal_only \
    --output_dir outputs_full_api
```

Expected output: `outputs_full_api/openai-gpt-4o-yesno-graph-inject-codename-causal_only/`
Low priority: with-graph only got to 0.141; graph-inject unlikely to improve significantly.


## o1 Sampling Design

o1 is expensive — run on a representative 100-trace sample instead of all 393.

**Sample stats** (saved to `data/o1_sample_indices.json`):

| Category | Full rate | Sample rate | Dev | Sample# | Full# |
|---|---|---|---|---|---|
| 1.1 | 0.438 | 0.440 | +0.002 | 44 | 172 |
| 1.2 | 0.018 | 0.020 | +0.002 | 2 | 7 |
| 1.3 | 0.552 | 0.560 | +0.008 | 56 | 217 |
| 1.4 | 0.112 | 0.110 | -0.002 | 11 | 44 |
| 1.5 | 0.440 | 0.440 | -0.000 | 44 | 173 |
| 2.1 | 0.074 | 0.070 | -0.004 | 7 | 28 |
| 2.2 | 0.422 | 0.420 | -0.002 | 42 | 166 |
| 2.3 | 0.326 | 0.330 | +0.004 | 33 | 128 |
| 2.4 | 0.053 | 0.050 | -0.003 | 5 | 21 |
| 2.6 | 0.702 | 0.700 | -0.002 | 70 | 276 |
| 3.1 | 0.326 | 0.320 | -0.006 | 32 | 128 |
| 3.2 | 0.214 | 0.210 | -0.004 | 21 | 84 |
| 3.3 | 0.293 | 0.290 | -0.003 | 28 | 115 |

---

## Scoring

```bash
# After each run completes:
python eval/calculate_scores_yesno.py --pred_dir <output_dir>/<model_subdir>

# For o1 (sampled): needs --sample_indices flag or filter GT manually before scoring.
```

---

## Final Comparison Table (current state)

| Model | Type | Baseline | +with-graph code+name | +graph-inject code+name |
|---|---|---|---|---|
| Mistral-Small-24B | open-source | 0.3773 | **0.5295** | 0.3701 |
| Gemma-3-27B | open-source | 0.1419 | 0.1323 | — |
| GPT-oss 20B | open API | 0.1766 | 0.1872 | 0.1884 |
| QwQ-32B | open thinking | 0.1608 | 0.1513 | **0.1717** |
| GPT-4o | closed-source | 0.0894 | 0.1410 | TBD |
| o1 | closed thinking | 0.0908† | TBD | — |

† Scored on 100-trace sample only; not directly comparable to full-393 numbers.
