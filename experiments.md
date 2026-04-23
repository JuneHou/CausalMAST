# MAST Benchmark — Evaluation Experiment Log

All experiments use AG2 traces from `data/annotation/annotation_ag2_filtered.jsonl` (393 traces).
Ground truth: `mast_annotation` field (human-labeled binary per category).

---

## Task Definition

**Yes/no classification**: Given a multi-agent trace, predict which of the 13 MAST error categories are present (binary yes/no per category).

GT average: **3.97 positive labels per trace** (categories are not rare).

**GT support per category:**

| Code | Name | Support | GT Rate |
|---|---|---|---|
| 1.1 | Disobey Task Specification | 172 | 43.8% |
| 1.2 | Disobey Role Specification | 7 | 1.8% |
| 1.3 | Step Repetition | 217 | 55.2% |
| 1.4 | Loss of Conversation History | 44 | 11.2% |
| 1.5 | Unaware of Termination Conditions | 173 | 44.0% |
| 2.1 | Conversation Reset | 29 | 7.4% |
| 2.2 | Fail to Ask for Clarification | 166 | 42.2% |
| 2.3 | Task Derailment | 128 | 32.6% |
| 2.4 | Information Withholding | 21 | 5.3% |
| 2.6 | Action-Reasoning Mismatch | 276 | 70.2% |
| 3.1 | Premature Termination | 128 | 32.6% |
| 3.2 | Weak Verification | 84 | 21.4% |
| 3.3 | No or Incorrect Verification | 115 | 29.3% |

---

## Prompt Versions

| Version | Description |
|---|---|
| **v1** (backup) | Definitions → Examples → Answer format → Trace. Prompt reordered Apr 20 (`f187add`); definitions before the trace. Scripts preserved at `eval/backup/`. |
| **v2** (current) | Same order as v1 + bold-fix in parser: `re.sub(r'\*\*(yes\|no)\*\*', ...)` to handle `**yes**`/`**no**` markdown bold output. |

Both v1 and v2 have definitions before the trace. The functional difference is the parser fix only.

---

## Evaluation Scripts

| Script | Purpose |
|---|---|
| `eval/run_eval_yesno_vllm.py` | Baseline yes/no (vLLM in-process) |
| `eval/run_eval_with_graph.py` | E3: static causal graph context injected before yes/no questions |
| `eval/run_eval_graph_inject.py` | E4: 2-pass dynamic graph injection |
| `eval/code_name/run_eval_with_graph.py` | E3 variant: edges use `code(name)` format |
| `eval/code_name/run_eval_graph_inject.py` | E4 variant: edges use `code(name)` format |
| `eval/reparse_predictions.py` | Re-parse existing raw_response JSONs with updated parser (no re-inference) |
| `eval/calculate_scores_yesno.py` | Scores predictions vs. GT; reports P/R/F1 + weighted/macro F1 |

---

## Results Summary

| # | Model | Variant | Prompt | W-F1 | Macro F1 | Traces | Notes |
|---|---|---|---|---|---|---|---|
| 1 | Gemma 3 27B | baseline | v1 | **0.4366** | 0.3085 | 393 | |
| 2 | Gemma 3 27B | with-graph (causal_only) | v1 | **0.4594** | 0.3191 | 393 | |
| 3 | Gemma 3 27B | graph-inject (causal_only) | v1 | **0.4379** | 0.3098 | 393 | |
| 4 | Mistral Small 3.1 24B | baseline | v2 | **0.3773** | 0.2686 | 393 | |
| 5 | Mistral Small 3.1 24B | with-graph (causal_only) | v2 | **0.4758** | 0.3284 | 393 | best single result |
| 6 | Mistral Small 3.1 24B | graph-inject (causal_only) | v2 | **0.3715** | 0.2619 | 393 | below baseline |
| 7 | GPT-4o | baseline | v1 | 0.0946 | 0.0656 | 393 | model behavior issue |
| 8 | GPT-4o | baseline | v2 | 0.0894 | 0.0618 | 393 | prompt reorder had no effect |
| 9 | GPT-oss 120B | baseline | v1 | 0.1784 | 0.1274 | 393 | |
| 10 | GPT-oss 120B | with-graph (causal_only) | v1 | 0.1649 | 0.1225 | **285** | incomplete run |

---

## Per-Category Breakdown

### Gemma 3 27B — baseline

| Code | P | R | F1 | Det% |
|---|---|---|---|---|
| 1.1 | 0.362 | 0.122 | 0.183 | 14.8% |
| 1.2 | 0.000 | 0.000 | 0.000 | 0.0% |
| 1.3 | 0.552 | 1.000 | 0.712 | 100.0% |
| 1.4 | 0.131 | 0.773 | 0.224 | 65.9% |
| 1.5 | 0.452 | 0.740 | 0.561 | 72.0% |
| 2.1 | 0.062 | 0.035 | 0.044 | 4.1% |
| 2.2 | 0.000 | 0.000 | 0.000 | 0.0% |
| 2.3 | 0.325 | 0.992 | 0.489 | 99.5% |
| 2.4 | 0.055 | 0.952 | 0.105 | 91.9% |
| 2.6 | 0.702 | 1.000 | 0.825 | 100.0% |
| 3.1 | 0.327 | 1.000 | 0.492 | 99.8% |
| 3.2 | 0.234 | 0.941 | 0.374 | 86.0% |
| 3.3 | 0.000 | 0.000 | 0.000 | 0.0% |

Gemma **over-detects broadly**: 1.3/2.3/2.6/3.1 near 100% recall but low precision. Completely misses 1.2, 2.2, 3.3.

### Gemma 3 27B — with-graph (causal_only)

| Code | P | R | F1 | Det% | ΔF1 |
|---|---|---|---|---|---|
| 1.1 | 0.443 | 0.273 | 0.338 | 27.0% | **+0.155** |
| 1.3 | 0.552 | 1.000 | 0.712 | 100.0% | 0.000 |
| 1.5 | 0.449 | 0.873 | 0.593 | 85.5% | +0.032 |
| 2.2 | 0.571 | 0.024 | 0.046 | 1.8% | +0.046 |
| 2.6 | 0.702 | 1.000 | 0.825 | 100.0% | 0.000 |
| 3.3 | 0.000 | 0.000 | 0.000 | 0.0% | 0.000 |

Graph helped mainly 1.1 (+0.155 F1). Categories already at 100% recall are unchanged (no room to improve). Overall W-F1: 0.4366 → 0.4594 (+0.023).

### Gemma 3 27B — graph-inject (causal_only)

Nearly identical to baseline (W-F1: 0.4379 vs 0.4366). Pass 1 already saturates recall for most categories — no categories remain for Pass 2 to discover.

---

### Mistral Small 3.1 24B — baseline

| Code | P | R | F1 | Det% |
|---|---|---|---|---|
| 1.1 | 0.481 | 0.227 | 0.308 | 20.6% |
| 1.2 | 0.000 | 0.000 | 0.000 | 1.0% |
| 1.3 | 0.536 | 0.410 | 0.465 | 42.2% |
| 1.4 | 0.092 | 0.182 | 0.122 | 22.1% |
| 1.5 | 0.469 | 0.480 | 0.474 | 45.0% |
| 2.1 | 0.000 | 0.000 | 0.000 | 1.5% |
| 2.2 | 0.452 | 0.759 | 0.566 | 71.0% |
| 2.3 | 0.283 | 0.102 | 0.149 | 11.7% |
| 2.4 | 0.033 | 0.048 | 0.039 | 7.6% |
| 2.6 | 0.728 | 0.417 | 0.530 | 40.2% |
| 3.1 | 0.309 | 0.102 | 0.153 | 10.7% |
| 3.2 | 0.221 | 0.988 | 0.362 | 95.4% |
| 3.3 | 0.279 | 0.383 | 0.322 | 40.2% |

Mistral **under-detects** most categories (opposite of Gemma). Reasonable precision but low recall for 1.1, 1.3, 2.3, 2.6, 3.1.

### Mistral Small 3.1 24B — with-graph (causal_only)

| Code | P | R | F1 | Det% | ΔF1 |
|---|---|---|---|---|---|
| 1.1 | 0.429 | 0.442 | 0.435 | 45.0% | **+0.127** |
| 1.2 | 0.000 | 0.000 | 0.000 | 0.0% | 0.000 |
| 1.3 | 0.559 | 0.894 | 0.688 | 88.3% | **+0.223** |
| 1.4 | 0.098 | 0.477 | 0.162 | 54.7% | +0.040 |
| 1.5 | 0.451 | 0.873 | 0.595 | 85.2% | **+0.121** |
| 2.1 | 0.000 | 0.000 | 0.000 | 1.3% | 0.000 |
| 2.2 | 0.386 | 0.398 | 0.392 | 43.5% | -0.174 |
| 2.3 | 0.324 | 0.258 | 0.287 | 25.9% | **+0.138** |
| 2.4 | 0.015 | 0.095 | 0.026 | 33.6% | -0.013 |
| 2.6 | 0.692 | 0.790 | 0.738 | 80.2% | **+0.208** |
| 3.1 | 0.305 | 0.195 | 0.238 | 20.9% | +0.085 |
| 3.2 | 0.208 | 0.655 | 0.316 | 67.2% | -0.046 |
| 3.3 | 0.289 | 0.609 | 0.392 | 61.6% | **+0.070** |

Graph boosts recall for 1.1, 1.3, 1.5, 2.3, 2.6, 3.3. 2.2 declines (already high recall; graph pushes over-detection). Overall W-F1: 0.3773 → **0.4758** (+0.0985).

### Mistral Small 3.1 24B — graph-inject (causal_only)

| Code | P | R | F1 | Det% | ΔF1 vs baseline |
|---|---|---|---|---|---|
| 1.1 | 0.520 | 0.151 | 0.234 | 12.7% | -0.074 |
| 1.3 | 0.572 | 0.714 | 0.635 | 69.0% | +0.170 |
| 1.5 | 0.398 | 0.474 | 0.433 | 52.4% | -0.041 |
| 2.2 | 0.423 | 0.464 | 0.443 | 46.3% | -0.123 |
| 2.6 | 0.641 | 0.453 | 0.531 | 49.6% | +0.001 |
| 3.2 | 0.208 | 0.845 | 0.333 | 87.0% | -0.029 |
| 3.3 | 0.331 | 0.365 | 0.347 | 32.3% | +0.025 |

W-F1: 0.3715 — **below both baseline (0.3773) and with-graph (0.4758)**. 2-pass approach underperforms static context for this model and edge set.

---

### GPT-4o — baseline

| Prompt | W-F1 | Macro F1 | Avg response |
|---|---|---|---|
| v1 | 0.0946 | 0.0656 | ~319 chars |
| v2 | 0.0894 | 0.0618 | ~319 chars |

Prompt reorder had **zero effect**. Model gives one-sentence summaries and marks almost everything `no`. Only 3.2 (Weak Verification, 35%) detected at any meaningful rate. RLHF conservatism; non-reasoning model. Not a viable judge for this task without stronger intervention.

### GPT-oss 120B — baseline

W-F1=0.1784. Better than GPT-4o but well below Mistral and Gemma. Strong on 2.6 (F1=0.393) and 1.1 (F1=0.280). With-graph run incomplete (285/393) and slightly lower (W-F1=0.1649) — not comparable.

---

## Key Findings

### F1: Prompt order matters for open-weights models, not GPT-4o
Mistral **baseline** W-F1 improved stepwise: 0.119 (trace before definitions) → 0.239 (definitions first) → 0.377 (definitions first + bold parse fix). These are baseline-only numbers; no graph is involved. GPT-4o showed no change (0.0946 → 0.0894) under the same prompt reorder. GPT-4o's underperformance is model-intrinsic.

For the **with-graph** experiment, three fixes were applied simultaneously before re-running (markdown header bug, graph edge format name→code, bold parse fix), so the individual contribution of the format change cannot be isolated. The combined effect was 0.0291 → 0.4758.

### F2: Bold markdown parsing bug silently zeroed ~50% of Mistral predictions
~50% of Mistral outputs used `**yes**`/`**no**` formatting. The regex didn't match bold-wrapped values, silently parsing them as 0. Gemma, GPT-4o, GPT-oss all used plain text and were unaffected.
- **Fix**: `re.sub(r'\*\*(yes|no)\*\*', r'\1', ...)` added to `parse_response()` in all scripts.
- **Recovery tool**: `eval/reparse_predictions.py` — updates existing JSONs without re-running inference.
- **Impact**: outputs_v2 baseline 0.1564 → 0.3773; with-graph 0.0291 → 0.4758.

### F3: `#` markdown header in graph guidance triggered full markdown output mode
`format_graph_guidance()` originally started with `"# Causal Error Patterns..."`. This H1 header caused Mistral to respond in full markdown (`### Analysis of the Trace`, `#### C.`), ignoring `@@` delimiters — all predictions parsed as 0. Gemma was unaffected.
- **Fix**: Changed to plain `"CAUSAL ERROR PATTERNS..."` (no `#`).

### F4: Static graph context (E3) outperforms dynamic 2-pass injection (E4) for both models
| Model | Baseline | With-graph (E3) | Graph-inject (E4) |
|---|---|---|---|
| Gemma 3 27B | 0.4366 | **0.4594** | 0.4379 |
| Mistral Small 24B | 0.3773 | **0.4758** | 0.3715 |

E4 underperforms baseline for Mistral and barely beats baseline for Gemma. The static upfront context lets the model find errors it would otherwise miss; dynamic injection depends on Pass 1 having already detected the source error, which limits its reach. E3 is both simpler and more effective.

### F5: Gemma over-detects; Mistral under-detects — complementary failure patterns
Gemma: high recall, low precision (1.3/2.3/2.6/3.1 at ~100% recall). Misses 1.2, 2.2, 3.3 entirely.
Mistral: balanced but conservative. Misses 1.2, 2.1, 2.4 consistently. Good on 2.2, 3.2.
The two models have opposite biases. An ensemble or calibration approach could exploit both.

### F6: GPT-oss 120B (0.1784) underperforms Mistral Small 24B (0.3773)
Despite 5× more parameters. Possible causes: different RLHF alignment, API-based inference limitations, or the incomplete with-graph run. Requires further investigation.

---

## Bugs Found and Fixed

| Bug | Affected | Fix |
|---|---|---|
| `**yes**`/`**no**` bold not parsed | Mistral baseline + with-graph (outputs_v2) | `re.sub` in `parse_response()` in all scripts; `reparse_predictions.py` for existing outputs |
| `# Causal Error Patterns` triggered markdown mode | Mistral with-graph (before fix) | Changed to `CAUSAL ERROR PATTERNS` (plain text) |
| 2.2 example used `[step_XX]` format identical to real traces | Both runs while present | Permanently removed — format conflict breaks `@@` output |
| Phantom categories 1.6/2.5/2.7 in example answer | All scripts before `f187add` | Removed from example answer in all scripts |

---

## Pending / Planned

| Experiment | Script | Status |
|---|---|---|
| Mistral — code_name with-graph | `eval/code_name/run_eval_with_graph.py --causal_only` | planned |
| Mistral — code_name graph-inject | `eval/code_name/run_eval_graph_inject.py --causal_only` | planned |
| GPT-4o with-graph | re-run with new prompt + causal_only graph | planned |
| GPT-oss 120B with-graph | complete remaining 108 traces | planned |
| Patch and rescore old Mistral | `eval/reparse_predictions.py --pred_dir outputs/mistralai-Mistral-Small-3.1-24B-v2-yesno-baseline` | pending |

---

## Output Directory Map

```
outputs/                                   ← v1 prompt (bold fix NOT applied to parser)
  gemma-3-27b-it-yesno-baseline/                          W-F1=0.4366
  gemma-3-27b-it-yesno-with-graph-causal_only/            W-F1=0.4594
  gemma-3-27b-it-yesno-graph-inject-causal_only/          W-F1=0.4379
  gpt-oss-120b-yesno-baseline/                            W-F1=0.1784
  gpt-oss-120b-yesno-with-graph-causal_only/              W-F1=0.1649  (285/393)
  openai-gpt-4o-yesno-baseline/                           W-F1=0.0946

outputs_v2/                                ← v2 prompt (bold fix applied)
  mistralai-Mistral-Small-3.1-24B-v2-yesno-baseline/      W-F1=0.3773
  mistralai-Mistral-Small-3.1-24B-v2-yesno-with-graph-causal_only/   W-F1=0.4758
  mistralai-Mistral-Small-3.1-24B-v2-yesno-graph-inject-causal_only/ W-F1=0.3715
  openai-gpt-4o-yesno-baseline/                           W-F1=0.0894
```
