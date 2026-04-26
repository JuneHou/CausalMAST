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
| **v1** (backup) | Definitions → Examples → Answer format → Trace. Prompt reordered Apr 20 (`f187add`); definitions before the trace. Scripts preserved at `eval/backup/`. Graph edges use **name-only** format: `Disobey Task Specification -> No or Incorrect Verification`. |
| **v2** (current) | Same order as v1 + bold-fix in parser: `re.sub(r'\*\*(yes\|no)\*\*', ...)` to handle `**yes**`/`**no**` markdown bold output. Graph edges use **code-only** format: `1.1 -> 3.3`. |
| **output_full** | v2 prompt. Graph edges use **code+name** format: `1.1(Disobey Task Specification) -> 3.3(No or Incorrect Verification)`. |

Both v1 and v2 have definitions before the trace. v1→v2: parser fix + edge format changed from name-only to code-only. v2→output_full: edge format changed from code-only to code+name.

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
| 11 | Gemma 3 27B | baseline | v2 | 0.1419 | 0.0827 | 393 | outputs_v2; correct score — rows 1–3 used wrong annotation file* |
| 12 | Gemma 3 27B | with-graph code (causal_only) | v2 | 0.1489 | 0.0962 | 393 | outputs_v2 |
| 13 | Gemma 3 27B | graph-inject code (causal_only) | v2 | 0.1476 | 0.0880 | 393 | outputs_v2 |
| 14 | Gemma 3 27B | with-graph codename (causal_only) | v2 | 0.1323 | 0.0991 | 393 | output_full; worse than code format |
| 15 | GPT-oss 20B | baseline | v2 | 0.1766 | 0.1247 | 393 | outputs_v2 |
| 16 | GPT-oss 20B | with-graph code (causal_only) | v2 | 0.1643 | 0.1176 | 393 | outputs_v2; below baseline |
| 17 | GPT-oss 20B | graph-inject code (causal_only) | v2 | 0.1705 | 0.1233 | 393 | outputs_v2 |
| 18 | GPT-oss 20B | with-graph codename (causal_only) | v2 | 0.1872 | 0.1314 | 393 | output_full |
| 19 | GPT-oss 20B | graph-inject codename (causal_only) | v2 | 0.1884 | 0.1400 | 393 | output_full |
| 20 | Mistral Small 24B | with-graph codename (causal_only) | v2 | **0.5295** | 0.3722 | 393 | output_full; **best overall** |
| 21 | Mistral Small 24B | graph-inject codename (causal_only) | v2 | 0.3701 | 0.2586 | 393 | output_full; below baseline |
| 22 | GPT-4o | with-graph codename (causal_only) | v2 | 0.1410 | 0.1033 | 393 | outputs_full_api; graph lifts from 0.0894 but model still severely under-detects |
| 23 | o1 | baseline | v2 | 0.0908 | 0.0581 | **100** | outputs_o1; 100-trace sample; worse than GPT-4o; RLHF conservatism extreme |
| 24 | QwQ-32B | baseline (thinking) | v2 | 0.1608 | 0.1202 | 393 | outputs_think; best of the non-Mistral models |
| 25 | QwQ-32B | with-graph (causal_only, thinking) | v2 | 0.1513 | 0.1058 | 393 | outputs_think; graph hurts slightly vs baseline |
| 26 | QwQ-32B | graph-inject (causal_only, thinking) | v2 | **0.1717** | 0.1364 | 393 | outputs_think; **only model where E4 > E3 > baseline** |
| 27 | Mistral Small 24B | with-graph codename (t0.5, 27 edges) | v2 | 0.5237 | 0.3694 | 393 | outputs_corr; **below causal_only** — extra edges add noise |
| 28 | Mistral Small 24B | baseline | detect | 0.3699 | 0.2571 | 393 | outputs_detect; detect prompt; -0.007 vs old baseline |
| 29 | Mistral Small 24B | with-graph codename (causal_only) | detect | 0.3730 | 0.2811 | 393 | outputs_detect; detect prompt; **-0.157 vs old E3** — graph benefit eliminated |

\* Rows 1–3 used an older `annotation_ag2_filtered.jsonl` where every trace index mapped to a different conversation. Rows 11–14 are the corrected Gemma scores on the current annotation file.

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

### Mistral Small 3.1 24B — with-graph codename (causal_only) ← best overall

Edge format: `1.1(Disobey Task Specification) -> 3.3(No or Incorrect Verification)  (strength: X.XX)`

| Code | P | R | F1 | Det% | ΔF1 vs code format |
|---|---|---|---|---|---|
| 1.1 | 0.435 | 0.640 | 0.518 | 64.4% | **+0.083** |
| 1.2 | 0.000 | 0.000 | 0.000 | 0.0% | 0.000 |
| 1.3 | 0.552 | 0.899 | 0.684 | 89.8% | -0.004 |
| 1.4 | 0.085 | 0.409 | 0.141 | 53.9% | -0.021 |
| 1.5 | 0.450 | 0.913 | 0.603 | 89.3% | +0.008 |
| 2.1 | 0.000 | 0.000 | 0.000 | 0.5% | 0.000 |
| 2.2 | 0.434 | 0.657 | 0.523 | 63.9% | **+0.131** |
| 2.3 | 0.335 | 0.414 | 0.371 | 40.2% | **+0.084** |
| 2.4 | 0.043 | 0.381 | 0.077 | 47.6% | +0.051 |
| 2.6 | 0.714 | 0.870 | 0.784 | 85.5% | +0.046 |
| 3.1 | 0.329 | 0.422 | 0.370 | 41.7% | **+0.132** |
| 3.2 | 0.209 | 0.750 | 0.327 | 76.6% | +0.011 |
| 3.3 | 0.302 | 0.826 | 0.442 | 80.2% | **+0.050** |

Codename format drives large recall gains on 1.1, 2.2, 2.3, 3.1 by making edge targets unambiguous. Overall W-F1: 0.4758 → **0.5295** (+0.054).

### Mistral Small 3.1 24B — graph-inject codename (causal_only)

| Code | P | R | F1 | Det% | ΔF1 vs baseline |
|---|---|---|---|---|---|
| 1.1 | 0.520 | 0.151 | 0.234 | 12.7% | -0.074 |
| 1.3 | 0.565 | 0.737 | 0.640 | 72.0% | +0.175 |
| 1.5 | 0.403 | 0.468 | 0.433 | 51.1% | -0.041 |
| 2.2 | 0.433 | 0.470 | 0.451 | 45.8% | -0.115 |
| 2.6 | 0.645 | 0.475 | 0.547 | 51.7% | +0.017 |
| 3.2 | 0.207 | 0.845 | 0.333 | 87.3% | -0.029 |
| 3.3 | 0.292 | 0.330 | 0.310 | 33.1% | -0.012 |

W-F1: 0.3701 — below both baseline (0.3773) and with-graph codename (0.5295). Same E3 > E4 pattern as code format.

---

### Gemma 3 27B — baseline (v2, corrected)

| Code | P | R | F1 | Det% |
|---|---|---|---|---|
| 1.1 | 0.000 | 0.000 | 0.000 | 0.5% |
| 1.2 | 0.000 | 0.000 | 0.000 | 0.3% |
| 1.3 | 0.541 | 0.332 | 0.411 | 33.8% |
| 1.4 | 0.000 | 0.000 | 0.000 | 0.0% |
| 1.5 | 0.323 | 0.058 | 0.098 | 7.9% |
| 2.1 | 0.000 | 0.000 | 0.000 | 0.0% |
| 2.2 | 0.000 | 0.000 | 0.000 | 0.8% |
| 2.3 | 0.200 | 0.023 | 0.042 | 3.8% |
| 2.4 | 0.000 | 0.000 | 0.000 | 0.5% |
| 2.6 | 0.698 | 0.217 | 0.331 | 21.9% |
| 3.1 | 0.200 | 0.008 | 0.015 | 1.3% |
| 3.2 | 0.205 | 0.095 | 0.130 | 9.9% |
| 3.3 | 0.214 | 0.026 | 0.047 | 3.6% |

Severe under-detection on correct annotation file. Only 1.3 (34% det) and 2.6 (22% det) fire with any reliability. Completely misses 1.1, 1.2, 1.4, 2.1, 2.2, 2.4. Contrast with the inflated v1 numbers (rows 1–3) which used the wrong annotation file.

### Gemma 3 27B — with-graph code (causal_only, v2)

| Code | P | R | F1 | Det% | ΔF1 vs baseline |
|---|---|---|---|---|---|
| 1.1 | 1.000 | 0.006 | 0.012 | 0.3% | +0.012 |
| 1.3 | 0.483 | 0.327 | 0.390 | 37.4% | -0.021 |
| 1.5 | 0.333 | 0.075 | 0.123 | 9.9% | +0.025 |
| 2.2 | 0.333 | 0.012 | 0.023 | 1.5% | +0.023 |
| 2.3 | 0.333 | 0.086 | 0.137 | 8.4% | +0.095 |
| 2.6 | 0.714 | 0.163 | 0.265 | 16.0% | -0.066 |
| 3.2 | 0.250 | 0.131 | 0.172 | 11.2% | +0.042 |
| 3.3 | 0.269 | 0.061 | 0.099 | 6.6% | +0.052 |

Graph guidance provides minimal lift. W-F1: 0.1419 → 0.1489 (+0.007). 2.6 drops due to reduced detection rate. Base detection too sparse for graph to propagate meaningfully.

### Gemma 3 27B — graph-inject code (causal_only, v2)

Nearly identical to baseline. W-F1: 0.1476 vs baseline 0.1419 (+0.006). Per-category changes are negligible — Pass 1 rarely detects source nodes so Pass 2 almost never fires.

### Gemma 3 27B — with-graph codename (causal_only, output_full)

| Code | P | R | F1 | Det% | ΔF1 vs code format |
|---|---|---|---|---|---|
| 1.3 | 0.512 | 0.198 | 0.286 | 21.4% | -0.104 |
| 1.5 | 0.442 | 0.110 | 0.176 | 10.9% | +0.053 |
| 2.6 | 0.680 | 0.123 | 0.209 | 12.7% | -0.056 |
| 3.2 | 0.256 | 0.131 | 0.173 | 10.9% | +0.001 |
| 3.3 | 0.250 | 0.104 | 0.147 | 12.2% | +0.048 |

Codename format **hurts** Gemma: W-F1 drops 0.1489 → 0.1323 (-0.017). 1.3 falls sharply (detection 37%→21%). Verbose edge descriptions appear to suppress Gemma's already-sparse reliable categories without unlocking new ones.

---

### GPT-oss 20B — baseline (v2)

| Code | P | R | F1 | Det% |
|---|---|---|---|---|
| 1.1 | 0.433 | 0.151 | 0.224 | 15.3% |
| 1.2 | 0.000 | 0.000 | 0.000 | 4.1% |
| 1.3 | 0.512 | 0.198 | 0.286 | 21.4% |
| 1.4 | 0.000 | 0.000 | 0.000 | 1.0% |
| 1.5 | 0.333 | 0.023 | 0.043 | 3.1% |
| 2.1 | 0.000 | 0.000 | 0.000 | 0.8% |
| 2.2 | 0.484 | 0.090 | 0.152 | 7.9% |
| 2.3 | 0.182 | 0.016 | 0.029 | 2.8% |
| 2.4 | 0.000 | 0.000 | 0.000 | 1.5% |
| 2.6 | 0.701 | 0.170 | 0.274 | 17.0% |
| 3.1 | 0.182 | 0.016 | 0.029 | 2.8% |
| 3.2 | 0.233 | 0.286 | 0.257 | 26.2% |
| 3.3 | 0.354 | 0.304 | 0.327 | 25.2% |

Similar profile to Gemma v2: very low recall on most categories. Strongest on 3.3 (0.327) and 3.2 (0.257). Good precision where it fires. W-F1=0.1766.

### GPT-oss 20B — with-graph code (causal_only, v2)

W-F1=0.1643 — below baseline (0.1766). Graph guidance does not help; pattern consistent with Gemma v2. Base detection too sparse for graph edges to trigger.

### GPT-oss 20B — graph-inject code (causal_only, v2)

W-F1=0.1705 — modest improvement over baseline (+0.006) and with-graph (+0.006). Still well below Mistral. Per-category changes negligible.

### GPT-oss 20B — with-graph codename (causal_only, output_full)

| Code | P | R | F1 | Det% | ΔF1 vs code format |
|---|---|---|---|---|---|
| 1.1 | 0.460 | 0.169 | 0.247 | 16.0% | +0.043 |
| 1.3 | 0.521 | 0.175 | 0.262 | 18.6% | 0.000 |
| 2.6 | 0.678 | 0.214 | 0.325 | 22.1% | +0.064 |
| 3.2 | 0.219 | 0.274 | 0.243 | 26.7% | +0.031 |
| 3.3 | 0.349 | 0.383 | 0.365 | 32.1% | +0.073 |

W-F1=0.1872 vs code format 0.1643 (+0.023). Modest improvement, mainly from 3.3 and 2.6. GPT-oss 20B lies between Gemma (hurt by codename) and Mistral (strongly helped).

### GPT-oss 20B — graph-inject codename (causal_only, output_full)

W-F1=0.1884 — marginal improvement over with-graph codename (0.1872). No meaningful difference between E3 and E4 for this model.

---

### GPT-4o — baseline

| Prompt | W-F1 | Macro F1 | Avg response |
|---|---|---|---|
| v1 | 0.0946 | 0.0656 | ~319 chars |
| v2 | 0.0894 | 0.0618 | ~319 chars |

Prompt reorder had **zero effect**. Model gives one-sentence summaries and marks almost everything `no`. Only 3.2 (Weak Verification, 35%) detected at any meaningful rate. RLHF conservatism; non-reasoning model. Not a viable judge for this task without stronger intervention.

### GPT-4o — with-graph codename (causal_only)

| Code | P | R | F1 | Det% | ΔF1 vs baseline |
|---|---|---|---|---|---|
| 1.1 | 0.458 | 0.064 | 0.112 | 6.1% | +0.112 |
| 1.3 | 0.500 | 0.115 | 0.187 | 12.7% | +0.187 |
| 1.5 | 0.423 | 0.064 | 0.111 | 6.6% | — |
| 2.2 | 0.462 | 0.108 | 0.176 | 9.9% | — |
| 2.6 | 0.733 | 0.120 | 0.206 | 11.5% | — |
| 3.2 | 0.223 | 0.393 | 0.285 | 37.7% | modest |
| 3.3 | 0.319 | 0.191 | 0.239 | 17.6% | — |

W-F1=0.141 (+0.047 vs baseline 0.094). Causal graph gives a meaningful boost but the fundamental problem persists — extreme under-detection (<15% for most categories). 3.2 is the only category with reasonable recall (39%). The pattern mirrors GPT-4o baseline: one-category anchor + near-zero recall elsewhere.

---

### o1 — baseline (100-trace sample)

| Code | P | R | F1 | Det% |
|---|---|---|---|---|
| 1.1 | 0.167 | 0.023 | 0.040 | 6.0% |
| 1.3 | 1.000 | 0.054 | 0.102 | 3.0% |
| 1.5 | 0.000 | 0.000 | 0.000 | 1.0% |
| 2.2 | 0.500 | 0.048 | 0.087 | 4.0% |
| 2.6 | 0.909 | 0.143 | 0.247 | 11.0% |
| 3.3 | 0.444 | 0.138 | 0.211 | 9.0% |

W-F1=0.0908 — **worse than GPT-4o baseline**. Precision is very high when it fires (1.000 on 1.3, 0.909 on 2.6) but recall is catastrophically low. o1's chain-of-thought reasoning makes it even more conservative than GPT-4o. Completely misses 1.2, 1.4, 2.1, 2.3, 2.4, 3.1. Not a viable judge without structural prompt changes.

---

### QwQ-32B — baseline (thinking)

| Code | P | R | F1 | Det% |
|---|---|---|---|---|
| 1.1 | 0.446 | 0.192 | 0.268 | 18.8% |
| 1.3 | 0.455 | 0.046 | 0.084 | 5.6% |
| 1.5 | 0.265 | 0.052 | 0.087 | 8.7% |
| 2.2 | 0.520 | 0.078 | 0.136 | 6.4% |
| 2.3 | 0.286 | 0.031 | 0.056 | 3.6% |
| 2.6 | 0.676 | 0.181 | 0.286 | 18.8% |
| 3.1 | 0.333 | 0.047 | 0.082 | 4.6% |
| 3.2 | 0.198 | 0.274 | 0.230 | 29.5% |
| 3.3 | 0.299 | 0.226 | 0.257 | 22.1% |

W-F1=0.1608. Best non-Mistral model. Decent precision on 2.6 and 1.1 but low recall everywhere except 3.2 and 3.3. Pattern: thinking budget suppresses detection — the model reasons itself out of positives.

### QwQ-32B — with-graph (causal_only, thinking)

| Code | P | R | F1 | Det% | ΔF1 vs baseline |
|---|---|---|---|---|---|
| 1.1 | 0.520 | 0.151 | 0.234 | 12.7% | -0.034 |
| 2.6 | 0.737 | 0.203 | 0.318 | 19.3% | +0.032 |
| 3.2 | 0.217 | 0.250 | 0.232 | 24.7% | +0.002 |
| 3.3 | 0.296 | 0.139 | 0.189 | 13.7% | -0.068 |

W-F1=0.1513 — **below baseline (0.1608)**. Graph context makes QwQ more conservative, not less. Upfront structure appears to tighten its reasoning, reducing detection. Same E3<baseline pattern as Gemma.

### QwQ-32B — graph-inject (causal_only, thinking)

| Code | P | R | F1 | Det% | ΔF1 vs baseline |
|---|---|---|---|---|---|
| 1.1 | 0.459 | 0.163 | 0.240 | 15.5% | -0.028 |
| 1.2 | 0.111 | 0.143 | 0.125 | 2.3% | +0.125 |
| 2.6 | 0.721 | 0.225 | 0.343 | 21.9% | +0.057 |
| 3.2 | 0.230 | 0.310 | 0.264 | 28.8% | +0.034 |
| 3.3 | 0.316 | 0.209 | 0.251 | 19.3% | -0.006 |

W-F1=0.1717 — **best QwQ variant and only model where E4 > E3 > baseline**. 2-pass injection works better than static context for thinking models: the targeted second-pass signal fits the reasoning flow better than flooding with upfront structure. 1.2 fires (F1=0.125) — rare for any model. 2.6 best recall of all QwQ runs.

---

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

### F12: Detection-oriented prompt (evidence requirement) eliminates graph benefit and hurts overall
"detect" prompt adds section C (evidence per yes: step IDs + quote + reason) and removes task-completion question (old B).

| Setup | W-F1 | Notes |
|---|---|---|
| Old baseline | 0.3773 | original prompt |
| Old E3 with-graph | **0.5295** | original prompt; +0.152 over baseline |
| New detect baseline | 0.3699 | evidence prompt; -0.007 vs old baseline |
| New detect E3 with-graph | 0.3730 | evidence prompt; **-0.157 vs old E3**; graph barely helps |

The evidence requirement in section C forces the model to commit to citing specific step IDs before saying yes. For subtle or distributed categories (2.6, 3.1, 2.3), the model can't find a clean quotable step and defaults to no. Detection rates collapse: 2.6 drops from 85.5%→31.3%, 3.1 from 41.7%→1.5%, 2.3 from 40.2%→3.8%, 1.1 from 64.4%→14.8%. The causal graph is also neutered — its hints no longer translate to detections because the model now requires traceable evidence before acting on them. The old prompt's phrasing ("only mark if you can identify a specific example") was already the right calibration: it requires mental evidence without forcing the model to write it out, avoiding recall collapse. **Original prompt is better.**

### F11: Causal-only edges (7) outperform full stability graph (27 edges, threshold=0.5) for Mistral E3
| Variant | W-F1 | Macro F1 | 3.3 F1 | 2.3 F1 |
|---|---|---|---|---|
| E3 causal_only | **0.5295** | **0.3722** | **0.442** | 0.371 |
| E3 t0.5 | 0.5237 | 0.3694 | 0.358 | **0.418** |

More edges help 2.3 (+0.047) and marginally 1.4/2.4/3.1, but hurt 3.3 sharply (-0.084) — extra edges push false positives into 3.3, dropping its precision from 0.302→0.266. The 7 intervention-validated causal edges are better signal than 27 bootstrap-stable ones. E4 with t0.5 is not worth running: E4 already loses ~0.16 W-F1 to E3 for Mistral, and a noisier graph won't recover that.

### Mistral Small 24B — with-graph codename (t0.5, 27 edges)

| Code | P | R | F1 | Det% | ΔF1 vs causal_only |
|---|---|---|---|---|---|
| 1.1 | 0.451 | 0.564 | 0.501 | 54.7% | -0.017 |
| 1.3 | 0.549 | 0.880 | 0.676 | 88.5% | -0.008 |
| 1.4 | 0.090 | 0.455 | 0.150 | 56.7% | +0.009 |
| 1.5 | 0.449 | 0.867 | 0.592 | 85.0% | -0.011 |
| 2.2 | 0.447 | 0.632 | 0.524 | 59.8% | +0.001 |
| 2.3 | 0.330 | 0.570 | 0.418 | 56.2% | **+0.047** |
| 2.4 | 0.052 | 0.429 | 0.092 | 44.3% | +0.015 |
| 2.6 | 0.698 | 0.880 | 0.779 | 88.5% | -0.005 |
| 3.1 | 0.326 | 0.453 | 0.379 | 45.3% | +0.009 |
| 3.2 | 0.210 | 0.798 | 0.333 | 81.2% | +0.006 |
| 3.3 | 0.266 | 0.548 | 0.358 | 60.3% | **-0.084** |

Overall W-F1: 0.5295 → 0.5237 (-0.006). Extra edges (mainly additional paths to 3.3) flood that category with false positives, hurting precision more than recall gains elsewhere compensate. causal_only graph remains optimal.

---

### F8: RLHF/thinking conservatism makes GPT-4o and o1 non-viable without structural changes
GPT-4o with-graph codename: 0.141 vs baseline 0.094 — causal graph helps (+0.047) but under-detection persists. o1 baseline on 100 traces: **0.091 — worse than GPT-4o**. Reasoning models over-deliberate and consistently talk themselves out of positive labels. The precision when they do fire is high (o1: 0.91 on 2.6, 1.0 on 1.3) but recall is catastrophically low (~5–14%). No prompt-level intervention has fixed this pattern across all GPT-4o/o1 experiments. This is a model-intrinsic alignment issue, not a prompt issue.

### F9: QwQ-32B is the only model where E4 (graph-inject) > E3 (with-graph) > baseline
| Model | Baseline | E3 with-graph | E4 graph-inject |
|---|---|---|---|
| Mistral Small 24B | 0.3773 | **0.5295** | 0.3701 |
| Gemma 3 27B | 0.1419 | 0.1489 | 0.1476 |
| GPT-oss 20B | 0.1766 | 0.1872 | 0.1884 |
| QwQ-32B | 0.1608 | 0.1513 | **0.1717** |

For all non-QwQ models, E3 ≥ E4. QwQ reverses this: static upfront context suppresses its thinking process, while targeted 2nd-pass injection fits naturally into its reasoning flow. This is consistent with thinking models needing targeted signals rather than broad upfront context.

### F10: Thinking models (QwQ, o1) do not outperform instruction-tuned models on this task
QwQ-32B best (E4): W-F1=0.1717. o1 baseline: W-F1=0.0908. Both below Mistral Small 24B (0.5295) and even below Mistral baseline (0.3773). Reasoning capability does not translate to better multi-label error classification — if anything it increases conservatism and reduces recall.

### F7: Edge format comparison — name-only vs code-only vs code+name

Three-format comparison for with-graph (causal_only), W-F1:

| Model | name-only (outputs/) | code-only (outputs_v2) | code+name (output_full) |
|---|---|---|---|
| Mistral Small 24B | 0.0291 ⚠ | 0.4758 | **0.5295** |
| Gemma 3 27B | — ✗ | 0.1489 | 0.1323 |
| GPT-oss (with-graph) | 0.1649 ⚠ | 0.1643 | **0.1872** |
| GPT-4o (baseline only) | 0.0946 | 0.0894 | — |

⚠ **Mistral name-only (0.0291)**: not a clean format comparison — three bugs were fixed simultaneously (markdown header bug, edge format change, bold parser fix). The 0.0291→0.4758 jump cannot be attributed to format alone.

✗ **Gemma name-only**: invalid — run against a different annotation file version; results discarded.

⚠ **GPT-oss name-only (0.1649)**: different model (120B) from code-only/code+name experiments (20B), and incomplete (285/393). Not directly comparable.

**Conclusion from clean comparisons (code-only vs code+name, same model, same annotation file):**

| Model | code-only | code+name | Verdict |
|---|---|---|---|
| Mistral Small 24B | 0.4758 | **0.5295** (+0.054) | code+name wins |
| Gemma 3 27B | **0.1489** | 0.1323 (-0.017) | code-only wins |
| GPT-oss 20B | 0.1643 | **0.1872** (+0.023) | code+name wins |

**Direction**: code+name is the right direction for models with adequate base detection (Mistral, GPT-oss). Gemma is the exception — but its problem is under-detection, not format; no edge format fixes zero recall. The name-only format has no clean evidence in its favor and carries higher token cost without a proven benefit.

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
| Mistral — codename with-graph | `eval/code_name/run_eval_with_graph.py --causal_only` | ✓ done (output_full, W-F1=0.5295) |
| Mistral — codename graph-inject | `eval/code_name/run_eval_graph_inject.py --causal_only` | ✓ done (output_full, W-F1=0.3701) |
| Gemma — codename with-graph | `eval/code_name/run_eval_with_graph.py --causal_only` | ✓ done (output_full, W-F1=0.1323) |
| GPT-oss 20B — codename with-graph | `eval/code_name/run_eval_with_graph.py --causal_only` | ✓ done (output_full, W-F1=0.1872) |
| GPT-oss 20B — codename graph-inject | `eval/code_name/run_eval_graph_inject.py --causal_only` | ✓ done (output_full, W-F1=0.1884) |
| QwQ-32B — baseline (thinking) | `eval/thinking/run_eval_yesno_vllm.py --enable_thinking` | ✓ done (outputs_think, W-F1=0.1608) |
| QwQ-32B — with-graph (thinking) | `eval/thinking/run_eval_with_graph.py --causal_only --enable_thinking` | ✓ done (outputs_think, W-F1=0.1513) |
| QwQ-32B — graph-inject (thinking) | `eval/thinking/run_eval_graph_inject.py --causal_only --enable_thinking` | ✓ done (outputs_think, W-F1=0.1717) |
| GPT-4o — with-graph codename | `eval/full_run_eval_with_graph_api.py --model openai/gpt-4o --causal_only` | ✓ done (outputs_full_api, W-F1=0.141) |
| o1 — baseline (100-trace sample) | `eval/run_eval_yesno_api.py --model openai/o1 --sample_indices ...` | ✓ done (outputs_o1, W-F1=0.0908) |
| GPT-4o — graph-inject codename | API inject script | planned (low priority; baseline perf too low) |
| o1 — with-graph codename (100-trace sample) | `eval/full_run_eval_with_graph_api.py --model openai/o1 --sample_indices ...` | planned (expensive; likely still poor) |
| GPT-oss 120B with-graph | complete remaining 108 traces | planned |
| Mistral — full stability graph (E3, threshold=0.5) | `eval/full_run_eval_with_graph.py --edge_threshold 0.5` | ✓ done (outputs_corr, W-F1=0.5237 — worse than causal_only) |

---

## Output Directory Map

```
outputs/                                   ← v1 prompt; graph edges use name-only format (Name -> Name); Gemma rows INVALID (wrong annotation file)
  gemma-3-27b-it-yesno-baseline/                          W-F1=0.4366  ← INVALID
  gemma-3-27b-it-yesno-with-graph-causal_only/            W-F1=0.4594  ← INVALID
  gemma-3-27b-it-yesno-graph-inject-causal_only/          W-F1=0.4379  ← INVALID
  gpt-oss-120b-yesno-baseline/                            W-F1=0.1784
  gpt-oss-120b-yesno-with-graph-causal_only/              W-F1=0.1649  (285/393)
  openai-gpt-4o-yesno-baseline/                           W-F1=0.0946

outputs_v2/                                ← v2 prompt (bold fix applied); graph edges use code-only format (1.1 -> 3.3)
  gemma-3-27b-it-yesno-baseline/                          W-F1=0.1419
  gemma-3-27b-it-yesno-with-graph-causal_only/            W-F1=0.1489
  gemma-3-27b-it-yesno-graph-inject-causal_only/          W-F1=0.1476
  gpt-oss-20b-yesno-baseline/                             W-F1=0.1766
  gpt-oss-20b-yesno-with-graph-causal_only/               W-F1=0.1643
  gpt-oss-20b-yesno-graph-inject-causal_only/             W-F1=0.1705
  mistralai-Mistral-Small-3.1-24B-v2-yesno-baseline/      W-F1=0.3773
  mistralai-Mistral-Small-3.1-24B-v2-yesno-with-graph-causal_only/   W-F1=0.4758
  mistralai-Mistral-Small-3.1-24B-v2-yesno-graph-inject-causal_only/ W-F1=0.3715
  openai-gpt-4o-yesno-baseline/                           W-F1=0.0894

output_full/                               ← v2 prompt; graph edges use code+name format (1.1(Name) -> 3.3(Name))
  gemma-3-27b-it-yesno-with-graph-codename-causal_only/              W-F1=0.1323
  gpt-oss-20b-yesno-with-graph-codename-causal_only/                 W-F1=0.1872
  gpt-oss-20b-yesno-graph-inject-codename-causal_only/               W-F1=0.1884
  mistralai-Mistral-Small-3.1-24B-v2-yesno-with-graph-codename-causal_only/   W-F1=0.5295  ← best overall
  mistralai-Mistral-Small-3.1-24B-v2-yesno-graph-inject-codename-causal_only/ W-F1=0.3701

outputs_full_api/                          ← v2 prompt; code+name format; API inference
  openai-gpt-4o-yesno-with-graph-codename-causal_only/               W-F1=0.1410

outputs_o1/                                ← v2 prompt; 100-trace stratified sample
  openai-o1-yesno-baseline/                               W-F1=0.0908  (100/393)

outputs_think/                             ← v2 prompt; code+name format; thinking enabled (QwQ-32B)
  QwQ-32B-yesno-baseline-thinking/                        W-F1=0.1608
  QwQ-32B-yesno-with-graph-causal_only-thinking/          W-F1=0.1513
  QwQ-32B-yesno-graph-inject-causal_only-thinking/        W-F1=0.1717
```
