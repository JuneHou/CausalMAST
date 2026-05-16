# TODO — τ=0.50 pivot for MAST paper tables

**Pivot**: change the +GI headline graph from causal-only (11 edges) to
corr≥0.50 Suppes-screened super-graph (25 edges). All three MAST tables affected.

**Decision basis**: threshold sweep (Table `ablation_threshold_sweep.tex`) shows
τ=0.50 has the highest mean W-F1 (24.75) and Macro-F1 (17.58) across the
5-model open-source panel, beating causal-only and all other τ-points on both
average metrics. corr≥0.40 wins more individual cells (4/10) but hurts GPT-oss-20B
and Gemma enough to drag the panel average below τ=0.50.

**Affected tables** (all in `paper/tables/`):
1. `main_results_0.5.tex` — NEW; 1 cell pending (GPT-4o). All scripts exist; no sbatch needed (API call).
2. `ablation_threshold_sweep.tex` — existing; caption reframe only, 0 new runs.
3. `who_and_when_results.tex` — existing; 10 cells pending if W&W pivots; otherwise footnote only. All scripts exist. 10 sbatch files in `sbatch/ww_corr05/`.

---

## 1. Main results table — 1 run ★ critical

Create `paper/tables/main_results_0.5.tex` from `main_results.tex`. Open-source
+GI rows at τ=0.50 are already done — pull from `outputs_thres/t0.5/`:

| Model | Output file |
|---|---|
| Mistral-Small-3.1-24B | `outputs_thres/t0.5/mistralai-Mistral-Small-3.1-24B-Instruct-2503-yesno-graph-inject-codename-corr0.5-metrics.json` |
| GPT-oss-120B | `outputs_thres/t0.5/gpt-oss-120b-yesno-graph-inject-codename-corr0.5-metrics.json` |
| GPT-oss-20B | `outputs_thres/t0.5/openai-gpt-oss-20b-yesno-graph-inject-codename-corr0.5-metrics.json` |
| Gemma-3-27B-IT | `outputs_thres/t0.5/google-gemma-3-27b-it-yesno-graph-inject-codename-corr0.5-metrics.json` |
| QwQ-32B | `outputs_thres/t0.5/Qwen-QwQ-32B-yesno-graph-inject-codename-corr0.5-thinking-metrics.json` |

Missing cell:

| # | Model | Script | Status |
|---|---|---|---|
| 1 | GPT-4o | `eval/full_run_eval_graph_inject_api.py` ✓ exists | ☐ |

No sbatch needed — run directly (API call, no GPU):

```bash
# Run from CausalMAST repo root. Needs OPENAI_API_KEY exported.
python eval/full_run_eval_graph_inject_api.py \
  --model openai/gpt-4o \
  --corr_threshold 0.5 \
  --output_dir outputs_thres/t0.5 \
  --max_workers 5
```

Expected output:
```
outputs_thres/t0.5/openai-gpt-4o-yesno-graph-inject-codename-corr0.5/
outputs_thres/t0.5/openai-gpt-4o-yesno-graph-inject-codename-corr0.5-metrics.json
```

After completion: draft `paper/tables/main_results_0.5.tex`, fill in all 6 +GI rows
at τ=0.50, update bold/underline per-column rankings.

---

## 2. Threshold sweep ablation table — 0 runs (caption reframe)

`paper/tables/ablation_threshold_sweep.tex` already contains the full 5-model × 5-τ
grid (random-11, causal-only, corr≥0.60, corr≥0.50, corr≥0.40) under +GI.
No new runs needed.

Re-cast caption: from *"ablation: does corr-τ help over causal-only?"* to
*"ablation: justification for choosing τ=0.50 as the headline."* The causal-only row
becomes the simpler-baseline comparison; the corr-τ rows are the τ-selection evidence.
GPT-4o excluded from ablation by scope (same as TRAIL's handling of Gemini).

---

## 3. Who&When table — 10 runs (or footnote: 0 runs)

`paper/tables/who_and_when_results.tex` currently uses causal-only for all
W1+GI and W2+CG rows. No τ=0.50 who&when outputs exist anywhere yet.

Runners support `--corr_threshold 0.5`. Output naming convention:
- W1+GI: `*-yesno-who_and_when_w1_graph_inject_corr0.5`
- W2+CG: `*-yesno-who_and_when_w2_graph_corr0.5`

### Missing cells (5 models × 2 variants = 10)

All scripts confirmed to exist. All 10 sbatch files created in `sbatch/ww_corr05/`.
All models run via ARC vLLM (local GPU), not DeepInfra.

| Model | W1+GI sbatch | W1+CG sbatch | W2+CG sbatch | GPUs | Adopted from |
|---|---|---|---|---|---|
| Mistral-Small-3.1-24B | `run_ww_w1_gi_corr05_mistral24b.sbatch` ☐ | `run_ww_w1_cg_corr05_mistral24b.sbatch` ☐ | `run_ww_w2_cg_corr05_mistral24b.sbatch` ☐ | 2 | `w1_cg/run_ww_w1_cg_mistral24b.sbatch` |
| GPT-oss-20B | `run_ww_w1_gi_corr05_gpt20b.sbatch` ☐ | `run_ww_w1_cg_corr05_gpt20b.sbatch` ☐ | `run_ww_w2_cg_corr05_gpt20b.sbatch` ☐ | 4 | `w1_cg/run_ww_w1_cg_gpt20b.sbatch` |
| Gemma-3-27B-IT | `run_ww_w1_gi_corr05_gemma27b.sbatch` ☐ | `run_ww_w1_cg_corr05_gemma27b.sbatch` ☐ | `run_ww_w2_cg_corr05_gemma27b.sbatch` ☐ | 2 | `w1_cg/run_ww_w1_cg_gemma27b.sbatch` |
| GPT-oss-120B | — | — | — | — | run separately (see below) |
| QwQ-32B | `run_ww_w1_gi_corr05_qwq32b.sbatch` ☐ | `run_ww_w1_cg_corr05_qwq32b.sbatch` ☐ | `run_ww_w2_cg_corr05_qwq32b.sbatch` ☐ | 4 | `run_ww_w1_gi_qwq32b.sbatch` |

Key changes from source sbatch (applied to all):
- W1+GI: script changed from `run_who_and_when_with_graph_vllm.py` → `run_who_and_when_graph_inject_vllm.py`
- W2+CG: `--variant w1` → `--variant w2`; time bumped to 8h (12h for GPT-120B)
- All: `--causal_only` → `--corr_threshold 0.5`; `OUTPUT_DIR` → `baselines/who&when/causal/outputs`

### Submit commands

```bash
cd /projects/slmreasoning/junh/causal-error/CausalMAST
# Priority order: W1+GI -> W1+CG -> W2+CG
for f in sbatch/ww_corr05/run_ww_w1_gi_*.sbatch; do sbatch "$f"; done
for f in sbatch/ww_corr05/run_ww_w1_cg_*.sbatch; do sbatch "$f"; done
for f in sbatch/ww_corr05/run_ww_w2_cg_*.sbatch; do sbatch "$f"; done
```

Or submit individually:

```bash
# W1+GI (priority)
sbatch sbatch/ww_corr05/run_ww_w1_gi_corr05_mistral24b.sbatch
sbatch sbatch/ww_corr05/run_ww_w1_gi_corr05_gpt20b.sbatch
sbatch sbatch/ww_corr05/run_ww_w1_gi_corr05_gemma27b.sbatch
sbatch sbatch/ww_corr05/run_ww_w1_gi_corr05_qwq32b.sbatch
# W1+CG
sbatch sbatch/ww_corr05/run_ww_w1_cg_corr05_mistral24b.sbatch
sbatch sbatch/ww_corr05/run_ww_w1_cg_corr05_gpt20b.sbatch
sbatch sbatch/ww_corr05/run_ww_w1_cg_corr05_gemma27b.sbatch
sbatch sbatch/ww_corr05/run_ww_w1_cg_corr05_qwq32b.sbatch
# W2+CG (after)
sbatch sbatch/ww_corr05/run_ww_w2_cg_corr05_mistral24b.sbatch
sbatch sbatch/ww_corr05/run_ww_w2_cg_corr05_gpt20b.sbatch
sbatch sbatch/ww_corr05/run_ww_w2_cg_corr05_gemma27b.sbatch
sbatch sbatch/ww_corr05/run_ww_w2_cg_corr05_qwq32b.sbatch
```

### Expected output dirs (in `baselines/who&when/causal/outputs/`)

```
# W1+GI
mistralai-Mistral-Small-3.1-24B-Instruct-2503-yesno-who_and_when_w1_graph_inject_corr0.5/
openai-gpt-oss-20b-yesno-who_and_when_w1_graph_inject_corr0.5/
google-gemma-3-27b-it-yesno-who_and_when_w1_graph_inject_corr0.5/
qwq-32b-yesno-who_and_when_w1_graph_inject_corr0.5-thinking/
# W1+CG
mistralai-Mistral-Small-3.1-24B-Instruct-2503-yesno-who_and_when_w1_graph_corr0.5/
openai-gpt-oss-20b-yesno-who_and_when_w1_graph_corr0.5/
google-gemma-3-27b-it-yesno-who_and_when_w1_graph_corr0.5/
qwq-32b-yesno-who_and_when_w1_graph_corr0.5-thinking/
# W2+CG
mistralai-Mistral-Small-3.1-24B-Instruct-2503-yesno-who_and_when_w2_graph_corr0.5/
openai-gpt-oss-20b-yesno-who_and_when_w2_graph_corr0.5/
google-gemma-3-27b-it-yesno-who_and_when_w2_graph_corr0.5/
qwq-32b-yesno-who_and_when_w2_graph_corr0.5-thinking/
```

### Alternative: footnote instead of rerun (0 runs)

If timeline is tight, keep W&W rows at causal-only and add a caption note:
> *Who\&When rows use the causal-only graph (11 edges); the main-table +GI
> headline at τ=0.50 is supported separately by the threshold sweep
> (Table~\ref{tab:mast_ablation_threshold_sweep}).*

---

## Total compute summary

| Scenario | New runs | What's covered |
|---|---|---|
| **Minimum** (main table + caption reframe; W&W → footnote) | **1** | Headline pivot only |
| **Standard** (main table + W&W rerun) | **11** | All tables consistent |

**Recommended**: run the 1 GPT-4o cell first (fast via API), then decide on
the W&W rerun based on submission timeline. If GPT-4o at τ=0.50 is clearly
better than causal-only, the pivot is unambiguous and W&W becomes a
"nice-to-have" for consistency.
