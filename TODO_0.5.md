# TODO — τ=0.50 pivot for MAST paper tables

**Pivot**: change the +GI headline graph from causal-only (11 edges) to
corr≥0.50 Suppes-screened super-graph (25 edges). All three MAST tables affected.

**Decision basis**: threshold sweep (Table `ablation_threshold_sweep.tex`) shows
τ=0.50 has the highest mean **gain over the causal-only baseline** across the
5-model open-source panel on both headline metrics (+1.35 W-F1 and +0.87
Macro-F1; next best τ=0.40 at +0.95 W-F1 and +0.55 Macro-F1; τ=0.60 is
negative on both at −0.18 W-F1 and −0.42 Macro-F1). Mean raw W-F1 / Macro-F1
agree with the gain criterion — the selection is robust to either aggregation.
corr≥0.40 wins more individual cells but hurts GPT-oss-20B and Gemma enough
to drag the panel-averaged gain below τ=0.50.

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
| GPT-oss-120B (ARC) | `outputs_thres/t0.5/gpt-oss-120b-yesno-graph-inject-codename-corr0.5-metrics.json` — produced via `eval/full_run_eval_graph_inject_api_arc.py` (no GPU); re-run command if needed: see "Re-running the 120B main-table cell" below |
| GPT-oss-20B | `outputs_thres/t0.5/openai-gpt-oss-20b-yesno-graph-inject-codename-corr0.5-metrics.json` |
| Gemma-3-27B-IT | `outputs_thres/t0.5/google-gemma-3-27b-it-yesno-graph-inject-codename-corr0.5-metrics.json` |
| QwQ-32B | `outputs_thres/t0.5/Qwen-QwQ-32B-yesno-graph-inject-codename-corr0.5-thinking-metrics.json` |

#### Re-running the 120B main-table cell (ARC API, no sbatch)

The 120B `+GI corr0.5` metrics file is already on disk. If a re-run is needed
(e.g., to refresh after a graph artifact change), run from the MAST repo root
after sourcing the ARC auth key:

```bash
# Auth: ARC_LLM_API_KEY env var (e.g., source path/to/arc_llm_api.sh)
# ARC default model is `gpt-oss-120b` (bare name, no `openai/` prefix);
# --model can be omitted to use the default.
python eval/full_run_eval_graph_inject_api_arc.py \
    --corr_threshold 0.5 \
    --output_dir outputs_thres/t0.5
```

ARC fairshare limits (30 rpm, 1000 rph, 3000 per 3hr) apply. A full 393-trace
+GI run does Pass-1 once per trace plus Pass-2 on the subset whose Pass-1
detections propagate; expect ~30-40 min real-time. Resumable (skips existing
per-trace files).

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

Note: MAST has no location labels. Do not pass `--span_index` to MAST runners.

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
| GPT-oss-120B (ARC) | bash (see below) | bash (see below) | bash (see below) | n/a | run via ARC API — no sbatch, no GPU |
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

### GPT-oss-120B — ARC API (bash, no sbatch, no GPU)

ARC is API-only, so 120B does not get an sbatch wrapper. Run directly from the
MAST repo root after sourcing the ARC auth key. The ARC scripts default
`--model` to `gpt-oss-120b` (bare name, no `openai/` prefix), so it can be
omitted.

```bash
# ============================================================
# Auth: source path/to/arc_llm_api.sh   → sets ARC_LLM_API_KEY
# Rate limits: 30 rpm, 1000 rph, 3000 per 3hr (3-rule sliding window)
# Resumable: skips existing per-trace files in --output_dir
# ============================================================

# W1+GI (Pass-1 + targeted Pass-2; ~30-40 min real-time over 393 traces)
python "baselines/who&when/causal/run_who_and_when_graph_inject_api_arc.py" \
    --variant w1 --corr_threshold 0.5 \
    --output_dir "baselines/who&when/causal/outputs"

# W1+CG (one-pass)
python "baselines/who&when/causal/run_who_and_when_with_graph_api_arc.py" \
    --variant w1 --corr_threshold 0.5 \
    --output_dir "baselines/who&when/causal/outputs"

# W2+CG (N+1 calls per trace; ARC hourly cap will dominate ~2-3 hr real-time)
python "baselines/who&when/causal/run_who_and_when_with_graph_api_arc.py" \
    --variant w2 --corr_threshold 0.5 \
    --output_dir "baselines/who&when/causal/outputs"
```

These three runs produce:
```
gpt-oss-120b-yesno-who_and_when_w1_graph_inject_corr0.5/
gpt-oss-120b-yesno-who_and_when_w1_graph_corr0.5/
gpt-oss-120b-yesno-who_and_when_w2_graph_corr0.5/
```

Scoring after each run completes:
```bash
python eval/calculate_scores_yesno.py \
    --pred_dir "baselines/who&when/causal/outputs/gpt-oss-120b-yesno-who_and_when_w1_graph_inject_corr0.5"
# (and analogously for the other two)
```

### Expected output dirs (in `baselines/who&when/causal/outputs/`)

```
# W1+GI
mistralai-Mistral-Small-3.1-24B-Instruct-2503-yesno-who_and_when_w1_graph_inject_corr0.5/
openai-gpt-oss-20b-yesno-who_and_when_w1_graph_inject_corr0.5/
google-gemma-3-27b-it-yesno-who_and_when_w1_graph_inject_corr0.5/
qwq-32b-yesno-who_and_when_w1_graph_inject_corr0.5-thinking/
gpt-oss-120b-yesno-who_and_when_w1_graph_inject_corr0.5/                  # ARC
# W1+CG
mistralai-Mistral-Small-3.1-24B-Instruct-2503-yesno-who_and_when_w1_graph_corr0.5/
openai-gpt-oss-20b-yesno-who_and_when_w1_graph_corr0.5/
google-gemma-3-27b-it-yesno-who_and_when_w1_graph_corr0.5/
qwq-32b-yesno-who_and_when_w1_graph_corr0.5-thinking/
gpt-oss-120b-yesno-who_and_when_w1_graph_corr0.5/                         # ARC
# W2+CG
mistralai-Mistral-Small-3.1-24B-Instruct-2503-yesno-who_and_when_w2_graph_corr0.5/
openai-gpt-oss-20b-yesno-who_and_when_w2_graph_corr0.5/
google-gemma-3-27b-it-yesno-who_and_when_w2_graph_corr0.5/
qwq-32b-yesno-who_and_when_w2_graph_corr0.5-thinking/
gpt-oss-120b-yesno-who_and_when_w2_graph_corr0.5/                         # ARC
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

---

# Follow-up — Closed-source cross-benchmark coverage

**Pivot**: in the combined main-results table
(`/data/wang/junh/githubs/-EMNLP-2026-CASCADE-Causal-Error/tables/main_results_combine.tex`),
split the merged "Closed-source" row so Gemini and GPT-4o each cover all three benchmark
columns (TRAIL-GAIA, TRAIL-SWE-Bench, MAST).

This file owns the **MAST-side Gemini** runs. TRAIL-side GPT-4o runs are in
`/data/wang/junh/githubs/trail-benchmark/TODO_0.35.md` (§6).

**Backend decision**: route Gemini through **OpenRouter** (mirrors the TRAIL pattern in
`benchmarking/eval/run_eval_graph_inject_router.py`). MAST currently has no
`*_router.py` runners — two new scripts need to be added.

## 4. New scripts (port from TRAIL) — 2 files ★ prereq

| New file | Adapted from | Purpose |
|---|---|---|
| `eval/run_eval_yesno_router.py` | `eval/run_eval_yesno_api.py` (litellm baseline) + TRAIL `run_eval_graph_inject_router.py` (OpenRouter args) | Baseline yes/no eval via OpenRouter |
| `eval/full_run_eval_graph_inject_router.py` | TRAIL `benchmarking/eval/run_eval_graph_inject_router.py` + MAST `eval/full_run_eval_graph_inject_api.py` (MAST 13-leaf taxonomy, two-pass, no `--span_index`) | +GI eval via OpenRouter |

OpenRouter args to mirror from TRAIL router (lines 299-374 of `run_eval_graph_inject_router.py`):
- `--openrouter_api_key_env OPENROUTER_API_KEY` (default)
- `--openrouter_base_url https://openrouter.ai/api/v1`
- `--temperature 1.0`, `--max_workers 5`, `--default_max_completion_tokens 15000`
- Strip the `openrouter/` prefix when calling `litellm.token_counter` (TRAIL line 54)
- Model id format: `openrouter/google/gemini-2.5-flash`, `openrouter/google/gemini-2.5-pro`

## 5. MAST Gemini cells — 4 runs ★ critical

| # | Model | Method | Script |
|---|---|---|---|
| 1 | openrouter/google/gemini-2.5-flash | Baseline | `run_eval_yesno_router.py` (new) |
| 2 | openrouter/google/gemini-2.5-pro   | Baseline | `run_eval_yesno_router.py` (new) |
| 3 | openrouter/google/gemini-2.5-flash | +GI(τ=0.5) | `full_run_eval_graph_inject_router.py` (new) |
| 4 | openrouter/google/gemini-2.5-pro   | +GI(τ=0.5) | `full_run_eval_graph_inject_router.py` (new) |

Commands (run from MAST repo root after `export OPENROUTER_API_KEY=...`):

```bash
# ============================================================
# Baseline (no graph) — 2 cells
# ============================================================
for model in openrouter/google/gemini-2.5-flash openrouter/google/gemini-2.5-pro; do
  python eval/run_eval_yesno_router.py \
    --model "$model" --max_workers 5
done

# ============================================================
# +GI at τ=0.50 — 2 cells
# ============================================================
for model in openrouter/google/gemini-2.5-flash openrouter/google/gemini-2.5-pro; do
  python eval/full_run_eval_graph_inject_router.py \
    --model "$model" \
    --corr_threshold 0.5 \
    --output_dir outputs_thres/t0.5 \
    --max_workers 5
done
```

Expected output dirs (`<model_tag>` = model with `/` → `-`):
```
outputs/openrouter-google-gemini-2.5-{flash,pro}-yesno-baseline/
outputs_thres/t0.5/openrouter-google-gemini-2.5-{flash,pro}-yesno-graph-inject-codename-corr0.5/
```

After completion: score with `eval/calculate_scores_yesno.py`, then add Gemini-Flash
and Gemini-Pro rows to `main_results_mast.tex` and update the closed-source block in
`main_results_combine.tex` to split into separate Gemini + GPT-4o rows.

## Cross-benchmark total

| Side | Runs | TODO file |
|---|---|---|
| TRAIL (GPT-4o) | 4 | `trail-benchmark/TODO_0.35.md` (§6) |
| MAST (this file, §5) | 4 | `MAST/TODO_0.5.md` |
| **New scripts** | 2 | `MAST/TODO_0.5.md` (§4) |
| **Total runs** | **8** | |
