# TODO — +CG main-results gap fill (MAST)

**Goal**: add a +CG row at τ=0.50 (corr-union 25 edges, `corr0.5`) to
`main_results_mast.tex` for all 5 open-source models, so each model block has
3 rows: Baseline / +CG(τ=0.50) / +GI(τ=0.50) (= \our).

**Current state**: `outputs_thres_cg/t0.5/` covers Gemma, GPT-oss-120B, and
GPT-oss-20B with the correct corr-union graph. Two models are missing.

Closed-source (GPT-4o) is out of scope for this file.

## Missing cells — 2 runs

| # | Model | Backend | Note |
|---|---|---|---|
| 1 | mistralai/Mistral-Small-3.1-24B-Instruct-2503 | vLLM | causal-only +CG exists in `outputs_full/`; τ=0.5 corr-union is missing |
| 2 | Qwen/QwQ-32B | vLLM (with `--enable_thinking`) | causal-only +CG exists in `outputs_think/`; τ=0.5 corr-union is missing |

## Commands

`eval/full_run_eval_with_graph.py` already supports `--corr_threshold` (line
321), so no runner extension needed.

```bash
# Run from MAST repo root
# Output lands in: outputs_thres_cg/t0.5/<model_tag>-yesno-with-graph-codename-corr0.5/

# ============================================================
# (1) Mistral-Small-3.1-24B via vLLM
# ============================================================
CUDA_VISIBLE_DEVICES=0,1 python eval/full_run_eval_with_graph.py \
  --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
  --tp 2 \
  --corr_threshold 0.5 \
  --output_dir outputs_thres_cg/t0.5

# ============================================================
# (2) QwQ-32B via vLLM with thinking
# IMPORTANT: --max_model_len 40960 — matches past known-good config
# (sbatch/run_ww_w1_gi_qwq32b.sbatch). The runner default of 128000
# would OOM the KV cache on A40.
# IMPORTANT: pick 4 GPUs with <500 MB used by other processes. CUDA-graph
# capture in capture_end has spiky memory; even 2-3 GB of contention from
# another user on the same GPU eats the headroom and causes OOM at init.
# Use `nvidia-smi` to find a clean group. Fallback: add
# `--gpu_memory_utilization 0.7` to leave extra room.
# ============================================================
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval/full_run_eval_with_graph.py \
  --model Qwen/QwQ-32B \
  --tp 4 \
  --enable_thinking \
  --max_model_len 40960 \
  --corr_threshold 0.5 \
  --output_dir outputs_thres_cg/t0.5
```

## Expected output paths

```
outputs_thres_cg/t0.5/mistralai-Mistral-Small-3.1-24B-Instruct-2503-yesno-with-graph-codename-corr0.5/
outputs_thres_cg/t0.5/QwQ-32B-yesno-with-graph-codename-corr0.5-thinking/
```

(QwQ output suffix gets `-thinking` automatically when `--enable_thinking` is on,
matching the existing `outputs_corr/QwQ-32B-yesno-with-graph-codename-t0.2-thinking/`
naming.)

## Scoring + table integration

```bash
python eval/calculate_scores_yesno.py --pred_dir outputs_thres_cg/t0.5/<dir>
```

After both cells finish:
- Pull metrics into a new `+CG (τ=0.50)` row per model block in
  `/data/wang/junh/githubs/-EMNLP-2026-CASCADE-Causal-Error/tables/main_results_mast.tex`
- Re-rank bold/underline per column

## Cross-reference

TRAIL has 4 parallel missing cells (Mistral-3.1 ×2 splits + QwenLong ×2 splits).
See `/data/wang/junh/githubs/trail-benchmark/TODO_CG.md`. Combined total:
**6 runs across both benchmarks**.
