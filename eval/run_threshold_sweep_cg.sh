#!/usr/bin/env bash
# eval/run_threshold_sweep_cg.sh — MAST graph-richness ablation sweep (+CG only)
#
# Sister of eval/run_threshold_sweep.sh that drives the +CG (one-pass, in-prompt
# graph guidance) method across the same threshold list as the +GI sweep.
# Pair this with the +GI sweep to compare how each injection architecture
# scales with edge count under the same edge sets (§4.5 claim).
#
# Threshold list (matches the +GI sweep so cells line up):
#   random       -> 11 random non-Suppes edges, seed=42 (null-graph control)
#   corr >= 0.60 -> ~18 union edges  (just above causal-only count)
#   corr >= 0.50 -> ~25 union edges  (mid-sweep)
#   corr >= 0.40 -> ~29 union edges  (saturating end)
#
# Each threshold point gets its own subdirectory under output_dir:
#   outputs_thres_cg/t_random11_seed42/
#   outputs_thres_cg/t0.6/
#   outputs_thres_cg/t0.5/
#   outputs_thres_cg/t0.4/
#
# IMPORTANT DIFFERENCES FROM eval/run_threshold_sweep.sh:
#   1. Inner runner is the +CG family (one-pass) instead of +GI (two-pass):
#        vllm      -> eval/full_run_eval_with_graph.py
#        litellm   -> eval/full_run_eval_with_graph_api.py
#        deepinfra -> eval/full_run_eval_with_graph_api_deepinfra.py
#        arc       -> eval/full_run_eval_with_graph_api_arc.py
#   2. The litellm runner (Gemini/GPT-4o/Anthropic) does NOT expose
#      --random_edges; the 'random' threshold is SKIPPED for litellm with a
#      warning. All other backends include random.
#   3. Output root defaults to outputs_thres_cg/ (parallel to outputs_thres/).
#   4. --span_index is accepted as a CLI-parity no-op by every +CG runner
#      (MAST has no location prediction), so it never appears here.
#
# Usage (run from MAST/):
#   eval/run_threshold_sweep_cg.sh <model> [gpus] [output_dir] [backend]
#
# Examples:
#   eval/run_threshold_sweep_cg.sh mistralai/Mistral-Small-3.1-24B-Instruct-2503 4,5,6,7
#   eval/run_threshold_sweep_cg.sh Qwen/QwQ-32B 0,1,2,3
#   eval/run_threshold_sweep_cg.sh openai/gpt-4o "" outputs_thres_cg litellm
#   eval/run_threshold_sweep_cg.sh openai/gpt-oss-20b "" outputs_thres_cg     # deepinfra
#   eval/run_threshold_sweep_cg.sh gpt-oss-120b "" outputs_thres_cg           # arc

set -euo pipefail

MODEL="${1:?Usage: $0 <model> [gpus] [output_dir] [backend]}"

# Reject the legacy <method> positional (cg|gi) — only +CG is supported by
# this script. Without this guard, "$0 <model> cg 4,5,6,7" silently sets
# GPUS=cg and OUTDIR=4,5,6,7.
if [[ "${2:-}" == "gi" || "${2:-}" == "cg" ]]; then
  echo "ERROR: legacy <method> arg detected ('${2}'). This script is +CG-only;" >&2
  echo "       drop that arg. New usage: $0 <model> [gpus] [output_dir] [backend]" >&2
  exit 1
fi

GPUS="${2:-0,1,2,3}"
OUTDIR="${3:-outputs_thres_cg}"
BACKEND="${4:-}"

# Infer backend from model name if not specified (mirrors run_threshold_sweep.sh)
if [[ -z "$BACKEND" ]]; then
  case "$MODEL" in
    gemini/*|openai/gpt-4*|openai/o*|anthropic/*) BACKEND="litellm"   ;;
    openai/gpt-oss-*|google/*)                    BACKEND="deepinfra" ;;
    */*)                                          BACKEND="vllm"      ;;
    *)                                            BACKEND="arc"       ;;
  esac
fi

# Select +CG inner script per backend.
case "$BACKEND" in
  vllm)      CG_SCRIPT="eval/full_run_eval_with_graph.py"               ;;
  litellm)   CG_SCRIPT="eval/full_run_eval_with_graph_api.py"           ;;
  deepinfra) CG_SCRIPT="eval/full_run_eval_with_graph_api_deepinfra.py" ;;
  arc)       CG_SCRIPT="eval/full_run_eval_with_graph_api_arc.py"       ;;
  *) echo "ERROR: backend must be vllm | litellm | deepinfra | arc (got: $BACKEND)" >&2; exit 1 ;;
esac

if [[ ! -f "$CG_SCRIPT" ]]; then
  echo "ERROR: inner script not found: $CG_SCRIPT" >&2
  exit 1
fi

# litellm +CG runner does not support --random_edges (only causal_only and
# corr_threshold). Skip random for that backend instead of failing.
case "$BACKEND" in
  litellm) SUPPORTS_RANDOM=0 ;;
  *)       SUPPORTS_RANDOM=1 ;;
esac

# Tensor-parallel size from GPU list (vLLM only)
TP=""
if [[ "$BACKEND" == "vllm" ]]; then
  IFS=',' read -ra _gpus <<< "$GPUS"
  TP="${#_gpus[@]}"
fi

# Per-model max_model_len (vLLM only; same empirical KV-cache budgets as the
# +GI sweep — see comments at the top of eval/run_threshold_sweep.sh).
case "$MODEL" in
  Tongyi-Zhiwen/QwenLong-L1-32B*)                 MAX_MODEL_LEN=128000 ;;
  mistralai/Mistral-Small-3.1-24B-Instruct-2503*) MAX_MODEL_LEN=108000 ;;
  google/gemma-3-27b-it*|openai/gemma-3-27b-it*)  MAX_MODEL_LEN=108000 ;;
  openai/gpt-oss-20b*|openai/gpt-oss-120b*)       MAX_MODEL_LEN=108000 ;;
  Qwen/QwQ-32B*)                                  MAX_MODEL_LEN=40960  ;;
  *)                                              MAX_MODEL_LEN=""     ;;
esac

# Thinking flag for reasoning models (vLLM only; API runners infer from name)
ENABLE_THINKING=""
case "$MODEL" in
  Qwen/QwQ-32B*) ENABLE_THINKING="--enable_thinking" ;;
esac

# Sweep points; override with THRESHOLDS env var
THRESHOLDS=(${THRESHOLDS:-random 0.6 0.5 0.4})

LOGDIR="${OUTDIR}/_sweep_logs"
mkdir -p "$LOGDIR"

MODEL_TAG="${MODEL//\//-}"
THINKING_SUFFIX=""
[[ -n "$ENABLE_THINKING" ]] && THINKING_SUFFIX="-thinking"

echo "============================================================"
echo "  MAST graph-richness threshold sweep (+CG)"
echo "  model      : $MODEL"
echo "  backend    : $BACKEND"
echo "  thresholds : ${THRESHOLDS[*]}"
echo "  output_dir : $OUTDIR  (per-threshold subdirs created inside)"
echo "  logs       : $LOGDIR"
echo "============================================================"

# ------------------------------------------------------------------
# Helper: invoke one inner script for one threshold point
# ------------------------------------------------------------------
run_one() {
  local script="$1"
  local t="$2"
  local thres_outdir="$3"
  local log="$4"
  local graph_flags=()

  case "$t" in
    random) graph_flags+=(--random_edges --random_n 11 --random_seed 42) ;;
    *)      graph_flags+=(--corr_threshold "$t") ;;
  esac

  if [[ "$BACKEND" == "vllm" ]]; then
    local vllm_flags=(--tp "$TP")
    [[ -n "$MAX_MODEL_LEN" ]] && vllm_flags+=(--max_model_len "$MAX_MODEL_LEN")
    CUDA_VISIBLE_DEVICES="$GPUS" python "$script" \
      --model "$MODEL" \
      --output_dir "$thres_outdir" \
      "${vllm_flags[@]}" \
      ${ENABLE_THINKING} \
      "${graph_flags[@]}" \
      2>&1 | tee "$log"
  else
    python "$script" \
      --model "$MODEL" \
      --output_dir "$thres_outdir" \
      ${ENABLE_THINKING} \
      "${graph_flags[@]}" \
      2>&1 | tee "$log"
  fi
}

# ------------------------------------------------------------------
# Threshold -> subdir + graph_tag mapping (shared by sweep and scoring)
# ------------------------------------------------------------------
get_sub_and_tag() {
  local t="$1"
  case "$t" in
    random) echo "t_random11_seed42 random11_seed42" ;;
    *)      echo "t${t} corr${t}" ;;
  esac
}

# ------------------------------------------------------------------
# Sweep
# ------------------------------------------------------------------
for t in "${THRESHOLDS[@]}"; do
  read -r sub graph_tag <<< "$(get_sub_and_tag "$t")"
  thres_outdir="${OUTDIR}/${sub}"
  mkdir -p "$thres_outdir"

  echo
  echo "============================================================"
  echo "[$(date +%T)] threshold=$t  graph_tag=$graph_tag"
  echo "             output -> $thres_outdir"
  echo "============================================================"

  if [[ "$t" == "random" && "$SUPPORTS_RANDOM" != "1" ]]; then
    log="${LOGDIR}/${MODEL_TAG}-cg-${graph_tag}.log"
    echo "WARN: +CG ${BACKEND} runner ($CG_SCRIPT) has no --random_edges flag" | tee "$log"
    echo "      — skipping 'random' threshold for this backend." | tee -a "$log"
    continue
  fi

  log="${LOGDIR}/${MODEL_TAG}-cg-${graph_tag}.log"
  echo "[+CG] log -> $log"
  run_one "$CG_SCRIPT" "$t" "$thres_outdir" "$log"
done

# ------------------------------------------------------------------
# Score all completed output dirs for this model
# ------------------------------------------------------------------
echo
echo "============================================================"
echo "[$(date +%T)] All thresholds done. Scoring ..."
echo "============================================================"

for t in "${THRESHOLDS[@]}"; do
  if [[ "$t" == "random" && "$SUPPORTS_RANDOM" != "1" ]]; then
    continue
  fi
  read -r sub graph_tag <<< "$(get_sub_and_tag "$t")"
  thres_outdir="${OUTDIR}/${sub}"

  cg_dir="${thres_outdir}/${MODEL_TAG}-yesno-with-graph-codename-${graph_tag}${THINKING_SUFFIX}"
  if [[ -d "$cg_dir" ]]; then
    echo
    echo "+CG  t=$t  $cg_dir"
    python eval/calculate_scores_yesno.py --pred_dir "$cg_dir"
  else
    echo "WARNING: +CG dir not found (skipping score): $cg_dir" >&2
  fi
done

echo
echo "Sweep complete. Output tree:"
find "$OUTDIR" -maxdepth 3 -name "*-metrics.json" | sort
