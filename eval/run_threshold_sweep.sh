#!/usr/bin/env bash
# eval/run_threshold_sweep.sh — MAST graph-richness ablation sweep (+GI only)
#
# Sweeps corr_threshold over {0.60, 0.50, 0.40} for +GI (two-pass dynamic
# graph injection) on the MAST benchmark (N=393 AG2 traces). Also runs
# random-11 null-graph control.
#
# Threshold score (matches TRAIL's ablation):
#   score(A->B) = sqrt(precedence(A,B) * Delta_PR(A,B))
# This is already implemented in load_graph_edges() via
# math.sqrt(e["precedence"] * e["pr_delta"]) on suppes_graph.json.
#
# Sweep points (from paper/ablation_graph_richness.tex):
#   random11_seed42 -- 11 random non-Suppes edges, seed=42 (null control)
#   corr >= 0.60    -- ~18 union edges  (just above causal-only count)
#   corr >= 0.50    -- ~25 union edges  (mid-sweep)
#   corr >= 0.40    -- ~29 union edges  (saturating end)
#
# Each threshold point gets its own subdirectory under output_dir:
#   outputs_thres/t_random11_seed42/
#   outputs_thres/t0.6/
#   outputs_thres/t0.5/
#   outputs_thres/t0.4/
# (mirrors TRAIL's benchmarking/eval/run_threshold_sweep.sh structure)
#
# Note: pass 0.6/0.5/0.4 (not 0.60/0.50/0.40) because Python's float
# formatting strips trailing zeros in the output dir name
# (f"corr{0.60}" -> "corr0.6").
#
# Usage (run from MAST/):
#   eval/run_threshold_sweep.sh <model> [gpus] [output_dir] [backend]
#
#   <model>      HuggingFace model id, API model id, or bare model name
#   [gpus]       CUDA_VISIBLE_DEVICES string (default: 0,1,2,3); ignored for
#                non-vllm backends
#   [output_dir] base directory (default: outputs_thres); per-threshold
#                subdirs t<value>/ are created automatically inside it
#   [backend]    vllm | litellm | deepinfra | arc
#                If omitted, inferred from model name (mirrors TRAIL):
#                  gemini/*, openai/gpt-4*, openai/o*, anthropic/* -> litellm
#                  openai/gpt-oss-*, google/*                       -> deepinfra
#                  <any-other>/<name>                                -> vllm
#                  <bare-name-no-slash> (e.g. gpt-oss-120b)         -> arc
#                Override with the 4th arg when the default doesn't fit.
#                deepinfra requires DEEPINFRA_API_KEY; arc requires ARC_LLM_API_KEY.
#
# Override sweep points via env:
#   THRESHOLDS="0.6 0.5" eval/run_threshold_sweep.sh ...
#
# Examples:
#   eval/run_threshold_sweep.sh mistralai/Mistral-Small-3.1-24B-Instruct-2503 4,5,6,7
#   eval/run_threshold_sweep.sh Qwen/QwQ-32B 0,1,2,3
#   eval/run_threshold_sweep.sh openai/gpt-4o "" outputs_thres litellm
#   eval/run_threshold_sweep.sh openai/gpt-oss-20b "" outputs_thres        # deepinfra
#   eval/run_threshold_sweep.sh gpt-oss-120b "" outputs_thres              # arc

set -euo pipefail

MODEL="${1:?Usage: $0 <model> [gpus] [output_dir] [backend]}"

# Reject the legacy <method> positional (cg|gi) — only +GI is supported now.
# Without this guard, "$0 <model> gi 4,5,6,7" silently sets GPUS=gi and
# OUTDIR=4,5,6,7, which then crashes vLLM with an opaque pydantic error.
if [[ "${2:-}" == "gi" || "${2:-}" == "cg" ]]; then
  echo "ERROR: legacy <method> arg detected ('${2}'). The sweep is now +GI-only;" >&2
  echo "       drop that arg. New usage: $0 <model> [gpus] [output_dir] [backend]" >&2
  exit 1
fi

GPUS="${2:-0,1,2,3}"
OUTDIR="${3:-outputs_thres}"
BACKEND="${4:-}"

# Infer backend from model name if not specified (mirrors TRAIL's sweep logic)
if [[ -z "$BACKEND" ]]; then
  case "$MODEL" in
    gemini/*|openai/gpt-4*|openai/o*|anthropic/*) BACKEND="litellm"   ;;
    openai/gpt-oss-*|google/*)                    BACKEND="deepinfra" ;;
    */*)                                          BACKEND="vllm"      ;;
    *)                                            BACKEND="arc"       ;;
  esac
fi

# Select inner script (+GI only). Inner scripts mirror the TRAIL layout in
# benchmarking/eval/. Missing scripts must be ported from
# baselines/who&when/causal/run_who_and_when_graph_inject_api_{arc,deepinfra}.py
# (which already exist in MAST) into eval/.
case "$BACKEND" in
  vllm)      GI_SCRIPT="eval/full_run_eval_graph_inject.py"               ;;
  litellm)   GI_SCRIPT="eval/full_run_eval_graph_inject_api.py"           ;;
  deepinfra) GI_SCRIPT="eval/full_run_eval_graph_inject_api_deepinfra.py" ;;
  arc)       GI_SCRIPT="eval/full_run_eval_graph_inject_api_arc.py"       ;;
  *) echo "ERROR: backend must be vllm | litellm | deepinfra | arc (got: $BACKEND)" >&2; exit 1 ;;
esac

if [[ ! -f "$GI_SCRIPT" ]]; then
  echo "ERROR: inner script not found: $GI_SCRIPT" >&2
  echo "       backend=$BACKEND requires this file. Port the equivalent" >&2
  echo "       baselines/who&when/causal/run_who_and_when_graph_inject_api_{arc,deepinfra}.py" >&2
  echo "       into eval/, mirroring full_run_eval_graph_inject.py's MAST-side prompts." >&2
  exit 1
fi

# Tensor-parallel size from GPU list (vLLM only)
TP=""
if [[ "$BACKEND" == "vllm" ]]; then
  IFS=',' read -ra _gpus <<< "$GPUS"
  TP="${#_gpus[@]}"
fi

# Per-model max_model_len (vLLM only; empirical KV-cache budgets)
case "$MODEL" in
  Tongyi-Zhiwen/QwenLong-L1-32B*)                 MAX_MODEL_LEN=128000 ;;
  mistralai/Mistral-Small-3.1-24B-Instruct-2503*) MAX_MODEL_LEN=108000 ;;
  google/gemma-3-27b-it*|openai/gemma-3-27b-it*)  MAX_MODEL_LEN=108000 ;;
  openai/gpt-oss-20b*|openai/gpt-oss-120b*)       MAX_MODEL_LEN=108000 ;;
  Qwen/QwQ-32B*)                                  MAX_MODEL_LEN=40960  ;;
  *)                                              MAX_MODEL_LEN=""     ;;
esac

# Thinking flag for reasoning models
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
echo "  MAST graph-richness threshold sweep (+GI)"
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
    random) graph_flags+=(--random_edges --random_n 11 --random_seed 111) ;;
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
      --span_index \
      "${graph_flags[@]}" \
      2>&1 | tee "$log"
  else
    python "$script" \
      --model "$MODEL" \
      --output_dir "$thres_outdir" \
      ${ENABLE_THINKING} \
      --span_index \
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

  log="${LOGDIR}/${MODEL_TAG}-gi-${graph_tag}.log"
  echo "[+GI] log -> $log"
  run_one "$GI_SCRIPT" "$t" "$thres_outdir" "$log"
done

# ------------------------------------------------------------------
# Score all completed output dirs for this model
# ------------------------------------------------------------------
echo
echo "============================================================"
echo "[$(date +%T)] All thresholds done. Scoring ..."
echo "============================================================"

for t in "${THRESHOLDS[@]}"; do
  read -r sub graph_tag <<< "$(get_sub_and_tag "$t")"
  thres_outdir="${OUTDIR}/${sub}"

  gi_dir="${thres_outdir}/${MODEL_TAG}-yesno-graph-inject-codename-${graph_tag}${THINKING_SUFFIX}"
  if [[ -d "$gi_dir" ]]; then
    echo
    echo "+GI  t=$t  $gi_dir"
    python eval/calculate_scores_yesno.py --pred_dir "$gi_dir"
  else
    echo "WARNING: +GI dir not found (skipping score): $gi_dir" >&2
  fi
done

echo
echo "Sweep complete. Output tree:"
find "$OUTDIR" -maxdepth 3 -name "*-metrics.json" | sort
