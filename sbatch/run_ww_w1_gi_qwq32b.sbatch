#!/bin/bash
#SBATCH --job-name=mast-ww-w1gi-qwq32b
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH -o slurm.%x.%j.out
#SBATCH -e slurm.%x.%j.err

set -euo pipefail

module load Miniconda3

ENV_PATH="/projects/slmreasoning/junh/envs/causal/"
PYTHON="$ENV_PATH/bin/python"
REPO_DIR="/projects/slmreasoning/junh/causal-error/CausalMAST"
# Model is cached locally — no HuggingFace download needed.
MODEL_PATH="/common/data/models/qwen--QwQ-32B"
MODEL_NAME="qwq-32b"
OUTPUT_DIR="$REPO_DIR/baselines/who&when/outputs"

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd "$REPO_DIR"

echo "START: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
"$PYTHON" -V

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# Who&When W1+graph_inject (+GI): QwQ-32B  (393 MAST AG2 traces)
#
# Notes:
#   - --enable_thinking: required for QwQ; script auto-bumps max_tokens -> 16000
#   - --max_model_len 108000: W1 sends the full trace in one call (not per-step),
#     so input can be 50-80k tokens; 108000 - 16000 (thinking) = 92k input budget
#   - --batch_size 4: conservative for large thinking outputs
#   - --tp auto-detected from CUDA_VISIBLE_DEVICES (= 4)
#   - W1+GI = Pass1 (1 call/trace) + Pass2 (1 call for triggered traces)
#     -> ~500-700 total calls vs W2's N_steps*393; estimated ~4h
#   - Output dir: baselines/who&when/outputs/qwq-32b-yesno-who_and_when_w1_graph_inject_causal_only-thinking/
# ---------------------------------------------------------------------------

"$PYTHON" -u "baselines/who&when/causal/run_who_and_when_graph_inject_vllm.py" \
  --model         "$MODEL_PATH" \
  --model_tag     "$MODEL_NAME" \
  --variant       w1 \
  --causal_only \
  --enable_thinking \
  --max_model_len 40960 \
  --batch_size    4 \
  --output_dir    "$OUTPUT_DIR" \
  "$@"

EC=$?
if [ $EC -ne 0 ]; then
  echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "Exit code: $EC"
  exit $EC
fi

# ---------------------------------------------------------------------------
# Score results
# ---------------------------------------------------------------------------
"$PYTHON" -u eval/calculate_scores_yesno.py \
  --annotation data/annotation/annotation_ag2_filtered.jsonl \
  --pred_dir "$OUTPUT_DIR/${MODEL_NAME}-yesno-who_and_when_w1_graph_inject_causal_only-thinking"

echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Exit code: 0"
exit 0
