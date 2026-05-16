#!/bin/bash
#SBATCH --job-name=mast-ww-w2wg-qwq32b
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=8:00:00
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
# Who&When W2+with_graph (+CG): QwQ-32B  (393 MAST AG2 traces)
#
# Notes:
#   - --enable_thinking: required for QwQ; script auto-bumps max_tokens -> 16000
#   - --max_model_len 65536: per-step W2 call = cumulative history (~20k tok)
#     + prompt (~2k) + thinking+answer (~16k) -> ~38k peak; 65536 gives headroom
#   - --batch_size 4: smaller batch for large thinking outputs avoids OOM on KV cache
#   - --tp auto-detected from CUDA_VISIBLE_DEVICES (= 4)
#   - Output dir: baselines/who&when/outputs/qwq-32b-yesno-who_and_when_w2_graph_causal_only-thinking/
# ---------------------------------------------------------------------------

"$PYTHON" -u "baselines/who&when/causal/run_who_and_when_with_graph_vllm.py" \
  --model         "$MODEL_PATH" \
  --model_tag     "$MODEL_NAME" \
  --variant       w2 \
  --causal_only \
  --enable_thinking \
  --max_model_len 65536 \
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
  --pred_dir "$OUTPUT_DIR/${MODEL_NAME}-yesno-who_and_when_w2_graph_causal_only-thinking"

echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Exit code: 0"
exit 0
