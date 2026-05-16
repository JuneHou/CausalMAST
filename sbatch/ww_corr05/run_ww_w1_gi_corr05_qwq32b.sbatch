#!/bin/bash
#SBATCH --job-name=mast-ww-w1gi-c05-qwq32b
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
MODEL_PATH="/common/data/models/qwen--QwQ-32B"
MODEL_NAME="qwq-32b"
OUTPUT_DIR="$REPO_DIR/baselines/who&when/causal/outputs"

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd "$REPO_DIR"

echo "START: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
"$PYTHON" -V

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# Who&When W1+GI (corr>=0.50, 25 edges): QwQ-32B
# Adopted from sbatch/run_ww_w1_gi_qwq32b.sbatch:
#   graph:  --causal_only -> --corr_threshold 0.5
#   output: baselines/who&when/outputs -> baselines/who&when/causal/outputs
# Output dir: qwq-32b-yesno-who_and_when_w1_graph_inject_corr0.5-thinking/
# ---------------------------------------------------------------------------

"$PYTHON" -u "baselines/who&when/causal/run_who_and_when_graph_inject_vllm.py" \
  --model         "$MODEL_PATH" \
  --model_tag     "$MODEL_NAME" \
  --variant       w1 \
  --corr_threshold 0.5 \
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

"$PYTHON" -u eval/calculate_scores_yesno.py \
  --annotation data/annotation/annotation_ag2_filtered.jsonl \
  --pred_dir "$OUTPUT_DIR/${MODEL_NAME}-yesno-who_and_when_w1_graph_inject_corr0.5-thinking"

echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Exit code: 0"
exit 0
