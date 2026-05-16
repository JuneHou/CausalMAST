#!/bin/bash
#SBATCH --job-name=mast-ww-w2cg-c05-mistral24b
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH -o slurm.%x.%j.out
#SBATCH -e slurm.%x.%j.err

set -euo pipefail

module load Miniconda3

ENV_PATH="/projects/slmreasoning/junh/envs/causal/"
PYTHON="$ENV_PATH/bin/python"
REPO_DIR="/projects/slmreasoning/junh/causal-error/CausalMAST"
MODEL_PATH="/common/data/models/mistralai--Mistral-Small-3.1-24B-Instruct-2503"
MODEL_NAME="mistralai-Mistral-Small-3.1-24B-Instruct-2503"
OUTPUT_DIR="$REPO_DIR/baselines/who&when/causal/outputs"

export CUDA_VISIBLE_DEVICES=0,1

cd "$REPO_DIR"

echo "START: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
"$PYTHON" -V

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# Who&When W2+CG (corr>=0.50, 25 edges): Mistral-Small-3.1-24B
# Adopted from sbatch/w1_cg/run_ww_w1_cg_mistral24b.sbatch:
#   variant: w1 -> w2  (step-by-step OR-aggregate)
#   graph:   --causal_only -> --corr_threshold 0.5
#   output:  baselines/who&when/outputs -> baselines/who&when/causal/outputs
# Output dir: mistralai-Mistral-Small-3.1-24B-Instruct-2503-yesno-who_and_when_w2_graph_corr0.5/
# ---------------------------------------------------------------------------

"$PYTHON" -u "baselines/who&when/causal/run_who_and_when_with_graph_vllm.py" \
  --model         "$MODEL_PATH" \
  --model_tag     "$MODEL_NAME" \
  --variant       w2 \
  --corr_threshold 0.5 \
  --max_model_len 61440 \
  --batch_size    8 \
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
  --pred_dir "$OUTPUT_DIR/${MODEL_NAME}-yesno-who_and_when_w2_graph_corr0.5"

echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Exit code: 0"
exit 0
