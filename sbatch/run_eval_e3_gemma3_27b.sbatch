#!/bin/bash
#SBATCH --job-name=mast-e3-gemma3-27b
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
MODEL_PATH="/common/data/models/google--gemma-3-27b-it"
MODEL_NAME="gemma-3-27b-it"

cd "$REPO_DIR"

echo "START: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
"$PYTHON" -V

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# E3: Static causal graph in prompt (single-pass), in-process vLLM
# ---------------------------------------------------------------------------
"$PYTHON" -u eval/run_eval_with_graph.py \
  --model "$MODEL_PATH" \
  --model_tag "$MODEL_NAME" \
  --tp 2 \
  --input data/annotation/annotation_ag2_filtered.jsonl \
  --output_dir outputs \
  --batch_size 8 \
  --max_model_len 32768 \
  --causal_only \
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
  --pred_dir "outputs/${MODEL_NAME}-yesno-with-graph-causal_only"

echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Exit code: 0"
exit 0
