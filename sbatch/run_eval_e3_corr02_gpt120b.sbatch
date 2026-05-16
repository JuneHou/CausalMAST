#!/bin/bash
#SBATCH --job-name=mast-e3-corr02-gpt120b
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
MODEL_PATH="/common/data/models/openai--gpt-oss-120b"
MODEL_NAME="gpt-oss-120b"

cd "$REPO_DIR"

echo "START: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
"$PYTHON" -V

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# E3: Static graph, codename format, corr threshold 0.20 (18 edges)
# ---------------------------------------------------------------------------
"$PYTHON" -u eval/full_run_eval_with_graph.py \
  --model "$MODEL_PATH" \
  --model_tag "$MODEL_NAME" \
  --tp 4 \
  --input data/annotation/annotation_ag2_filtered.jsonl \
  --output_dir outputs_corr \
  --batch_size 4 \
  --max_model_len 32768 \
  --edge_threshold 0.20 \
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
  --pred_dir "outputs_corr/${MODEL_NAME}-yesno-with-graph-codename-t0.2"

echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Exit code: 0"
exit 0
