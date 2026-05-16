#!/bin/bash
#SBATCH --job-name=mast-ww-w1cg-mistral24b
#SBATCH --account=slmreasoning
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH -o slurm.%x.%j.out
#SBATCH -e slurm.%x.%j.err

set -euo pipefail

module load Miniconda3

ENV_PATH="/projects/slmreasoning/junh/envs/causal/"
PYTHON="$ENV_PATH/bin/python"
REPO_DIR="/projects/slmreasoning/junh/causal-error/CausalMAST"
MODEL_PATH="/common/data/models/mistralai--Mistral-Small-3.1-24B-Instruct-2503"
MODEL_NAME="mistralai-Mistral-Small-3.1-24B-Instruct-2503"
OUTPUT_DIR="$REPO_DIR/baselines/who&when/outputs"

export CUDA_VISIBLE_DEVICES=0,1

cd "$REPO_DIR"

echo "START: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
"$PYTHON" -V

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# Who&When W1+with_graph (+CG): Mistral-Small-3.1-24B  (393 MAST AG2 traces)
# Using local cache — no HuggingFace download needed.
# Output: mistralai-Mistral-Small-3.1-24B-Instruct-2503-yesno-who_and_when_w1_graph_causal_only/
# ---------------------------------------------------------------------------

"$PYTHON" -u "baselines/who&when/causal/run_who_and_when_with_graph_vllm.py" \
  --model         "$MODEL_PATH" \
  --model_tag     "$MODEL_NAME" \
  --variant       w1 \
  --causal_only \
  --max_model_len 108000 \
  --batch_size    8 \
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
  --pred_dir "$OUTPUT_DIR/${MODEL_NAME}-yesno-who_and_when_w1_graph_causal_only"

echo "END: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Exit code: 0"
exit 0
