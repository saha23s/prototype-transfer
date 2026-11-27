#!/usr/bin/env bash
# GPU check of the source-free adaptation baselines before the full runs: every method at the real batch size (64) on a
# few batches, with peak GPU memory in the log. From the repo root:
#   sbatch scripts/slurm/smoke_sfda.sbatch
# Then, if it passes, one 3-fold array per method (adapt, extract, canonical evaluation):
#   TRAIN_SCRIPT=scripts/method/adapt_sfda.py sbatch --array=0-2 --constraint="48gb|80gb" --time=04:00:00 \
#       scripts/slurm/run_recipe.sbatch sfda-tent sfda-tent --method tent
# Outputs: outputs/{results,checkpoints}/smoke/ (git-ignored).
#SBATCH --job-name=sfda-smoke
#SBATCH --gres=gpu:1
#SBATCH --constraint="48gb|80gb"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME="$PWD/models/hf" HF_HUB_OFFLINE=1
[[ -d .venv ]] && source .venv/bin/activate
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
python -c 'import torch; assert torch.cuda.is_available(), "no GPU"'
python scripts/data/check_splits.py
for M in tent pseudo_label dropout_student notela shot; do
    SECONDS=0
    python scripts/method/adapt_sfda.py --method "$M" --dataset smoke --name "smoke-gpu-$M" --cv-fold 0 \
        --max-batches 3 --epochs 2 --num-workers "${SLURM_CPUS_PER_TASK:-8}"
    echo "smoke $M done in ${SECONDS}s"
    test -s "outputs/checkpoints/smoke/smoke-gpu-$M/best.pt"
    rm "outputs/checkpoints/smoke/smoke-gpu-$M/best.pt"   # 335 MB each; the log and metrics stay
done
echo "SFDA SMOKE PASSED"
