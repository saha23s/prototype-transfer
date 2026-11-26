#!/usr/bin/env bash
# The two supervised baselines trained on the full train splits and scored on TEST, with the recipe of the committed
# cross-validation baselines (--select last, seed 0, otherwise defaults). From the repo root:
#   sbatch scripts/slurm/final_baselines.sbatch
# 1. quick test: both domains, 2 batches per epoch, saved under dataset "smoke"
# 2. pretrained responses on the test splits (skipped if present)
# 3. soundscape-only (the oracle) first, then focal-only, each through run_recipe.sbatch (train, extract, canonical eval)
# Any failing step fails the job. Resources: 1 GPU, 8 CPUs, 32 GB RAM; one recipe run has taken 17 min to 2 h, by GPU type.
# Outputs: outputs/{results,checkpoints,features}/hsn/{soundscape-only,focal-only}, pretrained-xcl's test responses and
# canonical_test section, summaries/canonical_eval_{soundscape-only,focal-only}.json, outputs/{results,checkpoints}/smoke/
#SBATCH --job-name=final-baselines
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME="$PWD/models/hf" HF_HUB_OFFLINE=1
[[ -d .venv ]] && source .venv/bin/activate
export PYTHONPATH="$PWD/src"   # run this checkout's code, also from a git worktree that shares another checkout's .venv
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
SPL=data/HSN/splits BASE=outputs/features/hsn/pretrained-xcl/responses.pt
W=${SLURM_CPUS_PER_TASK:-8}
python -c 'import prototype_transfer, torch; print(prototype_transfer.__file__); assert torch.cuda.is_available(), "no GPU"'
python scripts/data/check_splits.py

for DOM in soundscape focal; do
    python scripts/train/train_protopnet.py --dataset smoke --name "smoke-final-$DOM" --train-domain "$DOM" --select last \
        --max-train-batches 2 --max-eval-samples 64 --num-workers "$W"
    test -s "outputs/checkpoints/smoke/smoke-final-$DOM/best.pt"
done

[[ -f $BASE ]] || python scripts/analysis/extract_prototype_activations.py pretrained-xcl --focal-manifest $SPL/focal_test.parquet \
    --soundscape-manifest $SPL/soundscape_test.parquet --min-soundscape-pos 5 --num-workers "$W"

bash scripts/slurm/run_recipe.sbatch soundscape-only oracle --train-domain soundscape --select last    # rungs as in the CV runs
bash scripts/slurm/run_recipe.sbatch focal-only baseline --train-domain focal --select last
echo "FINAL BASELINES DONE"
