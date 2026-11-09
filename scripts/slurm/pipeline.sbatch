#!/usr/bin/env bash
# Full pipeline on SLURM. Submit from the repo root:
#   sbatch scripts/slurm/pipeline.sbatch            # full run
#   sbatch scripts/slurm/pipeline.sbatch --smoke    # quick check
# Add your cluster's partition/account via `sbatch -p <partition> -A <account> ...`.
#SBATCH --job-name=hsn-protopnet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

srun scripts/run_pipeline.sh "$@"
