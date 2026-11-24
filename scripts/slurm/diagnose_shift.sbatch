#!/usr/bin/env bash
# Stage 2 GPU inference diagnostics for the PRETRAINED AudioProtoPNet (no training). Submit from the repo root:
#   sbatch scripts/slurm/diagnose_shift.sbatch                     # extract -> synthetic -> renorm -> analysis
#   TASKS="synthetic renorm" sbatch scripts/slurm/diagnose_shift.sbatch   # rerun only some tasks (caches are reused)
# Resources: 1 GPU (batch 16 needs ~4 GB), 8 CPUs for audio decoding, 32 GB RAM (waveform caches ~2 GB).
# Outputs: outputs/features/hsn/pretrained-xcl/diag_*.pt (gitignored), data/HSN/splits/soundscape_background.parquet,
#          outputs/results/hsn/summaries/diagnostics_stage2.json
#SBATCH --job-name=diag-shift
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME="$PWD/models/hf" HF_HUB_OFFLINE=1
[[ -d .venv ]] && source .venv/bin/activate
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
for t in ${TASKS:-extract synthetic renorm}; do
    python scripts/analysis/diagnose_shift.py --task "$t" --batch-size 16 --num-workers "${SLURM_CPUS_PER_TASK:-8}"
done
python scripts/analysis/analyze_diagnostics.py
