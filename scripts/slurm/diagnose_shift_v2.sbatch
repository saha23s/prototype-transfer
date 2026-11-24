#!/usr/bin/env bash
# Stage 2 follow-up (GPU inference, no training): does the synthetic shift get closer to the real one when
#   A  the focal window is random instead of the loudest 5 s       (--focal-window random)
#   B  the bird is made quieter over a fixed-level real background (--mix-mode attenuate), SNR down to -25 dB
#   C  both together
# Reuses outputs/features/hsn/pretrained-xcl/diag_responses.pt, diag_renorm.pt and data/HSN/splits/soundscape_background.parquet from
# scripts/slurm/diagnose_shift.sbatch. Submit from the repo root:  sbatch scripts/slurm/diagnose_shift_v2.sbatch
# Resources: 1 GPU (batch 16 ~4 GB), 8 CPUs for audio decoding, 32 GB RAM (waveform caches ~2 GB). ~35 min expected.
# Outputs: outputs/features/hsn/pretrained-xcl/diag_synthetic_{win,atten,win_atten}.pt, outputs/results/hsn/summaries/diagnostics_stage2_{...}.json
#SBATCH --job-name=diag-shift2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME="$PWD/models/hf" HF_HUB_OFFLINE=1
[[ -d .venv ]] && source .venv/bin/activate
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv

OV=(--overlap-rels -10 -5 0 5 --both-snrs -10 -20 --both-rels -5 -5)
run() {  # tag, extra args...
    local tag=$1; shift
    python scripts/analysis/diagnose_shift.py --task synthetic --batch-size 16 --num-workers "${SLURM_CPUS_PER_TASK:-8}" --tag "_$tag" "$@" "${OV[@]}"
    python scripts/analysis/analyze_diagnostics.py --synthetic "diag_synthetic_$tag.pt" --summary-name "diagnostics_stage2_$tag"
}
run win        --focal-window random --mix-mode add       --bg-snrs 0 -5 -10 -15 -20 -25
run atten      --focal-window peak_energy --mix-mode attenuate --bg-snrs 10 0 -5 -10 -15 -20 -25
run win_atten  --focal-window random --mix-mode attenuate --bg-snrs 10 0 -5 -10 -15 -20 -25
