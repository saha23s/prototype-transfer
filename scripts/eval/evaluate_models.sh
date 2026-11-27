#!/usr/bin/env bash
# Evaluate pretrained XCL, focal-only and soundscape-only models on the held-out TEST splits,
# then compare them. Run on filo after scripts/train/train_protopnet.py (both domains).
#   scripts/eval/evaluate_models.sh            (uses the GPU reservation wrapper if present)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HF_HOME="${HF_HOME:-$PWD/models/hf}"
[[ -d .venv ]] && source .venv/bin/activate
RUN=(); [[ -x "$HOME/.local/bin/local-training-3090" ]] && RUN=("$HOME/.local/bin/local-training-3090")

SPL=data/HSN/splits
extract() {  # name [checkpoint]
    local name=$1 ck=${2:-}
    "${RUN[@]}" python scripts/analysis/extract_prototype_activations.py \
        --focal-manifest $SPL/focal_test.parquet --soundscape-manifest $SPL/soundscape_test.parquet \
        --features-path outputs/features/responses_${name}.pt \
        --results-path outputs/results/prototype_transfer_${name}.json \
        --min-soundscape-pos 5 ${ck:+--checkpoint "$ck"}
}
extract pretrained-xcl
extract focal-only outputs/runs/focal-only/best.pt
extract soundscape-only outputs/runs/soundscape-only/best.pt

python scripts/eval/compare_models.py \
    --model pretrained-xcl xcl outputs/features/responses_pretrained-xcl.pt \
    --model focal-only focal outputs/features/responses_focal-only.pt \
    --model soundscape-only soundscape outputs/features/responses_soundscape-only.pt
