#!/usr/bin/env bash
# Evaluate pretrained XCL, focal-only and soundscape-only models on the held-out TEST splits,
# then compare them. Run on filo after scripts/train/train_protopnet.py (both domains).
#   scripts/eval/evaluate_models.sh            (uses the GPU reservation wrapper if present)
# Results go to the results store (docs/RESULTS.md): responses in outputs/features/hsn/<run_id>/, statistics in
# outputs/results/hsn/runs/<run_id>/, the comparisons in outputs/results/hsn/summaries/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HF_HOME="${HF_HOME:-$PWD/models/hf}"
[[ -d .venv ]] && source .venv/bin/activate
RUN=(); [[ -x "$HOME/.local/bin/local-training-3090" ]] && RUN=("$HOME/.local/bin/local-training-3090")

SPL=data/HSN/splits
extract() {  # run_id
    "${RUN[@]}" python scripts/analysis/extract_prototype_activations.py "$1" \
        --focal-manifest $SPL/focal_test.parquet --soundscape-manifest $SPL/soundscape_test.parquet \
        --min-soundscape-pos 5
}
extract pretrained-xcl
extract focal-only
extract soundscape-only

python scripts/eval/compare_models.py --model pretrained-xcl xcl --model focal-only focal --model soundscape-only soundscape

