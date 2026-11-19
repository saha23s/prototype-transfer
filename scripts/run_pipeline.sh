#!/usr/bin/env bash
# End-to-end pipeline: download -> preprocess -> inference -> figures.
#
#   scripts/run_pipeline.sh --smoke   # tiny run (16 clips/type), writes to outputs/smoke/
#   scripts/run_pipeline.sh           # full run (all focal shards), writes to outputs/
#
# Run on a compute node (see scripts/slurm/pipeline.sbatch), not a login node.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

SMOKE=0
[[ "${1:-}" == "--smoke" ]] && SMOKE=1

# Keep the HF model cache next to the data (models/ -> big disk) instead of ~/.cache
export HF_HOME="${HF_HOME:-$REPO_DIR/models/hf}"
[[ -d .venv ]] && source .venv/bin/activate

if (( SMOKE )); then
    DOWNLOAD_ARGS=()                                   # first focal shard + soundscapes
    INFER_ARGS=(--max-focal-samples 16 --max-soundscape-samples 16
                --output-dir outputs/smoke/results)
    FIG_DIR=outputs/smoke/figures
    RES_DIR=outputs/smoke/results
else
    DOWNLOAD_ARGS=(--all-focal)
    INFER_ARGS=(--max-focal-samples 1050 --max-soundscape-samples 1000)
    FIG_DIR=outputs/figures
    RES_DIR=outputs/results
fi

echo "[1/4] Download HSN"
python scripts/download_hsn.py "${DOWNLOAD_ARGS[@]}"
echo "[2/4] Preprocess (manifests)"
python scripts/preprocess_hsn.py
echo "[3/4] Inference"
python scripts/run_inference.py "${INFER_ARGS[@]}"
echo "[4/4] Figures"
python scripts/visualize_analysis.py \
    --cache-path "$RES_DIR/eval_cache.pt" \
    --summary-path "$RES_DIR/comparison_summary.json" \
    --figures-dir "$FIG_DIR"

echo "Done. Results: $RES_DIR  Figures: $FIG_DIR"
