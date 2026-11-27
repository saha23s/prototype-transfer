#!/usr/bin/env bash
# From raw data to the baseline numbers: download -> manifests -> canonical splits -> pretrained-model responses -> canonical evaluation.
#
#   scripts/run_pipeline.sh --smoke   # tiny check of the plumbing (16 clips per domain); writes only under the dataset name "smoke"
#   scripts/run_pipeline.sh           # full run (all focal shards); results in outputs/results/hsn/
#
# Run on a compute node (see scripts/slurm/pipeline.sbatch), not a login node. Training a model is a separate step
# (scripts/train/train_protopnet.py or scripts/slurm/run_recipe.sbatch); this script produces what every run is compared against.
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
    echo "[1/3] Download HSN (first shard)"
    python scripts/data/download_hsn.py "${DOWNLOAD_ARGS[@]}"
    echo "[2/3] Build manifests"
    python scripts/data/preprocess_hsn.py
    echo "[3/3] Pretrained-model responses on 16 clips per domain (dataset 'smoke'; the canonical splits are not built)"
    python scripts/analysis/extract_prototype_activations.py pretrained-xcl --dataset smoke \
        --focal-manifest data/HSN/preprocessed/focal_manifest.parquet \
        --soundscape-manifest data/HSN/preprocessed/soundscape_manifest.parquet \
        --max-focal-samples 16 --max-soundscape-samples 16 --batch-size 4 --num-workers 2 --min-soundscape-pos 1
    python scripts/report/results.py --dataset smoke show pretrained-xcl
    echo "Smoke run done: outputs/results/smoke/ (safe to delete)"
    exit 0
fi

echo "[1/5] Download HSN"
python scripts/data/download_hsn.py --all-focal
echo "[2/5] Build manifests (full)"
python scripts/data/preprocess_hsn.py --full
echo "[3/5] Canonical splits"
python scripts/data/make_splits.py
python scripts/data/check_splits.py
echo "[4/5] Pretrained-model responses on the test splits"
SPL=data/HSN/splits
python scripts/analysis/extract_prototype_activations.py pretrained-xcl \
    --focal-manifest $SPL/focal_test.parquet --soundscape-manifest $SPL/soundscape_test.parquet --min-soundscape-pos 5
echo "[5/5] Canonical evaluation (protocol v1) of the pretrained model"
python scripts/eval/canonical_eval.py --model pretrained-xcl --baseline pretrained-xcl

echo "Done. python scripts/report/results.py show pretrained-xcl"
