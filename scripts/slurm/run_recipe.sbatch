#!/usr/bin/env bash
# One run of the training-recipe ladder: train -> extract responses -> canonical evaluation. From the repo root:
#   sbatch scripts/slurm/run_recipe.sbatch NAME RUNG [train_protopnet.py args ...]                  # final run, scored on test
#   sbatch --array=0-2 scripts/slurm/run_recipe.sbatch NAME RUNG [train_protopnet.py args ...]      # 3-fold cross-validation
# Cross-validation (array task F = held-out fold): the run is called NAME-cvF, trains without fold F's soundscape sites
# (--cv-fold F) and is scored on them (canonical_eval --split cv --fold F). "{fold}" in the arguments becomes F, e.g.
#   sbatch --array=0-2 scripts/slurm/run_recipe.sbatch r1b R1b --train-domain focal --bg-kind real \
#       --bg-manifest data/HSN/splits/soundscape_background_cv{fold}.parquet
# When the three tasks are done:  python scripts/cv_aggregate.py NAME      (mean ± sd over folds)
# Other examples:
#   sbatch scripts/slurm/run_recipe.sbatch r0 R0 --train-domain focal --anchored
#   sbatch --array=0-2 scripts/slurm/run_recipe.sbatch lb500 label-budget-500 --train-domain soundscape --train-labels 500 --select last
# Needs data/HSN/splits (scripts/make_splits.py), outputs/features/responses_pretrained-xcl.pt (the baseline on test; see
# scripts/evaluate_models.sh) and, for background / pseudo-label rungs, the manifests of scripts/make_background_manifest.py
# (per fold with --fold F). Pretrained responses on each held-out fold are extracted here if missing.
# Resources: 1 GPU, 8 CPUs for audio decoding, 32 GB RAM; the earlier focal-only / soundscape-only runs took 13 / 23 min.
# Outputs: outputs/runs/RUN/, outputs/results/train_RUN.json, outputs/features/responses_RUN.pt (gitignored),
#          outputs/results/canonical_eval_RUN.json, a row in outputs/runs/registry.json (RUN = NAME or NAME-cvF).
#SBATCH --job-name=recipe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ $# -lt 2 ]]; then
    sed -n '2,4p' "${BASH_SOURCE[0]}" >&2; exit 2
fi
NAME=$1 RUNG=$2; shift 2
export HF_HOME="$PWD/models/hf" HF_HUB_OFFLINE=1
[[ -d .venv ]] && source .venv/bin/activate
SPL=data/HSN/splits FEAT=outputs/features
FOLD=${SLURM_ARRAY_TASK_ID:-}
if [[ -n $FOLD ]]; then
    RUN=$NAME-cv$FOLD
    set -- "${@//\{fold\}/$FOLD}"
    TRAIN_EXTRA=(--cv-fold "$FOLD")
    SC_MANIFEST=$SPL/soundscape_cv${FOLD}_heldout.parquet FOCAL_MANIFEST=$SPL/focal_val.parquet
    BASE=$FEAT/responses_pretrained-xcl_cv$FOLD.pt EVAL=(--split cv --fold "$FOLD")
else
    RUN=$NAME
    TRAIN_EXTRA=()
    SC_MANIFEST=$SPL/soundscape_test.parquet FOCAL_MANIFEST=$SPL/focal_test.parquet
    BASE=$FEAT/responses_pretrained-xcl.pt EVAL=()
fi
extract() {  # features-path results-path [checkpoint]
    python scripts/extract_prototype_activations.py --focal-manifest "$FOCAL_MANIFEST" --soundscape-manifest "$SC_MANIFEST" \
        --features-path "$1" --results-path "$2" --min-soundscape-pos 5 ${3:+--checkpoint "$3"}
}
if [[ ! -f $BASE ]]; then   # pretrained responses on the held-out fold are cheap to extract; the test baseline must exist
    [[ -n $FOLD ]] || { echo "missing $BASE (baseline responses; see scripts/evaluate_models.sh)" >&2; exit 1; }
    extract "$BASE" "outputs/results/prototype_transfer_pretrained-xcl_cv$FOLD.json"
fi
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/train_protopnet.py --name "$RUN" --rung "$RUNG" --num-workers "${SLURM_CPUS_PER_TASK:-8}" "${TRAIN_EXTRA[@]}" "$@"
extract "$FEAT/responses_$RUN.pt" "outputs/results/prototype_transfer_$RUN.json" "outputs/runs/$RUN/best.pt"

MODELS=(--model pretrained-xcl "$BASE")
[[ -z $FOLD && -f $FEAT/responses_soundscape-only.pt ]] && MODELS+=(--model soundscape-only "$FEAT/responses_soundscape-only.pt")
MODELS+=(--model "$RUN" "$FEAT/responses_$RUN.pt")
python scripts/canonical_eval.py "${MODELS[@]}" --baseline pretrained-xcl --oracle soundscape-only "${EVAL[@]}" \
    --out "outputs/results/canonical_eval_$RUN.json"
