#!/usr/bin/env bash
# One run of the training-recipe ladder: train -> extract responses -> canonical evaluation. From the repo root:
#   sbatch scripts/slurm/run_recipe.sbatch NAME RUNG [train_protopnet.py args ...]                  # final run, scored on test
#   sbatch --array=0-2 scripts/slurm/run_recipe.sbatch NAME RUNG [train_protopnet.py args ...]      # 3-fold cross-validation
# Cross-validation (array task F = held-out fold): the run is called NAME-cvF, trains without fold F's soundscape sites
# (--cv-fold F) and is scored on them (canonical_eval --split cv --fold F). "{fold}" in the arguments becomes F, e.g.
#   sbatch --array=0-2 scripts/slurm/run_recipe.sbatch r1b R1b --train-domain focal --bg-kind real \
#       --bg-manifest data/HSN/splits/soundscape_background_cv{fold}.parquet
# When the three tasks are done:  python scripts/eval/cv_aggregate.py NAME      (mean ± sd over folds)
# Other examples:
#   sbatch scripts/slurm/run_recipe.sbatch r0 R0 --train-domain focal --anchored
#   sbatch --array=0-2 scripts/slurm/run_recipe.sbatch lb500 label-budget-500 --train-domain soundscape --train-labels 500 --select last
# Needs data/HSN/splits (scripts/data/make_splits.py), outputs/features/hsn/pretrained-xcl/responses.pt (the baseline on test; see
# scripts/eval/evaluate_models.sh) and, for background / pseudo-label rungs, the manifests of scripts/method/make_background_manifest.py
# (per fold with --fold F). Pretrained responses on each held-out fold are extracted here if missing.
# Resources: 1 GPU, 8 CPUs for audio decoding, 32 GB RAM; the earlier focal-only / soundscape-only runs took 13 / 23 min.
# Outputs (RUN = NAME or NAME-cvF; see docs/RESULTS.md): outputs/results/hsn/runs/RUN/ (config, train log, metrics incl. the
#          canonical_test or canonical_cv<F> section), outputs/features/hsn/RUN/responses.pt and outputs/checkpoints/hsn/RUN/
#          (gitignored), summaries/canonical_eval_RUN.json, and RUN's entry in outputs/results/hsn/registry.jsonl.
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
SPL=data/HSN/splits FEAT=outputs/features/hsn
FOLD=${SLURM_ARRAY_TASK_ID:-}
if [[ -n $FOLD ]]; then
    RUN=$NAME-cv$FOLD
    set -- "${@//\{fold\}/$FOLD}"
    TRAIN_EXTRA=(--cv-fold "$FOLD")
    SC_MANIFEST=$SPL/soundscape_cv${FOLD}_heldout.parquet FOCAL_MANIFEST=$SPL/focal_val.parquet
    BASE=$FEAT/pretrained-xcl/responses_cv$FOLD.pt BASE_TAG=(--tag "cv$FOLD") EVAL=(--split cv --fold "$FOLD")
else
    RUN=$NAME
    TRAIN_EXTRA=()
    SC_MANIFEST=$SPL/soundscape_test.parquet FOCAL_MANIFEST=$SPL/focal_test.parquet
    BASE=$FEAT/pretrained-xcl/responses.pt BASE_TAG=() EVAL=()
fi
extract() {  # run_id [extra args]   (a trained run is read from outputs/checkpoints/hsn/<run_id>/best.pt)
    local run=$1; shift
    python scripts/analysis/extract_prototype_activations.py "$run" --focal-manifest "$FOCAL_MANIFEST" --soundscape-manifest "$SC_MANIFEST" \
        --min-soundscape-pos 5 "$@"
}
if [[ ! -f $BASE ]]; then   # pretrained responses on the held-out fold are cheap to extract; the test baseline must exist
    [[ -n $FOLD ]] || { echo "missing $BASE (the pretrained model's responses on the test split). Create it once with:
  python scripts/analysis/extract_prototype_activations.py pretrained-xcl --focal-manifest $SPL/focal_test.parquet --soundscape-manifest $SPL/soundscape_test.parquet --min-soundscape-pos 5" >&2; exit 1; }
    extract pretrained-xcl "${BASE_TAG[@]}"
fi
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/train/train_protopnet.py --name "$RUN" --rung "$RUNG" --num-workers "${SLURM_CPUS_PER_TASK:-8}" "${TRAIN_EXTRA[@]}" "$@"
extract "$RUN"

MODELS=(--model pretrained-xcl "$BASE")
[[ -z $FOLD && -f $FEAT/soundscape-only/responses.pt ]] && MODELS+=(--model soundscape-only)
MODELS+=(--model "$RUN")
python scripts/eval/canonical_eval.py "${MODELS[@]}" --baseline pretrained-xcl --oracle soundscape-only "${EVAL[@]}" \
    --summary-name "canonical_eval_$RUN"
