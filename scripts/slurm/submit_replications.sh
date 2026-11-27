#!/usr/bin/env bash
# Cross-validation reruns that test the weaker conclusions of the first round. From the repo root:
#   scripts/slurm/submit_replications.sh                 # print the sbatch commands (default; submits nothing)
#   scripts/slurm/submit_replications.sh --submit        # submit them
#   scripts/slurm/submit_replications.sh [--submit] seeds|frozen     # only one of the two groups
# seeds   seeds 1 and 2 (seed 0 exists) of plain focal fine-tuning, real background + overlap, and 250 labels. The seed also
#         draws the labeled clips, so the 250-label reruns vary the subset too.          18 runs, about 1.25-2 h each
# frozen  the label curve with a frozen backbone (100 ... 2000 and all labels, seed 0).  18 runs, about 0.3-0.6 h each (estimate)
# Each line is a 3-fold array job of scripts/slurm/run_recipe.sbatch (train, extract, canonical evaluation against the
# pretrained model and soundscape-only-cvF). Afterwards: python scripts/eval/cv_aggregate.py NAME for every NAME below.
# Run from a clean checkout so that the saved run configs record the code they came from.
set -euo pipefail
SUBMIT=0
if [[ ${1:-} == --submit ]]; then SUBMIT=1; shift; fi
GROUP=${1:-all}
[[ $GROUP == all || $GROUP == seeds || $GROUP == frozen ]] || { echo "unknown group '$GROUP' (seeds|frozen)" >&2; exit 2; }

job() {  # time_limit NAME RUNG [train_protopnet.py args ...]
    local limit=$1; shift
    local cmd=(sbatch --array=0-2 --time="$limit" scripts/slurm/run_recipe.sbatch "$@")
    if (( SUBMIT )); then "${cmd[@]}"; else printf '%q ' "${cmd[@]}"; echo; fi
}

if [[ $GROUP != frozen ]]; then
    for S in 1 2; do
        job 03:00:00 "focal-only-s$S" baseline --train-domain focal --select last --seed "$S"
        job 03:00:00 "r3-s$S" R3 --train-domain focal --select last --seed "$S" --bg-kind real \
            --bg-manifest 'data/HSN/splits/soundscape_background_cv{fold}.parquet' --overlap-prob 0.5
        job 03:00:00 "lb250-s$S" label-budget-250 --train-domain soundscape --select last --seed "$S" --train-labels 250
    done
fi
if [[ $GROUP != seeds ]]; then
    for N in 100 250 500 1000 2000; do
        job 02:00:00 "fz-lb$N" "frozen-label-budget-$N" --train-domain soundscape --select last --anchored --train-labels "$N"
    done
    job 02:00:00 fz-soundscape-only frozen-oracle --train-domain soundscape --select last --anchored
fi
