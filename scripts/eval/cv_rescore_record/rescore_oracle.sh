#!/usr/bin/env bash
# Joint canonical evaluation of every method per CV fold, with that fold's soundscape-only model as the oracle.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../../.."
source .venv/bin/activate
SP=$1
for F in 0 1 2; do
    models=(--model pretrained-xcl "outputs/features/hsn/pretrained-xcl/responses_cv$F.pt")
    for n in soundscape-only focal-only r0 r1a r1b r2 r3 r4 tf-quantile tf-reweight; do models+=(--model "$n-cv$F"); done
    python scripts/eval/canonical_eval.py --split cv --fold "$F" "${models[@]}" --baseline pretrained-xcl \
        --oracle "soundscape-only-cv$F" --summary-name "canonical_eval_all-cv$F" > "$SP/rescore_cv$F.log" 2>&1
    echo "$(date +%H:%M) scored fold $F"
done
python "$HERE/split_joint.py"
for n in soundscape-only focal-only r0 r1a r1b r2 r3 r4 tf-quantile tf-reweight; do
    python scripts/eval/cv_aggregate.py "$n"
done
echo "RESCORE DONE"
