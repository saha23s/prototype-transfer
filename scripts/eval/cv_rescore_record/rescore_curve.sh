#!/usr/bin/env bash
# Joint canonical evaluation per CV fold of the label curve and the frozen-backbone runs, with that fold's oracle.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../../.."
source .venv/bin/activate
SP=$1
NAMES="lb100 lb250 lb500 lb1000 lb2000 r0x-nopush r0x-paperortho r0x-nopush-paperortho"
for F in 0 1 2; do
    models=(--model pretrained-xcl "outputs/features/hsn/pretrained-xcl/responses_cv$F.pt")
    for n in soundscape-only r0 $NAMES; do models+=(--model "$n-cv$F"); done
    python scripts/eval/canonical_eval.py --split cv --fold "$F" "${models[@]}" --baseline pretrained-xcl \
        --oracle "soundscape-only-cv$F" --summary-name "canonical_eval_curve-cv$F" > "$SP/curve_cv$F.log" 2>&1
    echo "$(date +%H:%M) scored fold $F"
done
python "$HERE/split_curve.py"
for n in $NAMES; do python scripts/eval/cv_aggregate.py "$n"; done
echo "RESCORE DONE"
