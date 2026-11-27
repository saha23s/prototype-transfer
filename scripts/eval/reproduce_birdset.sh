#!/usr/bin/env bash
# Score the published AudioProtoPNet-5-XCL on HSN: all 12,000 soundscape clips (compare with the model card, docs/BIRDSET.md),
# then the 25 test sites of each adaptation/test partition (the no-adaptation reference for UDA).
# Run from the repo root after scripts/data/make_splits.py --partitions 3. On filo: WRAP=~/.local/bin/local-training-3090.
set -euo pipefail
W=${WRAP:-}
UV=${UV:-uv}
SPLITS=data/HSN/splits
$W $UV run python scripts/analysis/extract_prototype_activations.py pretrained-xcl --tag full --min-soundscape-pos 1
$UV run python scripts/eval/birdset_metrics.py pretrained-xcl --tag full
for k in 0 1 2; do
  $W $UV run python scripts/analysis/extract_prototype_activations.py pretrained-xcl --tag p$k --min-soundscape-pos 1 \
    --focal-manifest $SPLITS/focal_test.parquet --soundscape-manifest $SPLITS/soundscape_p${k}_test.parquet
  $UV run python scripts/eval/birdset_metrics.py pretrained-xcl --tag p$k
done
$UV run python scripts/report/results.py show pretrained-xcl
