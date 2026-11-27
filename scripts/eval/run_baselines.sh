#!/usr/bin/env bash
# Wave 1 of the baselines: every method on partitions 0-2, fixed defaults. Run from the repo root.
# On filo: WRAP=~/.local/bin/local-training-3090 UV=~/.local/bin/uv bash scripts/eval/run_baselines.sh
set -euo pipefail
W=${WRAP:-}
UV=${UV:-uv}
for k in 0 1 2; do
  $W $UV run python scripts/eval/adapt_baselines.py --partition $k --method source
  $W $UV run python scripts/eval/adapt_baselines.py --partition $k --method tent   --params norm
  $W $UV run python scripts/eval/adapt_baselines.py --partition $k --method pl     --params norm
  $W $UV run python scripts/eval/adapt_baselines.py --partition $k --method oracle --params norm
  $W $UV run python scripts/eval/adapt_baselines.py --partition $k --method oracle --params all --epochs 3
done
