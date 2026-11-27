#!/usr/bin/env bash
# Preparation and quick test before the cross-validation wave. Submit from the repo root:
#   sbatch scripts/slurm/prep_cv.sbatch
# 1. pretrained responses on the held-out sites of each fold (skipped if present)
# 2. per-fold background / pseudo-label manifests. Fold F adapts on the other two folds' held-out clips (together exactly
#    soundscape_cv{F}_train); only audio_path and probs are read. Checked against fold F's held-out clips.
# 3. quick test: the oracle and R4 settings on fold 0, 2 batches per epoch, saved under dataset "smoke"
# Any failing step fails the job. Outputs: outputs/features/hsn/pretrained-xcl/responses_cv{0,1,2}.pt,
# data/HSN/splits/soundscape_{background,pseudo}_cv{0,1,2}.parquet, outputs/{results,checkpoints}/smoke/
#SBATCH --job-name=prep-cv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME="$PWD/models/hf" HF_HUB_OFFLINE=1
[[ -d .venv ]] && source .venv/bin/activate
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv
SPL=data/HSN/splits FEAT=outputs/features/hsn/pretrained-xcl
W=${SLURM_CPUS_PER_TASK:-8}
python scripts/data/check_splits.py

for F in 0 1 2; do
    [[ -f $FEAT/responses_cv$F.pt ]] || python scripts/analysis/extract_prototype_activations.py pretrained-xcl --tag "cv$F" \
        --focal-manifest $SPL/focal_val.parquet --soundscape-manifest $SPL/soundscape_cv${F}_heldout.parquet \
        --min-soundscape-pos 5 --num-workers "$W"
done

for F in 0 1 2; do
    SOURCES=()
    for K in 0 1 2; do
        [[ $K == "$F" ]] || SOURCES+=(--source "$FEAT/responses_cv$K.pt:soundscape")
    done
    python scripts/method/make_background_manifest.py --fold "$F" "${SOURCES[@]}"
done
python - <<'EOF'
from pathlib import Path
import pandas as pd
from prototype_transfer.data.splits import check_no_heldout
spl = Path("data/HSN/splits")
for f in range(3):
    for kind in ("background", "pseudo"):
        m = spl / f"soundscape_{kind}_cv{f}.parquet"
        check_no_heldout(m, spl / f"soundscape_cv{f}_heldout.parquet", kind)
        df = pd.read_parquet(m)
        assert set(df.audio_path) <= set(pd.read_parquet(spl / f"soundscape_cv{f}_train.parquet").audio_path), m
        assert "hsn_classes" not in df and "targets" not in df, f"{m} carries labels"
        print(f"ok {m.name}: {len(df)} clips, none held out")
EOF

smoke() {  # name, train_protopnet.py args...
    local name=$1; shift
    python scripts/train/train_protopnet.py --dataset smoke --name "$name" --cv-fold 0 --select last \
        --max-train-batches 2 --max-eval-samples 64 --num-workers "$W" "$@"
    test -s "outputs/checkpoints/smoke/$name/best.pt"
}
smoke smoke-oracle --train-domain soundscape
smoke smoke-r4 --train-domain focal --bg-kind real --bg-manifest $SPL/soundscape_background_cv0.parquet \
    --overlap-prob 0.5 --pseudo-manifest $SPL/soundscape_pseudo_cv0.parquet
echo "PREP PASSED"
