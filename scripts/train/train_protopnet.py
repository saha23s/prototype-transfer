#!/usr/bin/env python3
"""Train a 21-class HSN AudioProtoPNet on ONE domain (focal or soundscape).

Initialised from the pretrained XCL model (backbone + the 105 HSN prototypes + last-layer rows).
Phases (ProtoPNet-style):
  1. warm:  prototypes + last layer, backbone frozen
  2. joint: everything (backbone at a low LR); keep the best epoch by IN-DOMAIN val cmAP
  3. push:  project each prototype onto its most similar training patch of its own class
  4. last:  last layer only, prototypes and backbone frozen
Model selection only ever looks at the training domain's val split; the other domain's val cmAP is
logged for monitoring transfer during training. --select last instead keeps the last epoch of each phase (a fixed
schedule that uses no validation labels); the last epoch's val metrics are always reported next to the selected ones.

Optional recipe components, all off by default so earlier runs stay reproducible (see README, "Training recipe"):
  --bg-kind noise|real     background mixing on focal training clips (src/prototype_transfer/method/mixing.py); real needs --bg-manifest
  --overlap-prob P         multi-label mixup with a second focal species (labels = union)
  --pseudo-manifest FILE   soft pseudo-label loss on unlabeled soundscape-train clips (teacher = pretrained model)
  --train-labels N         train on N random labeled clips with the number of optimizer steps held constant
  --cv-fold F              cross-validation fold F (K=3 folds over the non-test sites, scripts/data/make_splits.py): soundscape
                           train = the other folds' sites, soundscape "val" = the held-out fold, which is only logged
                           (never used to select); scored by scripts/eval/canonical_eval.py --split cv --fold F
  --anchored (= --joint-epochs 0): backbone stays frozen; only prototypes and last layer train, then push and last layer

Outputs (results store, see docs/RESULTS.md):
         outputs/results/<dataset>/runs/<name>/{config.json, metrics.json, train_log.csv, push.json}
         outputs/results/<dataset>/registry.jsonl (the run's index entry: rung, seed, commit, headline val metrics)
         outputs/checkpoints/<dataset>/<name>/best.pt
Launch on filo through the GPU reservation wrapper:
  ~/.local/bin/local-training-3090 python scripts/train/train_protopnet.py --train-domain focal

The training logic is in prototype_transfer.training.trainer.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prototype_transfer.evaluation import results
from prototype_transfer.training import trainer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-domain", choices=trainer.DOMAINS, required=True)
    p.add_argument("--name", default=None, help="run name (default: <train-domain>-only)")
    p.add_argument("--dataset", default=results.DEFAULT_DATASET, help="results are stored under outputs/results/<dataset>/")
    p.add_argument("--splits-dir", type=Path, default=Path("data/HSN/splits"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--warm-epochs", type=int, default=2)
    p.add_argument("--joint-epochs", type=int, default=12)
    p.add_argument("--last-epochs", type=int, default=4)
    p.add_argument("--lr-backbone", type=float, default=2e-5)
    p.add_argument("--lr-prototypes", type=float, default=1e-3)
    p.add_argument("--lr-last", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ortho-weight", type=float, default=0.01)
    p.add_argument("--rung", default="custom", help="recipe rung label recorded in the registry (e.g. R0, R1b, label-budget-500)")
    p.add_argument("--select", choices=["best", "last"], default="best",
                   help="best: epoch with the best in-domain val cmAP (default). last: last epoch of each phase, no validation labels used")
    p.add_argument("--cv-fold", type=int, default=None, help="cross-validation fold held out from training (0..K-1, see scripts/data/make_splits.py)")
    p.add_argument("--anchored", action="store_true", help="skip the joint phase (same as --joint-epochs 0): backbone frozen throughout")
    p.add_argument("--train-labels", type=int, default=None,
                   help="use N random labeled train clips; steps per epoch stay those of the full train set (sampling with replacement)")
    p.add_argument("--bg-kind", choices=["none", "noise", "real"], default="none", help="background mixing on focal training clips")
    p.add_argument("--bg-manifest", type=Path, default=None, help="parquet with audio_path of background clips (make_background_manifest.py)")
    p.add_argument("--bg-prob", type=float, default=0.5)
    p.add_argument("--snr-min", type=float, default=-5.0)
    p.add_argument("--snr-max", type=float, default=20.0)
    p.add_argument("--overlap-prob", type=float, default=0.0, help="probability of overlaying a second focal species")
    p.add_argument("--overlap-gain-min", type=float, default=-10.0)
    p.add_argument("--overlap-gain-max", type=float, default=0.0)
    p.add_argument("--pseudo-manifest", type=Path, default=None, help="parquet with audio_path + teacher_probs (make_background_manifest.py)")
    p.add_argument("--pseudo-weight", type=float, default=0.2)
    p.add_argument("--pseudo-batch-size", type=int, default=16)
    p.add_argument("--max-train-batches", type=int, default=None, help="debug: limit steps per epoch (and the clips used for the push)")
    p.add_argument("--max-eval-samples", type=int, default=None, help="debug: limit val clips")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    try:   # the trainer reads its manifests through splits.split_manifest, which refuses non-canonical splits
        trainer.validate(args)
    except ValueError as e:
        p.error(str(e))
    trainer.train(args)


if __name__ == "__main__":
    main()
