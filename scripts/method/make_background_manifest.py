#!/usr/bin/env python3
"""Label-free manifests for shift-matched training, from the pretrained model's responses on soundscape-train.

Reads pretrained responses (scripts/analysis/diagnose_shift.py --task extract; by default the soundscape_train entry of
outputs/features/hsn/pretrained-xcl/diag_responses.pt, which stores no annotations) and writes, for the clips of soundscape-train:
  background manifest  the clips whose max species probability is in the lowest --quantile (the model finds them empty);
                       columns audio_path, max_prob.  Used by train_protopnet.py --bg-kind real (R1b, R3, R4).
  pseudo manifest      every clip with the teacher's (pretrained) probabilities, columns audio_path, teacher_probs
                       (21 floats).  Used by train_protopnet.py --pseudo-manifest (R4).
No soundscape annotation is read (only audio_path and probs of each entry); the logic is prototype_transfer.method.manifests.

Cross-validation: --fold F keeps only the clips of soundscape_cv{F}_train (the sites NOT held out in fold F), takes the
quantile over those clips and writes soundscape_background_cv{F}.parquet / soundscape_pseudo_cv{F}.parquet, so no held-out
audio is used for adaptation. The pool is train + val sites, so pass the val responses too:
  --source outputs/features/hsn/pretrained-xcl/diag_responses.pt:soundscape_train --source outputs/features/hsn/pretrained-xcl/diag_responses_val.pt:soundscape_val
(the latter from diagnose_shift.py --task extract --eval-split val).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from prototype_transfer.data.splits import split_manifest
from prototype_transfer.method import manifests

logger = logging.getLogger("make_background_manifest")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", metavar="FILE:KEY", default=None,
                   help="responses file and entry to read, repeatable (default outputs/features/hsn/pretrained-xcl/diag_responses.pt:soundscape_train)")
    p.add_argument("--fold", type=int, default=None, help="cross-validation fold: keep only soundscape_cv<F>_train clips, write *_cv<F> files")
    p.add_argument("--splits-dir", type=Path, default=Path("data/HSN/splits"))
    p.add_argument("--quantile", type=float, default=0.30, help="fraction of soundscape-train clips kept as background")
    p.add_argument("--out-dir", type=Path, default=Path("data/HSN/splits"))
    args = p.parse_args()

    test_clips = pd.read_parquet(split_manifest(args.splits_dir, "soundscape", "test"))["audio_path"]
    probs, paths = manifests.load_responses(args.source or ["outputs/features/hsn/pretrained-xcl/diag_responses.pt:soundscape_train"],
                                            test_clips)
    suffix = ""
    if args.fold is not None:
        keep_paths = set(pd.read_parquet(args.splits_dir / f"soundscape_cv{args.fold}_train.parquet")["audio_path"])
        probs, paths = manifests.restrict_to_fold(probs, paths, keep_paths, args.fold)
        suffix = f"_cv{args.fold}"
    bg, pseudo, cutoff = manifests.build_manifests(probs, paths, args.quantile)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bg.to_parquet(args.out_dir / f"soundscape_background{suffix}.parquet", index=False)
    pseudo.to_parquet(args.out_dir / f"soundscape_pseudo{suffix}.parquet", index=False)
    logger.info(f"{len(bg)}/{len(paths)} background clips (max prob <= {cutoff:.4f}); "
                f"{len(pseudo)} teacher soft labels -> {args.out_dir}")


if __name__ == "__main__":
    main()
