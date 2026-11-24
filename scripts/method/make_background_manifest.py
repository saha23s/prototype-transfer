#!/usr/bin/env python3
"""Label-free manifests for shift-matched training, from the pretrained model's responses on soundscape-train.

Reads pretrained responses (scripts/analysis/diagnose_shift.py --task extract; by default the soundscape_train entry of
outputs/features/hsn/pretrained-xcl/diag_responses.pt, which stores no annotations) and writes, for the clips of soundscape-train:
  background manifest  the clips whose max species probability is in the lowest --quantile (the model finds them empty);
                       columns audio_path, max_prob.  Used by train_protopnet.py --bg-kind real (R1b, R3, R4).
  pseudo manifest      every clip with the teacher's (pretrained) probabilities, columns audio_path, teacher_probs
                       (21 floats).  Used by train_protopnet.py --pseudo-manifest (R4).
No soundscape annotation is read (only audio_path and probs of each entry); the selection rule is prototype_transfer.method.mixing.select_background.

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

import numpy as np
import pandas as pd
import torch


from prototype_transfer.method.mixing import select_background

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("make_background_manifest")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", metavar="FILE:KEY", default=None,
                   help="responses file and entry to read, repeatable (default outputs/features/hsn/pretrained-xcl/diag_responses.pt:soundscape_train)")
    p.add_argument("--fold", type=int, default=None, help="cross-validation fold: keep only soundscape_cv<F>_train clips, write *_cv<F> files")
    p.add_argument("--splits-dir", type=Path, default=Path("data/HSN/splits"))
    p.add_argument("--quantile", type=float, default=0.30, help="fraction of soundscape-train clips kept as background")
    p.add_argument("--out-dir", type=Path, default=Path("data/HSN/splits"))
    args = p.parse_args()

    probs_l, paths = [], []
    for src in args.source or ["outputs/features/hsn/pretrained-xcl/diag_responses.pt:soundscape_train"]:
        file, key = src.rsplit(":", 1)
        assert not key.startswith("soundscape_test"), "the test split is never used for adaptation"
        entry = torch.load(file, map_location="cpu", weights_only=False)[key]
        probs_l.append(np.asarray(entry["probs"])); paths += list(entry["audio_path"])      # only probs and paths: annotations are never read
    probs = np.concatenate(probs_l)                                                          # [N, 21]
    assert len(probs) == len(paths) and len(set(paths)) == len(paths), "duplicate or misaligned clips"
    suffix = ""
    if args.fold is not None:
        keep_paths = set(pd.read_parquet(args.splits_dir / f"soundscape_cv{args.fold}_train.parquet")["audio_path"])
        missing = keep_paths - set(paths)
        assert not missing, f"{len(missing)} clips of soundscape_cv{args.fold}_train have no responses; add the val responses with --source"
        sel = np.array([q in keep_paths for q in paths])
        probs, paths, suffix = probs[sel], [q for q, k in zip(paths, sel) if k], f"_cv{args.fold}"
    max_prob = probs.max(axis=1)
    keep = select_background(max_prob, args.quantile)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bg = pd.DataFrame({"audio_path": np.array(paths)[keep], "max_prob": max_prob[keep]})
    pseudo = pd.DataFrame({"audio_path": paths, "teacher_probs": list(probs.astype(np.float32))})
    bg.to_parquet(args.out_dir / f"soundscape_background{suffix}.parquet", index=False)
    pseudo.to_parquet(args.out_dir / f"soundscape_pseudo{suffix}.parquet", index=False)
    logger.info(f"{len(bg)}/{len(paths)} background clips (max prob <= {np.quantile(max_prob, args.quantile):.4f}); "
                f"{len(pseudo)} teacher soft labels -> {args.out_dir}")


if __name__ == "__main__":
    main()
