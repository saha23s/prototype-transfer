#!/usr/bin/env python3
"""CLI for the stage 2 domain-shift diagnostics of the pretrained AudioProtoPNet; the tasks (extract, synthetic,
renorm) are documented and implemented in prototype_transfer.analysis.shift_diagnostics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from prototype_transfer.analysis import shift_diagnostics as sd
from prototype_transfer.data.preprocessing import AudioPreprocessor
from prototype_transfer.evaluation import results
from prototype_transfer.models.classifier import HSNProtoPNetClassifier

logger = logging.getLogger("diagnose_shift")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=sd.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=list(sd.TASKS), required=True)
    p.add_argument("--eval-split", choices=list(sd.EVAL_SETS), default="test",
                   help="test: focal_test + soundscape_test (committed results). val: focal_val + soundscape_val, outputs get a _val suffix")
    p.add_argument("--dataset", default=results.DEFAULT_DATASET)
    p.add_argument("--out-dir", type=Path, default=None, help="default: outputs/features/<dataset>/pretrained-xcl/")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bg-quantile", type=float, default=0.30, help="fraction of soundscape-train clips kept as background")
    p.add_argument("--tag", default="", help="suffix of the synthetic output file, e.g. _lowsnr -> diag_synthetic_lowsnr.pt")
    p.add_argument("--focal-window", choices=["peak_energy", "random", "center", "start"], default="peak_energy",
                   help="how the 5 s focal window is chosen in the synthetic task (the committed results use peak_energy)")
    p.add_argument("--mix-mode", choices=["add", "attenuate"], default="add",
                   help="add: boost the background to reach the SNR (committed run); attenuate: keep the background level, quieten the bird")
    p.add_argument("--bg-snrs", type=float, nargs="*", default=list(sd.BG_SNR_DB), help="background SNRs (dB)")
    p.add_argument("--overlap-rels", type=float, nargs="*", default=list(sd.OVERLAP_REL_DB), help="overlay levels (dB re clip)")
    p.add_argument("--both-snrs", type=float, nargs="*", default=[b[0] for b in sd.BOTH], help="combined conditions: background SNRs (dB) ...")
    p.add_argument("--both-rels", type=float, nargs="*", default=[b[1] for b in sd.BOTH], help="... paired one-to-one with these overlay levels (dB)")
    p.add_argument("--pool", type=int, default=600, help="background / overlay clips held in memory for the synthetic task")
    p.add_argument("--max-n", type=int, default=None, help="debug: random N clips of every split (writes nothing to data/)")
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    logger.info(f"device {device}; args {vars(args)}")
    args.out_dir = args.out_dir or results.Store(args.dataset).features("pretrained-xcl")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = HSNProtoPNetClassifier(device=device)
    pre = sd.to_device(AudioPreprocessor(), device)
    sd.TASKS[args.task](args, model, pre)
    logger.info("done")


if __name__ == "__main__":
    main()
