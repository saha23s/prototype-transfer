#!/usr/bin/env python3
"""Adapt the pretrained AudioProtoPNet to soundscapes with a published source-free adaptation baseline (needs a GPU).

Inductive (protocol v1): the model is adapted on UNLABELED soundscape audio of the training sites (soundscape_train, or
soundscape_cv{F}_train with --cv-fold F), frozen, and then scored on sites it never heard. No labels are read, nothing
is selected on a validation set (the last epoch is kept), and the settings are the reference authors' audio defaults
(prototype_transfer.method.sfda.METHODS); a flag that changes one is recorded in the run's config.

Methods: tent, pseudo_label, dropout_student, notela, shot. What is ported and what differs from the reference:
docstrings of prototype_transfer/method/sfda.py and prototype_transfer/training/adapt.py.

Writes outputs/checkpoints/<dataset>/<name>/best.pt (the final model; the name is what the extractor expects) and the
run's `adaptation` section, config and per-epoch log in the results store. To adapt, extract and score in one job, use
the training job script with this script in place of the trainer (array = 3-fold cross-validation):
  TRAIN_SCRIPT=scripts/method/adapt_sfda.py sbatch --array=0-2 scripts/slurm/run_recipe.sbatch \
      sfda-notela sfda-notela --method notela
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from prototype_transfer.data.splits import split_manifest
from prototype_transfer.evaluation import results
from prototype_transfer.method import sfda
from prototype_transfer.models.protopnet import build_hsn_protopnet, save_checkpoint
from prototype_transfer.training.adapt import UnlabeledSoundscapes, adapt

logger = logging.getLogger("adapt_sfda")
OVERRIDES = {"lr": float, "threshold": float, "alpha": float, "lam": float, "knn": int, "beta": float}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=sorted(sfda.METHODS), required=True)
    p.add_argument("--name", default=None, help="run id (default: sfda-<method>[-cv<F>])")
    p.add_argument("--rung", default=None, help="label recorded in the registry (default: sfda-<method>)")
    p.add_argument("--dataset", default=results.DEFAULT_DATASET)
    p.add_argument("--splits-dir", type=Path, default=Path("data/HSN/splits"))
    p.add_argument("--cv-fold", type=int, default=None, help="adapt without this fold's sites")
    p.add_argument("--epochs", type=int, default=10, help="reference value: 10")
    p.add_argument("--batch-size", type=int, default=64, help="reference value: 64")
    p.add_argument("--drop-path", type=float, default=0.2, help="stochastic-depth probability of the student noise")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None, help="debug: adapt on a random subset of this many batches")
    for key, kind in OVERRIDES.items():
        p.add_argument(f"--{key}", type=kind, default=None, help=f"override the method's {key}")
    args = p.parse_args()

    cfg = dict(sfda.METHODS[args.method])
    for key in OVERRIDES:
        value = getattr(args, key)
        if value is not None:
            if key not in cfg:
                p.error(f"--{key} is not a setting of {args.method}")
            cfg[key] = value
    name = args.name or f"sfda-{args.method}" + ("" if args.cv_fold is None else f"-cv{args.cv_fold}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = split_manifest(args.splits_dir, "soundscape", "train", args.cv_fold)   # refuses non-canonical splits
    dataset = UnlabeledSoundscapes(manifest)
    n_clips = len(dataset)
    if args.max_batches:
        n_clips = min(n_clips, args.max_batches * args.batch_size)
        dataset.paths = [dataset.paths[i] for i in np.random.default_rng(args.seed).permutation(len(dataset))[:n_clips]]
        assert len(dataset) == n_clips

    store = results.Store(args.dataset)
    prov = results.provenance()
    effective = {"method": args.method, **cfg, "epochs": args.epochs, "batch_size": args.batch_size,
                 "drop_path": args.drop_path if cfg["dropout"] else 0.0, "n_adapt_clips": n_clips,
                 "adapt_manifest": str(manifest), "trainable": "backbone LayerNorm scale and shift", "uses_labels": False}
    config = {"args": vars(args), "effective": effective, **prov}
    index = {"rung": args.rung or f"sfda-{args.method}", "seed": args.seed, "cv_fold": args.cv_fold,
             "commit": prov["commit"], "dirty": prov["dirty"], "legacy": False, "note": None}
    results.save_results(store, name, "adaptation", {"status": "running"}, config=config,
                         entry={**index, "status": "running", "started": results.now()})

    history: list[dict] = []

    def log_epoch(record: dict) -> None:
        history.append(record)
        results.write_train_log(store, name, history)

    model = build_hsn_protopnet()
    adapt(model, dataset, args.method, cfg, epochs=args.epochs, batch_size=args.batch_size, drop_path=args.drop_path,
          device=device, seed=args.seed, num_workers=args.num_workers, on_epoch=log_epoch)
    if device == "cuda":
        effective["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        logger.info(f"peak GPU memory allocated: {effective['peak_gpu_gb']} GB")
    ckpt = store.checkpoints(name) / "best.pt"
    save_checkpoint(model, ckpt, {"phase": "sfda", "method": args.method, "epoch": args.epochs})
    entry = results.save_results(store, name, "adaptation", {**effective, "final_loss": history[-1]["loss"]},
                                 config=config, log=history,
                                 entry={**index, "status": "done", "finished": results.now()})
    logger.info(f"Done. checkpoint {ckpt}; results {entry['paths']['metrics']}")


if __name__ == "__main__":
    main()
