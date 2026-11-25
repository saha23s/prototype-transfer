"""Training of a 21-class HSN AudioProtoPNet on ONE domain (focal or soundscape).

Initialised from the pretrained XCL model (backbone + the 105 HSN prototypes + last-layer rows).
Phases (ProtoPNet-style):
  1. warm:  prototypes + last layer, backbone frozen
  2. joint: everything (backbone at a low LR); keep the best epoch by IN-DOMAIN val cmAP
  3. push:  project each prototype onto its most similar training patch of its own class
  4. last:  last layer only, prototypes and backbone frozen
Model selection only ever looks at the training domain's val split; the other domain's val cmAP is logged for
monitoring transfer during training. select="last" instead keeps the last epoch of each phase (a fixed schedule that
uses no validation labels).

`train(args)` takes the argparse namespace of scripts/train/train_protopnet.py (which documents every option);
`validate(args)` checks the option combinations that would make a run invalid.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from prototype_transfer.data.datasets import HSNInferenceDataset
from prototype_transfer.data.splits import check_no_heldout, split_manifest
from prototype_transfer.evaluation import results
from prototype_transfer.evaluation.metrics import compute_cmap, compute_macro_roc_auc
from prototype_transfer.method.train_dataset import HSNPseudoDataset, HSNTrainDataset
from prototype_transfer.models.protopnet import (build_hsn_protopnet, load_checkpoint, param_groups, push_prototypes,
                                                 save_checkpoint, set_trainable)
from prototype_transfer.training.losses import orthogonality_loss

logger = logging.getLogger(__name__)
DOMAINS = ("focal", "soundscape")


def validate(args: argparse.Namespace) -> None:
    """Raise ValueError for option combinations that would make a run invalid (CV leakage, unsupported pairings)."""
    is_sc = args.train_domain == "soundscape"
    if args.bg_kind == "real" and args.bg_manifest is None:
        raise ValueError("--bg-kind real needs --bg-manifest")
    if args.pseudo_manifest is not None and is_sc:
        raise ValueError("--pseudo-manifest adds unlabeled soundscape clips to FOCAL training; it is not for --train-domain soundscape")
    if args.cv_fold is not None:
        if is_sc and args.select != "last":
            raise ValueError("--cv-fold with --train-domain soundscape needs --select last: the held-out fold must not choose a checkpoint")
        for flag, path in (("--bg-manifest", args.bg_manifest), ("--pseudo-manifest", args.pseudo_manifest)):
            if path is not None:
                check_no_heldout(path, split_manifest(args.splits_dir, "soundscape", "val", args.cv_fold), flag)


def cycle(loader: DataLoader):
    """Endless iterator over a DataLoader (re-shuffled every pass)."""
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    ys, ps = [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(input_values=b["spectrogram"].to(device)).logits
        ps.append(torch.sigmoid(logits.float()).cpu().numpy())
        ys.append(b["target"].numpy() > 0.5)
    y, p = np.concatenate(ys), np.concatenate(ps)
    cmap, _ = compute_cmap(y.astype(int), p)
    auc, _ = compute_macro_roc_auc(y.astype(int), p)
    return {"cmAP": cmap, "macro_auc": auc}


def run_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, sched, device: str, ortho_w: float,
              pseudo_iter=None, pseudo_w: float = 0.0) -> float:
    # The model's own asymmetric multi-label loss (remote code), as used to train AudioProtoPNet.
    # reduction="sum" because the class defaults to "mean"; with / n_lab below, the term is per clip, like the pseudo-label term.
    asl = sys.modules[type(model).__module__].AsymmetricLossMultiLabel(reduction="sum")
    model.train()
    tot, n = 0.0, 0
    for b in loader:
        x, y = b["spectrogram"].to(device, non_blocking=True), b["target"].to(device, non_blocking=True)
        n_lab = x.shape[0]
        if pseudo_iter is not None:   # one forward pass over the labeled batch and the unlabeled soundscape batch
            pb = next(pseudo_iter)
            x = torch.cat([x, pb["spectrogram"].to(device, non_blocking=True)])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(input_values=x).logits
        logits = logits.float()
        loss = asl(logits[:n_lab], y) / n_lab   # asymmetric loss in fp32 (sum-reduced -> per clip)
        if pseudo_iter is not None:   # soft BCE to the teacher's probabilities, summed over classes, mean over clips
            soft = pb["target"].to(device, non_blocking=True)
            loss = loss + pseudo_w * F.binary_cross_entropy_with_logits(logits[n_lab:], soft, reduction="none").sum(1).mean()
        if ortho_w > 0:
            loss = loss + ortho_w * orthogonality_loss(model.head)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss ({float(loss)}) at {b['audio_path'][:3]}...; stopping instead of training on NaNs")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 5.0)
        opt.step()
        if sched is not None:
            sched.step()
        tot += float(loss) * n_lab; n += n_lab
    return tot / max(1, n)


def make_opt(model: nn.Module, lrs: dict[str, float], steps: int, wd: float):
    groups = [{"params": ps, "lr": lrs[name], "weight_decay": 0.0 if name == "prototypes" else wd}
              for name, ps in param_groups(model).items() if lrs.get(name, 0) > 0]
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=max(1, steps),
                                                pct_start=0.1, anneal_strategy="cos") if steps else None
    return opt, sched


def train(args: argparse.Namespace) -> None:
    """Run the four phases. Results go to the results store (evaluation/results.py): config.json, train_log.csv, push.json and
    the `train` metrics section under outputs/results/<dataset>/runs/<name>/, plus the registry entry; checkpoints go to
    outputs/checkpoints/<dataset>/<name>/."""
    if args.anchored:
        args.joint_epochs = 0
    is_sc = args.train_domain == "soundscape"
    manifest = lambda domain, part: split_manifest(args.splits_dir, domain, part, args.cv_fold)
    name = args.name or f"{args.train_domain}-only"
    store = results.Store(args.dataset)
    ckpt_dir = store.checkpoints(name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = HSNTrainDataset(manifest(args.train_domain, "train"), is_soundscape=is_sc,
                               bg_kind=args.bg_kind, bg_manifest=args.bg_manifest, bg_prob=args.bg_prob,
                               snr_min=args.snr_min, snr_max=args.snr_max, overlap_prob=args.overlap_prob,
                               overlap_gain_min=args.overlap_gain_min, overlap_gain_max=args.overlap_gain_max)
    if args.max_train_batches:
        train_ds.df = train_ds.df.sample(n=min(len(train_ds.df), args.max_train_batches * args.batch_size),
                                         random_state=args.seed).reset_index(drop=True)
    n_full = len(train_ds.df)
    sampler = None
    if args.train_labels:   # fewer distinct labeled clips, same number of optimizer steps (label-budget curve)
        train_ds.df = train_ds.df.sample(n=min(n_full, args.train_labels), random_state=args.seed).reset_index(drop=True)
        sampler = RandomSampler(train_ds, replacement=True, num_samples=n_full // args.batch_size * args.batch_size)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                          num_workers=args.num_workers, drop_last=True, pin_memory=True, persistent_workers=True)
    pseudo_iter = None
    if args.pseudo_manifest is not None:   # unlabeled soundscape-train clips + teacher soft labels; no annotations are read
        pseudo_dl = DataLoader(HSNPseudoDataset(args.pseudo_manifest), batch_size=args.pseudo_batch_size, shuffle=True,
                               num_workers=max(1, args.num_workers // 2), drop_last=True, pin_memory=True, persistent_workers=True)
        pseudo_iter = cycle(pseudo_dl)
    effective = {"cv_fold": args.cv_fold, "mixing": train_ds.mixing_config(), "select": args.select, "joint_epochs": args.joint_epochs,
                 "train_labels": args.train_labels, "n_train_clips": len(train_ds.df), "steps_per_epoch": len(train_dl),
                 "pseudo_weight": args.pseudo_weight if pseudo_iter is not None else 0.0, "pseudo_manifest": str(args.pseudo_manifest)}
    prov = results.provenance()
    config = {"args": vars(args), "effective": effective, **prov}
    logger.info(f"resolved args: {vars(args)}")
    logger.info(f"effective settings: {effective}; git {prov['commit']} dirty={prov['dirty']}")
    index = {"rung": args.rung, "seed": args.seed, "cv_fold": args.cv_fold, "commit": prov["commit"], "dirty": prov["dirty"]}
    results.save_results(store, name, "train", {"status": "running"}, config=config,
                         entry={**index, "status": "running", "started": results.now()})
    val_dl = {d: DataLoader(HSNInferenceDataset(manifest(d, "val"), is_soundscape=d == "soundscape",
                                                max_samples=args.max_eval_samples),
                            batch_size=64, num_workers=args.num_workers) for d in DOMAINS}

    model = build_hsn_protopnet().to(device)
    history = []

    def log_eval(phase: str, epoch: int, loss: float | None) -> dict:
        ev = {d: evaluate(model, val_dl[d], device) for d in DOMAINS}
        rec = {"phase": phase, "epoch": epoch, "train_loss": loss, "time": time.time(),
               **{f"val_{d}_{k}": v for d in DOMAINS for k, v in ev[d].items()}}
        history.append(rec)
        results.write_train_log(store, name, history)
        logger.info(f"[{phase} {epoch}] loss={loss if loss is None else round(loss, 4)} "
                    + " ".join(f"{d}: cmAP {ev[d]['cmAP']:.3f} AUC {ev[d]['macro_auc']:.3f}" for d in DOMAINS))
        return ev

    in_dom = args.train_domain
    ev = log_eval("init", 0, None)
    train_epoch = lambda opt, sched, ortho: run_epoch(model, train_dl, opt, sched, device, ortho, pseudo_iter, args.pseudo_weight)

    # 1. warm
    set_trainable(model, {"prototypes", "last_layer"})
    steps = len(train_dl) * args.warm_epochs
    opt, sched = make_opt(model, {"prototypes": args.lr_prototypes, "last_layer": args.lr_last}, steps, args.weight_decay)
    for e in range(1, args.warm_epochs + 1):
        ev = log_eval("warm", e, train_epoch(opt, sched, args.ortho_weight))

    # 2. joint (--select best: best epoch by in-domain val cmAP; --select last: the last epoch)
    joint_ckpt = ckpt_dir / ("joint_best.pt" if args.select == "best" else "joint_last.pt")
    if args.joint_epochs == 0:   # anchored: nothing to train jointly, carry the warm-phase model forward
        save_checkpoint(model, joint_ckpt, {"epoch": 0, "val": ev})
    set_trainable(model, {"backbone", "prototypes", "last_layer"})
    steps = len(train_dl) * args.joint_epochs
    opt, sched = make_opt(model, {"backbone": args.lr_backbone, "prototypes": args.lr_prototypes / 10,
                                  "last_layer": args.lr_last / 10}, steps, args.weight_decay)
    best, best_ep = -1.0, 0
    for e in range(1, args.joint_epochs + 1):
        ev = log_eval("joint", e, train_epoch(opt, sched, args.ortho_weight))
        if args.select == "last" or ev[in_dom]["cmAP"] > best:
            best, best_ep = ev[in_dom]["cmAP"], e
            save_checkpoint(model, joint_ckpt, {"epoch": e, "val": ev})
    logger.info(f"Selected joint epoch {best_ep} (--select {args.select}; {in_dom} val cmAP {best:.3f})")
    model = load_checkpoint(joint_ckpt, device)

    # 3. push onto training patches of the training domain (deterministic windows)
    push_ds = HSNInferenceDataset(manifest(args.train_domain, "train"), is_soundscape=is_sc,
                                  max_samples=args.max_train_batches * args.batch_size if args.max_train_batches else None)
    push_info = push_prototypes(model, push_ds, device, num_workers=args.num_workers)
    log_eval("push", 0, None)

    # 4. last layer
    set_trainable(model, {"last_layer"})
    steps = len(train_dl) * args.last_epochs
    opt, sched = make_opt(model, {"last_layer": args.lr_last}, steps, 0.0)
    best, best_ev = -1.0, None
    for e in range(1, args.last_epochs + 1):
        ev = log_eval("last", e, train_epoch(opt, sched, 0.0))
        if args.select == "last" or ev[in_dom]["cmAP"] > best:
            best, best_ev = ev[in_dom]["cmAP"], ev
            save_checkpoint(model, ckpt_dir / "best.pt", {"phase": "last", "epoch": e, "val": ev})
    if args.last_epochs == 0:
        best_ev = history[-1]
        save_checkpoint(model, ckpt_dir / "best.pt", {"phase": "push", "val": best_ev})

    final_val = {d: {k: history[-1][f"val_{d}_{k}"] for k in ("cmAP", "macro_auc")} for d in DOMAINS}   # last epoch, whatever --select chose
    headline = {f"{which}_{d}_{k}": v[d][k] for which, v in (("val", best_ev), ("last_val", final_val))
                for d in DOMAINS for k in ("cmAP", "macro_auc")}
    entry = results.save_results(store, name, "train", {"train_domain": in_dom, "select": args.select, "selected_val": best_ev,
                                                        "last_epoch_val": final_val},
                                 headline=headline, config=config, log=history, files={"push.json": push_info},
                                 entry={**index, "status": "done", "finished": results.now()})
    logger.info(f"Done. checkpoint {ckpt_dir / 'best.pt'}; results {entry['paths']['metrics']}")
