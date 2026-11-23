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
  --bg-kind noise|real     background mixing on focal training clips (src/mixing.py); real needs --bg-manifest
  --overlap-prob P         multi-label mixup with a second focal species (labels = union)
  --pseudo-manifest FILE   soft pseudo-label loss on unlabeled soundscape-train clips (teacher = pretrained model)
  --train-labels N         train on N random labeled clips with the number of optimizer steps held constant
  --cv-fold F              cross-validation fold F (K=3 folds over the non-test sites, scripts/make_splits.py): soundscape
                           train = the other folds' sites, soundscape "val" = the held-out fold, which is only logged
                           (never used to select); scored by scripts/canonical_eval.py --split cv --fold F
  --anchored (= --joint-epochs 0): backbone stays frozen; only prototypes and last layer train, then push and last layer

Outputs: outputs/runs/<name>/{best.pt, history.json, push.json, args.json}
         outputs/results/train_<name>.json (selected and last-epoch val metrics, both domains)
         outputs/runs/registry.json (run name -> rung, seed, commit, settings)
Launch on filo through the GPU reservation wrapper:
  ~/.local/bin/local-training-3090 python scripts/train_protopnet.py --train-domain focal
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference_dataset import HSNInferenceDataset
from src.metrics import compute_cmap, compute_macro_roc_auc
from src.protopnet import (build_hsn_protopnet, load_checkpoint, orthogonality_loss, param_groups,
                           push_prototypes, save_checkpoint, set_trainable)
from src.train_dataset import HSNPseudoDataset, HSNTrainDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_protopnet")
DOMAINS = ("focal", "soundscape")
REGISTRY = Path("outputs/runs/registry.json")


def update_registry(name: str, entry: dict) -> None:
    """Merge `entry` into outputs/runs/registry.json under the run name (run name -> rung, seed, commit, settings)."""
    reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    reg[name] = {**reg.get(name, {}), **entry}
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=1, sort_keys=True))


def git_state() -> dict:
    """Commit the run was launched from and whether the tree had uncommitted changes (None if not in a git repo)."""
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def split_manifest(args: argparse.Namespace, domain: str, part: str) -> Path:
    """Manifest of `domain`'s train / val part. With --cv-fold, soundscape train is the other folds' sites and soundscape
    "val" is the held-out fold (monitoring only)."""
    if args.cv_fold is not None and domain == "soundscape":
        return args.splits_dir / f"soundscape_cv{args.cv_fold}_{'train' if part == 'train' else 'heldout'}.parquet"
    return args.splits_dir / f"{domain}_{part}.parquet"


def check_no_heldout(manifest: Path, heldout: Path, what: str) -> None:
    """Refuse to train if `manifest` lists any clip of the held-out fold (its audio must not be used for adaptation either)."""
    leaked = set(pd.read_parquet(manifest)["audio_path"]) & set(pd.read_parquet(heldout)["audio_path"])
    if leaked:
        raise SystemExit(f"{what} {manifest} contains {len(leaked)} clips of the held-out fold ({heldout}); "
                         "build it with scripts/make_background_manifest.py --fold F")


def cycle(loader):
    """Endless iterator over a DataLoader (re-shuffled every pass)."""
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
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


def run_epoch(model, loader, opt, sched, device, ortho_w: float, pseudo_iter=None, pseudo_w: float = 0.0) -> float:
    # The model's own asymmetric multi-label loss (remote code), as used to train AudioProtoPNet.
    asl = sys.modules[type(model).__module__].AsymmetricLossMultiLabel()
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


def make_opt(model, lrs: dict[str, float], steps: int, wd: float):
    groups = [{"params": ps, "lr": lrs[name], "weight_decay": 0.0 if name == "prototypes" else wd}
              for name, ps in param_groups(model).items() if lrs.get(name, 0) > 0]
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=max(1, steps),
                                                pct_start=0.1, anneal_strategy="cos") if steps else None
    return opt, sched


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-domain", choices=DOMAINS, required=True)
    p.add_argument("--name", default=None, help="run name (default: <train-domain>-only)")
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
    p.add_argument("--cv-fold", type=int, default=None, help="cross-validation fold held out from training (0..K-1, see scripts/make_splits.py)")
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
    if args.anchored:
        args.joint_epochs = 0
    is_sc = args.train_domain == "soundscape"
    if args.bg_kind == "real" and args.bg_manifest is None:
        p.error("--bg-kind real needs --bg-manifest")
    if args.pseudo_manifest is not None and is_sc:
        p.error("--pseudo-manifest adds unlabeled soundscape clips to FOCAL training; it is not for --train-domain soundscape")
    if args.cv_fold is not None:
        if is_sc and args.select != "last":
            p.error("--cv-fold with --train-domain soundscape needs --select last: the held-out fold must not choose a checkpoint")
        for flag, path in (("--bg-manifest", args.bg_manifest), ("--pseudo-manifest", args.pseudo_manifest)):
            if path is not None:
                check_no_heldout(path, split_manifest(args, "soundscape", "val"), flag)
    name = args.name or f"{args.train_domain}-only"
    run_dir = Path("outputs/runs") / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = HSNTrainDataset(split_manifest(args, args.train_domain, "train"), is_soundscape=is_sc,
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
    git = git_state()
    (run_dir / "args.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()}, "effective": effective, **git}, indent=1))
    logger.info(f"resolved args: {vars(args)}")
    logger.info(f"effective settings: {effective}; git {git}")
    update_registry(name, {"rung": args.rung, "seed": args.seed, "train_domain": args.train_domain, "select": args.select, "cv_fold": args.cv_fold,
                           "status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), **git,
                           "settings": effective, "args": str(run_dir / "args.json")})
    val_dl = {d: DataLoader(HSNInferenceDataset(split_manifest(args, d, "val"), is_soundscape=d == "soundscape",
                                                max_samples=args.max_eval_samples),
                            batch_size=64, num_workers=args.num_workers) for d in DOMAINS}

    model = build_hsn_protopnet().to(device)
    history = []

    def log_eval(phase: str, epoch: int, loss: float | None) -> dict:
        ev = {d: evaluate(model, val_dl[d], device) for d in DOMAINS}
        rec = {"phase": phase, "epoch": epoch, "train_loss": loss, "time": time.time(),
               **{f"val_{d}_{k}": v for d in DOMAINS for k, v in ev[d].items()}}
        history.append(rec)
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))
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
    joint_ckpt = run_dir / ("joint_best.pt" if args.select == "best" else "joint_last.pt")
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
    push_ds = HSNInferenceDataset(split_manifest(args, args.train_domain, "train"), is_soundscape=is_sc,
                                  max_samples=args.max_train_batches * args.batch_size if args.max_train_batches else None)
    push_info = push_prototypes(model, push_ds, device, num_workers=args.num_workers)
    (run_dir / "push.json").write_text(json.dumps(push_info, indent=1))
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
            save_checkpoint(model, run_dir / "best.pt", {"phase": "last", "epoch": e, "val": ev, "push": "push.json"})
    if args.last_epochs == 0:
        best_ev = history[-1]
        save_checkpoint(model, run_dir / "best.pt", {"phase": "push", "val": best_ev})

    res = Path("outputs/results") / f"train_{name}.json"
    res.parent.mkdir(parents=True, exist_ok=True)
    final_val = {d: {k: history[-1][f"val_{d}_{k}"] for k in ("cmAP", "macro_auc")} for d in DOMAINS}   # last epoch, whatever --select chose
    res.write_text(json.dumps({"name": name, "train_domain": in_dom, "select": args.select, "selected_val": best_ev,
                               "last_epoch_val": final_val, "checkpoint": str(run_dir / "best.pt"),
                               "history": str(run_dir / "history.json")}, indent=1))
    update_registry(name, {"status": "done", "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": str(res),
                           "checkpoint": str(run_dir / "best.pt"), "canonical_eval": f"outputs/results/canonical_eval_{name}.json"})
    logger.info(f"Done. checkpoint {run_dir / 'best.pt'}; results {res}")


if __name__ == "__main__":
    main()
