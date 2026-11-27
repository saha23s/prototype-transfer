#!/usr/bin/env python3
"""Train a 21-class HSN AudioProtoPNet on ONE domain (focal or soundscape).

Initialised from the pretrained XCL model (backbone + the 105 HSN prototypes + last-layer rows).
Phases (ProtoPNet-style):
  1. warm:  prototypes + last layer, backbone frozen
  2. joint: everything (backbone at a low LR); keep the best epoch by IN-DOMAIN val cmAP
  3. push:  project each prototype onto its most similar training patch of its own class
  4. last:  last layer only, prototypes and backbone frozen
Model selection only ever looks at the training domain's val split; the other domain's val cmAP is
logged for monitoring transfer during training.

Outputs (results store, see docs/RESULTS.md):
         outputs/results/<dataset>/runs/<name>/{config.json, metrics.json, train_log.csv, push.json}
         outputs/results/<dataset>/registry.jsonl (the run's index entry: seed, commit, headline val metrics)
         outputs/checkpoints/<dataset>/<name>/{best.pt, joint_best.pt}
Launch on filo through the GPU reservation wrapper:
  ~/.local/bin/local-training-3090 python scripts/train/train_protopnet.py --train-domain focal
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


from prototype_transfer.data.datasets import HSNInferenceDataset
from prototype_transfer.evaluation.metrics import compute_cmap, compute_macro_roc_auc
from prototype_transfer.data.splits import verify_split
from prototype_transfer.evaluation import results
from prototype_transfer.models.protopnet import (build_hsn_protopnet, load_checkpoint, orthogonality_loss, param_groups,
                           push_prototypes, save_checkpoint, set_trainable)
from prototype_transfer.method.train_dataset import HSNTrainDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_protopnet")
DOMAINS = ("focal", "soundscape")


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


def run_epoch(model, loader, opt, sched, device, ortho_w: float) -> float:
    # The model's own asymmetric multi-label loss (remote code), as used to train AudioProtoPNet.
    asl = sys.modules[type(model).__module__].AsymmetricLossMultiLabel()
    model.train()
    tot, n = 0.0, 0
    for b in loader:
        x, y = b["spectrogram"].to(device, non_blocking=True), b["target"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(input_values=x).logits
        loss = asl(logits.float(), y) / x.shape[0]   # asymmetric loss in fp32 (sum-reduced -> per clip)
        if ortho_w > 0:
            loss = loss + ortho_w * orthogonality_loss(model.head)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 5.0)
        opt.step()
        if sched is not None:
            sched.step()
        tot += float(loss) * x.shape[0]; n += x.shape[0]
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
    p.add_argument("--max-train-batches", type=int, default=None, help="debug: limit steps per epoch")
    p.add_argument("--max-eval-samples", type=int, default=None, help="debug: limit val clips")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    for d in DOMAINS:   # refuse splits that differ from the committed ones (configs/splits/hsn.json)
        for part in ("train", "val"):
            try:
                verify_split(args.splits_dir / f"{d}_{part}.parquet")
            except ValueError as e:
                p.error(str(e))
    name = args.name or f"{args.train_domain}-only"
    store = results.Store(args.dataset)
    ckpt_dir = store.checkpoints(name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    prov = results.provenance()
    config = {"args": vars(args), **prov}
    index = {"seed": args.seed, "commit": prov["commit"], "dirty": prov["dirty"]}
    results.save_results(store, name, "train", {"status": "running"}, config=config,
                         entry={**index, "status": "running", "started": results.now()})
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    is_sc = args.train_domain == "soundscape"
    train_ds = HSNTrainDataset(args.splits_dir / f"{args.train_domain}_train.parquet", is_soundscape=is_sc)
    if args.max_train_batches:
        train_ds.df = train_ds.df.sample(n=min(len(train_ds.df), args.max_train_batches * args.batch_size),
                                         random_state=args.seed).reset_index(drop=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          drop_last=True, pin_memory=True, persistent_workers=True)
    val_dl = {d: DataLoader(HSNInferenceDataset(args.splits_dir / f"{d}_val.parquet", is_soundscape=d == "soundscape",
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
    log_eval("init", 0, None)

    # 1. warm
    set_trainable(model, {"prototypes", "last_layer"})
    steps = len(train_dl) * args.warm_epochs
    opt, sched = make_opt(model, {"prototypes": args.lr_prototypes, "last_layer": args.lr_last}, steps, args.weight_decay)
    for e in range(1, args.warm_epochs + 1):
        log_eval("warm", e, run_epoch(model, train_dl, opt, sched, device, args.ortho_weight))

    # 2. joint (select best by in-domain val cmAP)
    set_trainable(model, {"backbone", "prototypes", "last_layer"})
    steps = len(train_dl) * args.joint_epochs
    opt, sched = make_opt(model, {"backbone": args.lr_backbone, "prototypes": args.lr_prototypes / 10,
                                  "last_layer": args.lr_last / 10}, steps, args.weight_decay)
    best, best_ep = -1.0, 0
    for e in range(1, args.joint_epochs + 1):
        ev = log_eval("joint", e, run_epoch(model, train_dl, opt, sched, device, args.ortho_weight))
        if ev[in_dom]["cmAP"] > best:
            best, best_ep = ev[in_dom]["cmAP"], e
            save_checkpoint(model, ckpt_dir / "joint_best.pt", {"epoch": e, "val": ev})
    logger.info(f"Best joint epoch {best_ep} ({in_dom} val cmAP {best:.3f})")
    model = load_checkpoint(ckpt_dir / "joint_best.pt", device)

    # 3. push onto training patches of the training domain (deterministic windows)
    push_ds = HSNInferenceDataset(args.splits_dir / f"{args.train_domain}_train.parquet", is_soundscape=is_sc)
    push_info = push_prototypes(model, push_ds, device, num_workers=args.num_workers)
    log_eval("push", 0, None)

    # 4. last layer
    set_trainable(model, {"last_layer"})
    steps = len(train_dl) * args.last_epochs
    opt, sched = make_opt(model, {"last_layer": args.lr_last}, steps, 0.0)
    best, best_ev = -1.0, None
    for e in range(1, args.last_epochs + 1):
        ev = log_eval("last", e, run_epoch(model, train_dl, opt, sched, device, 0.0))
        if ev[in_dom]["cmAP"] > best:
            best, best_ev = ev[in_dom]["cmAP"], ev
            save_checkpoint(model, ckpt_dir / "best.pt", {"phase": "last", "epoch": e, "val": ev, "push": "push.json"})
    if args.last_epochs == 0:
        best_ev = history[-1]
        save_checkpoint(model, ckpt_dir / "best.pt", {"phase": "push", "val": best_ev})

    selected = {d: {k: best_ev[d][k] if d in best_ev else best_ev[f"val_{d}_{k}"] for k in ("cmAP", "macro_auc")} for d in DOMAINS}
    headline = {f"val_{d}_{k}": v for d in DOMAINS for k, v in selected[d].items()}
    entry = results.save_results(store, name, "train", {"train_domain": in_dom, "selected_val": selected},
                                 headline=headline, config=config, log=history, files={"push.json": push_info},
                                 entry={**index, "status": "done", "finished": results.now()})
    logger.info(f"Done. checkpoint {ckpt_dir / 'best.pt'}; results {entry['paths']['metrics']}")


if __name__ == "__main__":
    main()
