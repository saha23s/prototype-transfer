#!/usr/bin/env python3
"""Inductive source-free adaptation baselines for AudioProtoPNet-5-XCL on one HSN partition.

Adapts the pretrained 21-class model on the partition's 75 adaptation sites (labels unused, except by `oracle`), then
scores the partition's 25 test sites with BirdSet's metrics (evaluation.metrics.birdset_scores, docs/BIRDSET.md):

  python scripts/eval/adapt_baselines.py --partition 0 --method tent      # all of them: scripts/eval/run_baselines.sh
  methods:  source   no adaptation (sanity check: must match `birdset_p<k>` of pretrained-xcl)
            tent     minimise the mean binary entropy of the sigmoid outputs (Tent, multi-label form)
            pl       hard pseudo-labels from the source model (p > --threshold), BCE; labels fixed before adapting
            oracle   supervised BCE on the adaptation clips' true labels (upper bound, not a UDA method)
  --params  norm     LayerNorm affine parameters of the backbone only (ConvNeXt has no BatchNorm, so AdaBN is a no-op)
            all      every parameter
Hyperparameters are fixed defaults, not tuned yet (they are to be tuned on BirdSet POW, never on HSN test sites). The model stays in eval
mode throughout: LayerNorm keeps no running statistics, so this only switches off stochastic depth.

Results: section `birdset_p<k>` of run `uda-<method>-<params>[-<suffix>]` in the results store.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prototype_transfer.data.datasets import HSNInferenceDataset
from prototype_transfer.data.splits import check_no_heldout, partition_manifest
from prototype_transfer.evaluation import results
from prototype_transfer.evaluation.metrics import birdset_scores
from prototype_transfer.models.protopnet import build_hsn_protopnet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("adapt_baselines")


def norm_params(model: nn.Module) -> list[nn.Parameter]:
    ps = []
    for name, m in model.model.named_modules():
        if isinstance(m, nn.LayerNorm) or "layernorm" in type(m).__name__.lower() or "layer_norm" in name.lower():
            ps += [p for p in m.parameters(recurse=False)]
    return ps


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for b in loader:   # scored in fp32: bf16 shifts cmAP by ~1e-3; adaptation steps stay in bf16
        logits = model(input_values=b["spectrogram"].to(device)).logits
        ps.append(torch.sigmoid(logits.float()).cpu().numpy())
        ys.append(b["target"].numpy())
    return np.concatenate(ys).astype(int), np.concatenate(ps)


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--partition", type=int, required=True)
    a.add_argument("--method", choices=("source", "tent", "pl", "oracle"), required=True)
    a.add_argument("--params", choices=("norm", "all"), default="norm")
    a.add_argument("--lr", type=float, default=None, help="default: 1e-4 (norm) / 1e-5 (all)")
    a.add_argument("--epochs", type=int, default=1)
    a.add_argument("--batch-size", type=int, default=32)
    a.add_argument("--threshold", type=float, default=0.5, help="pl: pseudo-label threshold")
    a.add_argument("--num-workers", type=int, default=8)
    a.add_argument("--splits-dir", type=Path, default=Path("data/HSN/splits"))
    a.add_argument("--suffix", default="", help="appended to the run id, e.g. lr3e-4")
    a.add_argument("--seed", type=int, default=0)
    args = a.parse_args()
    lr = args.lr or (1e-4 if args.params == "norm" else 1e-5)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adapt_m = partition_manifest(args.splits_dir, args.partition, "adapt")   # verified against configs/splits/hsn.json
    test_m = partition_manifest(args.splits_dir, args.partition, "test")
    adapt_ds = HSNInferenceDataset(adapt_m, is_soundscape=True)
    test_ds = HSNInferenceDataset(test_m, is_soundscape=True)
    check_no_heldout(adapt_m, test_m, "adaptation set")
    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh, num_workers=args.num_workers,
                                   pin_memory=True, drop_last=sh)
    adapt_dl, test_dl = dl(adapt_ds, True), dl(test_ds, False)

    model = build_hsn_protopnet().to(device).eval()
    t0 = time.time()
    log = []
    if args.method != "source":
        for p in model.parameters():
            p.requires_grad_(False)
        params = norm_params(model) if args.params == "norm" else list(model.parameters())
        if not params:
            raise SystemExit("no LayerNorm parameters found in the backbone")
        for p in params:
            p.requires_grad_(True)
        logger.info(f"adapting {sum(p.numel() for p in params):,} parameters ({args.params}) with lr {lr}")
        opt = torch.optim.Adam(params, lr=lr)
        pseudo = None
        if args.method == "pl":   # fixed hard pseudo-labels from the source model, keyed by file name
            _, p_src = predict(model, dl(adapt_ds, False), device)
            pseudo = dict(zip(adapt_ds.df["audio_path"].map(lambda s: Path(s).name), (p_src > args.threshold).astype(np.float32)))
            logger.info(f"pseudo-labels: {float((p_src > args.threshold).mean()):.4f} positive rate, "
                        f"{int((p_src > args.threshold).any(1).sum())}/{len(p_src)} clips with >=1 label")
        for ep in range(args.epochs):
            tot, n = 0.0, 0
            for b in adapt_dl:
                x = b["spectrogram"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits = model(input_values=x).logits.float()
                if args.method == "tent":
                    pr = torch.sigmoid(logits)
                    loss = -(pr * F.logsigmoid(logits) + (1 - pr) * F.logsigmoid(-logits)).sum(1).mean()
                else:
                    y = (torch.from_numpy(np.stack([pseudo[f] for f in b["filename"]])) if args.method == "pl"
                         else b["target"]).to(device)
                    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="sum") / x.shape[0]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss) * x.shape[0]; n += x.shape[0]
            log.append({"epoch": ep + 1, "loss": tot / max(1, n), "time": time.time() - t0})
            logger.info(f"epoch {ep + 1}: loss {tot / max(1, n):.4f}")

    y, p = predict(model, test_dl, device)
    m = {**birdset_scores(y, p), "method": args.method, "params": args.params, "lr": lr, "epochs": args.epochs,
         "threshold": args.threshold if args.method == "pl" else None, "seconds": time.time() - t0}
    run_id = f"uda-{args.method}" + ("" if args.method == "source" else f"-{args.params}") + (f"-{args.suffix}" if args.suffix else "")
    results.save_results(results.Store(), run_id, f"birdset_p{args.partition}", m, kind="train",
                         headline={k: m[k] for k in ("cmAP", "AUROC", "T1_labelled_clips")}, log=log or None,
                         config={"args": vars(args), **results.provenance()}, entry={"seed": args.seed})
    logger.info(f"{run_id} p{args.partition}: cmAP {m['cmAP']:.4f} AUROC {m['AUROC']:.4f} T1 {m['T1_labelled_clips']:.4f}")


if __name__ == "__main__":
    main()
