#!/usr/bin/env python3
"""Extract per-prototype responses of AudioProtoPNet on HSN focal and soundscape clips.

RUN_ID names the model: `pretrained-xcl` (the published model) or a trained run, whose checkpoint is taken from
outputs/checkpoints/<dataset>/<run_id>/best.pt unless --checkpoint says otherwise. Writes (results store, docs/RESULTS.md):
  outputs/features/<dataset>/<run_id>/responses[_TAG].pt                 responses for both domains + prototype vectors/weights
  outputs/results/<dataset>/runs/<run_id>/prototype_transfer[_TAG].json  per-prototype focal vs soundscape statistics
                                                                         (+ a `prototype_transfer[_tag]` section in metrics.json)
--tag keeps a second extraction of the same model apart (e.g. the pretrained model on a cross-validation fold: --tag cv0).

Uses the uncapped manifests by default (scripts/data/preprocess_hsn.py --full).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch


from prototype_transfer.data.datasets import HSNInferenceDataset
from prototype_transfer.models.classifier import HSN_SPECIES_LIST, HSNProtoPNetClassifier
from prototype_transfer.analysis.prototype_responses import extract_prototype_responses, prototype_transfer_stats
from prototype_transfer.evaluation import results as store_lib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_prototype_activations")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pre = Path("data/HSN/preprocessed")
    p.add_argument("run_id", help="`pretrained-xcl`, or a run trained with scripts/train/train_protopnet.py")
    p.add_argument("--dataset", default=store_lib.DEFAULT_DATASET)
    p.add_argument("--focal-manifest", type=Path, default=pre / "focal_manifest_full.parquet")
    p.add_argument("--soundscape-manifest", type=Path, default=pre / "soundscape_manifest_full.parquet")
    p.add_argument("--max-focal-samples", type=int, default=None)
    p.add_argument("--max-soundscape-samples", type=int, default=None)
    p.add_argument("--tag", default="", help="suffix for the output files, e.g. cv0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--min-soundscape-pos", type=int, default=10,
                   help="Species need at least this many soundscape positives to be analysed")
    p.add_argument("--device", default="auto")
    p.add_argument("--checkpoint", default=None,
                   help="21-class checkpoint (default: the run's best.pt; `pretrained-xcl` uses the published model)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    logger.info(f"Using device: {device}")
    store = store_lib.Store(args.dataset)
    checkpoint = args.checkpoint or store.checkpoints(args.run_id) / "best.pt"
    if args.run_id == "pretrained-xcl" and args.checkpoint is None:
        checkpoint = None
    elif not Path(checkpoint).exists():
        raise SystemExit(f"{checkpoint} not found: train {args.run_id!r} first, or pass --checkpoint")
    model = HSNProtoPNetClassifier(device=device, checkpoint=str(checkpoint) if checkpoint else None)

    results = {}
    for name, manifest, is_sc, cap in [
        ("focal", args.focal_manifest, False, args.max_focal_samples),
        ("soundscape", args.soundscape_manifest, True, args.max_soundscape_samples),
    ]:
        ds = HSNInferenceDataset(manifest, is_soundscape=is_sc, max_samples=cap)
        results[name] = extract_prototype_responses(
            model, ds, batch_size=args.batch_size, num_workers=args.num_workers, desc=name)

    weights = model.hsn_last_layer_weights().detach().cpu().numpy()
    payload = {
        **results,
        "species_list": HSN_SPECIES_LIST,
        "last_layer_weights": weights,                                          # (21, 5)
        "prototype_vectors": model.hsn_prototype_vectors().detach().cpu().numpy(),  # (21, 5, D)
    }
    sfx = f"_{args.tag}" if args.tag else ""
    features_path = store.features(args.run_id) / f"responses{sfx}.pt"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, features_path)
    logger.info(f"Saved responses -> {features_path}")

    stats = prototype_transfer_stats(
        results["focal"], results["soundscape"], weights, HSN_SPECIES_LIST,
        min_soundscape_pos=args.min_soundscape_pos)
    stats["n_clips"] = {k: int(len(v["targets"])) for k, v in results.items()}
    kind = store_lib.read_registry(store).get(args.run_id, {}).get("kind", "reference")
    entry = store_lib.save_results(
        store, args.run_id, f"prototype_transfer{sfx}",
        {"stats_file": f"prototype_transfer{sfx}.json", "n_clips": stats["n_clips"], "min_soundscape_pos": stats["min_soundscape_pos"],
         "focal_manifest": args.focal_manifest, "soundscape_manifest": args.soundscape_manifest},
        kind=kind, files={f"prototype_transfer{sfx}.json": stats})
    logger.info(f"Saved stats -> {entry['paths']['files']}")


if __name__ == "__main__":
    main()
