#!/usr/bin/env python3
"""Extract per-prototype responses of AudioProtoPNet on HSN focal and soundscape clips.

Writes (see AGENTS.md):
  outputs/features/prototype_responses.pt   responses for both domains + prototype vectors/weights
  outputs/results/prototype_transfer.json   per-prototype focal vs soundscape statistics

Uses the uncapped manifests by default (scripts/preprocess_hsn.py --full).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference_dataset import HSNInferenceDataset
from src.model import HSN_SPECIES_LIST, HSNProtoPNetClassifier
from src.prototype_analysis import extract_prototype_responses, prototype_transfer_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_prototype_activations")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pre = Path("data/HSN/preprocessed")
    p.add_argument("--focal-manifest", type=Path, default=pre / "focal_manifest_full.parquet")
    p.add_argument("--soundscape-manifest", type=Path, default=pre / "soundscape_manifest_full.parquet")
    p.add_argument("--max-focal-samples", type=int, default=None)
    p.add_argument("--max-soundscape-samples", type=int, default=None)
    p.add_argument("--features-path", type=Path, default=Path("outputs/features/prototype_responses.pt"))
    p.add_argument("--results-path", type=Path, default=Path("outputs/results/prototype_transfer.json"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--min-soundscape-pos", type=int, default=10,
                   help="Species need at least this many soundscape positives to be analysed")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    logger.info(f"Using device: {device}")
    model = HSNProtoPNetClassifier(device=device)

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
    args.features_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.features_path)
    logger.info(f"Saved responses -> {args.features_path}")

    stats = prototype_transfer_stats(
        results["focal"], results["soundscape"], weights, HSN_SPECIES_LIST,
        min_soundscape_pos=args.min_soundscape_pos)
    stats["n_clips"] = {k: int(len(v["targets"])) for k, v in results.items()}
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(stats, indent=2))
    logger.info(f"Saved stats -> {args.results_path}")


if __name__ == "__main__":
    main()
