#!/usr/bin/env python3
"""Run AudioProtoPNet inference and evaluation on BirdSet HSN Soundscape vs Focal recordings.

Evaluates:
  1. Focal recordings (clean Xeno-Canto bird songs).
  2. Soundscape recordings (continuous High Sierra Nevada PAM clips).
Computes official metrics:
  - cmAP (Class-mean Average Precision)
  - Macro ROC-AUC
  - Top-1 Accuracy
  - Macro F1
Saves results to outputs/results/ and caches activations for visual analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import HSNProtoPNetClassifier, HSN_SPECIES_LIST
from src.inference_dataset import HSNInferenceDataset
from src.metrics import evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_inference")

def run_evaluation(
    model: HSNProtoPNetClassifier,
    dataset: Dataset,
    batch_size: int = 16,
    device: str = "cpu",
    desc: str = "Evaluation",
) -> dict[str, Any]:
    """Execute model inference over dataset and compute metrics and activation summaries."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_targets = []
    all_probs = []
    all_logits = []
    all_class_proto_max = []
    sample_metadata = []

    # Store a few representative samples with full spatial heatmaps for visual analysis
    cached_heatmaps = []

    logger.info(f"Starting {desc} ({len(dataset)} samples)...")
    for batch_idx, batch in enumerate(tqdm(loader, desc=desc)):
        specs = batch["spectrogram"]  # (B, 1, 256, 626)
        targets = batch["target"]    # (B, 21)

        out = model(specs, return_heatmaps=(batch_idx < 10))  # cache heatmaps for first 10 batches

        probs = out["probs"].detach().cpu().numpy()
        logits = out["logits"].detach().cpu().numpy()
        proto_max = out["class_max_activations"].detach().cpu().numpy()

        all_probs.append(probs)
        all_logits.append(logits)
        all_targets.append(targets.numpy())
        all_class_proto_max.append(proto_max)

        for i in range(len(specs)):
            sample_metadata.append({
                "filename": batch["filename"][i],
                "audio_path": batch["audio_path"][i],
                "species": batch["species"][i],
            })

            # Cache up to 15 positive examples with spatial heatmaps for plotting
            if len(cached_heatmaps) < 15 and torch.sum(targets[i]) > 0 and "prototype_activations" in out:
                active_classes = torch.where(targets[i] == 1)[0].tolist()
                for c_idx in active_classes:
                    cached_heatmaps.append({
                        "filename": batch["filename"][i],
                        "species": HSN_SPECIES_LIST[c_idx],
                        "class_idx": c_idx,
                        "spectrogram": specs[i].clone().cpu(),
                        # heatmaps for 5 prototypes of this class: (5, 16, 38)
                        "prototype_heatmaps": out["prototype_activations"][i, c_idx].clone().cpu(),
                        "prob": float(probs[i, c_idx]),
                        "proto_scores": out["prototype_max_activations"][i, c_idx].clone().cpu().numpy().tolist(),
                    })

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_probs, axis=0)
    y_proto_max = np.concatenate(all_class_proto_max, axis=0)

    metrics = evaluate_predictions(y_true, y_pred, class_names=HSN_SPECIES_LIST)
    metrics["mean_prototype_activation"] = float(np.mean(y_proto_max))
    metrics["max_prototype_activation"] = float(np.max(y_proto_max))

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proto_max": y_proto_max,
        "cached_heatmaps": cached_heatmaps,
        "sample_metadata": sample_metadata,
    }


def parse_args() -> argparse.Namespace:
    # Defaults for batch size / device / output dir come from the YAML config (CLI overrides).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=Path("configs/model_protonet.yaml"))
    cfg_path, _ = pre.parse_known_args()
    with open(cfg_path.config) as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Evaluate AudioProtoPNet on BirdSet HSN", parents=[pre])
    parser.add_argument("--focal-manifest", type=Path,
                        default=Path("data/HSN/preprocessed/focal_manifest.parquet"))
    parser.add_argument("--soundscape-manifest", type=Path,
                        default=Path("data/HSN/preprocessed/soundscape_manifest.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path(cfg["evaluation"]["output_dir"]))
    parser.add_argument("--batch-size", type=int, default=cfg["model"]["batch_size"])
    parser.add_argument("--max-focal-samples", type=int, default=250)
    parser.add_argument("--max-soundscape-samples", type=int, default=250)
    parser.add_argument("--device", type=str, default=cfg["model"]["device"],
                        help="'auto' (cuda if available, else cpu), 'cpu', or 'cuda'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {args.device}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    model = HSNProtoPNetClassifier(device=args.device)

    # 2. Evaluate Focal
    focal_dataset = HSNInferenceDataset(
        manifest_path=args.focal_manifest,
        is_soundscape=False,
        max_samples=args.max_focal_samples,
    )
    focal_res = run_evaluation(
        model=model,
        dataset=focal_dataset,
        batch_size=args.batch_size,
        device=args.device,
        desc="Focal Evaluation",
    )

    # 3. Evaluate Soundscape
    soundscape_dataset = HSNInferenceDataset(
        manifest_path=args.soundscape_manifest,
        is_soundscape=True,
        max_samples=args.max_soundscape_samples,
    )
    sc_res = run_evaluation(
        model=model,
        dataset=soundscape_dataset,
        batch_size=args.batch_size,
        device=args.device,
        desc="Soundscape Evaluation",
    )

    # 4. Save Metrics
    focal_metrics_path = output_dir / "focal_metrics.json"
    sc_metrics_path = output_dir / "soundscape_metrics.json"
    with open(focal_metrics_path, "w") as f:
        json.dump(focal_res["metrics"], f, indent=2)
    with open(sc_metrics_path, "w") as f:
        json.dump(sc_res["metrics"], f, indent=2)

    comparison_summary = {
        "focal": {
            "num_samples": focal_res["metrics"]["num_samples"],
            "cmAP": focal_res["metrics"]["cmAP"],
            "macro_roc_auc": focal_res["metrics"]["macro_roc_auc"],
            "top1_accuracy": focal_res["metrics"]["top1_accuracy"],
            "mean_prototype_activation": focal_res["metrics"]["mean_prototype_activation"],
        },
        "soundscape": {
            "num_samples": sc_res["metrics"]["num_samples"],
            "cmAP": sc_res["metrics"]["cmAP"],
            "macro_roc_auc": sc_res["metrics"]["macro_roc_auc"],
            "top1_accuracy": sc_res["metrics"]["top1_accuracy"],
            "mean_prototype_activation": sc_res["metrics"]["mean_prototype_activation"],
        },
    }
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(comparison_summary, f, indent=2)

    # 5. Cache Activations & Data for Visual Analysis
    cache_path = output_dir / "eval_cache.pt"
    torch.save({
        "focal": {
            "y_true": focal_res["y_true"],
            "y_pred": focal_res["y_pred"],
            "y_proto_max": focal_res["y_proto_max"],
            "cached_heatmaps": focal_res["cached_heatmaps"],
        },
        "soundscape": {
            "y_true": sc_res["y_true"],
            "y_pred": sc_res["y_pred"],
            "y_proto_max": sc_res["y_proto_max"],
            "cached_heatmaps": sc_res["cached_heatmaps"],
        },
        "species_list": HSN_SPECIES_LIST,
    }, cache_path)

    logger.info("================ EVALUATION SUMMARY ================")
    logger.info(f"Focal cmAP:       {focal_res['metrics']['cmAP']:.4f} | ROC-AUC: {focal_res['metrics']['macro_roc_auc']:.4f} | Top-1: {focal_res['metrics']['top1_accuracy']:.4f}")
    logger.info(f"Soundscape cmAP:  {sc_res['metrics']['cmAP']:.4f} | ROC-AUC: {sc_res['metrics']['macro_roc_auc']:.4f} | Top-1: {sc_res['metrics']['top1_accuracy']:.4f}")
    logger.info(f"Results successfully saved to {output_dir}")


if __name__ == "__main__":
    main()
