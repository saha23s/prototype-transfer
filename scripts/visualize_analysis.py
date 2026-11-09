#!/usr/bin/env python3
"""Generate publication-quality visual analysis comparing Soundscape vs Focal recordings.

Creates:
  1. Side-by-side Spectrogram Comparison (focal vs soundscape).
  2. Explainable Prototype Activation Overlays.
  3. Prototype Activation Strength Distributions (KDE & boxplots).
  4. Benchmark Performance Metrics Comparison (cmAP, AUC, Top-1).
  5. Spectral Power Distribution across frequency bands (SNR analysis).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import seaborn as sns
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualization import (
    plot_focal_vs_soundscape_comparison,
    plot_prototype_heatmap_overlay,
    plot_activation_distributions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("visualize_analysis")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"figure.dpi": 300, "font.family": "sans-serif"})


def generate_metrics_comparison_figure(
    summary_path: Path,
    output_path: Path,
) -> None:
    """Plot bar chart comparing cmAP, ROC-AUC, and Top-1 Accuracy."""
    with open(summary_path) as f:
        data = json.load(f)

    metrics = ["cmAP", "macro_roc_auc", "top1_accuracy"]
    metric_labels = ["cmAP (Class-mean AP)", "Macro ROC-AUC", "Top-1 Accuracy"]
    focal_vals = [data["focal"][m] for m in metrics]
    sc_vals = [data["soundscape"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, focal_vals, width, label="Focal (Xeno-Canto)", color="#2b6cb0", edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, sc_vals, width, label="Soundscape (High Sierra PAM)", color="#c53030", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Score (0.0 - 1.0)", fontweight="bold")
    ax.set_title("AudioProtoPNet-5-BirdSet-XCL: Focal vs Soundscape Domain Gap", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    # Add score annotations above bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved metrics comparison figure -> {output_path}")


def generate_frequency_energy_profile(
    focal_specs: list[torch.Tensor],
    soundscape_specs: list[torch.Tensor],
    output_path: Path,
) -> None:
    """Plot average spectral energy vs frequency comparing focal and soundscapes."""
    # Convert spectrograms to numpy
    f_arr = [s.squeeze().numpy() for s in focal_specs[:50]]
    s_arr = [s.squeeze().numpy() for s in soundscape_specs[:50]]

    # Mean energy across time frames for each mel frequency bin
    f_mean_freq = np.mean([np.mean(s, axis=1) for s in f_arr], axis=0)
    s_mean_freq = np.mean([np.mean(s, axis=1) for s in s_arr], axis=0)

    # Mel bins to approximate frequency in kHz (0 to 16 kHz)
    freqs_khz = np.linspace(0, 16, len(f_mean_freq))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(freqs_khz, f_mean_freq, label="Focal (Xeno-Canto)", color="#2b6cb0", linewidth=2.2)
    ax.plot(freqs_khz, s_mean_freq, label="Soundscape (PAM)", color="#c53030", linewidth=2.2, linestyle="--")

    ax.set_xlabel("Frequency (kHz)", fontweight="bold")
    ax.set_ylabel("Normalized Spectral Power (dB scale)", fontweight="bold")
    ax.set_title("Frequency Power Spectrum: High-SNR Focal vs Ambient Soundscape", fontsize=13, fontweight="bold")
    ax.axvspan(0, 1.5, color="gray", alpha=0.15, label="Low-Frequency Wind/Ambient Floor (<1.5 kHz)")
    ax.axvspan(2.0, 7.0, color="#38a169", alpha=0.12, label="Core Avian Vocalization Band (2-7 kHz)")
    ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved frequency energy profile -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visual analysis figures")
    parser.add_argument("--cache-path", type=Path, default=Path("outputs/results/eval_cache.pt"))
    parser.add_argument("--summary-path", type=Path, default=Path("outputs/results/comparison_summary.json"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = args.figures_dir.resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading evaluation cache from {args.cache_path}...")
    # weights_only=False: the cache holds numpy arrays and is written by run_inference.py (trusted).
    cache = torch.load(args.cache_path, map_location="cpu", weights_only=False)

    focal_data = cache["focal"]
    sc_data = cache["soundscape"]
    species_list = cache["species_list"]

    # 1. Benchmark Metrics Figure
    if args.summary_path.exists():
        generate_metrics_comparison_figure(
            args.summary_path,
            figures_dir / "fig1_benchmark_metrics_comparison.png",
        )

    # 2. Side-by-side Spectrogram Comparison
    f_heatmaps = focal_data.get("cached_heatmaps", [])
    s_heatmaps = sc_data.get("cached_heatmaps", [])

    if f_heatmaps and s_heatmaps:
        # Match species present in both
        f_by_sp = {item["species"]: item for item in f_heatmaps}
        s_by_sp = {item["species"]: item for item in s_heatmaps}
        common_species = [sp for sp in species_list if sp in f_by_sp and sp in s_by_sp]

        if not common_species:
            # Fall back to top available
            f_samples = f_heatmaps[:4]
            s_samples = s_heatmaps[:4]
        else:
            selected_species = common_species[:4]
            f_samples = [f_by_sp[sp] for sp in selected_species]
            s_samples = [s_by_sp[sp] for sp in selected_species]

        f_specs = [item["spectrogram"] for item in f_samples]
        f_titles = [f"{item['species']} (Prob: {item['prob']:.2f})" for item in f_samples]
        s_specs = [item["spectrogram"] for item in s_samples]
        s_titles = [f"{item['species']} (Prob: {item['prob']:.2f})" for item in s_samples]

        plot_focal_vs_soundscape_comparison(
            focal_specs=f_specs,
            focal_titles=f_titles,
            soundscape_specs=s_specs,
            soundscape_titles=s_titles,
            output_path=figures_dir / "fig2_spectrogram_focal_vs_soundscape.png",
        )

        # 3. Prototype Activation Overlay (Best representative focal and soundscape)
        best_focal = max(f_heatmaps, key=lambda x: x["prob"])
        best_proto_f = int(np.argmax(best_focal["proto_scores"]))
        plot_prototype_heatmap_overlay(
            spectrogram=best_focal["spectrogram"],
            heatmap=best_focal["prototype_heatmaps"][best_proto_f],
            species_name=best_focal["species"],
            proto_idx=best_proto_f,
            recording_type="Focal (Clean)",
            output_path=figures_dir / "fig3_prototype_overlay_focal.png",
        )

        best_sc = max(s_heatmaps, key=lambda x: x["prob"])
        best_proto_s = int(np.argmax(best_sc["proto_scores"]))
        plot_prototype_heatmap_overlay(
            spectrogram=best_sc["spectrogram"],
            heatmap=best_sc["prototype_heatmaps"][best_proto_s],
            species_name=best_sc["species"],
            proto_idx=best_proto_s,
            recording_type="Soundscape (Noisy/Ambient)",
            output_path=figures_dir / "fig4_prototype_overlay_soundscape.png",
        )

        # 4. Frequency Energy Profile
        generate_frequency_energy_profile(
            focal_specs=[item["spectrogram"] for item in f_heatmaps],
            soundscape_specs=[item["spectrogram"] for item in s_heatmaps],
            output_path=figures_dir / "fig5_spectral_energy_profile.png",
        )

    # 5. Prototype Activation Distribution
    plot_activation_distributions(
        focal_activations=focal_data["y_proto_max"],
        soundscape_activations=sc_data["y_proto_max"],
        species_names=species_list,
        output_path=figures_dir / "fig6_activation_distributions.png",
    )

    logger.info("All visual analysis figures successfully generated!")


if __name__ == "__main__":
    main()
