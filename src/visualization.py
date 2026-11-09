"""Visualization and visual analysis tools for Soundscape vs Focal bird recordings.

Generates high-resolution publication-grade figures comparing:
  1. Mel spectrogram representations of Focal vs Soundscape audio.
  2. AudioProtoPNet prototype activation heatmaps overlaid on spectrograms.
  3. Spectral energy and SNR distributions across frequency bands.
  4. Class-level confidence and prototype activation intensity profiles.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

logger = logging.getLogger("visualization")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def plot_spectrogram(
    spectrogram: np.ndarray | torch.Tensor,
    title: str = "Log-Mel Spectrogram",
    sample_rate: int = 32000,
    hop_length: int = 256,
    n_mels: int = 256,
    cmap: str = "magma",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a single log-mel spectrogram with time (seconds) and frequency (kHz) axes."""
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.squeeze().detach().cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    n_mels, n_frames = spectrogram.shape
    duration = (n_frames * hop_length) / sample_rate
    nyquist_khz = sample_rate / 2000.0

    im = ax.imshow(
        spectrogram,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=[0, duration, 0, nyquist_khz],
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (kHz)")
    plt.colorbar(im, ax=ax, format="%+2.0f dB")
    return ax


def plot_focal_vs_soundscape_comparison(
    focal_specs: list[np.ndarray | torch.Tensor],
    focal_titles: list[str],
    soundscape_specs: list[np.ndarray | torch.Tensor],
    soundscape_titles: list[str],
    output_path: Path | str,
    sample_rate: int = 32000,
    hop_length: int = 256,
) -> None:
    """Side-by-side visual comparison of focal recordings vs soundscape recordings."""
    num_pairs = min(len(focal_specs), len(soundscape_specs))
    fig, axes = plt.subplots(num_pairs, 2, figsize=(14, 3.2 * num_pairs), squeeze=False)

    for i in range(num_pairs):
        # Focal on left column
        f_spec = focal_specs[i]
        if isinstance(f_spec, torch.Tensor):
            f_spec = f_spec.squeeze().detach().cpu().numpy()
        duration_f = (f_spec.shape[1] * hop_length) / sample_rate
        im0 = axes[i, 0].imshow(
            f_spec, origin="lower", aspect="auto", cmap="magma",
            extent=[0, duration_f, 0, 16]
        )
        axes[i, 0].set_title(f"Focal: {focal_titles[i]}", fontsize=11, fontweight="bold", color="#1a365d")
        axes[i, 0].set_ylabel("Frequency (kHz)")
        if i == num_pairs - 1:
            axes[i, 0].set_xlabel("Time (s)")
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # Soundscape on right column
        s_spec = soundscape_specs[i]
        if isinstance(s_spec, torch.Tensor):
            s_spec = s_spec.squeeze().detach().cpu().numpy()
        duration_s = (s_spec.shape[1] * hop_length) / sample_rate
        im1 = axes[i, 1].imshow(
            s_spec, origin="lower", aspect="auto", cmap="magma",
            extent=[0, duration_s, 0, 16]
        )
        axes[i, 1].set_title(f"Soundscape: {soundscape_titles[i]}", fontsize=11, fontweight="bold", color="#742a2a")
        if i == num_pairs - 1:
            axes[i, 1].set_xlabel("Time (s)")
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

    plt.suptitle("BirdSet HSN: Focal (Xeno-Canto) vs Soundscape (High Sierra Nevada PAM)", fontsize=14, y=1.005)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved focal vs soundscape comparison to {output_path}")


def plot_prototype_heatmap_overlay(
    spectrogram: np.ndarray | torch.Tensor,
    heatmap: np.ndarray | torch.Tensor,
    species_name: str,
    proto_idx: int,
    recording_type: str,
    output_path: Path | str,
    sample_rate: int = 32000,
    hop_length: int = 256,
) -> None:
    """Overlay prototype activation heatmap on top of spectrogram."""
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.squeeze().detach().cpu().numpy()
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.squeeze().detach().cpu().numpy()

    # Interpolate heatmap to match spectrogram dimensions
    import scipy.ndimage
    n_mels, n_frames = spectrogram.shape
    zoom_y = n_mels / heatmap.shape[0]
    zoom_x = n_frames / heatmap.shape[1]
    upsampled_heatmap = scipy.ndimage.zoom(heatmap, (zoom_y, zoom_x), order=1)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    duration = (n_frames * hop_length) / sample_rate

    # 1. Base Spectrogram
    im1 = ax1.imshow(spectrogram, origin="lower", aspect="auto", cmap="magma", extent=[0, duration, 0, 16])
    ax1.set_title(f"Log-Mel Spectrogram ({recording_type} - {species_name})", fontweight="bold")
    ax1.set_ylabel("Freq (kHz)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 2. Prototype Heatmap
    im2 = ax2.imshow(upsampled_heatmap, origin="lower", aspect="auto", cmap="viridis", extent=[0, duration, 0, 16])
    ax2.set_title(f"Prototype #{proto_idx} Activation Map (Max: {heatmap.max():.2f})", fontweight="bold")
    ax2.set_ylabel("Freq (kHz)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 3. Superimposed Overlay
    ax3.imshow(spectrogram, origin="lower", aspect="auto", cmap="gray", extent=[0, duration, 0, 16])
    im3 = ax3.imshow(upsampled_heatmap, origin="lower", aspect="auto", cmap="jet", alpha=0.55, extent=[0, duration, 0, 16])
    ax3.set_title("Superimposed Explainability Overlay", fontweight="bold")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Freq (kHz)")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved prototype overlay to {output_path}")


def plot_activation_distributions(
    focal_activations: np.ndarray,
    soundscape_activations: np.ndarray,
    species_names: list[str],
    output_path: Path | str,
) -> None:
    """Compare distribution of prototype activation strengths between focal and soundscapes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Boxplot of activations for top species
    n_display = min(8, len(species_names))
    focal_data = []
    soundscape_data = []
    labels = []

    for i in range(n_display):
        sp = species_names[i]
        f_vals = focal_activations[:, i] if focal_activations.ndim == 2 else focal_activations
        s_vals = soundscape_activations[:, i] if soundscape_activations.ndim == 2 else soundscape_activations
        focal_data.append(f_vals)
        soundscape_data.append(s_vals)
        labels.append(sp)

    # Kernel Density Estimation comparing overall activations
    sns.kdeplot(focal_activations.flatten(), label="Focal (Xeno-Canto)", color="#2b6cb0", fill=True, alpha=0.4, ax=ax1)
    sns.kdeplot(soundscape_activations.flatten(), label="Soundscape (PAM)", color="#c53030", fill=True, alpha=0.4, ax=ax1)
    ax1.set_title("Overall Prototype Activation Density", fontweight="bold")
    ax1.set_xlabel("Activation Strength")
    ax1.set_ylabel("Density")
    ax1.legend()

    # Per-species median activation comparison
    focal_medians = [np.median(d) for d in focal_data]
    soundscape_medians = [np.median(d) for d in soundscape_data]
    x = np.arange(n_display)
    width = 0.35

    ax2.bar(x - width/2, focal_medians, width, label="Focal Median", color="#2b6cb0")
    ax2.bar(x + width/2, soundscape_medians, width, label="Soundscape Median", color="#c53030")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_title("Median Activation by Species", fontweight="bold")
    ax2.set_ylabel("Median Prototype Activation")
    ax2.legend()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved activation distribution figure to {output_path}")
