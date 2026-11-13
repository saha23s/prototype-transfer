"""Prototype-level focal -> soundscape transfer analysis for AudioProtoPNet on HSN.

Two halves:
  * ``extract_prototype_responses``: run the model over a manifest and keep, for every
    clip, the max activation / peak location of all 105 HSN prototypes, plus full
    heatmaps and latent peak patches for (clip, true class) pairs.
  * ``prototype_transfer_stats``: per-prototype statistics comparing the two domains
    (activation on positives, detection AUC, share of the class logit, peak frequency).

Activation grid: 16 frequency rows x 38 time columns over the (256 mel x 626 frame)
spectrogram, i.e. one cell ~ 16 mel bins x ~0.13 s.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.model import HSNProtoPNetClassifier

logger = logging.getLogger("prototype_analysis")

SAMPLE_RATE = 32_000
N_MELS = 256
HOP_LENGTH = 256


# ----------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------
@torch.no_grad()
def extract_prototype_responses(
    model: HSNProtoPNetClassifier,
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 8,
    desc: str = "Extract",
) -> dict[str, Any]:
    """Run the model over ``dataset`` and collect prototype responses.

    Returns a dict of numpy arrays:
      targets (N, 21) bool, probs (N, 21), proto_max (N, 21, 5),
      peak_freq / peak_time (N, 21, 5) uint8,
      pos_clip / pos_class (M,) int: index pairs of positive (clip, class) entries,
      pos_heatmaps (M, 5, H, W) float16, pos_peak_feats (M, 5, D) float16,
      audio_path / species (N,) lists.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    keys = ("targets", "probs", "proto_max", "peak_freq", "peak_time")
    acc: dict[str, list] = {k: [] for k in keys}
    pos_clip, pos_class, pos_heat, pos_feat = [], [], [], []
    audio_paths: list[str] = []
    species: list[str] = []

    offset = 0
    for batch in tqdm(loader, desc=desc):
        r = model.prototype_responses(batch["spectrogram"])
        targets = batch["target"].numpy() > 0.5
        acc["targets"].append(targets)
        acc["probs"].append(r["probs"].cpu().numpy())
        acc["proto_max"].append(r["proto_max"].cpu().numpy())
        acc["peak_freq"].append(r["peak_freq"].cpu().numpy().astype(np.uint8))
        acc["peak_time"].append(r["peak_time"].cpu().numpy().astype(np.uint8))

        bi, ci = np.nonzero(targets)
        if len(bi):
            bi_t, ci_t = torch.from_numpy(bi), torch.from_numpy(ci)
            pos_heat.append(r["heatmaps"][bi_t, ci_t].half().cpu().numpy())
            pos_feat.append(r["peak_feats"][bi_t, ci_t].half().cpu().numpy())
            pos_clip.append(bi + offset)
            pos_class.append(ci)

        audio_paths.extend(batch["audio_path"])
        species.extend(batch["species"])
        offset += len(targets)

    out = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    if pos_clip:
        out["pos_clip"] = np.concatenate(pos_clip)
        out["pos_class"] = np.concatenate(pos_class)
        out["pos_heatmaps"] = np.concatenate(pos_heat)
        out["pos_peak_feats"] = np.concatenate(pos_feat)
    out["audio_path"] = audio_paths
    out["species"] = species
    logger.info(f"{desc}: {offset} clips, {len(out.get('pos_clip', []))} positive (clip, class) pairs")
    return out


# ----------------------------------------------------------------------------
# Grid geometry
# ----------------------------------------------------------------------------
def _hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_bin_centers_hz(n_mels: int = N_MELS, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Center frequency (Hz) of each mel bin of torchaudio's MelScale (HTK, f_min=0)."""
    pts = np.linspace(_hz_to_mel(0.0), _hz_to_mel(sample_rate / 2), n_mels + 2)
    return _mel_to_hz(pts[1:-1])


def grid_row_centers_khz(n_rows: int = 16, n_mels: int = N_MELS) -> np.ndarray:
    """Center frequency (kHz) of each activation-grid frequency row."""
    centers = mel_bin_centers_hz(n_mels)
    per_row = n_mels // n_rows
    return np.array([centers[r * per_row:(r + 1) * per_row].mean() for r in range(n_rows)]) / 1000.0


def grid_cell_to_spec_slice(
    row: int, col: int, n_rows: int = 16, n_cols: int = 38,
    n_mels: int = N_MELS, n_frames: int = 626, pad_cells: int = 2,
) -> tuple[slice, slice]:
    """Spectrogram (mel, frame) slices covering a grid cell plus ``pad_cells`` of context."""
    mh, fw = n_mels / n_rows, n_frames / n_cols
    r0, r1 = max(0, row - pad_cells), min(n_rows, row + pad_cells + 1)
    c0, c1 = max(0, col - pad_cells), min(n_cols, col + pad_cells + 1)
    return slice(int(r0 * mh), int(r1 * mh)), slice(int(c0 * fw), int(c1 * fw))


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def _safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    return float(roc_auc_score(y, s))


def _median_or_none(x: np.ndarray) -> float | None:
    return float(np.median(x)) if len(x) else None


def prototype_transfer_stats(
    focal: dict[str, Any],
    soundscape: dict[str, Any],
    weights: np.ndarray,
    species_list: list[str],
    min_soundscape_pos: int = 10,
) -> dict[str, Any]:
    """Per-prototype focal vs soundscape statistics.

    For each species c and prototype k (positives = clips labelled with c):
      act_pos / act_neg:   median max-activation on positives / negatives
      auc:                 ROC-AUC of the prototype's max activation as a detector of c
                           (focal negatives = other species' focal clips;
                            soundscape negatives = all other soundscape clips)
      logit_share:         mean fraction of the class evidence sum_k w_k*a_k on positives
                           contributed by this prototype ("which prototype carries the decision")
      peak_khz:            median frequency (kHz) of the activation peak on positives
    """
    row_khz = grid_row_centers_khz()
    domains = {"focal": focal, "soundscape": soundscape}
    out: dict[str, Any] = {"species": {}, "min_soundscape_pos": min_soundscape_pos}

    for c, sp in enumerate(species_list):
        w = weights[c]
        sp_entry: dict[str, Any] = {"last_layer_weights": w.tolist(), "domains": {}}
        for name, d in domains.items():
            y = d["targets"][:, c]
            act = d["proto_max"][:, c, :]  # (N, 5)
            pos, neg = act[y], act[~y]
            contrib = pos * w[None, :]
            share = contrib / np.clip(contrib.sum(axis=1, keepdims=True), 1e-8, None)
            peak_rows = d["peak_freq"][y, c, :].astype(int)
            sp_entry["domains"][name] = {
                "n_pos": int(y.sum()),
                "class_auc": _safe_auc(y, d["probs"][:, c]),
                "prototypes": [
                    {
                        "act_pos": _median_or_none(pos[:, k]),
                        "act_neg": _median_or_none(neg[:, k]),
                        "auc": _safe_auc(y, act[:, k]),
                        "logit_share": float(share[:, k].mean()) if len(pos) else None,
                        "peak_khz": _median_or_none(row_khz[peak_rows[:, k]]),
                    }
                    for k in range(act.shape[1])
                ],
            }
        sp_entry["analysed"] = sp_entry["domains"]["soundscape"]["n_pos"] >= min_soundscape_pos
        out["species"][sp] = sp_entry
    return out
