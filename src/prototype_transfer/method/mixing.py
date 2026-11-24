"""Waveform mixing used by the shift-matched training augmentations (background and overlap mixing).

Pure functions on [1, T] waveforms (no I/O, no model), so they can be unit-tested (tests/test_mixing.py).
All randomness comes from an explicit numpy Generator.

SNR convention: snr_db = 10 log10( power(signal) / power(added source) ), power = mean square over time.
"""

from __future__ import annotations

import numpy as np
import torch

NOISE_KINDS = ("white", "pink", "brown")   # spectral slope 1/f^beta with beta = 0, 1, 2
_EPS = 1e-12


def power(x: torch.Tensor) -> torch.Tensor:
    """Mean square over time. x [..., T] -> [...]."""
    return x.pow(2).mean(dim=-1)


def mix_at_snr(x: torch.Tensor, b: torch.Tensor, snr_db: float) -> torch.Tensor:
    """x + g * b with g chosen so that power(x) / power(g * b) = snr_db.  x, b: [1, T]."""
    g = torch.sqrt(power(x) / (power(b).clamp_min(_EPS) * 10 ** (snr_db / 10)))
    return x + g.unsqueeze(-1) * b


def peak_guard(x: torch.Tensor) -> torch.Tensor:
    """Scale down only if max |x| > 1 (waveforms are float in [-1, 1]); never scales up."""
    return x / x.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)


def coloured_noise(n: int, kind: str, rng: np.random.Generator) -> torch.Tensor:
    """[1, n] noise with power spectrum ~ 1/f^beta (white 0, pink 1, brown 2), normalised to unit power."""
    beta = {"white": 0.0, "pink": 1.0, "brown": 2.0}[kind]
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.arange(1, len(spec) + 1, dtype=np.float64)
    spec = spec / f ** (beta / 2)                 # amplitude ~ f^(-beta/2) -> power ~ f^(-beta)
    x = torch.from_numpy(np.fft.irfft(spec, n=n)).float().unsqueeze(0)
    return x / power(x).sqrt().clamp_min(_EPS)


def mix_background(x: torch.Tensor, b: torch.Tensor, snr_min: float, snr_max: float, rng: np.random.Generator) -> tuple[torch.Tensor, float]:
    """Add background b to x at an SNR drawn uniformly from [snr_min, snr_max] dB. Returns (mixture, snr_db)."""
    snr = float(rng.uniform(snr_min, snr_max))
    return mix_at_snr(x, b, snr), snr


def mix_overlap(x: torch.Tensor, y: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, gain_min: float, gain_max: float,
                rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Overlay a second source y on x, `rel` dB relative to x (rel ~ U[gain_min, gain_max], <= 0 keeps x dominant).

    Multi-label mixup: the target is the union (elementwise max) of both label vectors; at most two sources.
    Returns (mixture, target, rel_db)."""
    rel = float(rng.uniform(gain_min, gain_max))
    return mix_at_snr(x, y, -rel), torch.maximum(t_x, t_y), rel


def select_background(max_prob: np.ndarray, quantile: float) -> np.ndarray:
    """Label-free background selector: boolean mask of the clips whose max species probability under the pretrained
    model is in the lowest `quantile` fraction. Uses model outputs only, never annotations."""
    return max_prob <= np.quantile(max_prob, quantile)
