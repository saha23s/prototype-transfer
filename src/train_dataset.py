"""Training dataset with augmentation for one-domain ProtoPNet training.

Focal recordings: random 5-s crop (partial decode). Soundscape clips: the 5-s clip itself.
Augmentations (train only): random gain, time/frequency masking on the normalised log-mel.
Optional shift-matched augmentations for focal training (all off by default, so existing runs are reproducible):
background mixing (generic coloured noise or real soundscape-train clips the pretrained model finds empty) and overlap
mixing (a second focal species, labels = union); see src/mixing.py. HSNPseudoDataset serves unlabeled soundscape-train
clips with cached teacher probabilities as soft targets.
Evaluation uses src.inference_dataset.HSNInferenceDataset (peak-energy window for focal).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from src.mixing import NOISE_KINDS, coloured_noise, mix_background, mix_overlap, peak_guard
from src.model import HSN_SPECIES_LIST
from src.preprocessing import AudioPreprocessor

CLASS_TO_IDX = {sp: i for i, sp in enumerate(HSN_SPECIES_LIST)}


def targets_of(row: pd.Series, is_soundscape: bool) -> torch.Tensor:
    t = torch.zeros(len(HSN_SPECIES_LIST))
    if is_soundscape:
        for c in row["hsn_classes"]:
            t[CLASS_TO_IDX[c]] = 1.0
    else:
        t[int(row["target_class"])] = 1.0
    return t


class HSNTrainDataset(Dataset):
    def __init__(self, manifest: Path | str, is_soundscape: bool, augment: bool = True,
                 time_masks: int = 2, max_time: int = 60, freq_masks: int = 2, max_freq: int = 24,
                 bg_kind: str = "none", bg_manifest: Path | str | None = None, bg_prob: float = 0.5,
                 snr_min: float = -5.0, snr_max: float = 20.0,
                 overlap_prob: float = 0.0, overlap_gain_min: float = -10.0, overlap_gain_max: float = 0.0) -> None:
        """bg_kind: none | noise (white/pink/brown) | real (random 5 s of a clip listed in bg_manifest, column audio_path).
        With probability bg_prob the background is added at an SNR ~ U[snr_min, snr_max] dB; with probability
        overlap_prob a second focal clip of a different species is overlaid at a level ~ U[overlap_gain_min,
        overlap_gain_max] dB relative to the clip and the target becomes the union. Both act on the waveform after
        the 5 s crop and before the gain augmentation (overlap first, then background), and only on focal training."""
        self.df = pd.read_parquet(manifest).reset_index(drop=True)
        self.is_soundscape = is_soundscape
        self.augment = augment
        self.pre = AudioPreprocessor()
        self.tm, self.mt, self.fm, self.mf = time_masks, max_time, freq_masks, max_freq
        assert bg_kind in ("none", "noise", "real"), bg_kind
        if bg_kind == "real":
            assert bg_manifest is not None, "bg_kind='real' needs bg_manifest"
            self.bg_paths = pd.read_parquet(bg_manifest)["audio_path"].tolist()
            assert self.bg_paths, f"{bg_manifest} lists no background clips"
        self.bg_kind, self.bg_prob, self.snr = bg_kind, bg_prob, (snr_min, snr_max)
        self.ov_prob, self.ov_gain = overlap_prob, (overlap_gain_min, overlap_gain_max)
        if is_soundscape:
            assert bg_kind == "none" and overlap_prob == 0, "background / overlap mixing is for focal training clips only"
        self._rng: np.random.Generator | None = None

    def mixing_config(self) -> dict[str, Any]:
        """The effective mixing settings, printed at the start of a run and saved with it."""
        return {"bg_kind": self.bg_kind, "bg_prob": self.bg_prob if self.bg_kind != "none" else 0.0, "snr_db": list(self.snr),
                "overlap_prob": self.ov_prob, "overlap_gain_db": list(self.ov_gain)}

    @property
    def rng(self) -> np.random.Generator:
        """Per-process generator, seeded from torch's (per-worker, per-run) seed on first use."""
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
        return self._rng

    def _target(self, row: pd.Series) -> torch.Tensor:
        return targets_of(row, self.is_soundscape)

    def _background(self) -> torch.Tensor:
        if self.bg_kind == "noise":
            return coloured_noise(self.pre.num_samples, NOISE_KINDS[self.rng.integers(len(NOISE_KINDS))], self.rng)
        path = self.bg_paths[self.rng.integers(len(self.bg_paths))]
        return self.pre.extract_soundscape_window(self.pre.load_and_resample(path))

    def _overlay(self, species: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A random focal-train clip of a different species: (random 5 s crop, its target)."""
        other = np.nonzero(self.df["target_class"].to_numpy() != species)[0]
        row = self.df.iloc[int(self.rng.choice(other))]
        return self._load_crop(row["audio_path"]), targets_of(row, False)

    def __len__(self) -> int:
        return len(self.df)

    def _load_crop(self, path: str) -> torch.Tensor:
        n = self.pre.num_samples
        info = sf.info(path)
        sr = info.samplerate
        need = int(np.ceil(n * sr / self.pre.sample_rate))
        start = 0 if info.frames <= need else np.random.randint(0, info.frames - need + 1)
        data, sr = sf.read(path, start=start, frames=need, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T).mean(0, keepdim=True)
        if sr != self.pre.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.pre.sample_rate)
        if wav.shape[-1] < n:
            wav = torch.nn.functional.pad(wav, (0, n - wav.shape[-1]))
        return wav[:, :n]

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.df.iloc[i]
        path = row["audio_path"]
        if self.is_soundscape:
            wav = self.pre.extract_soundscape_window(self.pre.load_and_resample(path))
        else:
            wav = self._load_crop(path)
        target = self._target(row)
        if self.augment and self.ov_prob > 0 and self.rng.random() < self.ov_prob:
            other, t_other = self._overlay(int(row["target_class"]))
            wav, target, _ = mix_overlap(wav, other, target, t_other, *self.ov_gain, self.rng)
        if self.augment and self.bg_kind != "none" and self.rng.random() < self.bg_prob:
            wav, _ = mix_background(wav, self._background(), *self.snr, self.rng)
        if self.augment and (self.bg_kind != "none" or self.ov_prob > 0):
            wav = peak_guard(wav)
        if self.augment:
            wav = wav * float(10 ** (np.random.uniform(-6, 6) / 20))
        spec = self.pre.transform_to_spectrogram(wav)  # (1, 256, 626)
        if self.augment:
            fill = float(spec.mean())
            for _ in range(self.tm):
                w = np.random.randint(0, self.mt + 1); t0 = np.random.randint(0, spec.shape[-1] - w + 1)
                spec[..., t0:t0 + w] = fill
            for _ in range(self.fm):
                h = np.random.randint(0, self.mf + 1); f0 = np.random.randint(0, spec.shape[-2] - h + 1)
                spec[..., f0:f0 + h, :] = fill
        return {"spectrogram": spec, "target": target, "audio_path": path}


class HSNPseudoDataset(HSNTrainDataset):
    """Unlabeled soundscape-train clips with cached soft targets (R4): the teacher's sigmoid probabilities.

    The manifest (scripts/make_background_manifest.py) has audio_path and teacher_probs (21 floats per clip) and no
    annotations; they are never read. Gets the existing masking and gain noise, no background / overlap mixing."""

    def __init__(self, manifest: Path | str, **kw: Any) -> None:
        super().__init__(manifest, is_soundscape=True, **kw)
        assert "teacher_probs" in self.df, f"{manifest} has no teacher_probs column"
        assert "hsn_classes" not in self.df, "a pseudo-label manifest must not carry annotations"

    def _target(self, row: pd.Series) -> torch.Tensor:
        return torch.tensor(np.asarray(row["teacher_probs"], dtype=np.float32))
