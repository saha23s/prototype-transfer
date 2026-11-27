"""Training dataset with augmentation for one-domain ProtoPNet training.

Focal recordings: random 5-s crop (partial decode). Soundscape clips: the 5-s clip itself.
Augmentations (train only): random gain, time/frequency masking on the normalised log-mel.
Evaluation uses prototype_transfer.data.datasets.HSNInferenceDataset (peak-energy window for focal).
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

from prototype_transfer.models.classifier import HSN_SPECIES_LIST
from prototype_transfer.data.preprocessing import AudioPreprocessor

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
                 time_masks: int = 2, max_time: int = 60, freq_masks: int = 2, max_freq: int = 24) -> None:
        self.df = pd.read_parquet(manifest).reset_index(drop=True)
        self.is_soundscape = is_soundscape
        self.augment = augment
        self.pre = AudioPreprocessor()
        self.tm, self.mt, self.fm, self.mf = time_masks, max_time, freq_masks, max_freq

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
        return {"spectrogram": spec, "target": targets_of(row, self.is_soundscape), "audio_path": path}
