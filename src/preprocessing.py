"""Audio preprocessing and feature extraction for BirdSet HSN and AudioProtoPNet.

Implements exact log-mel spectrogram transformations and normalization matching
the DBD-research-group/AudioProtoPNet-5-BirdSet-XCL preprocessor configuration:
  - Sample rate: 32,000 Hz
  - Clip duration: 5.0 seconds (160,000 samples)
  - STFT: n_fft=2048, hop_length=256, n_stft=1025
  - Mel scale: n_mels=256, power=2.0
  - Decibel scaling: AmplitudeToDB(top_db=80)
  - Standardization: mean=-13.369, std=13.162
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torchaudio
from torchaudio import transforms

logger = logging.getLogger("preprocessing")

# AudioProtoPNet default specifications
DEFAULT_SAMPLE_RATE = 32_000
DEFAULT_CLIP_SECONDS = 5.0
DEFAULT_NUM_SAMPLES = int(DEFAULT_SAMPLE_RATE * DEFAULT_CLIP_SECONDS)  # 160,000

# Spectrogram specifications
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 256
DEFAULT_N_MELS = 256
DEFAULT_N_STFT = 1025
DEFAULT_POWER = 2.0
DEFAULT_TOP_DB = 80.0

# Standardization parameters (empirical values from preprocessor_config.json)
DEFAULT_MEAN = -13.369
DEFAULT_STD = 13.162


class AudioPreprocessor:
    """Preprocesses audio waveforms into standardized log-mel spectrograms for AudioProtoPNet."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        clip_duration: float = DEFAULT_CLIP_SECONDS,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        n_mels: int = DEFAULT_N_MELS,
        n_stft: int = DEFAULT_N_STFT,
        power: float = DEFAULT_POWER,
        top_db: float = DEFAULT_TOP_DB,
        mean: float = DEFAULT_MEAN,
        std: float = DEFAULT_STD,
    ) -> None:
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.num_samples = int(round(sample_rate * clip_duration))

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_stft = n_stft
        self.power = power
        self.top_db = top_db
        self.mean = mean
        self.std = std

        # Torchaudio transform pipeline
        self.spec_transform = transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=self.power,
        )
        self.mel_scale = transforms.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            n_stft=self.n_stft,
        )
        self.db_scale = transforms.AmplitudeToDB(
            stype="power",
            top_db=self.top_db,
        )

    def load_and_resample(self, path: str | Path) -> torch.Tensor:
        """Load audio file and resample to target sample rate, averaging to mono."""
        import soundfile as sf
        try:
            data, sr = sf.read(str(path), dtype="float32")
            waveform = torch.from_numpy(data)
            if waveform.ndim > 1:
                # Transpose from (samples, channels) to (channels, samples)
                waveform = waveform.t()
                waveform = waveform.mean(dim=0, keepdim=True)
            elif waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
        except Exception:
            waveform, sr = torchaudio.load(str(path))
            if waveform.ndim > 1 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            elif waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        return waveform

    def extract_focal_window(
        self,
        waveform: torch.Tensor,
        mode: str = "peak_energy",
        event_time: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Extract a 5-second window from a variable-length focal recording.

        Args:
            waveform: Audio tensor of shape (1, num_samples).
            mode: Windowing strategy:
                - 'peak_energy': Find the 5s window with highest RMS energy (isolates bird vocalization).
                - 'event': Center on annotated timestamp (start, end).
                - 'center': Take middle 5 seconds.
                - 'start': Take initial 5 seconds.
            event_time: Optional (start_sec, end_sec) tuple.
        """
        total_samples = waveform.shape[-1]
        target_len = self.num_samples

        if total_samples <= target_len:
            # Pad with zeros to target length
            pad_amount = target_len - total_samples
            return torch.nn.functional.pad(waveform, (0, pad_amount))

        if mode == "event" and event_time is not None:
            start_sec, end_sec = event_time
            mid_sec = (start_sec + end_sec) / 2.0
            mid_sample = int(round(mid_sec * self.sample_rate))
            start_sample = max(0, mid_sample - target_len // 2)
            if start_sample + target_len > total_samples:
                start_sample = max(0, total_samples - target_len)
            return waveform[:, start_sample : start_sample + target_len]

        if mode == "peak_energy":
            # Compute rolling RMS energy with 1-second steps to locate bird vocalization
            step = self.sample_rate // 2  # 0.5 second stride
            num_windows = max(1, (total_samples - target_len) // step + 1)
            best_start = 0
            best_energy = -1.0
            for i in range(num_windows):
                start = i * step
                end = start + target_len
                window = waveform[:, start:end]
                energy = float(torch.sum(window**2))
                if energy > best_energy:
                    best_energy = energy
                    best_start = start
            return waveform[:, best_start : best_start + target_len]

        if mode == "center":
            start = (total_samples - target_len) // 2
            return waveform[:, start : start + target_len]

        # Default: start
        return waveform[:, :target_len]

    def extract_soundscape_window(self, waveform: torch.Tensor) -> torch.Tensor:
        """Process a 5-second soundscape clip (pad or truncate to exact length)."""
        target_len = self.num_samples
        total_samples = waveform.shape[-1]
        if total_samples < target_len:
            return torch.nn.functional.pad(waveform, (0, target_len - total_samples))
        return waveform[:, :target_len]

    def transform_to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Transform 5-second waveform into standardized log-mel spectrogram.

        Args:
            waveform: Tensor of shape (1, 160000) or (batch, 1, 160000).

        Returns:
            Standardized log-mel spectrogram tensor of shape (batch, 1, 256, 626)
            or (1, 256, 626).
        """
        if waveform.ndim == 2:
            # (1, num_samples) -> ensure batch dimension for consistency
            squeeze_output = True
            audio = waveform
        elif waveform.ndim == 3:
            squeeze_output = False
            audio = waveform.squeeze(1)
        else:
            squeeze_output = False
            audio = waveform

        spec = self.spec_transform(audio)
        mel = self.mel_scale(spec)
        mel_db = self.db_scale(mel)
        mel_norm = (mel_db - self.mean) / self.std

        if squeeze_output:
            # Output: (1, n_mels, time_frames) e.g. (1, 256, 626)
            return mel_norm
        else:
            # Output: (batch, 1, n_mels, time_frames)
            return mel_norm.unsqueeze(1)

    def process_file(
        self,
        filepath: str | Path,
        is_soundscape: bool = False,
        focal_mode: str = "peak_energy",
        event_time: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Complete preprocessing pipeline from file to standardized spectrogram tensor."""
        waveform = self.load_and_resample(filepath)
        if is_soundscape:
            clip = self.extract_soundscape_window(waveform)
        else:
            clip = self.extract_focal_window(waveform, mode=focal_mode, event_time=event_time)
        return self.transform_to_spectrogram(clip)
