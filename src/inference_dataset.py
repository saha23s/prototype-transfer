"""Manifest-backed dataset used by the inference/analysis scripts.

Reads a manifest written by scripts/preprocess_hsn.py and returns standardized
log-mel spectrograms (1, 256, 626) with 21-class HSN multi-hot targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.model import HSN_SPECIES_LIST
from src.preprocessing import AudioPreprocessor

CLASS_TO_IDX = {sp: idx for idx, sp in enumerate(HSN_SPECIES_LIST)}
NUM_CLASSES = len(HSN_SPECIES_LIST)


class HSNInferenceDataset(Dataset):
    """PyTorch Dataset loading preprocessed audio clips for inference."""

    def __init__(
        self,
        manifest_path: Path | str,
        is_soundscape: bool,
        max_samples: int | None = None,
    ) -> None:
        self.df = pd.read_parquet(manifest_path)
        if max_samples is not None and len(self.df) > max_samples:
            self.df = self.df.iloc[:max_samples].copy().reset_index(drop=True)
        self.is_soundscape = is_soundscape
        self.preprocessor = AudioPreprocessor()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]

        # Target vector (21 classes)
        target = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        if self.is_soundscape:
            classes = row.get("hsn_classes", [])
            if isinstance(classes, (list, tuple, np.ndarray)):
                for c in classes:
                    if c in CLASS_TO_IDX:
                        target[CLASS_TO_IDX[c]] = 1.0
        else:
            cls_idx = row.get("target_class")
            if cls_idx is not None and not np.isnan(cls_idx):
                target[int(cls_idx)] = 1.0

        # Preprocess spectrogram (1, 256, 626)
        spec = self.preprocessor.process_file(
            audio_path,
            is_soundscape=self.is_soundscape,
            focal_mode="peak_energy",
        )

        species_val = row.get("ebird_code")
        if species_val is None or (isinstance(species_val, float) and np.isnan(species_val)):
            classes = row.get("hsn_classes", [])
            species_str = ",".join(classes) if len(classes) > 0 else "background"
        else:
            species_str = str(species_val)

        return {
            "spectrogram": spec,
            "target": target,
            "audio_path": audio_path,
            "filename": Path(audio_path).name,
            "species": species_str,
        }
