#!/usr/bin/env python3
"""Pre-processing script for BirdSet HSN audio into standardized log-mel spectrograms.

Applies the determined pre-processing specification:
  - 32,000 Hz resampling, mono conversion
  - 5.0 second fixed clip duration (160,000 samples)
  - Peak-energy windowing for variable-length focal recordings
  - Exact STFT parameters: n_fft=2048, hop_length=256, n_mels=256, top_db=80
  - Normalization: (spec_db - (-13.369)) / 13.162
  - Generates unified manifests and pre-extracted spectrogram caches in scratch.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import AudioPreprocessor, DEFAULT_SAMPLE_RATE, DEFAULT_CLIP_SECONDS
from src.model import HSN_SPECIES_LIST

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess_hsn")

CLASS_TO_IDX = {sp: idx for idx, sp in enumerate(HSN_SPECIES_LIST)}


def build_focal_manifest(
    metadata_path: Path,
    audio_dir: Path,
    output_manifest_path: Path,
    max_samples_per_class: int | None = None,
) -> pd.DataFrame:
    """Build standardized manifest for focal recordings with 21-class one-hot/multi-hot labels."""
    logger.info(f"Reading focal metadata from {metadata_path}...")
    df = pd.read_parquet(metadata_path)

    # Check which audio files actually exist
    existing_files = {p.name: p for p in audio_dir.glob("*.ogg")}
    df = df[df["filepath"].isin(existing_files)].copy()
    logger.info(f"Focal records with existing audio: {len(df)}")

    # Assign local integer label
    df["target_class"] = df["ebird_code"].map(CLASS_TO_IDX)
    df = df.dropna(subset=["target_class"])
    df["target_class"] = df["target_class"].astype(int)

    # Optional balanced subsampling if requested
    if max_samples_per_class is not None:
        sampled_dfs = []
        for sp, group in df.groupby("ebird_code"):
            n_take = min(len(group), max_samples_per_class)
            sampled_dfs.append(group.sample(n=n_take, random_state=42))
        df = pd.concat(sampled_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
        logger.info(f"Subsampled focal dataset to {len(df)} records ({max_samples_per_class} per class max)")

    df["audio_path"] = df["filepath"].apply(lambda f: str(audio_dir / f))
    df["modality"] = "focal"

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_manifest_path, index=False)
    logger.info(f"Saved focal manifest ({len(df)} rows) -> {output_manifest_path}")
    return df


def build_soundscape_manifest(
    metadata_path: Path,
    audio_dir: Path,
    output_manifest_path: Path,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Build standardized manifest for soundscape 5s recordings."""
    logger.info(f"Reading soundscape metadata from {metadata_path}...")
    df = pd.read_parquet(metadata_path)

    # The index in HSN_metadata_test_5s is the filename (e.g. HSN_001_..._000_005.ogg)
    if "filepath" not in df.columns:
        df["filepath"] = df.index

    existing_files = {p.name: p for p in audio_dir.glob("*.ogg")}
    df = df[df["filepath"].isin(existing_files)].copy()
    logger.info(f"Soundscape records with existing audio: {len(df)}")

    df["audio_path"] = df["filepath"].apply(lambda f: str(audio_dir / f))
    df["modality"] = "soundscape"

    # Multi-label species parsing
    def get_active_classes(row) -> list[str]:
        val = row.get("ebird_code_multilabel")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return []
        if isinstance(val, (list, tuple, np.ndarray)):
            return [str(x) for x in val if str(x) in CLASS_TO_IDX]
        if isinstance(val, str):
            return [x.strip() for x in val.split(",") if x.strip() in CLASS_TO_IDX]
        return []

    df["hsn_classes"] = df.apply(get_active_classes, axis=1)
    df["num_hsn_classes"] = df["hsn_classes"].apply(len)

    if max_samples is not None and len(df) > max_samples:
        # Stratify to ensure clips with target bird vocalizations are well represented
        with_birds = df[df["num_hsn_classes"] > 0]
        without_birds = df[df["num_hsn_classes"] == 0]
        n_pos = min(len(with_birds), max_samples // 2)
        n_neg = min(len(without_birds), max_samples - n_pos)
        df_sub = pd.concat([
            with_birds.sample(n=n_pos, random_state=42),
            without_birds.sample(n=n_neg, random_state=42),
        ]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        df = df_sub
        logger.info(f"Subsampled soundscape dataset to {len(df)} records ({n_pos} positive, {n_neg} background)")

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_manifest_path, index=False)
    logger.info(f"Saved soundscape manifest ({len(df)} rows) -> {output_manifest_path}")
    return df


def precompute_spectrogram_features(
    manifest_df: pd.DataFrame,
    output_tensor_dir: Path,
    is_soundscape: bool,
    batch_size: int = 50,
) -> None:
    """Precompute and save normalized log-mel spectrogram tensors."""
    output_tensor_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = AudioPreprocessor()
    logger.info(f"Precomputing spectrogram features -> {output_tensor_dir} (is_soundscape={is_soundscape})...")

    records = manifest_df.to_dict("records")
    for r in tqdm(records, desc="Precomputing features"):
        audio_path = r["audio_path"]
        stem = Path(audio_path).stem
        save_path = output_tensor_dir / f"{stem}.pt"
        if save_path.exists():
            continue

        try:
            spec = preprocessor.process_file(
                filepath=audio_path,
                is_soundscape=is_soundscape,
                focal_mode="peak_energy",
            )
            # Save as float32 tensor
            torch.save(spec.clone().detach().cpu(), save_path)
        except Exception as e:
            logger.warning(f"Error processing {audio_path}: {e}")

    logger.info("Precomputing features complete!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess BirdSet HSN dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data/HSN"),
                        help="HSN data directory (default: data/HSN)")
    parser.add_argument("--focal-samples-per-class", type=int, default=50,
                        help="Max focal samples per species for evaluation (default: 50)")
    parser.add_argument("--soundscape-samples", type=int, default=1000,
                        help="Max soundscape samples for evaluation (default: 1000)")
    parser.add_argument("--cache-tensors", action="store_true",
                        help="Precompute and save spectrogram .pt files to scratch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    raw_dir = data_dir / "raw"
    preprocessed_dir = data_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    metadata_train = raw_dir / "metadata" / "HSN_metadata_train.parquet"
    metadata_test_5s = raw_dir / "metadata" / "HSN_metadata_test_5s.parquet"
    focal_audio_dir = raw_dir / "audio_focal"
    soundscape_audio_dir = raw_dir / "audio_soundscape"

    # 1. Build Manifests
    focal_manifest_path = preprocessed_dir / "focal_manifest.parquet"
    soundscape_manifest_path = preprocessed_dir / "soundscape_manifest.parquet"

    df_focal = build_focal_manifest(
        metadata_path=metadata_train,
        audio_dir=focal_audio_dir,
        output_manifest_path=focal_manifest_path,
        max_samples_per_class=args.focal_samples_per_class,
    )

    df_soundscape = build_soundscape_manifest(
        metadata_path=metadata_test_5s,
        audio_dir=soundscape_audio_dir,
        output_manifest_path=soundscape_manifest_path,
        max_samples=args.soundscape_samples,
    )

    logger.info(f"Preprocessed manifests ready: Focal={len(df_focal)}, Soundscape={len(df_soundscape)}")

    if args.cache_tensors:
        precompute_spectrogram_features(df_focal, preprocessed_dir / "focal_features", is_soundscape=False)
        precompute_spectrogram_features(df_soundscape, preprocessed_dir / "soundscape_features", is_soundscape=True)

    logger.info("Pre-processing script finished successfully.")


if __name__ == "__main__":
    main()
