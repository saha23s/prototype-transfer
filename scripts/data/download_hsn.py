#!/usr/bin/env python3
"""Download script for BirdSet HSN (High Sierra Nevada) 21 species dataset.

Downloads:
  1. Metadata parquet files (train, test_5s, test).
  2. Soundscape 5s audio archive (HSN_test5s_shard_0001.tar.gz).
  3. Focal audio archive shards (HSN_train_shard_0001.tar.gz ...).
Extracts archives to scratch storage and validates dataset structure.
"""

from __future__ import annotations

import argparse
import logging
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Sequence
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_hsn")

BASE_URL = "https://huggingface.co/datasets/DBD-research-group/BirdSet/resolve/data/HSN"

METADATA_FILES = [
    "HSN_metadata_train.parquet",
    "HSN_metadata_test_5s.parquet",
    "HSN_metadata_test.parquet",
]

SOUNDSCAPE_ARCHIVES = [
    "HSN_test5s_shard_0001.tar.gz",
]

FOCAL_SHARD_TEMPLATE = "HSN_train_shard_{shard:04d}.tar.gz"
TOTAL_FOCAL_SHARDS = 7


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest_path: Path, force: bool = False) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not force and dest_path.stat().st_size > 0:
        logger.info(f"File already exists (size: {dest_path.stat().st_size / 1e6:.2f} MB): {dest_path}")
        return dest_path

    logger.info(f"Downloading {url} -> {dest_path}")
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "BirdSet-Downloader/1.0"})
    
    with urllib.request.urlopen(req) as response:
        total_size = int(response.info().get("Content-Length", 0))
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest_path.name, total=total_size) as bar:
            with open(tmp_path, "wb") as f_out:
                while True:
                    buffer = response.read(1024 * 1024)
                    if not buffer:
                        break
                    f_out.write(buffer)
                    bar.update(len(buffer))

    tmp_path.rename(dest_path)
    logger.info(f"Successfully saved {dest_path} ({dest_path.stat().st_size / 1e6:.2f} MB)")
    return dest_path


def extract_archive(tar_path: Path, target_dir: Path, delete_archive: bool = False) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {tar_path} -> {target_dir}")
    extracted_count = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        logger.info(f"Archive contains {len(members)} files. Extracting...")
        for member in tqdm(members, desc=f"Extracting {tar_path.name}"):
            # Prevent path traversal vulnerabilities
            if member.name.startswith("/") or ".." in member.name:
                continue
            tar.extract(member, path=target_dir)
            extracted_count += 1
            
    logger.info(f"Extracted {extracted_count} files into {target_dir}")
    if delete_archive:
        logger.info(f"Removing archive {tar_path} to conserve disk space")
        tar_path.unlink()
    return extracted_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BirdSet HSN dataset")
    parser.add_argument("--base-dir", type=Path, default=Path("data/HSN"),
                        help="Base directory for HSN dataset (default: data/HSN)")
    parser.add_argument("--focal-shards", type=int, nargs="+", default=[1],
                        help="Train focal shard indices to download (default: 1; max 7)")
    parser.add_argument("--all-focal", action="store_true",
                        help="Download all 7 focal shards (total ~5.6 GB)")
    parser.add_argument("--skip-soundscape", action="store_true",
                        help="Skip downloading soundscape 5s clips")
    parser.add_argument("--delete-archives", action="store_true",
                        help="Delete tar.gz archives after extraction")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download existing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    logger.info(f"Target HSN base directory: {base_dir}")

    raw_dir = base_dir / "raw"
    metadata_dir = raw_dir / "metadata"
    soundscape_dir = raw_dir / "audio_soundscape"
    focal_dir = raw_dir / "audio_focal"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    soundscape_dir.mkdir(parents=True, exist_ok=True)
    focal_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download Metadata
    logger.info("=== Phase 1: Downloading Metadata Parquet Files ===")
    for meta_file in METADATA_FILES:
        url = f"{BASE_URL}/{meta_file}"
        dest = metadata_dir / meta_file
        download_file(url, dest, force=args.force)

    # 2. Download and Extract Soundscape
    if not args.skip_soundscape:
        logger.info("=== Phase 2: Downloading Soundscape (test_5s) Audio Archive ===")
        for sc_file in SOUNDSCAPE_ARCHIVES:
            url = f"{BASE_URL}/{sc_file}"
            dest = raw_dir / sc_file
            download_file(url, dest, force=args.force)
            extract_archive(dest, soundscape_dir, delete_archive=args.delete_archives)

    # 3. Download and Extract Focal Audio
    logger.info("=== Phase 3: Downloading Focal (train) Audio Archives ===")
    shards_to_download = list(range(1, TOTAL_FOCAL_SHARDS + 1)) if args.all_focal else args.focal_shards
    for shard_idx in shards_to_download:
        shard_name = FOCAL_SHARD_TEMPLATE.format(shard=shard_idx)
        url = f"{BASE_URL}/{shard_name}"
        dest = raw_dir / shard_name
        download_file(url, dest, force=args.force)
        extract_archive(dest, focal_dir, delete_archive=args.delete_archives)

    # 4. Summary & Verification
    logger.info("=== Phase 4: Dataset Verification & Summary ===")
    soundscape_files = list(soundscape_dir.glob("*.ogg"))
    focal_files = list(focal_dir.glob("*.ogg"))
    logger.info(f"Total Soundscape (.ogg) files extracted: {len(soundscape_files)}")
    logger.info(f"Total Focal (.ogg) files extracted: {len(focal_files)}")

    train_meta_path = metadata_dir / "HSN_metadata_train.parquet"
    test_5s_meta_path = metadata_dir / "HSN_metadata_test_5s.parquet"
    if train_meta_path.exists():
        df_train = pd.read_parquet(train_meta_path)
        logger.info(f"Focal metadata records: {len(df_train)} across {df_train['ebird_code'].nunique()} species")
    if test_5s_meta_path.exists():
        df_test = pd.read_parquet(test_5s_meta_path)
        logger.info(f"Soundscape 5s metadata records: {len(df_test)}")

    logger.info("Download and extraction complete!")


if __name__ == "__main__":
    main()
