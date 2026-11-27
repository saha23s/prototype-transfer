#!/usr/bin/env python3
"""Write leakage-aware train/val/test manifests for one-domain ProtoPNet training.

Outputs data/HSN/splits/{focal,soundscape}_{train,val,test}.parquet and a summary
outputs/results/<dataset>/dataset/splits.json (per-species positives per split, sites, recordists).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd


from prototype_transfer.models.classifier import HSN_SPECIES_LIST
from prototype_transfer.data.splits import focal_split, soundscape_split
from prototype_transfer.evaluation import results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("make_splits")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    pre = Path("data/HSN/preprocessed")
    p.add_argument("--focal-manifest", type=Path, default=pre / "focal_manifest_full.parquet")
    p.add_argument("--soundscape-manifest", type=Path, default=pre / "soundscape_manifest_full.parquet")
    p.add_argument("--out-dir", type=Path, default=Path("data/HSN/splits"))
    p.add_argument("--dataset", default=results.DEFAULT_DATASET, help="the split summary goes to outputs/results/<dataset>/dataset/")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    store = results.Store(args.dataset)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    f = pd.read_parquet(args.focal_manifest)
    s = pd.read_parquet(args.soundscape_manifest)
    f["split"] = focal_split(f, seed=args.seed)
    s["split"], sc_info = soundscape_split(s, seed=args.seed)

    summary = {"seed": args.seed, "focal": {}, "soundscape": {"sites": sc_info["sites"],
               "eligible_species": sc_info["eligible_species"], "objective": sc_info["objective"]}}
    for name, df in (("focal", f), ("soundscape", s)):
        for split in ("train", "val", "test"):
            part = df[df["split"] == split].reset_index(drop=True)
            part.to_parquet(args.out_dir / f"{name}_{split}.parquet", index=False)
            if name == "focal":
                counts = part["ebird_code"].value_counts()
                extra = {"recordists": int(part["recordist"].nunique())}
            else:
                counts = part["hsn_classes"].explode().dropna().value_counts()
                extra = {"background_clips": int((part["num_hsn_classes"] == 0).sum())}
            summary[name][split] = {"n_clips": len(part), **extra,
                                    "positives": {sp: int(counts.get(sp, 0)) for sp in HSN_SPECIES_LIST}}
            logger.info(f"{name}/{split}: {len(part)} clips")
    # recordist leakage check
    rec = {sp: set(f.loc[f["split"] == sp, "recordist"]) for sp in ("train", "val", "test")}
    assert not (rec["train"] & rec["test"]) and not (rec["val"] & rec["test"]), "recordist leakage"
    results.atomic_write(store.dataset_dir / "splits.json", json.dumps(summary, indent=1))
    logger.info(f"Wrote {args.out_dir} and {store.dataset_dir} (soundscape objective {sc_info['objective']:.3f})")


if __name__ == "__main__":
    main()
