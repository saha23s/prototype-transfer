#!/usr/bin/env python3
"""Check that data/HSN/splits holds the committed splits (fingerprints in configs/splits/hsn.json), or pin new ones.

  python scripts/data/check_splits.py           # exit 1 and list the differing files if the splits are not the committed ones
  python scripts/data/check_splits.py --write   # add fingerprints for split files not pinned yet (existing ones must match)
  python scripts/data/check_splits.py --write --force   # re-pin everything: only after a deliberate change of the splits

The fingerprint of a split is the sha256 of its sorted (file name, labels) lines plus the clip count, so it is the same on any
machine and for any pandas / pyarrow version. Training and prototype extraction run the same check on the manifests they read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from prototype_transfer.data import splits

NAMES = [f"{d}_{p}.parquet" for d in ("focal", "soundscape") for p in ("train", "val", "test")] + \
        [f"soundscape_p{k}_{p}.parquet" for k in range(splits.N_PARTITIONS) for p in ("adapt", "test")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits-dir", type=Path, default=Path("data/HSN/splits"))
    p.add_argument("--write", action="store_true", help="pin the fingerprints of split files that are not pinned yet")
    p.add_argument("--force", action="store_true", help="with --write: also replace fingerprints that differ")
    args = p.parse_args()
    lock = json.loads(splits.SPLITS_LOCK.read_text()) if splits.SPLITS_LOCK.exists() else {"dataset": "hsn", "seed": 0, "files": {}}
    want = lock["files"]
    present = [n for n in NAMES if (args.splits_dir / n).exists()]
    got = {n: splits.fingerprint(pd.read_parquet(args.splits_dir / n)) for n in present}
    bad = [n for n in present if n in want and got[n] != want[n]]
    missing = [n for n in NAMES if n not in present]
    for n in NAMES:
        state = "missing" if n in missing else "DIFF" if n in bad else "ok" if n in want else "new"
        size = f"{got[n]['n_clips']:6d} clips  {got[n]['sha256'][:12]}" if n in got else ""
        print(f"{state:8} {n:34} {size}")
    if args.write:
        if bad and not args.force:
            sys.exit(f"{len(bad)} split(s) differ from the pinned ones: {bad}. Use --force only after a deliberate change of the splits.")
        lock["files"] = {**want, **got} if args.force else {**got, **want}
        splits.SPLITS_LOCK.parent.mkdir(parents=True, exist_ok=True)
        splits.SPLITS_LOCK.write_text(json.dumps(lock, indent=1) + "\n")
        print(f"wrote {splits.SPLITS_LOCK} ({len(lock['files'])} splits)")
        return
    if bad or missing:
        sys.exit(f"{len(bad)} split(s) differ, {len(missing)} missing: {bad + missing}. Regenerate with scripts/data/make_splits.py (seed 0).")
    print("splits are the committed ones" + ("" if all(n in want for n in present) else " (some not pinned yet: run with --write)"))


if __name__ == "__main__":
    main()
