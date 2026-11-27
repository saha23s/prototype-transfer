#!/usr/bin/env python3
"""Check that data/HSN/splits holds the canonical splits (fingerprints in configs/splits/hsn.json), or record them.

  python scripts/data/check_splits.py           # exit 1 and list the differing files if the splits are not canonical
  python scripts/data/check_splits.py --write   # (re)write the fingerprints: only after a deliberate change of the splits,
                                                #  which also means a new protocol version

The fingerprint of a split is the sha256 of its sorted (file name, labels) lines plus the clip count, so it is the same on any
machine and for any pandas / pyarrow version. Training (scripts/train/train_protopnet.py) and scoring
(scripts/eval/canonical_eval.py) run the same check on the manifests they read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from prototype_transfer.data import splits

NAMES = [f"{d}_{p}.parquet" for d in ("focal", "soundscape") for p in ("train", "val", "test")] + \
        [f"soundscape_cv{f}_{p}.parquet" for f in range(3) for p in ("train", "heldout")]   # keyed by file name, as verify_split looks them up


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits-dir", type=Path, default=Path("data/HSN/splits"))
    p.add_argument("--write", action="store_true", help="record the fingerprints of the current splits")
    args = p.parse_args()
    got = {n: splits.fingerprint(pd.read_parquet(args.splits_dir / n)) for n in NAMES}
    if args.write:
        splits.SPLITS_LOCK.parent.mkdir(parents=True, exist_ok=True)
        splits.SPLITS_LOCK.write_text(json.dumps({"dataset": "hsn", "seed": 0, "files": got}, indent=1) + "\n")
        print(f"wrote {splits.SPLITS_LOCK} ({len(got)} splits)")
        return
    want = json.loads(splits.SPLITS_LOCK.read_text())["files"]
    bad = [n for n in NAMES if got[n] != want.get(n)]
    for n in NAMES:
        print(f"{'ok  ' if n not in bad else 'DIFF'} {n:34} {got[n]['n_clips']:6d} clips  {got[n]['sha256'][:12]}")
    if bad:
        sys.exit(f"{len(bad)} split(s) differ from the canonical ones: {bad}. Regenerate with scripts/data/make_splits.py (seed 0).")
    print("splits are canonical")


if __name__ == "__main__":
    main()
