#!/usr/bin/env python3
"""CLI: paired intervals between adapted models in cross-validation (CPU only); see prototype_transfer.evaluation.contrasts.
Saves summaries/cv_contrasts.json in the results store and leaves every run's saved evaluation untouched.

  python scripts/eval/paired_contrasts.py                       # the single-ingredient contrasts shown on the dashboard
  python scripts/eval/paired_contrasts.py --pair r3 focal-only  # chosen pairs (VARIANT CONTROL), saved under --summary-name
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prototype_transfer.evaluation import contrasts, results
from prototype_transfer.reporting.runs import CONTRASTS


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", nargs=2, action="append", metavar=("VARIANT", "CONTROL"))
    p.add_argument("--dataset", default=results.DEFAULT_DATASET)
    p.add_argument("--splits-dir", type=Path, default=Path("data/HSN/splits"))
    p.add_argument("--summary-name", default="cv_contrasts")
    args = p.parse_args()
    if args.pair and args.summary_name == "cv_contrasts":
        p.error("--pair needs its own --summary-name (cv_contrasts holds the dashboard's contrasts)")
    pairs = [tuple(x) for x in args.pair] if args.pair else [(v, c) for v, c, _ in CONTRASTS]
    store = results.Store(args.dataset)
    out = contrasts.evaluate_contrasts(store, pairs, args.splits_dir)
    entry = results.save_summary(store, args.summary_name, out, headline={"n_pairs": len(pairs)},
                                 entry={"note": "exploratory paired intervals between adapted models"})
    for e in out["pairs"]:
        for s in contrasts.SPECIES_SETS:
            cell = lambda x: f"{x['diff']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]"
            print(f"{e['variant']:16} - {e['control']:11} {s:7}  mean {cell(e[s]['mean'])}  folds " + "  ".join(map(cell, e[s]["folds"])))
    print(f"Saved {entry['paths']['summary']}")


if __name__ == "__main__":
    main()
