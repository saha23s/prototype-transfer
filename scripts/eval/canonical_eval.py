#!/usr/bin/env python3
"""CLI for the canonical evaluation (protocol v1); the protocol and the computation live in
prototype_transfer.evaluation.canonical. Prints a table and saves to the results store: each model's numbers as the run's
canonical_<split> section, and the whole comparison as summaries/canonical_eval[_val|_cv<fold>].json."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prototype_transfer.data.splits import verify_split
from prototype_transfer.evaluation import canonical, results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=canonical.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=results.DEFAULT_DATASET)
    p.add_argument("--model", nargs="+", action="append", metavar=("NAME", "RESPONSES_PT"), required=True,
                   help="RUN_ID [RESPONSES_PT]; the responses default to outputs/features/<dataset>/<run_id>/responses.pt")
    p.add_argument("--baseline", default="pretrained-xcl", help="model the success criteria are measured against")
    p.add_argument("--oracle", default="soundscape-only", help="supervised in-domain reference for the gap fraction (optional)")
    p.add_argument("--split", choices=["test", "val", "cv"], default="test",
                   help="which held-out split the responses were extracted on. test: soundscape_test + focal_test (the reported numbers). "
                        "val: soundscape_val + focal_val, for development without touching test. "
                        "cv: held-out sites of --fold + focal_val (cross-validation, K=3 folds over the non-test sites)")
    p.add_argument("--fold", type=int, default=None, help="cross-validation fold (required with --split cv)")
    p.add_argument("--focal-manifest", type=Path, default=None, help="gives the recordist of each focal clip (default: focal_test, or focal_val for --split val / cv)")
    p.add_argument("--soundscape-manifest", type=Path, default=None, help="clips the soundscape responses must match (default: soundscape_<split>, or soundscape_cv<fold>_heldout)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary-name", default=None, help="name of the summary file (default: canonical_eval[_val|_cv<fold>])")
    args = p.parse_args()
    if (args.split == "cv") != (args.fold is not None):
        p.error("--fold goes with --split cv (and only with it)")
    if any(len(m) > 2 for m in args.model):
        p.error("--model takes RUN_ID and an optional RESPONSES_PT")
    store = results.Store(args.dataset)
    spl = Path("data/HSN/splits")
    focal_manifest = args.focal_manifest or spl / ("focal_test.parquet" if args.split == "test" else "focal_val.parquet")
    soundscape_manifest = args.soundscape_manifest or spl / (f"soundscape_cv{args.fold}_heldout.parquet" if args.split == "cv" else f"soundscape_{args.split}.parquet")

    for manifest in (focal_manifest, soundscape_manifest):   # scores on non-canonical splits are not comparable
        try:
            verify_split(manifest)
        except ValueError as e:
            p.error(str(e))
    models = [canonical.Model(m[0], Path(m[1]) if len(m) == 2 else store.features(m[0]) / "responses.pt") for m in args.model]
    out, point, boot = canonical.evaluate(models, args.baseline, args.oracle, args.split, args.fold,
                                          focal_manifest, soundscape_manifest, args.n_boot, args.seed)
    entry = canonical.save(store, out, args.summary_name)
    print(canonical.summary_table(out, point, boot))
    print(f"Saved {entry['paths']['summary']} and the canonical_{canonical.tag_of(args.split, args.fold)} section of {', '.join(out['models'])} "
          f"(python scripts/report/results.py list)")


if __name__ == "__main__":
    main()
