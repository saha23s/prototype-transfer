#!/usr/bin/env python3
"""CLI: combine the per-fold canonical evaluations of one cross-validated run (CPU only); see
prototype_transfer.evaluation.cv. Saves summaries/cv_NAME.json in the results store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prototype_transfer.evaluation import cv, results


def main() -> None:
    p = argparse.ArgumentParser(description=cv.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="run name without the -cv<fold> suffix")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--baseline", default="pretrained-xcl")
    p.add_argument("--dataset", default=results.DEFAULT_DATASET)
    args = p.parse_args()
    store = results.Store(args.dataset)
    paths = [store.summary(f"canonical_eval_{args.name}-cv{f}") for f in range(args.folds)]
    missing = [str(q) for q in paths if not q.exists()]
    if missing:
        sys.exit(f"missing per-fold evaluations: {missing}")
    res = cv.aggregate([json.loads(q.read_text()) for q in paths], args.name, args.baseline)
    headline = {f"{k}_{stat}": res["run"][k][stat] for k in ("sc_ap_primary", "sc_auc_primary", "f_auc_primary") for stat in ("mean", "sd")}
    entry = results.save_summary(store, f"cv_{args.name}", res, headline=headline)
    print(cv.summary_table(res))
    print(f"Saved {entry['paths']['summary']}")


if __name__ == "__main__":
    main()
