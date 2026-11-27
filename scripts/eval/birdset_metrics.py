#!/usr/bin/env python3
"""Score responses.pt (from scripts/analysis/extract_prototype_activations.py) with BirdSet's soundscape metrics and save them
to the results store as section `birdset_<tag>` of the run.

  python scripts/eval/birdset_metrics.py RUN_ID --tag full    # reads outputs/features/hsn/RUN_ID/responses_full.pt

Metrics (prototype_transfer.evaluation.metrics.birdset_scores) on the soundscape clips, 21 HSN species:
  cmAP     mean AP over species with >= 1 positive clip
  AUROC    mean ROC-AUC over species with positives and negatives
  T1       top-1 prediction is one of the clip's labels, over clips with >= 1 label (T1_labelled_clips) and over all clips
Reference (AudioProtoPNet-5 model card, mean of 5 seeds): HSN cmAP 0.54, AUROC 0.92, T1-Acc 0.65.
scripts/eval/reproduce_birdset.sh runs the whole check; docs/BIRDSET.md has the numbers.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from prototype_transfer.evaluation import results
from prototype_transfer.evaluation.metrics import birdset_scores


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("run_id")
    a.add_argument("--tag", default="", help="responses_<tag>.pt; also names the section birdset_<tag>")
    a.add_argument("--dataset", default=results.DEFAULT_DATASET)
    args = a.parse_args()
    store = results.Store(args.dataset)
    sfx = f"_{args.tag}" if args.tag else ""
    z = torch.load(store.features(args.run_id) / f"responses{sfx}.pt", map_location="cpu", weights_only=False)
    s = z["soundscape"]
    m = birdset_scores(np.asarray(s["targets"]), np.asarray(s["probs"], dtype=float))
    kind = results.read_registry(store).get(args.run_id, {}).get("kind", "reference")
    results.save_results(store, args.run_id, f"birdset{sfx}", m, kind=kind,
                         headline={k: m[k] for k in ("cmAP", "AUROC", "T1_labelled_clips")})
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
