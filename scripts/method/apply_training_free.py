#!/usr/bin/env python3
"""Apply a training-free baseline to the pretrained model's cached responses of one cross-validation fold (CPU only).

Fitted for fold F on the unlabeled soundscape activations of the OTHER folds' held-out sites (together exactly
soundscape_cv{F}_train, checked) and on the focal-train activations; no labels are used. Applied to fold F's cached
focal_val and held-out soundscape activations. Writes outputs/features/<dataset>/<run_id>/responses.pt with run id
tf-<method>-cv<F>, and its settings to the results store. Score it with scripts/eval/canonical_eval.py:
  python scripts/eval/canonical_eval.py --split cv --fold F --baseline pretrained-xcl --summary-name canonical_eval_tf-quantile-cvF \
      --model pretrained-xcl outputs/features/hsn/pretrained-xcl/responses_cvF.pt --model tf-quantile-cvF
The methods are in prototype_transfer.method.training_free.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from prototype_transfer.data.splits import split_manifest
from prototype_transfer.evaluation import results
from prototype_transfer.method import training_free as tf

logger = logging.getLogger("apply_training_free")
N_FOLDS = 3
PROB_TOL = 1e-5   # adaptation off must reproduce the cached pretrained probabilities
KEEP = ("targets", "audio_path")   # copied through for scoring only; nothing fitted here sees them


def names(paths: list[str]) -> set[str]:
    return {Path(p).name for p in paths}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["quantile", "reweight"], required=True)
    p.add_argument("--fold", type=int, choices=range(N_FOLDS), required=True)
    p.add_argument("--dataset", default=results.DEFAULT_DATASET)
    p.add_argument("--splits-dir", type=Path, default=Path("data/HSN/splits"))
    p.add_argument("--n-quantiles", type=int, default=1001, help="quantile: number of knots per prototype")
    p.add_argument("--top-fraction", type=float, default=0.05, help="reweight: fraction of strongest activations in the contrast")
    args = p.parse_args()

    store = results.Store(args.dataset)
    base = store.features("pretrained-xcl")
    load = lambda path: torch.load(path, map_location="cpu", weights_only=False)
    diag = load(base / "diag_responses.pt")   # focal_train activations and the last-layer bias
    folds = {f: load(base / f"responses_cv{f}.pt") for f in range(N_FOLDS)}
    own = folds[args.fold]
    weights, bias = np.asarray(own["last_layer_weights"]), np.asarray(diag["bias"])
    assert own["species_list"] == diag["species"] and np.allclose(weights, diag["last_layer_weights"]), "different models"

    # target pool: activations of the other folds' held-out clips; only proto_max and audio_path are read here
    others = [f for f in range(N_FOLDS) if f != args.fold]
    target = np.concatenate([folds[f]["soundscape"]["proto_max"] for f in others])
    pool = [q for f in others for q in folds[f]["soundscape"]["audio_path"]]
    train = pd.read_parquet(split_manifest(args.splits_dir, "soundscape", "train", args.fold))
    assert len(pool) == len(train) and names(pool) == names(train["audio_path"]), "target pool is not the fold's training sites"
    assert not names(pool) & names(own["soundscape"]["audio_path"]), "held-out clips in the target pool"
    reference = diag["focal_train"]["proto_max"]
    assert names(diag["focal_train"]["audio_path"]) == names(pd.read_parquet(split_manifest(args.splits_dir, "focal", "train"))["audio_path"])

    for domain in ("focal", "soundscape"):
        err = np.abs(tf.class_probs(own[domain]["proto_max"], weights, bias) - own[domain]["probs"]).max()
        assert err < PROB_TOL, f"{domain}: cannot rebuild the pretrained probabilities (max error {err:.2e})"

    new_weights = weights
    if args.method == "quantile":
        knots = tf.fit_quantile_map(target, reference, args.n_quantiles)
        adapt = lambda a: tf.apply_quantile_map(a, *knots)
        settings = {"n_quantiles": args.n_quantiles, "zeros": "kept at zero"}
    else:
        new_weights = tf.reweight(weights, target, reference, args.top_fraction)
        adapt = lambda a: a
        settings = {"top_fraction": args.top_fraction}

    payload = {"species_list": own["species_list"], "last_layer_weights": new_weights}
    for domain in ("focal", "soundscape"):
        acts = adapt(own[domain]["proto_max"])
        payload[domain] = {**{k: own[domain][k] for k in KEEP}, "proto_max": acts,
                           "probs": tf.class_probs(acts, new_weights, bias).astype(own[domain]["probs"].dtype)}
        assert np.isfinite(payload[domain]["probs"]).all() and np.isfinite(acts).all(), f"{domain}: non-finite output"

    run_id = f"tf-{args.method}-cv{args.fold}"
    path = store.features(run_id) / "responses.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    prov = results.provenance()
    config = {"args": vars(args), "settings": settings, "target_clips": len(pool), "reference_clips": len(reference),
              "uses_labels": False, **prov}
    results.save_results(store, run_id, "adaptation", {"method": args.method, **settings, "target_clips": len(pool),
                                                       "reference_clips": len(reference)},
                         kind="reference", config=config,
                         entry={"rung": f"tf-{args.method}", "cv_fold": args.fold, "commit": prov["commit"], "dirty": prov["dirty"]})
    logger.info(f"{run_id}: fitted on {len(pool)} unlabeled soundscape and {len(reference)} focal clips -> {path}")


if __name__ == "__main__":
    main()
