#!/usr/bin/env python3
"""Export the prototype-transfer results into a self-contained dashboard data bundle.

By default the bundle covers every clip OUTSIDE the test split (--scope non-test), assembled from the pretrained model's
cached responses (see dashboard_data.non_test_payload); its statistics are saved as the prototype_transfer_nontest section of
pretrained-xcl. --scope all instead reads the full-manifest responses (responses_full.pt) and statistics
(prototype_transfer_full.json; extract_prototype_activations.py pretrained-xcl --tag full), which include the test sites.
It writes
  dashboard/data.json, data.js     per-species / per-prototype stats + findings
  dashboard/patches/<sp>_P<k>_<domain>_<rank>.png   top-activating spectrogram crops
and, with --scope all and if present, embeds the model_comparison summary (test-split comparison of pretrained XCL,
focal-only and soundscape-only models) as `model_comparison`, plus per-prototype test-split stats for each
model as `model_prototypes` (from the run's prototype_transfer.json; confusion stats need the run's responses.pt). `--comparison-only` refreshes just those blocks in an existing
dashboard/data.json, without the main features file or audio.

Patch export recomputes spectrograms from audio, so run it where data/ lives. With --scope non-test it also runs the model
(CPU is enough) on the few focal-train clips that are cropped, to find where each prototype fired.
The computation is in prototype_transfer.reporting.dashboard_data; this script is the command line.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from prototype_transfer.analysis.prototype_responses import prototype_transfer_stats
from prototype_transfer.evaluation import results
from prototype_transfer.models.classifier import HSNProtoPNetClassifier
from prototype_transfer.reporting import dashboard_data as dd


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=results.DEFAULT_DATASET)
    p.add_argument("--scope", choices=["non-test", "all"], default="non-test")
    p.add_argument("--splits-dir", type=Path, default=Path("data/HSN/splits"))
    p.add_argument("--min-soundscape-pos", type=int, default=10, help="--scope non-test: soundscape positives a species needs")
    p.add_argument("--features-path", type=Path, default=None, help="default: the pretrained model's responses_full.pt")
    p.add_argument("--results-path", type=Path, default=None, help="default: the pretrained model's prototype_transfer_full.json")
    p.add_argument("--out-dir", type=Path, default=Path("dashboard"))
    p.add_argument("--top", type=int, default=3, help="Patches per prototype and domain")
    p.add_argument("--skip-patches", action="store_true")
    p.add_argument("--comparison-path", type=Path, default=None, help="default: the model_comparison summary")
    p.add_argument("--comparison-only", action="store_true",
                   help="Only (re)embed model_comparison and model_prototypes into the existing <out-dir>/data.json")
    p.add_argument("--models", nargs="+", default=list(dd.MODEL_LABELS), help="Models for model_prototypes")
    args = p.parse_args()
    store = results.Store(args.dataset)
    args.features_path = args.features_path or store.features("pretrained-xcl") / "responses_full.pt"
    args.results_path = args.results_path or store.run_dir("pretrained-xcl") / "prototype_transfer_full.json"
    args.comparison_path = args.comparison_path or store.summary("model_comparison")

    if args.comparison_only:
        out = json.loads((args.out_dir / "data.json").read_text())
        out["model_comparison"] = dd.model_comparison_block(args.comparison_path)
        if out["model_comparison"] is None:
            raise SystemExit(f"{args.comparison_path} not found")
        out["model_prototypes"] = dd.model_prototypes_block(args.models, store)
        dd.write_bundle(out, args.out_dir)
        logging.info(f"Embedded {args.comparison_path} into {args.out_dir / 'data.json'}")
        return

    patch_dir = None if args.skip_patches else args.out_dir / "patches"
    if args.scope == "all":
        payload = torch.load(args.features_path, map_location="cpu", weights_only=False)
        stats = json.loads(args.results_path.read_text())
        # the visual findings describe the non-test crops, so they are not carried over to this scope
        out = dd.build_bundle(payload, stats, store, args.comparison_path, args.models, patch_dir, args.top, curated=[])
        out["meta"]["scope"] = "all"
    else:
        payload = dd.non_test_payload(store, args.splits_dir)
        sp_list = payload["species_list"]
        stats = prototype_transfer_stats(payload["focal"], payload["soundscape"], payload["last_layer_weights"], sp_list,
                                         min_soundscape_pos=args.min_soundscape_pos)
        stats["n_clips"] = {d: len(payload[d]["targets"]) for d in dd.DOMAINS}
        results.save_results(store, "pretrained-xcl", "prototype_transfer_nontest",
                             {"stats_file": "prototype_transfer_nontest.json", "n_clips": stats["n_clips"],
                              "min_soundscape_pos": stats["min_soundscape_pos"], "scope": "train and val splits of both domains"},
                             kind="reference", files={"prototype_transfer_nontest.json": stats})
        if patch_dir is not None:
            cols = [sp_list.index(s) for s, v in stats["species"].items() if v["analysed"]]
            model = HSNProtoPNetClassifier(device="cpu")
            n = dd.fill_peak_time(payload["focal"], cols, args.splits_dir, args.top, model)
            logging.info(f"found peak positions on {n} focal-train clips")
        # the per-model blocks are test-split results, so they are left out of the non-test bundle
        out = dd.build_bundle(payload, stats, store, None, [], patch_dir, args.top, curated=dd.CURATED_FINDINGS)
        out["meta"]["scope"] = "non-test"
    (args.out_dir / "patches").mkdir(parents=True, exist_ok=True)
    dd.write_bundle(out, args.out_dir)
    logging.info(f"Wrote {args.out_dir / 'data.json'} ({len(out['species'])} species)")


if __name__ == "__main__":
    main()
