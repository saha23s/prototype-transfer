#!/usr/bin/env python3
"""Compare models (pretrained XCL, focal-only, soundscape-only) on the held-out TEST splits.

Inputs: per-model responses (outputs/features/<dataset>/<run_id>/responses.pt) from extract_prototype_activations.py run on
data/HSN/splits/{focal,soundscape}_test.parquet. Species analysed: >= --min-pos positives in BOTH
test sets. Saves summaries/model_comparison.json in the results store and prints a summary table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from prototype_transfer.evaluation import results


DOMAINS = ("focal", "soundscape")


def domain_metrics(d: dict, cols: list[int]) -> dict:
    y, p, a = d["targets"][:, cols], d["probs"][:, cols], d["proto_max"][:, cols, :]
    ap = [average_precision_score(y[:, i], p[:, i]) for i in range(len(cols))]
    auc = [roc_auc_score(y[:, i], p[:, i]) for i in range(len(cols))]
    pauc = [roc_auc_score(y[:, i], a[:, i, k]) for i in range(len(cols)) for k in range(a.shape[2])]
    pos = np.concatenate([a[y[:, i], i, :].ravel() for i in range(len(cols))])
    neg = np.concatenate([a[~y[:, i], i, :].ravel() for i in range(len(cols))])
    return {"cmAP": float(np.mean(ap)), "macro_auc": float(np.mean(auc)),
            "per_species_ap": [float(v) for v in ap], "per_species_auc": [float(v) for v in auc],
            "proto_auc_median": float(np.median(pauc)), "proto_auc_frac_gt_0.8": float(np.mean(np.array(pauc) > 0.8)),
            "proto_auc": [float(v) for v in pauc],
            "act_pos_median": float(np.median(pos)), "act_neg_median": float(np.median(neg))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=results.DEFAULT_DATASET)
    p.add_argument("--model", nargs="+", action="append", metavar=("RUN_ID", "TRAIN_DOMAIN"), required=True,
                   help="RUN_ID TRAIN_DOMAIN [FEATURES_PT], e.g. --model focal-only focal (TRAIN_DOMAIN: focal|soundscape|xcl); "
                        "the responses default to outputs/features/<dataset>/<run_id>/responses.pt")
    p.add_argument("--min-pos", type=int, default=5)
    args = p.parse_args()
    if any(len(m) not in (2, 3) for m in args.model):
        p.error("--model takes RUN_ID TRAIN_DOMAIN [FEATURES_PT]")
    store = results.Store(args.dataset)
    payloads = {m[0]: (m[1], torch.load(m[2] if len(m) == 3 else store.features(m[0]) / "responses.pt", map_location="cpu", weights_only=False))
                for m in args.model}
    first = next(iter(payloads.values()))[1]
    sp_list = first["species_list"]
    n_pos = {d: first[d]["targets"].sum(axis=0) for d in DOMAINS}
    cols = [i for i in range(len(sp_list)) if all(n_pos[d][i] >= args.min_pos for d in DOMAINS)]
    out = {"species": [sp_list[i] for i in cols], "n_pos": {d: {sp_list[i]: int(n_pos[d][i]) for i in cols} for d in DOMAINS},
           "min_pos": args.min_pos, "models": {}}
    for name, (train_dom, pl) in payloads.items():
        for d in DOMAINS:  # same test clips for every model
            assert (pl[d]["targets"] == first[d]["targets"]).all(), f"{name}: different {d} test set"
        m = {d: domain_metrics(pl[d], cols) for d in DOMAINS}
        entry = {"train_domain": train_dom, **m}
        if train_dom in DOMAINS:
            other = "soundscape" if train_dom == "focal" else "focal"
            entry["in_domain"], entry["cross_domain"] = train_dom, other
            entry["gap_cmAP"] = m[train_dom]["cmAP"] - m[other]["cmAP"]
            entry["gap_proto_auc_median"] = m[train_dom]["proto_auc_median"] - m[other]["proto_auc_median"]
        out["models"][name] = entry
    saved = results.save_summary(store, "model_comparison", out, headline={"n_species": len(cols)})

    print(f"Species ({len(cols)}): {', '.join(out['species'])}")
    print(f"{'model':16} {'train':10} | {'focal cmAP':>10} {'sc cmAP':>8} | {'focal AUC':>9} {'sc AUC':>7} | "
          f"{'f protoAUC':>10} {'s protoAUC':>10}")
    for n, e in out["models"].items():
        print(f"{n:16} {e['train_domain']:10} | {e['focal']['cmAP']:10.3f} {e['soundscape']['cmAP']:8.3f} | "
              f"{e['focal']['macro_auc']:9.3f} {e['soundscape']['macro_auc']:7.3f} | "
              f"{e['focal']['proto_auc_median']:10.3f} {e['soundscape']['proto_auc_median']:10.3f}")
    print(f"Saved {saved['paths']['summary']}")


if __name__ == "__main__":
    main()
