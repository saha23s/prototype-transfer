#!/usr/bin/env python3
"""Canonical evaluation (protocol v1): the ONE way every model is scored on the held-out test splits.

Input per model: the response file written by scripts/extract_prototype_activations.py (keys focal / soundscape) or by
scripts/diagnose_shift.py extract (keys focal_test / soundscape_test), run on data/HSN/splits/{focal,soundscape}_test.parquet.
All models must be evaluated on identical clips (checked).

Protocol (fixed; see dashboard/plan.html):
  species      secondary = species with >= 5 positives in BOTH test sets (12); primary = >= 30 soundscape test positives (7)
  headline     soundscape macro AP on the primary species
  secondary    soundscape macro AUC (primary, 12), cmAP over the 12, focal macro AUC / AP (retention only)
  prototype    delta_k = AUC_soundscape - AUC_focal of prototype k's max activation as a detector of its species:
                 delta       soundscape negatives = every other soundscape clip (as in model_comparison.json)
                 delta_birds soundscape negatives = clips with another target species only (matches focal negatives,
                             which are other species' clips); removes empty-background clips from the negative set
  uncertainty  cluster bootstrap, B resamples, shared across models: soundscape by site (30), focal by recordist (110)
  comparison   paired differences and the plan's success criteria against --baseline (and --oracle)
Soundscape labels are used only here, for evaluation. Writes outputs/results/canonical_eval.json and prints a table.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import ap_cols, auc_cols, cluster_indices
from src.splits import site_of

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("canonical_eval")

PROTOCOL = "v1"
K = 5                    # prototypes per species
MIN_POS, PRIMARY_MIN_POS = 5, 30
FAIL, IMPROVE = -0.10, 0.02
GAP_FRACTION, FOCAL_TOL = 0.5, 0.01
SCALARS = ("sc_ap_primary", "sc_auc_primary", "sc_cmap_12", "sc_auc_12", "f_auc_primary", "f_ap_primary", "f_auc_12",
           "n_fail", "n_improve", "median_delta", "n_fail_birds", "median_delta_birds")


# ----------------------------------------------------------------------------- metrics
class Model:
    """Responses of one model on both test splits."""

    def __init__(self, name: str, path: Path) -> None:
        z = torch.load(path, map_location="cpu", weights_only=False)
        f, s = z.get("focal_test", z.get("focal")), z.get("soundscape_test", z.get("soundscape"))
        assert f is not None and s is not None, f"{path}: no focal / soundscape responses"
        self.name, self.species = name, list(z.get("species", z.get("species_list")))
        self.y_f, self.p_f, self.a_f = f["targets"].astype(bool), f["probs"], f["proto_max"]
        self.y_s, self.p_s, self.a_s = s["targets"].astype(bool), s["probs"], s["proto_max"]
        self.paths = {"focal": [Path(p).name for p in f["audio_path"]], "soundscape": [Path(p).name for p in s["audio_path"]]}


def metrics(m: Model, fi: np.ndarray, si: np.ndarray, cols: list[int], pri: list[int]) -> dict:
    """All scalar metrics of one model on the clip subsets fi (focal) and si (soundscape); `pri` indexes into `cols`."""
    yf, pf, af, ys, ps, as_ = m.y_f[fi], m.p_f[fi], m.a_f[fi], m.y_s[si], m.p_s[si], m.a_s[si]
    ap_s, auc_s = ap_cols(ps[:, cols], ys[:, cols]), auc_cols(ps[:, cols], ys[:, cols])
    ap_f, auc_f = ap_cols(pf[:, cols], yf[:, cols]), auc_cols(pf[:, cols], yf[:, cols])
    pa_f = auc_cols(af[:, cols, :].reshape(len(fi), -1), np.repeat(yf[:, cols], K, axis=1))   # [12 * 5]
    pa_s = auc_cols(as_[:, cols, :].reshape(len(si), -1), np.repeat(ys[:, cols], K, axis=1))
    delta = pa_s - pa_f
    n_other = ys.sum(1)
    pa_b = np.concatenate([  # soundscape negatives restricted to clips that contain another target species
        auc_cols(as_[(ys[:, c] | (n_other - ys[:, c] > 0))][:, c, :], np.repeat(ys[(ys[:, c] | (n_other - ys[:, c] > 0)), c][:, None], K, axis=1))
        for c in cols])
    delta_b = pa_b - pa_f
    out = {"sc_ap_primary": np.nanmean(ap_s[pri]), "sc_auc_primary": np.nanmean(auc_s[pri]), "sc_cmap_12": np.nanmean(ap_s),
           "sc_auc_12": np.nanmean(auc_s), "f_auc_primary": np.nanmean(auc_f[pri]), "f_ap_primary": np.nanmean(ap_f[pri]),
           "f_auc_12": np.nanmean(auc_f), "n_fail": int((delta < FAIL).sum()), "n_improve": int((delta > IMPROVE).sum()),
           "median_delta": float(np.nanmedian(delta)), "n_fail_birds": int((delta_b < FAIL).sum()),
           "median_delta_birds": float(np.nanmedian(delta_b))}
    return {k: float(v) for k, v in out.items()} | {"_delta": delta, "_delta_birds": delta_b, "_auc_f": pa_f, "_auc_s": pa_s, "_auc_s_birds": pa_b}


def ci(x: np.ndarray) -> list[float]:
    return [float(v) for v in np.nanpercentile(x, [2.5, 97.5])]


# ----------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs=2, action="append", metavar=("NAME", "RESPONSES_PT"), required=True)
    p.add_argument("--baseline", default="pretrained-xcl", help="model the success criteria are measured against")
    p.add_argument("--oracle", default="soundscape-only", help="supervised in-domain reference for the gap fraction (optional)")
    p.add_argument("--focal-manifest", type=Path, default=Path("data/HSN/splits/focal_test.parquet"), help="gives the recordist of each focal clip")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("outputs/results/canonical_eval.json"))
    args = p.parse_args()

    models = [Model(n, Path(f)) for n, f in args.model]
    ref = models[0]
    for m in models[1:]:  # identical species, labels and clips for every model
        assert m.species == ref.species and m.paths == ref.paths, f"{m.name} was evaluated on different clips than {ref.name}"
        assert (m.y_f == ref.y_f).all() and (m.y_s == ref.y_s).all(), f"{m.name}: targets differ"
    names = [m.name for m in models]
    assert args.baseline in names, f"--baseline {args.baseline} is not among the models {names}"
    oracle = args.oracle if args.oracle in names else None

    sp = ref.species
    npos_f, npos_s = ref.y_f.sum(0), ref.y_s.sum(0)
    cols = [c for c in range(len(sp)) if npos_f[c] >= MIN_POS and npos_s[c] >= MIN_POS]
    pri = [i for i, c in enumerate(cols) if npos_s[c] >= PRIMARY_MIN_POS]
    logger.info(f"{len(cols)} secondary species, {len(pri)} primary: {[sp[cols[i]] for i in pri]}")

    sites = site_of(pd.Series(ref.paths["soundscape"])).to_numpy()
    man = pd.read_parquet(args.focal_manifest)
    rec = dict(zip(man["audio_path"].map(lambda q: Path(q).name), man["recordist"].fillna("unknown")))
    recordists = np.array([rec[q] for q in ref.paths["focal"]])
    rng = np.random.default_rng(args.seed)
    boot_f, boot_s = cluster_indices(recordists, rng, args.n_boot), cluster_indices(sites, rng, args.n_boot)
    logger.info(f"{len(np.unique(sites))} test sites, {len(np.unique(recordists))} focal recordists, {args.n_boot} shared resamples")

    allf, alls = np.arange(len(ref.y_f)), np.arange(len(ref.y_s))
    point = {m.name: metrics(m, allf, alls, cols, pri) for m in models}
    boot = {m.name: {k: np.empty(args.n_boot) for k in SCALARS} for m in models}
    for b in range(args.n_boot):
        for m in models:
            r = metrics(m, boot_f[b], boot_s[b], cols, pri)
            for k in SCALARS:
                boot[m.name][k][b] = r[k]
        if (b + 1) % 100 == 0:
            logger.info(f"bootstrap {b + 1}/{args.n_boot}")

    out = {"protocol": PROTOCOL, "seed": args.seed, "n_boot": args.n_boot, "baseline": args.baseline, "oracle": oracle,
           "species": {"secondary": [sp[c] for c in cols], "primary": [sp[cols[i]] for i in pri],
                       "n_pos_soundscape": {sp[c]: int(npos_s[c]) for c in cols}, "n_pos_focal": {sp[c]: int(npos_f[c]) for c in cols}},
           "n_clips": {"focal": len(allf), "soundscape": len(alls)}, "n_clusters": {"recordists": int(len(np.unique(recordists))), "sites": int(len(np.unique(sites)))},
           "thresholds": {"fail": FAIL, "improve": IMPROVE, "gap_fraction": GAP_FRACTION, "focal_tolerance": FOCAL_TOL}, "models": {}}
    base_b, base_p = boot[args.baseline], point[args.baseline]
    for m in models:
        n = m.name
        e = {"point": {k: (int(point[n][k]) if k.startswith("n_") else point[n][k]) for k in SCALARS}, "ci95": {k: ci(boot[n][k]) for k in SCALARS},
             "prototypes": [{"species": sp[cols[j // K]], "k": j % K + 1, "auc_focal": float(point[n]["_auc_f"][j]),
                             "auc_soundscape": float(point[n]["_auc_s"][j]), "delta": float(point[n]["_delta"][j]),
                             "delta_birds": float(point[n]["_delta_birds"][j])} for j in range(len(cols) * K)]}
        if n != args.baseline:
            e["vs_baseline"] = {k: {"diff": point[n][k] - base_p[k], "ci95": ci(boot[n][k] - base_b[k])}
                                for k in ("sc_ap_primary", "sc_auc_primary", "sc_cmap_12", "f_auc_primary", "n_fail", "n_fail_birds")}
            if oracle and n != oracle:
                gap_pt = (point[n]["sc_ap_primary"] - base_p["sc_ap_primary"]) / (point[oracle]["sc_ap_primary"] - base_p["sc_ap_primary"])
                gap_b = (boot[n]["sc_ap_primary"] - base_b["sc_ap_primary"]) / (boot[oracle]["sc_ap_primary"] - base_b["sc_ap_primary"])
                e["gap_fraction"] = {"value": float(gap_pt), "ci95": ci(gap_b)}
                d = e["vs_baseline"]
                e["criteria"] = {
                    "1_ap_gap_closed": bool(gap_pt >= GAP_FRACTION and d["sc_ap_primary"]["ci95"][0] > 0),
                    "2_auc_not_below_baseline": bool(d["sc_auc_primary"]["diff"] >= 0),
                    "3_failing_prototypes_halved": bool(point[n]["n_fail"] <= base_p["n_fail"] // 2),
                    "4_focal_retained": bool(point[n]["f_auc_primary"] >= base_p["f_auc_primary"] - FOCAL_TOL)}
        out["models"][n] = e
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    fmt = lambda n, k, d=3: f"{point[n][k]:.{d}f} [{ci(boot[n][k])[0]:.{d}f},{ci(boot[n][k])[1]:.{d}f}]"
    print(f"\nCanonical evaluation {PROTOCOL}: primary = {out['species']['primary']}; {len(cols)} secondary; {args.n_boot} site/recordist-cluster resamples")
    print(f"{'model':18} {'sc AP primary':>22} {'sc AUC primary':>22} {'cmAP-12':>22} {'focal AUC prim':>22}  fail  fail(birds)")
    for m in models:
        n = m.name
        print(f"{n:18} {fmt(n, 'sc_ap_primary'):>22} {fmt(n, 'sc_auc_primary'):>22} {fmt(n, 'sc_cmap_12'):>22} {fmt(n, 'f_auc_primary'):>22}  "
              f"{int(point[n]['n_fail']):4d}  {int(point[n]['n_fail_birds']):4d}")
    for n, e in out["models"].items():
        if "criteria" in e:
            print(f"{n}: gap closed {e['gap_fraction']['value']:.2f} {e['gap_fraction']['ci95']}; criteria {e['criteria']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
