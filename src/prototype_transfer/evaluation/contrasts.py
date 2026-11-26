"""Paired intervals for the soundscape-AP difference between two adapted models in cross-validation.

Protocol v1 gives intervals only against the pretrained model. These are EXPLORATORY additions, not part of the protocol:
same metric (macro AP) and same cluster bootstrap over held-out sites, applied to `variant - control`. Per fold, the
resamples are the ones canonical.evaluate draws (same seed and order), so each model's interval against the baseline must
reproduce the saved one exactly; that is checked, and a mismatch stops the run. The interval of the 3-fold mean uses
independent draws per fold (the folds hold disjoint sites).

The intervals describe which held-out sites were scored, for the checkpoints as trained: they do not include
training-seed variation. Nothing is written to the per-run sections of the results store.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from prototype_transfer.data.splits import site_of
from prototype_transfer.evaluation import canonical, results
from prototype_transfer.evaluation.metrics import ap_cols, cluster_indices

logger = logging.getLogger(__name__)

BASELINE = "pretrained-xcl"
TOL = 1e-9
# primary: each fold's primary species (the protocol's). shared: species primary on every fold. fixed: species eligible
# (protocol "secondary") on every fold.
SPECIES_SETS = ("primary", "shared", "fixed")


def macro_ap(m: canonical.Model, idx: np.ndarray, cols: list[int]) -> float:
    """Soundscape macro AP over the species columns `cols`, on the clips `idx`."""
    return float(np.nanmean(ap_cols(m.p_s[idx][:, cols], m.y_s[idx][:, cols])))


def boot_ap(m: canonical.Model, draws: list[np.ndarray], col_sets: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """{species set: [n_boot] macro AP}; per-species AP is computed once per draw and averaged per set."""
    cols = sorted({c for cs in col_sets.values() for c in cs})
    ap = np.stack([ap_cols(m.p_s[i][:, cols], m.y_s[i][:, cols]) for i in draws])   # [n_boot, len(cols)]
    return {name: np.nanmean(ap[:, [cols.index(c) for c in cs]], axis=1) for name, cs in col_sets.items()}


def evaluate_contrasts(store: results.Store, pairs: list[tuple[str, str]], splits_dir: Path, n_folds: int = 3) -> dict:
    """Intervals of AP(variant) - AP(control) for every (variant, control) pair of cross-validated run names.

    Args:
        store: results store holding the saved per-fold evaluations and the cached responses.
        pairs: (variant, control) run names without the -cv<fold> suffix.
        splits_dir: split manifests (for each focal clip's recordist and the number of focal clusters).
        n_folds: number of cross-validation folds.

    Returns:
        JSON-serialisable dict: for each pair and species set (SPECIES_SETS), per-fold and 3-fold-mean point differences
        with 95% percentile intervals.
    """
    saved = lambda key, f: json.loads(store.summary(f"canonical_eval_{key}-cv{f}").read_text())
    names = sorted({k for p in pairs for k in p})
    first = {f: saved(names[0], f) for f in range(n_folds)}
    seed, n_boot = first[0]["seed"], first[0]["n_boot"]
    primary = [first[f]["species"]["primary"] for f in range(n_folds)]
    shared = [s for s in primary[0] if all(s in p for p in primary)]
    secondary = [first[f]["species"]["secondary"] for f in range(n_folds)]
    fixed = [s for s in secondary[0] if all(s in x for x in secondary)]
    pooled_rng = np.random.default_rng(seed)

    point: dict = {}     # (name, fold, set) -> AP
    fold_boot: dict = {}  # (name, fold, set) -> [n_boot], protocol draws
    pool_boot: dict = {}  # (name, fold, set) -> [n_boot], independent draws per fold
    for f in range(n_folds):
        paths = {BASELINE: store.features(BASELINE) / f"responses_cv{f}.pt",
                 **{k: store.features(f"{k}-cv{f}") / "responses.pt" for k in names}}
        models = {k: canonical.Model(k, p) for k, p in paths.items()}
        ref = models[BASELINE]
        assert all(m.species == ref.species and m.paths == ref.paths and (m.y_s == ref.y_s).all() for m in models.values()), \
            f"fold {f}: models were scored on different clips"
        codes = {"primary": primary[f], "shared": shared, "fixed": fixed}
        col_sets = {name: [ref.species.index(s) for s in codes[name]] for name in SPECIES_SETS}

        man = pd.read_parquet(splits_dir / "focal_val.parquet")
        rec = dict(zip(man["audio_path"].map(lambda q: Path(q).name), man["recordist"].fillna("unknown")))
        rng = np.random.default_rng(seed)
        cluster_indices(np.array([rec[q] for q in ref.paths["focal"]]), rng, n_boot)   # consumed first by the protocol
        sites = site_of(pd.Series(ref.paths["soundscape"])).to_numpy()
        draws = cluster_indices(sites, rng, n_boot)
        pooled_draws = cluster_indices(sites, pooled_rng, n_boot)

        everything = np.arange(len(ref.y_s))
        for k, m in models.items():
            for s, cols in col_sets.items():
                point[k, f, s] = macro_ap(m, everything, cols)
            for s, v in boot_ap(m, draws, col_sets).items():
                fold_boot[k, f, s] = v
            for s, v in boot_ap(m, pooled_draws, col_sets).items():
                pool_boot[k, f, s] = v
            if k != BASELINE:   # the check that these are the protocol's resamples, on the protocol's numbers
                e = saved(k, f)["models"][f"{k}-cv{f}"]
                got = canonical.ci(fold_boot[k, f, "primary"] - fold_boot[BASELINE, f, "primary"])
                assert abs(point[k, f, "primary"] - e["point"]["sc_ap_primary"]) < TOL, f"{k} fold {f}: AP differs from saved"
                assert np.allclose(got, e["vs_baseline"]["sc_ap_primary"]["ci95"], rtol=0, atol=TOL), \
                    f"{k} fold {f}: interval vs baseline {got} does not reproduce the saved one"
            logger.info(f"fold {f}: {k} done")

    out: dict = {"metric": "soundscape macro AP", "split": "cv", "n_boot": n_boot, "seed": seed, "shared_species": shared,
                 "fixed_species": fixed,
                 "exploratory": True, "note": "not part of protocol v1; training-seed variation not included", "pairs": []}
    for variant, control in pairs:
        entry: dict = {"variant": variant, "control": control}
        for s in SPECIES_SETS:
            folds = [{"diff": point[variant, f, s] - point[control, f, s],
                      "ci95": canonical.ci(fold_boot[variant, f, s] - fold_boot[control, f, s])} for f in range(n_folds)]
            mean_draws = np.mean([pool_boot[variant, f, s] - pool_boot[control, f, s] for f in range(n_folds)], axis=0)
            entry[s] = {"folds": folds, "mean": {"diff": float(np.mean([x["diff"] for x in folds])), "ci95": canonical.ci(mean_draws)}}
        out["pairs"].append(entry)
    return out
