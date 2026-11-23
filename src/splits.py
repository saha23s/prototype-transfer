"""Leakage-aware train/val/test splits for one-domain training on BirdSet HSN.

* Focal (Xeno-Canto): grouped by recordist (same person/equipment/locality -> near-duplicates),
  stratified by species with StratifiedGroupKFold.
* Soundscape (PAM): grouped by recording site (HSN_xxx; one 10-min recording per site, 120 adjacent
  5-s clips). Species are strongly clustered by site, so we search random site partitions for the one
  whose per-species positive fractions are closest to the target fractions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.model import HSN_SPECIES_LIST


def focal_split(df: pd.DataFrame, test_frac: float = 0.2, val_frac: float = 0.1, seed: int = 0) -> pd.Series:
    """Return a Series of 'train' / 'val' / 'test' aligned with df (grouped by recordist)."""
    y = df["target_class"].to_numpy()
    groups = df["recordist"].fillna("unknown").to_numpy()
    split = pd.Series("train", index=df.index)

    n_test = max(2, round(1 / test_frac))
    test_idx = next(StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed).split(df, y, groups))[1]
    split.iloc[test_idx] = "test"

    rest = np.nonzero((split == "train").to_numpy())[0]
    n_val = max(2, round((1 - test_frac) / val_frac))
    val_rel = next(StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed + 1)
                   .split(rest, y[rest], groups[rest]))[1]
    split.iloc[rest[val_rel]] = "val"
    return split


def site_of(filepath: pd.Series) -> pd.Series:
    return filepath.str.extract(r"^(HSN_\d+)_")[0]


def soundscape_split(df: pd.DataFrame, test_frac: float = 0.3, val_frac: float = 0.1, seed: int = 0,
                     n_trials: int = 20000, min_sites: int = 3) -> tuple[pd.Series, dict]:
    """Return ('train'/'val'/'test' Series aligned with df, info) grouped by recording site.

    Objective (species present at >= min_sites sites): first, every such species must have positives in
    both train and test; then minimise the mean deviation of its test (and, half-weighted, val) positive
    fraction from test_frac (val_frac). Val may lack rare species (it is only used for model selection).
    """
    site = site_of(df["filepath"])
    sites = np.array(sorted(site.unique()))
    # positives per (site, species)
    Y = np.zeros((len(df), len(HSN_SPECIES_LIST)), dtype=np.int64)
    idx = {sp: i for i, sp in enumerate(HSN_SPECIES_LIST)}
    for r, classes in enumerate(df["hsn_classes"]):
        for sp in classes:
            Y[r, idx[sp]] = 1
    per_site = pd.DataFrame(Y, columns=HSN_SPECIES_LIST).groupby(site.to_numpy()).sum().reindex(sites)
    totals = per_site.sum(axis=0)
    eligible = [sp for sp in HSN_SPECIES_LIST if (per_site[sp] > 0).sum() >= min_sites]

    rng = np.random.default_rng(seed)
    n_test, n_val = round(len(sites) * test_frac), round(len(sites) * val_frac)
    best, best_score = None, np.inf
    for _ in range(n_trials):
        perm = rng.permutation(len(sites))
        t, v = perm[:n_test], perm[n_test:n_test + n_val]
        ft = per_site.iloc[t][eligible].sum() / totals[eligible]
        fv = per_site.iloc[v][eligible].sum() / totals[eligible]
        ftr = 1 - ft - fv
        # 1st: every eligible species needs train AND test positives; 2nd: mean deviation from targets
        missing = int((ft == 0).sum() + (ftr == 0).sum())
        score = missing + np.abs(ft - test_frac).mean() + 0.5 * np.abs(fv - val_frac).mean()
        if score < best_score:
            best, best_score = (t, v), score
    t, v = best
    assign = {s: "train" for s in sites}
    assign.update({sites[i]: "test" for i in t})
    assign.update({sites[i]: "val" for i in v})
    split = site.map(assign)
    frac = {sp: {s: float(per_site.loc[[k for k, a in assign.items() if a == s], sp].sum() / totals[sp])
                 if totals[sp] else None for s in ("train", "val", "test")} for sp in HSN_SPECIES_LIST}
    info = {"sites": {s: sorted(k for k, a in assign.items() if a == s) for s in ("train", "val", "test")},
            "eligible_species": eligible, "objective": float(best_score), "positive_fraction": frac}
    return split, info
