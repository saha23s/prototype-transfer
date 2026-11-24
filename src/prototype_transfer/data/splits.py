"""Leakage-aware train/val/test splits for one-domain training on BirdSet HSN.

* Focal (Xeno-Canto): grouped by recordist (same person/equipment/locality -> near-duplicates),
  stratified by species with StratifiedGroupKFold.
* Soundscape (PAM): grouped by recording site (HSN_xxx; one 10-min recording per site, 120 adjacent
  5-s clips). Species are strongly clustered by site, so we search random site partitions for the one
  whose per-species positive fractions are closest to the target fractions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from prototype_transfer.models.classifier import HSN_SPECIES_LIST


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


def soundscape_cv_folds(df: pd.DataFrame, n_folds: int = 3, seed: int = 0, n_trials: int = 20000,
                        min_sites: int = 3) -> tuple[pd.Series, dict]:
    """Site-grouped cross-validation folds of a soundscape manifest (pass the non-test sites, i.e. train + val).

    Returns (fold id 0..n_folds-1 per clip aligned with df, info). Whole sites go to one fold, so the clips of a
    10-minute recording never straddle folds. Random search over site partitions with equal-size folds (sizes differ by
    at most one site), first requiring that every species present at >= min_sites sites has positives in every fold,
    then minimising the mean deviation of each fold's positive fraction from 1 / n_folds (same objective as
    soundscape_split)."""
    site = site_of(df["filepath"])
    sites = np.array(sorted(site.unique()))
    assert 1 < n_folds <= len(sites), f"n_folds={n_folds} must be between 2 and {len(sites)}"
    Y = np.zeros((len(df), len(HSN_SPECIES_LIST)), dtype=np.int64)
    idx = {sp: i for i, sp in enumerate(HSN_SPECIES_LIST)}
    for r, classes in enumerate(df["hsn_classes"]):
        for sp in classes:
            Y[r, idx[sp]] = 1
    per_site = pd.DataFrame(Y, columns=HSN_SPECIES_LIST).groupby(site.to_numpy()).sum().reindex(sites)
    eligible = [sp for sp in HSN_SPECIES_LIST if (per_site[sp] > 0).sum() >= min_sites]
    P = per_site[eligible].to_numpy(dtype=float)                       # [n_sites, n_eligible]
    totals = P.sum(axis=0)
    sizes = [len(sites) // n_folds + (g < len(sites) % n_folds) for g in range(n_folds)]
    bounds = np.cumsum([0] + sizes)
    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for _ in range(n_trials):
        perm = rng.permutation(len(sites))
        frac = np.stack([P[perm[bounds[g]:bounds[g + 1]]].sum(axis=0) / totals for g in range(n_folds)])   # [n_folds, n_eligible]
        score = int((frac == 0).sum()) + float(np.abs(frac - 1 / n_folds).mean())
        if score < best_score:
            best, best_score = perm, score
    fold_of = {sites[i]: g for g in range(n_folds) for i in best[bounds[g]:bounds[g + 1]]}
    info = {"sites": {str(g): sorted(s for s, f in fold_of.items() if f == g) for g in range(n_folds)},
            "eligible_species": eligible, "objective": float(best_score)}
    return site.map(fold_of), info


# ------------------------------------------------------------------------------------------- canonical splits
SPLITS_LOCK = Path("configs/splits/hsn.json")   # committed fingerprints of the canonical splits (relative to the repo root)


def fingerprint(df: pd.DataFrame) -> dict:
    """Identity of a split manifest that does not depend on where the data lives, row order or library versions:
    the sha256 of its sorted (file name, labels) lines, and its clip count. Labels are the species of a soundscape clip,
    or the species and recordist of a focal clip (both enter the evaluation)."""
    names = df["audio_path"].map(lambda q: Path(q).name)
    if "hsn_classes" in df:
        label = df["hsn_classes"].map(lambda c: ",".join(sorted(c)))
    else:
        label = df["ebird_code"].astype(str) + "|" + df["recordist"].fillna("unknown").astype(str)
    lines = sorted(f"{n}|{lab}" for n, lab in zip(names, label))
    return {"n_clips": len(df), "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest()}


def verify_split(manifest: Path, lock: Path = SPLITS_LOCK) -> None:
    """Refuse a canonical split manifest that differs from the committed fingerprint. Manifests that are not listed in the
    lock file (e.g. background manifests, which depend on a model) pass.

    Raises:
        ValueError: if the manifest's clips or labels differ from the canonical split; the message says how to regenerate.
    """
    expected = json.loads(lock.read_text())["files"] if lock.exists() else {}
    want = expected.get(Path(manifest).name)
    if want is None:
        return
    got = fingerprint(pd.read_parquet(manifest))
    if got != want:
        raise ValueError(
            f"{manifest} is not the canonical split: {got['n_clips']} clips (expected {want['n_clips']}), "
            f"checksum {'differs' if got['sha256'] != want['sha256'] else 'matches'}. Regenerate with "
            "`python scripts/data/make_splits.py` (seed 0) and run `python scripts/data/check_splits.py`; "
            "results on different splits are not comparable.")


def split_manifest(splits_dir: Path, domain: str, part: str, cv_fold: int | None = None) -> Path:
    """Manifest of `domain`'s train / val part. With a CV fold, soundscape train is the other folds' sites and
    soundscape "val" is the held-out fold (monitoring only). Raises ValueError if the file is not the canonical split."""
    if cv_fold is not None and domain == "soundscape":
        path = splits_dir / f"soundscape_cv{cv_fold}_{'train' if part == 'train' else 'heldout'}.parquet"
    else:
        path = splits_dir / f"{domain}_{part}.parquet"
    verify_split(path)
    return path


def check_no_heldout(manifest: Path, heldout: Path, what: str) -> None:
    """Refuse a manifest that lists any clip of the held-out fold (its audio must not be used for adaptation either).

    Raises:
        ValueError: if `manifest` and `heldout` share an audio_path.
    """
    leaked = set(pd.read_parquet(manifest)["audio_path"]) & set(pd.read_parquet(heldout)["audio_path"])
    if leaked:
        raise ValueError(f"{what} {manifest} contains {len(leaked)} clips of the held-out fold ({heldout}); "
                         "build it with scripts/method/make_background_manifest.py --fold F")
