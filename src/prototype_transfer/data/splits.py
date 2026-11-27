"""Leakage-aware splits of BirdSet HSN, their fingerprints, and the guards that keep held-out audio out of training.

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


# ------------------------------------------------------------------------------- adaptation / test partitions
N_PARTITIONS, ADAPT_TEST_FRAC = 3, 0.25


def adaptation_test_split(df: pd.DataFrame, partition: int, test_frac: float = ADAPT_TEST_FRAC, n_trials: int = 20000,
                          min_sites: int = 3) -> tuple[pd.Series, dict]:
    """Split a soundscape manifest by site into an unlabeled adaptation pool and test sites (inductive UDA protocol).

    Partition k is the site partition chosen with seed k by the same search as `soundscape_split` (every species present
    at >= min_sites sites has positives on both sides; per-species test fractions close to `test_frac`), with no val part.
    Returns ('adapt' / 'test' per clip aligned with df, info). Adaptation code may read only the audio of 'adapt' clips;
    labels are read only to score 'test' clips."""
    split, info = soundscape_split(df, test_frac=test_frac, val_frac=0.0, seed=partition, n_trials=n_trials, min_sites=min_sites)
    info["sites"] = {"adapt": info["sites"]["train"], "test": info["sites"]["test"]}
    info["positive_fraction"] = {sp: None if f is None else {"adapt": f["train"], "test": f["test"]}
                                 for sp, f in info["positive_fraction"].items()}
    return split.map({"train": "adapt", "test": "test"}), {"partition": partition, "test_frac": test_frac, **info}


def partition_manifest(splits_dir: Path, partition: int, part: str) -> Path:
    """data/HSN/splits/soundscape_p<k>_<adapt|test>.parquet, checked against the committed fingerprint."""
    assert part in ("adapt", "test"), part
    path = Path(splits_dir) / f"soundscape_p{partition}_{part}.parquet"
    verify_split(path)
    return path


# ------------------------------------------------------------------------------------- fingerprints and guards
SPLITS_LOCK = Path("configs/splits/hsn.json")   # committed fingerprints of the canonical splits (relative to the repo root)


def fingerprint(df: pd.DataFrame) -> dict:
    """Identity of a split manifest that does not depend on where the data lives, row order or library versions:
    the sha256 of its sorted (file name, labels) lines, and its clip count. Labels are the species of a soundscape clip,
    or the species and recordist of a focal clip."""
    names = df["audio_path"].map(lambda q: Path(q).name)
    if "hsn_classes" in df:
        label = df["hsn_classes"].map(lambda c: ",".join(sorted(c)))
    else:
        label = df["ebird_code"].astype(str) + "|" + df["recordist"].fillna("unknown").astype(str)
    lines = sorted(f"{n}|{lab}" for n, lab in zip(names, label))
    return {"n_clips": len(df), "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest()}


def verify_split(manifest: Path, lock: Path = SPLITS_LOCK) -> None:
    """Refuse a split manifest that differs from its committed fingerprint. Files not listed in the lock pass (derived
    manifests, or partitions not pinned yet: see scripts/data/check_splits.py --write).

    Raises:
        ValueError: if the manifest's clips or labels differ from the committed split; the message says how to regenerate.
    """
    expected = json.loads(Path(lock).read_text())["files"] if Path(lock).exists() else {}
    want = expected.get(Path(manifest).name)
    if want is None:
        return
    got = fingerprint(pd.read_parquet(manifest))
    if got != want:
        raise ValueError(
            f"{manifest} is not the committed split: {got['n_clips']} clips (expected {want['n_clips']}), "
            f"checksum {'differs' if got['sha256'] != want['sha256'] else 'matches'}. Regenerate with "
            "`python scripts/data/make_splits.py` (seed 0) and run `python scripts/data/check_splits.py`; "
            "results on different splits are not comparable.")


def check_no_heldout(manifest: Path, heldout: Path, what: str) -> None:
    """Refuse a manifest that lists any clip of a held-out manifest (test audio must not be used for adaptation either).

    Raises:
        ValueError: if `manifest` and `heldout` share a clip (compared by file name).
    """
    name = lambda m: set(pd.read_parquet(m)["audio_path"].map(lambda q: Path(q).name))
    leaked = name(manifest) & name(heldout)
    if leaked:
        raise ValueError(f"{what} {manifest} contains {len(leaked)} held-out clips of {heldout}; "
                         "adaptation may only use the audio of the adaptation sites")
