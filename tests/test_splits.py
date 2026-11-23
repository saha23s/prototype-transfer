"""Plain-assert tests for the cross-validation folds in src/splits.py.  Run: python tests/test_splits.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from train_protopnet import check_no_heldout  # noqa: E402
from src.model import HSN_SPECIES_LIST  # noqa: E402
from src.splits import site_of, soundscape_cv_folds  # noqa: E402


def _toy(n_sites: int = 31, clips: int = 6, seed: int = 0) -> pd.DataFrame:
    """n_sites sites x clips clips; each of 6 species lives at roughly 40% of the sites."""
    rng = np.random.default_rng(seed)
    home = {sp: rng.random(n_sites) < 0.4 for sp in HSN_SPECIES_LIST[:6]}
    rows = [{"filepath": f"HSN_{s:03d}_20220101_000000_{c}.ogg",
             "hsn_classes": [sp for sp, h in home.items() if h[s] and rng.random() < 0.5]}
            for s in range(n_sites) for c in range(clips)]
    return pd.DataFrame(rows)


def test_site_of():
    assert site_of(pd.Series(["HSN_012_20220101_a.ogg"])).tolist() == ["HSN_012"]


def test_folds_partition_the_sites_evenly_and_keep_sites_whole():
    df = _toy(31)
    fold, info = soundscape_cv_folds(df, n_folds=3, seed=0, n_trials=300)
    site = site_of(df["filepath"])
    assert sorted(fold.unique()) == [0, 1, 2]
    assert site.groupby(fold).nunique().sort_values().tolist() == [10, 10, 11]           # sizes differ by at most one site
    assert (fold.groupby(site).nunique() == 1).all()                                      # a site is in exactly one fold
    assert sorted(s for v in info["sites"].values() for s in v) == sorted(site.unique())  # every site listed exactly once


def test_folds_are_deterministic_and_seed_dependent():
    df = _toy(31)
    a, _ = soundscape_cv_folds(df, seed=1, n_trials=200)
    b, _ = soundscape_cv_folds(df, seed=1, n_trials=200)
    c, _ = soundscape_cv_folds(df, seed=2, n_trials=200)
    assert (a == b).all() and not (a == c).all()


def test_every_eligible_species_has_positives_in_every_fold():
    df = _toy(40)
    fold, info = soundscape_cv_folds(df, n_folds=3, seed=0, n_trials=500)
    for g in range(3):
        seen = set(df.loc[fold == g, "hsn_classes"].explode().dropna())
        assert set(info["eligible_species"]) <= seen, (g, set(info["eligible_species"]) - seen)


def test_training_refuses_manifests_with_heldout_clips():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        pd.DataFrame({"audio_path": ["a", "b"]}).to_parquet(d / "heldout.parquet", index=False)
        pd.DataFrame({"audio_path": ["c", "d"]}).to_parquet(d / "clean.parquet", index=False)
        pd.DataFrame({"audio_path": ["c", "b"]}).to_parquet(d / "leaky.parquet", index=False)
        check_no_heldout(d / "clean.parquet", d / "heldout.parquet", "--bg-manifest")           # fine
        try:
            check_no_heldout(d / "leaky.parquet", d / "heldout.parquet", "--bg-manifest")
            raise AssertionError("a manifest with held-out clips must be refused")
        except SystemExit as e:
            assert "held-out fold" in str(e)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
