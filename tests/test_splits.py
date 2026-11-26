"""Plain-assert tests for the cross-validation folds in src/prototype_transfer/data/splits.py.  Run: python tests/test_splits.py"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from prototype_transfer.models.classifier import HSN_SPECIES_LIST
from prototype_transfer.data.splits import check_no_heldout, fingerprint, site_of, soundscape_cv_folds, verify_split


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
        except ValueError as e:
            assert "held-out fold" in str(e)


def test_heldout_clips_are_found_under_another_data_root():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        pd.DataFrame({"audio_path": ["/here/HSN_001_a.ogg"]}).to_parquet(d / "heldout.parquet", index=False)
        pd.DataFrame({"audio_path": ["/there/HSN_001_a.ogg"]}).to_parquet(d / "leaky.parquet", index=False)
        try:
            check_no_heldout(d / "leaky.parquet", d / "heldout.parquet", "--bg-manifest")
        except ValueError as e:
            assert "held-out fold" in str(e)
        else:
            raise AssertionError("the same clip under another directory must be refused")


def _manifest(root: str, names: list[str], labels: list[list[str]]) -> pd.DataFrame:
    return pd.DataFrame({"audio_path": [f"{root}/{n}" for n in names], "hsn_classes": labels})


def test_fingerprint_ignores_location_and_order_but_not_clips_or_labels():
    names, labels = ["a.ogg", "b.ogg", "c.ogg"], [["x"], [], ["y", "x"]]
    base = fingerprint(_manifest("/data/one", names, labels))
    assert base == fingerprint(_manifest("/elsewhere", names[::-1], labels[::-1]))          # other machine, other row order
    assert base == fingerprint(_manifest("/data/one", names, [["x"], [], ["x", "y"]]))      # label order is irrelevant
    assert base != fingerprint(_manifest("/data/one", names, [["x"], ["x"], ["y", "x"]]))   # a changed label
    assert base != fingerprint(_manifest("/data/one", names[:2], labels[:2]))               # a missing clip
    focal = pd.DataFrame({"audio_path": ["/d/XC1.ogg"], "ebird_code": ["amepip"], "recordist": ["r1"]})
    assert fingerprint(focal) != fingerprint(focal.assign(recordist="r2"))                   # recordists enter the evaluation


def test_verify_split_refuses_a_different_split_and_passes_unlisted_files():
    import json
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        good = _manifest(str(tmp), ["a.ogg", "b.ogg"], [["x"], []])
        good.to_parquet(tmp / "soundscape_test.parquet")
        lock = tmp / "lock.json"
        lock.write_text(json.dumps({"files": {"soundscape_test.parquet": fingerprint(good)}}))
        verify_split(tmp / "soundscape_test.parquet", lock)
        _manifest(str(tmp), ["a.ogg", "b.ogg"], [["x"], ["y"]]).to_parquet(tmp / "soundscape_test.parquet")
        try:
            verify_split(tmp / "soundscape_test.parquet", lock)
        except ValueError as e:
            assert "not the canonical split" in str(e) and "make_splits.py" in str(e)
        else:
            raise AssertionError("a different split must be refused")
        good.to_parquet(tmp / "soundscape_background.parquet")     # derived manifests are not canonical splits
        verify_split(tmp / "soundscape_background.parquet", lock)


def test_verify_split_with_a_canonical_name_checks_renamed_files():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        full = _manifest("/a", ["HSN_001_a.ogg", "HSN_002_b.ogg"], [["whcspa"], []])
        lock = d / "lock.json"
        lock.write_text(json.dumps({"files": {"soundscape_test.parquet": fingerprint(full)}}))
        full.to_parquet(d / "renamed_copy.parquet", index=False)
        full.iloc[:1].to_parquet(d / "my_subset.parquet", index=False)
        verify_split(d / "renamed_copy.parquet", lock, canonical="soundscape_test.parquet")    # same clips and labels: fine
        for path, name, where, message in (("my_subset.parquet", "soundscape_test.parquet", lock, "is not the canonical split"),
                                           ("renamed_copy.parquet", "focal_val.parquet", lock, "no fingerprint"),
                                           ("renamed_copy.parquet", "soundscape_test.parquet", d / "absent.json", "no fingerprint")):
            try:
                verify_split(d / path, where, canonical=name)
            except ValueError as e:
                assert message in str(e), e
            else:
                raise AssertionError(f"{path} accepted as {name} (lock {where.name})")


def test_committed_lock_covers_every_split_the_code_can_ask_for():
    import json
    from prototype_transfer.data.splits import SPLITS_LOCK
    files = json.loads(SPLITS_LOCK.read_text())["files"]
    wanted = {f"{d}_{p}.parquet" for d in ("focal", "soundscape") for p in ("train", "val", "test")}
    wanted |= {f"soundscape_cv{f}_{p}.parquet" for f in range(3) for p in ("train", "heldout")}
    assert wanted == set(files), wanted ^ set(files)
    assert files["soundscape_test.parquet"]["n_clips"] == 3600 and files["focal_test.parquet"]["n_clips"] == 1157


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
