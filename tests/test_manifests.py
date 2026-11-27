"""Label-free manifests (prototype_transfer.method.manifests). Run: python tests/test_manifests.py"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from prototype_transfer.method import manifests
from prototype_transfer.training import trainer


def _responses(tmp: Path, name: str, n: int, start: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    entry = {"probs": rng.random((n, 21)).astype(np.float32), "audio_path": [f"/a/HSN_{start + i:03d}.ogg" for i in range(n)],
             "targets": rng.random((n, 21)) < 0.1}                     # present in the file, must never be read
    torch.save({"soundscape_train": entry, "soundscape_val": entry, "soundscape_test": entry}, tmp / name)
    return str(tmp / name)


def test_background_is_the_lowest_quantile_and_pseudo_keeps_every_clip():
    with tempfile.TemporaryDirectory() as t:
        probs, paths = manifests.load_responses([_responses(Path(t), "r.pt", 200, 0, 0) + ":soundscape_train"], [])
        bg, pseudo, cutoff = manifests.build_manifests(probs, paths, 0.3)
        assert len(pseudo) == 200 and len(bg) == 60
        assert bg["max_prob"].max() <= cutoff + 1e-9 and (probs.max(axis=1) <= cutoff).sum() >= 60
        assert list(pseudo.columns) == ["audio_path", "teacher_probs"] and pseudo["teacher_probs"].iloc[0].shape == (21,)
        assert set(bg["audio_path"]) <= set(paths)


def test_test_split_and_duplicate_clips_are_refused():
    with tempfile.TemporaryDirectory() as t:
        f = _responses(Path(t), "r.pt", 20, 0, 0)
        for bad in (lambda: manifests.load_responses([f + ":soundscape_test"], []),
                    lambda: manifests.load_responses([f + ":soundscape_train", f + ":soundscape_train"], [])):
            try:
                bad()
            except AssertionError:
                pass
            else:
                raise AssertionError("expected a refusal")


def test_test_clips_are_refused_whatever_the_key_or_directory():
    with tempfile.TemporaryDirectory() as t:
        f = _responses(Path(t), "r.pt", 20, 0, 0)                            # paths are /a/HSN_000.ogg, ...
        z = torch.load(f, weights_only=False)
        torch.save({"soundscape": z["soundscape_test"]}, f)                  # the key extract_prototype_activations.py writes
        try:
            manifests.load_responses([f + ":soundscape"], ["/other/root/HSN_007.ogg"])
        except AssertionError as e:
            assert "1 clips of the test split" in str(e)
        else:
            raise AssertionError("expected a refusal")


def test_training_refuses_test_clips_with_and_without_a_cv_fold():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        for name, clips in (("test", ["/a/x.ogg"]), ("val", ["/a/y.ogg"]), ("leaky", ["/b/x.ogg"]), ("clean", ["/b/z.ogg"])):
            pd.DataFrame({"audio_path": clips}).to_parquet(t / f"{name}.parquet", index=False)
        real = trainer.split_manifest
        trainer.split_manifest = lambda splits, domain, part, fold=None: t / f"{part}.parquet"
        try:
            for fold in (None, 0):
                for flag in ("bg_manifest", "pseudo_manifest"):
                    args = argparse.Namespace(train_domain="focal", bg_kind="none", bg_manifest=None, pseudo_manifest=None,
                                              select="last", cv_fold=fold, splits_dir=t)
                    setattr(args, flag, t / "clean.parquet")
                    trainer.validate(args)                                   # fine
                    setattr(args, flag, t / "leaky.parquet")
                    try:
                        trainer.validate(args)
                    except ValueError as e:
                        assert "test split" in str(e)
                    else:
                        raise AssertionError(f"{flag} with test clips accepted (cv_fold={fold})")
        finally:
            trainer.split_manifest = real


def test_fold_restriction_keeps_only_training_sites_and_needs_all_of_them():
    with tempfile.TemporaryDirectory() as t:
        f1, f2 = _responses(Path(t), "a.pt", 50, 0, 1), _responses(Path(t), "b.pt", 50, 50, 2)
        probs, paths = manifests.load_responses([f1 + ":soundscape_train", f2 + ":soundscape_val"], [])
        keep = set(paths[:30]) | set(paths[60:90])
        p2, q2 = manifests.restrict_to_fold(probs, paths, keep, 0)
        assert set(q2) == keep and len(p2) == 60 and np.array_equal(p2[0], probs[0])
        try:
            manifests.restrict_to_fold(probs[:50], paths[:50], keep, 0)       # val responses missing
        except AssertionError as e:
            assert "no responses" in str(e)
        else:
            raise AssertionError("expected a refusal")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"{len(tests)} tests OK")
