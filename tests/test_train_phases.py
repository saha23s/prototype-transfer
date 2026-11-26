"""Plain-assert test of trainer.train with --last-epochs 0: the reported scores belong to the saved checkpoint.
Run: python tests/test_train_phases.py

Models, datasets and epochs are stand-ins (no download, no GPU). The toy model is one number, which is also its val score,
so a checkpoint's weights show which scores it should be reported with.
"""

import argparse
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from prototype_transfer.evaluation import results
from prototype_transfer.training import trainer

EPOCH_SCORES = {"warm": [0.1], "joint": [0.5, 0.9, 0.3]}   # joint epoch 2 is the best, epoch 3 the last
PUSH_SCORE = 0.7


class Toy(nn.Module):
    def __init__(self, score: float = 0.0) -> None:
        super().__init__()
        self.score = nn.Parameter(torch.tensor(score, dtype=torch.float64))


class ToyData:
    def __init__(self, *args, **kwargs) -> None:
        self.df = pd.DataFrame({"i": range(4)})

    def mixing_config(self) -> dict:
        return {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> int:
        return i


def _set(model: nn.Module, score: float) -> None:
    with torch.no_grad():
        model.score.fill_(score)


def _train(tmp: str, joint_epochs: int, select: str, no_push: bool) -> tuple[dict, dict]:
    """Run trainer.train with 0 last-layer epochs; returns (the `train` metrics section, the saved best.pt)."""
    scores = iter(EPOCH_SCORES["warm"] + EPOCH_SCORES["joint"][:joint_epochs])

    def load(path, device="cpu"):
        return Toy(float(torch.load(path, weights_only=False)["state_dict"]["score"]))

    store = results.Store("toy", Path(tmp))
    results.save_results(store, "toy", "train", {}, entry={"legacy": True, "note": "copied old result"})
    stubs = {"HSNTrainDataset": ToyData, "HSNInferenceDataset": ToyData, "build_hsn_protopnet": Toy, "load_checkpoint": load,
             "split_manifest": lambda *a: Path(tmp) / "unused.parquet", "set_trainable": lambda model, groups: None,
             "make_opt": lambda model, lrs, steps, wd: (None, None),
             "run_epoch": lambda model, *a: _set(model, next(scores)) or 0.0,
             "push_prototypes": lambda model, *a, **k: _set(model, PUSH_SCORE) or [],
             "evaluate": lambda model, *a: {"cmAP": float(model.score), "macro_auc": float(model.score)}}
    args = argparse.Namespace(
        train_domain="focal", name="toy", dataset="toy", splits_dir=Path(tmp), cv_fold=None, seed=0, select=select,
        anchored=joint_epochs == 0, warm_epochs=1, joint_epochs=joint_epochs, last_epochs=0, no_push=no_push, rung="test",
        batch_size=2, num_workers=1, max_train_batches=None, max_eval_samples=None, train_labels=None, bg_kind="none",
        bg_manifest=None, bg_prob=0.0, snr_min=0.0, snr_max=0.0, overlap_prob=0.0, overlap_gain_min=0.0, overlap_gain_max=0.0,
        pseudo_manifest=None, pseudo_batch_size=2, pseudo_weight=0.0, lr_backbone=0.0, lr_prototypes=0.0, lr_last=0.0,
        weight_decay=0.0, ortho_weight=0.0, ortho_form="frobenius")
    real = {k: getattr(trainer, k) for k in stubs}
    real_store, real_prov = results.Store, results.provenance
    try:
        for k, v in stubs.items():
            setattr(trainer, k, v)
        results.Store = lambda dataset: store
        results.provenance = lambda: {"commit": None, "dirty": None, "env": {}}
        trainer.train(args)
    finally:
        for k, v in real.items():
            setattr(trainer, k, v)
        results.Store, results.provenance = real_store, real_prov
    entry = results.read_registry(store)["toy"]
    assert entry["legacy"] is False and entry["note"] is None, f"a freshly trained run is not the old copied one: {entry}"
    return (results.load_metrics(store, "toy")["sections"]["train"],
            torch.load(store.checkpoints("toy") / "best.pt", weights_only=False))


def test_zero_last_epochs_reports_the_scores_of_the_saved_checkpoint():
    cases = {  # (joint epochs, --select, --no-push) -> (score of the saved model, its phase, score of the last epoch run)
        "push": ((3, "best", False), (PUSH_SCORE, "push", PUSH_SCORE)),
        "no push, best": ((3, "best", True), (0.9, "joint", 0.3)),
        "no push, last": ((3, "last", True), (0.3, "joint", 0.3)),
        "anchored, no push": ((0, "best", True), (0.1, "warm", 0.1)),
        "anchored, push": ((0, "best", False), (PUSH_SCORE, "push", PUSH_SCORE))}
    for what, (options, (score, phase, last)) in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            metrics, ckpt = _train(tmp, *options)
        assert float(ckpt["state_dict"]["score"]) == score, f"{what}: wrong model saved"
        assert metrics["selected_val"]["focal"]["cmAP"] == score, f"{what}: {metrics['selected_val']}"
        assert ckpt["extra"]["phase"] == phase and ckpt["extra"]["val"]["focal"]["cmAP"] == score, f"{what}: {ckpt['extra']}"
        assert metrics["last_epoch_val"]["focal"]["cmAP"] == last, f"{what}: {metrics['last_epoch_val']}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests OK")
