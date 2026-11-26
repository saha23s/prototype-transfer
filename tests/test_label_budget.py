"""Plain-assert test of --train-labels in training/trainer.train.  Run: python tests/test_label_budget.py

With a label budget of N, every step that reads training labels (the training batches AND the prototype push) must see
only the N sampled clips. The real train() and dataset classes run on a temporary manifest; the model, the training
steps and evaluation are stubbed, so no audio, model download or GPU is needed. Nothing is written outside a temp dir.
"""

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from prototype_transfer.evaluation import results
from prototype_transfer.training import trainer

N_CLIPS, BUDGET = 40, 6


def _args(splits: Path, train_labels: int | None) -> argparse.Namespace:
    return argparse.Namespace(
        train_domain="soundscape", name="t", dataset="hsn", splits_dir=splits, batch_size=2, num_workers=1, warm_epochs=1,
        joint_epochs=1, last_epochs=1, lr_backbone=1e-5, lr_prototypes=1e-3, lr_last=1e-3, weight_decay=0.0,
        ortho_weight=0.01, ortho_form="frobenius", no_push=False, rung="test", select="last", cv_fold=None, anchored=False,
        train_labels=train_labels, bg_kind="none", bg_manifest=None, bg_prob=0.5, snr_min=-5.0, snr_max=20.0,
        overlap_prob=0.0, overlap_gain_min=-10.0, overlap_gain_max=0.0, pseudo_manifest=None, pseudo_weight=0.2,
        pseudo_batch_size=2, max_train_batches=None, max_eval_samples=None, seed=0)


class _Model:
    def to(self, device):
        return self


def _clips_seen(train_labels: int | None) -> tuple[list[str], list[str]]:
    """Run train() and return (audio paths of the training dataset, audio paths handed to push)."""
    seen: dict = {}
    ev = {"cmAP": 0.0, "macro_auc": 0.5}

    def fake_run_epoch(model, loader, *a, **k):
        seen["train"] = loader.dataset.df["audio_path"].tolist()
        return 0.0

    def fake_push(model, dataset, device, **k):
        seen["push"] = dataset.df["audio_path"].tolist()
        return []

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        df = pd.DataFrame({"audio_path": [f"clip_{i}.ogg" for i in range(N_CLIPS)], "hsn_classes": [["whcspa"]] * N_CLIPS})
        for domain in trainer.DOMAINS:
            for part in ("train", "val"):
                df.to_parquet(root / f"{domain}_{part}.parquet")
        stubs = {"build_hsn_protopnet": lambda: _Model(), "load_checkpoint": lambda *a: _Model(), "save_checkpoint": lambda *a: None,
                 "set_trainable": lambda *a: None, "make_opt": lambda *a: (None, None), "evaluate": lambda *a: ev,
                 "run_epoch": fake_run_epoch, "push_prototypes": fake_push,
                 "split_manifest": lambda splits, domain, part, fold=None: root / f"{domain}_{part}.parquet"}
        saved = {k: getattr(trainer, k) for k in stubs}
        real_store = results.Store
        try:
            for k, v in stubs.items():
                setattr(trainer, k, v)
            results.Store = lambda dataset: real_store(dataset, root / "outputs")
            trainer.train(_args(root, train_labels))
        finally:
            results.Store = real_store
            for k, v in saved.items():
                setattr(trainer, k, v)
    return seen["train"], seen["push"]


def test_push_sees_only_the_labeled_clips_of_the_budget():
    train, push = _clips_seen(BUDGET)
    assert len(train) == BUDGET and len(set(train)) == BUDGET
    extra = set(push) - set(train)
    assert not extra and sorted(push) == sorted(train), f"push read labels of {len(extra)} clips outside the {BUDGET}-label budget"


def test_without_a_budget_push_uses_the_whole_training_manifest():
    train, push = _clips_seen(None)
    assert len(train) == N_CLIPS and sorted(push) == sorted(train) == sorted(f"clip_{i}.ogg" for i in range(N_CLIPS))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
