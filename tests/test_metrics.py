"""Plain-assert tests for birdset_scores in src/prototype_transfer/evaluation/metrics.py.  Run: python tests/test_metrics.py"""
import numpy as np

from prototype_transfer.evaluation.metrics import birdset_scores


def test_top1_is_reported_over_labelled_and_all_clips():
    y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]])
    p = np.array([[.9, .1, .1], [.8, .2, .1], [.1, .1, .9], [.5, .1, .1]])
    m = birdset_scores(y, p)
    assert m["n_clips"] == 4 and m["n_labelled_clips"] == 2
    assert m["T1_labelled_clips"] == 0.5 and m["T1_all_clips"] == 0.25


def test_cmap_and_auroc_skip_species_without_positives():
    y = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]])
    p = np.array([[.9, .1, .5], [.1, .9, .5], [.8, .2, .5], [.2, .1, .5]])
    m = birdset_scores(y, p)
    assert m["n_species_with_positives"] == 2
    assert m["cmAP"] == 1.0 and m["AUROC"] == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
