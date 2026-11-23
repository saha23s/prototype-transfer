"""Plain-assert tests for scripts/canonical_eval.py (pytest is not a project dependency).  Run: python tests/test_canonical_eval.py"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from canonical_eval import K, Model, metrics  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import ap_cols, auc_cols, cluster_indices  # noqa: E402


def test_auc_cols_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    S = np.round(rng.random((300, 7)), 1)            # rounding creates many ties
    Y = rng.random((300, 7)) < 0.2
    Y[:, 3] = False                                  # a column with no positives -> NaN
    got = auc_cols(S, Y)
    for j in range(7):
        want = roc_auc_score(Y[:, j], S[:, j]) if Y[:, j].any() else np.nan
        assert (np.isnan(got[j]) and np.isnan(want)) or abs(got[j] - want) < 1e-12, (j, got[j], want)


def test_ap_cols_matches_sklearn():
    rng = np.random.default_rng(1)
    S, Y = rng.random((200, 4)), rng.random((200, 4)) < 0.3
    assert np.allclose(ap_cols(S, Y), [average_precision_score(Y[:, j], S[:, j]) for j in range(4)])


def _toy_model():
    """2 target species, 6 soundscape clips: species 0 positives {0,1}; clip 2 holds species 1 only; clips 3-5 are empty.
    Prototype activation of species 0 is high on its positives, HIGH on the other-species clip 2 and low on empty clips,
    so AUC against all negatives is high but AUC against other-species negatives only is low."""
    m = Model.__new__(Model)
    m.name, m.species = "toy", ["a", "b"]
    m.y_s = np.zeros((6, 2), bool); m.y_s[[0, 1], 0] = True; m.y_s[2, 1] = True
    m.a_s = np.zeros((6, 2, K)); m.a_s[[0, 1], 0, :] = 0.9; m.a_s[2, 0, :] = 0.95; m.a_s[3:, 0, :] = 0.05
    m.p_s = np.zeros((6, 2))
    m.y_f = np.zeros((4, 2), bool); m.y_f[[0, 1], 0] = True; m.y_f[[2, 3], 1] = True   # focal: negatives are the other species
    m.a_f = np.zeros((4, 2, K)); m.a_f[[0, 1], 0, :] = 0.9; m.a_f[[2, 3], 0, :] = 0.1
    m.p_f = np.zeros((4, 2))
    return m


def test_matched_negatives():
    m = _toy_model()
    r = metrics(m, np.arange(4), np.arange(6), cols=[0], pri=[0])
    assert np.allclose(r["_auc_f"], 1.0)               # focal: perfect
    assert np.allclose(r["_auc_s"], 3 / 4)             # each positive beats 3 of the 4 negatives (all but the high-scoring clip 2)
    assert np.allclose(r["_auc_s_birds"], 0.0)         # vs the single other-species clip, which scores higher than both positives
    assert np.allclose(r["_delta"], -0.25) and np.allclose(r["_delta_birds"], -1.0)
    assert r["n_fail"] == K and r["n_fail_birds"] == K and (r["_delta_birds"] < r["_delta"]).all()


def test_cluster_bootstrap_is_deterministic_and_whole_clusters():
    g = np.array(["a", "a", "b", "b", "b", "c"])
    a, b = cluster_indices(g, np.random.default_rng(3), 5), cluster_indices(g, np.random.default_rng(3), 5)
    assert all((x == y).all() for x, y in zip(a, b))
    for idx in a:                                       # every resample is a union of whole clusters
        for grp in "abc":
            members = set(np.nonzero(g == grp)[0])
            assert members <= set(idx) or not (members & set(idx))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
