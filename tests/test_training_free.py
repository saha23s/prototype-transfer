"""Plain-assert tests for src/prototype_transfer/method/training_free.py.  Run: python tests/test_training_free.py"""

import numpy as np

from prototype_transfer.method import training_free as tf


def _acts(n: int, scale: float, seed: int, zero_frac: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = (rng.random((n, 3, 5)) * scale + 1e-3).astype(np.float32)
    a[rng.random(a.shape) < zero_frac] = 0.0
    return a


def test_class_probs_matches_manual_value():
    a, w, b = np.full((1, 1, 2), 0.5), np.array([[1.0, 2.0]]), np.array([-1.5])
    assert np.isclose(tf.class_probs(a, w, b)[0, 0], 0.5)   # logit = 0.5 + 1.0 - 1.5 = 0


def test_quantile_map_recovers_a_rescaling_and_is_the_identity_on_equal_domains():
    ref = _acts(4000, 0.8, 0)
    same = tf.apply_quantile_map(ref, *tf.fit_quantile_map(ref, ref, 1001))
    assert np.allclose(same, ref, atol=1e-3)
    shrunk = 0.3 * ref   # the target is the reference with shrunken activations
    back = tf.apply_quantile_map(shrunk, *tf.fit_quantile_map(shrunk, ref, 1001))
    assert np.allclose(back, ref, atol=1e-3)


def test_quantile_map_keeps_zeros_and_never_reverses_the_order():
    target, ref = _acts(3000, 0.4, 1, zero_frac=0.4), _acts(2000, 0.9, 2, zero_frac=0.1)
    knots = tf.fit_quantile_map(target, ref, 101)
    x = _acts(500, 0.6, 3, zero_frac=0.3)
    y = tf.apply_quantile_map(x, *knots)
    assert (y[x == 0] == 0).all() and (y >= 0).all()
    for s in range(3):
        for k in range(5):
            order = np.argsort(x[:, s, k], kind="stable")
            assert (np.diff(y[order, s, k]) >= -1e-7).all()


def test_reweight_is_the_identity_on_equal_domains_and_keeps_species_totals():
    w = np.random.default_rng(4).random((3, 5)).astype(np.float32) + 0.3
    ref = _acts(1000, 0.8, 5)
    assert np.allclose(tf.reweight(w, ref, ref, 0.05), w, rtol=1e-6)
    out = tf.reweight(w, _acts(1000, 0.5, 6), ref, 0.05)
    assert np.allclose(out.sum(1), w.sum(1), rtol=1e-6) and (out >= 0).all()


def test_reweight_moves_weight_away_from_a_flattened_prototype():
    w = np.ones((3, 5), dtype=np.float32)
    ref = _acts(1000, 0.8, 7)
    target = ref.copy()
    target[:, 0, 0] = 0.5 * target[:, 0, 0]   # this prototype's responses stand out half as much on the target
    out = tf.reweight(w, target, ref, 0.05)
    assert out[0, 0] < out[0, 1:].min() and np.allclose(out[1:], 1.0)


def test_contrast_matches_manual_value():
    a = np.arange(20, dtype=float).reshape(20, 1, 1)
    assert np.isclose(tf.contrast(a, 0.10)[0, 0], 18.5 - 9.5)   # mean of {18, 19} - median


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
