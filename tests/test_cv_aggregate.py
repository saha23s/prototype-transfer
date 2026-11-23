"""Plain-assert tests for scripts/cv_aggregate.py.  Run: python tests/test_cv_aggregate.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cv_aggregate import KEYS, aggregate  # noqa: E402


def _eval(fold: int, run_ap: float, base_ap: float) -> dict:
    point = lambda ap: {k: (ap if k == "sc_ap_primary" else 0.5) for k in KEYS}
    return {"split": "cv", "fold": fold, "models": {
        f"r-cv{fold}": {"point": point(run_ap), "gap_fraction": {"value": 0.5}, "criteria": {"1": fold != 2, "2": True}},
        "pretrained-xcl": {"point": point(base_ap)}}}


def test_mean_sd_and_paired_difference():
    r = aggregate([_eval(0, 0.7, 0.6), _eval(1, 0.8, 0.6), _eval(2, 0.9, 0.7)], "r")
    assert np.isclose(r["run"]["sc_ap_primary"]["mean"], 0.8) and np.isclose(r["run"]["sc_ap_primary"]["sd"], 0.1)
    assert np.allclose(r["difference"]["sc_ap_primary"]["folds"], [0.1, 0.2, 0.2])        # paired per fold, not mean - mean
    assert r["criteria_folds_passed"] == {"1": 2, "2": 3} and np.isclose(r["gap_fraction"]["mean"], 0.5)


def test_wrong_fold_or_split_is_refused():
    good = [_eval(0, 0.7, 0.6), _eval(1, 0.8, 0.6)]
    for bad in ({**good[1], "fold": 0}, {**good[1], "split": "test"}):
        try:
            aggregate([good[0], bad], "r")
            raise AssertionError("expected a refusal")
        except AssertionError as e:
            assert "canonical_eval was run with" in str(e)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
