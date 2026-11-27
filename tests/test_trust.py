"""Plain-assert tests of the prototype-trust diagnostics (analysis/trust.py). Run: python tests/test_trust.py"""

import numpy as np
import pandas as pd
import torch
from torchaudio.functional import melscale_fbanks

from prototype_transfer.analysis import trust


def test_rank_one_fit_recovers_independent_witnesses():
    rng = np.random.default_rng(0)
    n, quality = 200_000, np.array([1.8, 1.4, 1.0, 0.7, 0.4])
    y = rng.random(n) < 0.2
    a = np.zeros((n, 1, 5))
    a[:, 0] = y[:, None] * quality + rng.standard_normal((n, 5))            # errors independent given y
    out = trust.agreement(a, y[:, None], np.arange(5.0)[None], ["x"])
    row = out["species"][0]
    assert abs(row["estimated_r"] - row["true_r"]) < 0.01 and abs(row["leftover_corr"]) < 0.01 and row["rank_estimate"] > 0.999


def test_a_shared_error_inflates_the_estimate_without_raising_the_residual():
    rng = np.random.default_rng(1)
    n = 100_000
    y = rng.random(n) < 0.2
    shared = rng.standard_normal(n)                                            # e.g. another bird that fools all five
    a = (0.6 * y + shared)[:, None, None] + rng.standard_normal((n, 1, 5)) * 0.7
    row = trust.agreement(a, y[:, None], np.arange(5.0)[None], ["x"])["species"][0]
    assert row["estimated_r"] > 2 * row["true_r"] and row["fit_residual"] < 0.02


def test_cell_window_matches_the_mel_filterbank():
    fb = melscale_fbanks(n_freqs=1025, f_min=0.0, f_max=16_000.0, n_mels=256, sample_rate=32_000)   # [1025, 256]
    hz = torch.linspace(0, 16_000, 1025)
    centre = hz[fb.argmax(dim=0)].numpy()                                      # centre frequency of every mel bin
    for row in (3, 8, 15):
        _, _, low, high = trust.cell_window(row, 0, widen=0)
        inside = centre[row * 16: (row + 1) * 16]
        assert low <= inside.min() + 40 and high >= inside.max() - 40, (row, low, high, inside.min(), inside.max())
    t0, t1, _, _ = trust.cell_window(0, 37, widen=0)
    assert abs(t1 - 5.0) < 1e-9 and abs((t1 - t0) - 5 / 38) < 1e-9
    assert trust.cell_window(5, 10, widen=1)[0] < trust.cell_window(5, 10, widen=0)[0]


def test_overlap_needs_time_and_frequency():
    window = (1.0, 1.2, 2000.0, 3000.0)
    box = lambda *b: np.array([b], dtype=float)
    assert trust.overlaps(window, box(1.1, 2.0, 2500, 4000))
    assert not trust.overlaps(window, box(1.3, 2.0, 2500, 4000))               # later in time
    assert not trust.overlaps(window, box(1.1, 2.0, 3500, 4000))               # higher in frequency
    assert not trust.overlaps(window, np.zeros((0, 4)))


def test_match_labels():
    assert trust.label_match(False, False) == "false"
    assert trust.label_match(True, True) == "correct" and trust.label_match(True, False) == "ambiguous"


def test_stratified_auc_compares_only_within_strata():
    margin = np.array([1.0, 0.0, 10.0, 9.0])
    is_false = np.array([True, False, True, False])
    stratum = np.array(["a", "a", "b", "b"])
    assert trust.stratified_auc(margin, is_false, stratum) == 1.0             # across strata it would be 0.75
    assert np.isnan(trust.stratified_auc(margin[:2], np.array([True, True]), stratum[:2]))


def _matches(contested_false: float, contested_correct: float, hit: float = 0.8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(9):
        for k in range(5):
            for i in range(100):
                false = i % 2 == 0
                contested = rng.random() < (contested_false if false else contested_correct)
                present = np.zeros(21, dtype=bool)
                present[1] = rng.random() < 0.3
                in_box = (not false) and rng.random() < hit
                rows.append({"species": f"s{s}", "prototype": k, "site": f"site{rng.integers(30)}", "own": rng.random(),
                             "margin": (0.05 if contested else -0.05) + rng.normal(0, 0.01), "rival_index": 1,
                             "rival_species": "r", "rival_present": bool(present[1]), "present_species": present,
                             "label": trust.label_match(not false, in_box), "in_own_box": in_box,
                             "chance_in_box": 0.3 if not false else np.nan})
    return pd.DataFrame(rows)


def test_criteria_pass_when_competition_targets_false_matches():
    out = trust.evaluate_competition(_matches(0.6, 0.1), n_boot=50)
    assert all(out["passed"][k] for k in ("c0", "c1", "c2", "c3")) and out["decision"].startswith("pursue")
    assert abs(out["point"]["c2"] - 0.5) < 0.05 and out["species_up"] == 9 and out["point"]["c3"] > 0.7


def test_criteria_fail_when_contest_is_unrelated_to_correctness():
    out = trust.evaluate_competition(_matches(0.4, 0.4), n_boot=50)
    assert out["passed"]["c1"] and not out["passed"]["c2"] and not out["passed"]["c3"]
    assert out["decision"].startswith("do not pursue")


def test_a_failed_localisation_check_makes_the_result_inconclusive():
    out = trust.evaluate_competition(_matches(0.6, 0.1, hit=0.32), n_boot=50)
    assert not out["passed"]["c0"] and out["decision"].startswith("inconclusive")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
