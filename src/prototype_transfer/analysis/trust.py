"""When can a prototype be trusted? Two diagnostics on the pretrained model (docs/TRUST_DIAGNOSTIC.md has the rules).

agreement    Can the reliability of a species' 5 prototypes be estimated without labels from how they agree? Under errors
             that are independent given presence y, corr(a_i, a_j) = r_i r_j (i != j), r_i = corr(a_i, y).
competition  At the grid cell where a prototype matches, does a prototype of another species match at least as strongly,
             more often when the match is false than when it is correct? Truth comes from time-frequency call boxes.

Pure functions (arrays in, numbers out); the CLI that loads data, runs the model and saves is
scripts/analysis/prototype_trust.py. Tests: tests/test_trust.py. Labels and boxes are used only to evaluate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

GRID_FREQ, GRID_TIME = 16, 38
CLIP_SECONDS, N_MELS, F_MAX = 5.0, 256, 16_000.0
MIN_POSITIVES, TOP_MATCHES = 30, 100
MIN_PER_CLASS = 10                                       # C2: species counted if it has this many false and correct matches
CRITERIA = {"c0_localisation": 0.15, "c1_contested_false": 0.20, "c2_difference": 0.10, "c2_species": 6,
            "c3_auc": 0.60, "c4_lift": 1.5}


# --------------------------------------------------------------------------------------------- agreement
def fit_rank_one(corr: np.ndarray, starts: int = 20, seed: int = 0) -> tuple[np.ndarray, float]:
    """r minimising sum_{i<j} (r_i r_j - corr_ij)^2 with |r_i| <= 1, signed so that sum r >= 0; and the RMS residual."""
    upper = np.triu_indices(len(corr), 1)
    rng = np.random.default_rng(seed)
    fits = [least_squares(lambda v: np.outer(v, v)[upper] - corr[upper], rng.uniform(0.1, 0.9, len(corr)), bounds=(-1, 1))
            for _ in range(starts)]
    best = min(fits, key=lambda f: f.cost)
    r = best.x if best.x.sum() >= 0 else -best.x
    return r, float(np.sqrt(np.mean(best.fun ** 2)))


def agreement(activations: np.ndarray, targets: np.ndarray, weights: np.ndarray, species: list[str],
              min_positives: int = MIN_POSITIVES) -> dict:
    """Label-free estimate of r per prototype against its true value, per species with enough positives.

    Args:
        activations: max prototype activations [N, 21, 5].
        targets: presence [N, 21] (used only to check the estimate).
        weights: the model's last-layer weights [21, 5], as a label-free alternative ranking.
    """
    upper = np.triu_indices(activations.shape[2], 1)
    rows, estimates, truths = [], [], []
    for s, name in enumerate(species):
        y = targets[:, s].astype(bool)
        if y.sum() < min_positives:
            continue
        a = activations[:, s]
        corr = np.corrcoef(a.T)
        true = np.array([np.corrcoef(a[:, k], y)[0, 1] for k in range(a.shape[1])])
        est, residual = fit_rank_one(corr)
        auc = np.array([roc_auc_score(y, a[:, k]) for k in range(a.shape[1])])
        estimates += list(est)
        truths += list(true)
        rows.append({"species": name, "positives": int(y.sum()), "prototype_corr": float(corr[upper].mean()),
                     "leftover_corr": float((corr - np.outer(true, true))[upper].mean()),
                     "estimated_r": float(est.mean()), "true_r": float(true.mean()), "fit_residual": residual,
                     "rank_estimate": float(spearmanr(est, auc).statistic),
                     "rank_weight": float(spearmanr(weights[s], auc).statistic)})
    return {"species": rows, "n_clips": int(len(activations)),
            "mean_estimated_r": float(np.mean(estimates)), "mean_true_r": float(np.mean(truths)),
            "median_rank_estimate": float(np.median([r["rank_estimate"] for r in rows])),
            "median_rank_weight": float(np.median([r["rank_weight"] for r in rows]))}


# --------------------------------------------------------------------------------------------- call boxes
def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=float) / 700.0)   # HTK, as torchaudio's MelScale default


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=float) / 2595.0) - 1.0)


def cell_window(row: int, col: int, widen: int = 1) -> tuple[float, float, float, float]:
    """(t_low, t_high [s], f_low, f_high [Hz]) of grid cell (row = frequency, 0 lowest; col = time), widened by
    `widen` cells on every side. Mel band edges are equally spaced on the mel scale from 0 to F_MAX."""
    dt = CLIP_SECONDS / GRID_TIME
    bins_per_row = N_MELS / GRID_FREQ
    mel_of_bin = lambda b: b * float(hz_to_mel(F_MAX)) / (N_MELS + 1)
    low, high = max(0, row - widen) * bins_per_row, min(GRID_FREQ, row + 1 + widen) * bins_per_row
    return ((col - widen) * dt, (col + 1 + widen) * dt, float(mel_to_hz(mel_of_bin(low))), float(mel_to_hz(mel_of_bin(high + 1))))


def overlaps(window: tuple[float, float, float, float], boxes: np.ndarray) -> bool:
    """Does the window intersect any box? boxes [n, 4] = (t_low, t_high, f_low, f_high), clip-relative seconds and Hz."""
    if len(boxes) == 0:
        return False
    t0, t1, f0, f1 = window
    return bool(((boxes[:, 0] < t1) & (boxes[:, 1] > t0) & (boxes[:, 2] < f1) & (boxes[:, 3] > f0)).any())


# --------------------------------------------------------------------------------------------- competition
def label_match(present: bool, in_own_box: bool) -> str:
    """present: the clip label or any box has the species; in_own_box: the match window overlaps such a box."""
    if not present:
        return "false"
    return "correct" if in_own_box else "ambiguous"


def stratified_auc(margin: np.ndarray, is_false: np.ndarray, stratum: np.ndarray) -> float:
    """P(margin of a false match > margin of a correct one) over pairs within the same stratum (ties count half),
    pooled over strata in proportion to their number of pairs. NaN without any pair."""
    wins = pairs = 0.0
    for s in np.unique(stratum):
        m = stratum == s
        n_false, n_true = int(is_false[m].sum()), int((~is_false[m]).sum())
        if n_false and n_true:
            wins += roc_auc_score(is_false[m], margin[m]) * n_false * n_true
            pairs += n_false * n_true
    return wins / pairs if pairs else float("nan")


def own_score_strata(matches: pd.DataFrame, n: int = 3) -> np.ndarray:
    """Stratum id = (prototype, third of its own-activation ranking among its matches)."""
    third = matches.groupby(["species", "prototype"])["own"].transform(
        lambda x: np.minimum((x.rank(method="first") - 1) * n // len(x), n - 1)).astype(int)
    return (matches["species"].astype(str) + "/" + matches["prototype"].astype(str) + "/" + third.astype(str)).to_numpy()


def _contrast(m: pd.DataFrame) -> dict[str, float]:
    """The resampled quantities of C0-C3 on a table of matches."""
    false, correct = m[m.label == "false"], m[m.label == "correct"]
    present = m[m.label != "false"]
    two = m[m.label != "ambiguous"]
    cf = float((false.margin >= 0).mean()) if len(false) else float("nan")
    cc = float((correct.margin >= 0).mean()) if len(correct) else float("nan")
    return {"c0": float(present.in_own_box.mean() - present.chance_in_box.mean()) if len(present) else float("nan"),
            "c1": cf, "contested_correct": cc, "c2": cf - cc,
            "c3": stratified_auc(two.margin.to_numpy(), (two.label == "false").to_numpy(), two.stratum.to_numpy()),
            "own_auc": stratified_auc(-two.own.to_numpy(), (two.label == "false").to_numpy(),
                                      (two.species.astype(str) + "/" + two.prototype.astype(str)).to_numpy())}


def evaluate_competition(matches: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """Apply criteria C0-C4 to the table of matches.

    Args:
        matches: one row per match with species, prototype, site, own, margin (rival - own), label
            (false | correct | ambiguous), in_own_box (bool), chance_in_box (probability of a hit with the prototype's
            match cell shuffled among its clips with the species), rival_species, rival_present (bool).
    """
    m = matches.copy()
    m["stratum"] = own_score_strata(m)
    point = _contrast(m)
    sites = np.array(sorted(m.site.unique()))
    rows_of = {s: np.flatnonzero(m.site.to_numpy() == s) for s in sites}
    rng = np.random.default_rng(seed)
    boot = pd.DataFrame([_contrast(m.iloc[np.concatenate([rows_of[s] for s in rng.choice(sites, len(sites))])])
                         for _ in range(n_boot)])
    ci = {k: [float(np.nanpercentile(boot[k], 2.5)), float(np.nanpercentile(boot[k], 97.5))] for k in boot}

    per_species = []
    for name, g in m.groupby("species", sort=False):
        n = g.label.value_counts()
        q = _contrast(g)
        per_species.append({"species": name, **{f"n_{k}": int(n.get(k, 0)) for k in ("false", "correct", "ambiguous")},
                            "contested_false": q["c1"], "contested_correct": q["contested_correct"], "difference": q["c2"],
                            "margin_auc": q["c3"],
                            "counted": bool(n.get("false", 0) >= MIN_PER_CLASS and n.get("correct", 0) >= MIN_PER_CLASS)})
    n_up = sum(r["counted"] and r["difference"] > 0 for r in per_species)

    false = m[m.label == "false"]
    contested = false[false.margin >= 0]
    # base rate: how often each contested match's rival species is present among the SAME species' false matches overall
    present_rate = {}
    for name, g in false.groupby("species"):
        cols = np.stack(g.present_species.to_numpy())                     # [n, 21] bool
        present_rate[name] = cols.mean(axis=0)
    base = float(np.mean([present_rate[r.species][r.rival_index] for r in contested.itertuples()])) if len(contested) else float("nan")
    seen = float(contested.rival_present.mean()) if len(contested) else float("nan")
    lift = seen / base if base and base > 0 else float("nan")

    passed = {"c0": bool(point["c0"] >= CRITERIA["c0_localisation"] and ci["c0"][0] > 0),
              "c1": bool(point["c1"] >= CRITERIA["c1_contested_false"]),
              "c2": bool(point["c2"] >= CRITERIA["c2_difference"] and ci["c2"][0] > 0 and n_up >= CRITERIA["c2_species"]),
              "c3": bool(point["c3"] >= CRITERIA["c3_auc"] and ci["c3"][0] > 0.5),
              "c4": bool(lift >= CRITERIA["c4_lift"])}
    if not passed["c0"]:
        decision = "inconclusive: match locations do not pass the localisation check"
    elif all(passed[k] for k in ("c1", "c2", "c3")):
        decision = "pursue: build and test a competition step"
    else:
        decision = "do not pursue competition among the 21 species in this form"
    return {"point": point, "ci95": ci, "species_up": int(n_up), "species_counted": int(sum(r["counted"] for r in per_species)),
            "c4": {"rival_present": seen, "base_rate": base, "lift": lift, "n_contested_false": int(len(contested))},
            "localisation": {"observed": float(m[m.label != "false"].in_own_box.mean()),
                             "chance": float(m[m.label != "false"].chance_in_box.mean())},
            "counts": {k: int(v) for k, v in m.label.value_counts().items()}, "per_species": per_species,
            "passed": passed, "decision": decision, "criteria": CRITERIA, "n_boot": n_boot, "n_sites": int(len(sites))}


# --------------------------------------------------------------------------------- checks added after the result
def pooled_pair_auc(value: np.ndarray, is_false: np.ndarray, codes: np.ndarray) -> float:
    """stratified_auc() computed from within-stratum average ranks (Mann-Whitney U), for integer stratum codes.
    Same number, fast enough to bootstrap several quantities."""
    rank = pd.Series(value).groupby(codes).rank().to_numpy()
    size = int(codes.max()) + 1
    n_false = np.bincount(codes, weights=is_false, minlength=size)
    n_true = np.bincount(codes, weights=~is_false, minlength=size)
    u = np.bincount(codes, weights=rank * is_false, minlength=size) - n_false * (n_false + 1) / 2
    pairs = float((n_false * n_true).sum())
    return float(u[n_false * n_true > 0].sum() / pairs) if pairs else float("nan")


def location_controls(matches: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """Two questions the pre-set criteria do not answer, asked AFTER seeing that they passed (so exploratory; they do not
    change the decision). Does the signal need the rival to be at the SAME location, or is another species anywhere in
    the clip as informative? And how much does the rival add to the prototype's own score within the same strata?

    Args:
        matches: as for evaluate_competition, plus rival_anywhere (largest activation of any other species' prototype
            anywhere in the clip).
    """
    m = matches.copy()
    m["stratum"] = own_score_strata(m)
    m = m[m.label != "ambiguous"].reset_index(drop=True)
    codes = pd.factorize(m.stratum)[0]
    is_false = (m.label == "false").to_numpy()
    same = (m.margin + m.own).to_numpy()
    values = {"margin_same_place": m.margin.to_numpy(), "margin_anywhere": m.rival_anywhere.to_numpy() - m.own.to_numpy(),
              "rival_same_place": same, "rival_anywhere": m.rival_anywhere.to_numpy(), "own_low": -m.own.to_numpy()}
    differences = (("margin_same_place", "margin_anywhere"), ("rival_same_place", "rival_anywhere"),
                   ("margin_same_place", "own_low"))

    def quantities(idx: np.ndarray) -> dict[str, float]:
        out = {k: pooled_pair_auc(v[idx], is_false[idx], codes[idx]) for k, v in values.items()}
        return out | {f"{a} - {b}": out[a] - out[b] for a, b in differences}

    point = quantities(np.arange(len(m)))
    sites = np.array(sorted(m.site.unique()))
    rows_of = {s: np.flatnonzero(m.site.to_numpy() == s) for s in sites}
    rng = np.random.default_rng(seed)
    boot = pd.DataFrame([quantities(np.concatenate([rows_of[s] for s in rng.choice(sites, len(sites))])) for _ in range(n_boot)])
    ci = {k: [float(np.nanpercentile(boot[k], 2.5)), float(np.nanpercentile(boot[k], 97.5))] for k in boot}
    contested = {name: {lab: float((values[name][is_false == flag] >= 0).mean()) for lab, flag in (("false", True), ("correct", False))}
                 for name in ("margin_same_place", "margin_anywhere")}
    return {"auc": point, "ci95": ci, "contested": contested, "n_matches": int(len(m)), "n_boot": n_boot,
            "added_after_result": True}
