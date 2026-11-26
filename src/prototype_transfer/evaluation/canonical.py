"""Canonical evaluation (protocol v1): the ONE way every model is scored on the held-out test splits.

Input per model: the response file written by scripts/analysis/extract_prototype_activations.py (keys focal / soundscape) or by
scripts/analysis/diagnose_shift.py extract (keys focal_test / soundscape_test), run on data/HSN/splits/{focal,soundscape}_test.parquet.
--split val scores soundscape_val (10 sites) against focal_val instead, so development never touches the test sites.
--split cv --fold F scores the held-out sites of cross-validation fold F (soundscape_cv{F}_heldout, scripts/data/make_splits.py)
against focal_val; the model must have been trained without those sites (scripts/train/train_protopnet.py --cv-fold F), and
scripts/eval/cv_aggregate.py combines the folds. The split (and fold) that was scored is recorded in the JSON.
All models must be evaluated on identical clips (checked), and on exactly the clips of the requested split.

Protocol (fixed; changing it means a new PROTOCOL version):
  species      secondary = species with >= 5 positives in BOTH test sets (12); primary = >= 30 soundscape test positives (7)
  headline     soundscape macro AP on the primary species
  secondary    soundscape macro AUC (primary, 12), cmAP over the 12, focal macro AUC / AP (retention only)
  prototype    delta_k = AUC_soundscape - AUC_focal of prototype k's max activation as a detector of its species:
                 delta       soundscape negatives = every other soundscape clip (as in model_comparison.json)
                 delta_birds soundscape negatives = clips with another target species only (matches focal negatives,
                             which are other species' clips); removes empty-background clips from the negative set
  uncertainty  cluster bootstrap, B resamples, shared across models: soundscape by site (30), focal by recordist (110)
  comparison   paired differences and the plan's success criteria against --baseline (and --oracle)
Soundscape labels are used only here, for evaluation. The CLI is scripts/eval/canonical_eval.py (prints a table and saves to
the results store, see `save`); this module holds the computation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from prototype_transfer.data.splits import site_of
from prototype_transfer.evaluation import results
from prototype_transfer.evaluation.metrics import ap_cols, auc_cols, cluster_indices

logger = logging.getLogger(__name__)

PROTOCOL = "v1"
K = 5                    # prototypes per species
MIN_POS, PRIMARY_MIN_POS = 5, 30
FAIL, IMPROVE = -0.10, 0.02
GAP_FRACTION, FOCAL_TOL = 0.5, 0.01
SCALARS = ("sc_ap_primary", "sc_auc_primary", "sc_cmap_12", "sc_auc_12", "f_auc_primary", "f_ap_primary", "f_auc_12",
           "n_fail", "n_improve", "median_delta", "n_fail_birds", "median_delta_birds")


# ----------------------------------------------------------------------------- metrics
class Model:
    """Responses of one model on both test splits."""

    def __init__(self, name: str, path: Path) -> None:
        z = torch.load(path, map_location="cpu", weights_only=False)
        f, s = z.get("focal_test", z.get("focal")), z.get("soundscape_test", z.get("soundscape"))
        assert f is not None and s is not None, f"{path}: no focal / soundscape responses"
        self.name, self.species = name, list(z.get("species", z.get("species_list")))
        self.y_f, self.p_f, self.a_f = f["targets"].astype(bool), f["probs"], f["proto_max"]
        self.y_s, self.p_s, self.a_s = s["targets"].astype(bool), s["probs"], s["proto_max"]
        self.paths = {"focal": [Path(p).name for p in f["audio_path"]], "soundscape": [Path(p).name for p in s["audio_path"]]}


def metrics(m: Model, fi: np.ndarray, si: np.ndarray, cols: list[int], pri: list[int]) -> dict:
    """All scalar metrics of one model on the clip subsets fi (focal) and si (soundscape); `pri` indexes into `cols`."""
    yf, pf, af, ys, ps, as_ = m.y_f[fi], m.p_f[fi], m.a_f[fi], m.y_s[si], m.p_s[si], m.a_s[si]
    ap_s, auc_s = ap_cols(ps[:, cols], ys[:, cols]), auc_cols(ps[:, cols], ys[:, cols])
    ap_f, auc_f = ap_cols(pf[:, cols], yf[:, cols]), auc_cols(pf[:, cols], yf[:, cols])
    pa_f = auc_cols(af[:, cols, :].reshape(len(fi), -1), np.repeat(yf[:, cols], K, axis=1))   # [12 * 5]
    pa_s = auc_cols(as_[:, cols, :].reshape(len(si), -1), np.repeat(ys[:, cols], K, axis=1))
    delta = pa_s - pa_f
    n_other = ys.sum(1)
    pa_b = np.concatenate([  # soundscape negatives restricted to clips that contain another target species
        auc_cols(as_[(ys[:, c] | (n_other - ys[:, c] > 0))][:, c, :], np.repeat(ys[(ys[:, c] | (n_other - ys[:, c] > 0)), c][:, None], K, axis=1))
        for c in cols])
    delta_b = pa_b - pa_f
    out = {"sc_ap_primary": np.nanmean(ap_s[pri]), "sc_auc_primary": np.nanmean(auc_s[pri]), "sc_cmap_12": np.nanmean(ap_s),
           "sc_auc_12": np.nanmean(auc_s), "f_auc_primary": np.nanmean(auc_f[pri]), "f_ap_primary": np.nanmean(ap_f[pri]),
           "f_auc_12": np.nanmean(auc_f), "n_fail": int((delta < FAIL).sum()), "n_improve": int((delta > IMPROVE).sum()),
           "median_delta": float(np.nanmedian(delta)), "n_fail_birds": int((delta_b < FAIL).sum()),
           "median_delta_birds": float(np.nanmedian(delta_b))}
    return {k: float(v) for k, v in out.items()} | {"_delta": delta, "_delta_birds": delta_b, "_auc_f": pa_f, "_auc_s": pa_s, "_auc_s_birds": pa_b}


def ci(x: np.ndarray) -> list[float]:
    return [float(v) for v in np.nanpercentile(x, [2.5, 97.5])]


# ----------------------------------------------------------------------------- evaluation
def evaluate(models: list[Model], baseline: str, oracle: str | None, split: str, fold: int | None,
             focal_manifest: Path, soundscape_manifest: Path, n_boot: int, seed: int) -> tuple[dict, dict, dict]:
    """Score `models` under protocol v1.

    Args:
        models: responses of every model on the same clips; the first one defines species, clips and labels.
        baseline: name of the model the success criteria are measured against (must be in `models`).
        oracle: name of the supervised in-domain reference for the gap fraction; ignored with a warning if it is not in `models`.
        split: "test", "val" or "cv" (recorded in the output; the responses must be exactly this split's clips).
        fold: cross-validation fold (only with split == "cv").
        focal_manifest: gives the recordist of each focal clip (the focal bootstrap cluster).
        soundscape_manifest: the clips the soundscape responses must match.
        n_boot, seed: cluster-bootstrap resamples (shared across models) and their seed.

    Returns:
        (out, point, boot): `out` is the JSON-serialisable result, `point[name]` the point metrics (incl. per-prototype
        arrays) and `boot[name][metric]` the bootstrap distributions, both needed by `summary_table`.
    """
    ref = models[0]
    for m in models[1:]:  # identical species, labels and clips for every model
        assert m.species == ref.species and m.paths == ref.paths, f"{m.name} was evaluated on different clips than {ref.name}"
        assert (m.y_f == ref.y_f).all() and (m.y_s == ref.y_s).all(), f"{m.name}: targets differ"
    # the responses must be exactly the clips of the requested split (soundscape labels are only ever read for that split)
    want = set(pd.read_parquet(soundscape_manifest)["audio_path"].map(lambda q: Path(q).name))
    assert set(ref.paths["soundscape"]) == want, (
        f"--split {split}: the soundscape responses are not the clips of {soundscape_manifest} "
        f"({len(set(ref.paths['soundscape']) - want)} extra, {len(want - set(ref.paths['soundscape']))} missing)")
    names = [m.name for m in models]
    assert baseline in names, f"--baseline {baseline} is not among the models {names}"
    if oracle and oracle not in names:
        logger.warning(f"--oracle {oracle} is not among the models {names}: gap fraction and success criteria are omitted")
        oracle = None

    sp = ref.species
    npos_f, npos_s = ref.y_f.sum(0), ref.y_s.sum(0)
    cols = [c for c in range(len(sp)) if npos_f[c] >= MIN_POS and npos_s[c] >= MIN_POS]
    pri = [i for i, c in enumerate(cols) if npos_s[c] >= PRIMARY_MIN_POS]
    logger.info(f"{len(cols)} secondary species, {len(pri)} primary: {[sp[cols[i]] for i in pri]}")

    sites = site_of(pd.Series(ref.paths["soundscape"])).to_numpy()
    man = pd.read_parquet(focal_manifest)
    rec = dict(zip(man["audio_path"].map(lambda q: Path(q).name), man["recordist"].fillna("unknown")))
    recordists = np.array([rec[q] for q in ref.paths["focal"]])
    rng = np.random.default_rng(seed)
    boot_f, boot_s = cluster_indices(recordists, rng, n_boot), cluster_indices(sites, rng, n_boot)
    logger.info(f"{len(np.unique(sites))} test sites, {len(np.unique(recordists))} focal recordists, {n_boot} shared resamples")

    allf, alls = np.arange(len(ref.y_f)), np.arange(len(ref.y_s))
    point = {m.name: metrics(m, allf, alls, cols, pri) for m in models}
    boot = {m.name: {k: np.empty(n_boot) for k in SCALARS} for m in models}
    for b in range(n_boot):
        for m in models:
            r = metrics(m, boot_f[b], boot_s[b], cols, pri)
            for k in SCALARS:
                boot[m.name][k][b] = r[k]
        if (b + 1) % 100 == 0:
            logger.info(f"bootstrap {b + 1}/{n_boot}")

    out = {"protocol": PROTOCOL, "split": split, "fold": fold, "seed": seed, "n_boot": n_boot, "baseline": baseline, "oracle": oracle,
           "species": {"secondary": [sp[c] for c in cols], "primary": [sp[cols[i]] for i in pri],
                       "n_pos_soundscape": {sp[c]: int(npos_s[c]) for c in cols}, "n_pos_focal": {sp[c]: int(npos_f[c]) for c in cols}},
           "n_clips": {"focal": len(allf), "soundscape": len(alls)}, "n_clusters": {"recordists": int(len(np.unique(recordists))), "sites": int(len(np.unique(sites)))},
           "thresholds": {"fail": FAIL, "improve": IMPROVE, "gap_fraction": GAP_FRACTION, "focal_tolerance": FOCAL_TOL}, "models": {}}
    base_b, base_p = boot[baseline], point[baseline]
    for m in models:
        n = m.name
        e = {"point": {k: (int(point[n][k]) if k.startswith("n_") else point[n][k]) for k in SCALARS}, "ci95": {k: ci(boot[n][k]) for k in SCALARS},
             "prototypes": [{"species": sp[cols[j // K]], "k": j % K + 1, "auc_focal": float(point[n]["_auc_f"][j]),
                             "auc_soundscape": float(point[n]["_auc_s"][j]), "delta": float(point[n]["_delta"][j]),
                             "delta_birds": float(point[n]["_delta_birds"][j])} for j in range(len(cols) * K)]}
        if n != baseline:
            e["vs_baseline"] = {k: {"diff": point[n][k] - base_p[k], "ci95": ci(boot[n][k] - base_b[k])}
                                for k in ("sc_ap_primary", "sc_auc_primary", "sc_cmap_12", "f_auc_primary", "n_fail", "n_fail_birds")}
            if oracle and n != oracle:
                gap_pt = (point[n]["sc_ap_primary"] - base_p["sc_ap_primary"]) / (point[oracle]["sc_ap_primary"] - base_p["sc_ap_primary"])
                gap_b = (boot[n]["sc_ap_primary"] - base_b["sc_ap_primary"]) / (boot[oracle]["sc_ap_primary"] - base_b["sc_ap_primary"])
                e["gap_fraction"] = {"value": float(gap_pt), "ci95": ci(gap_b)}
                d = e["vs_baseline"]
                e["criteria"] = {
                    "1_ap_gap_closed": bool(gap_pt >= GAP_FRACTION and d["sc_ap_primary"]["ci95"][0] > 0),
                    "2_auc_not_below_baseline": bool(d["sc_auc_primary"]["diff"] >= 0),
                    "3_failing_prototypes_halved": bool(point[n]["n_fail"] <= base_p["n_fail"] // 2),
                    "4_focal_retained": bool(point[n]["f_auc_primary"] >= base_p["f_auc_primary"] - FOCAL_TOL)}
        out["models"][n] = e
    return out, point, boot


def tag_of(split: str, fold: int | None) -> str:
    """Name of the metrics section / summary suffix of an evaluation: test, val or cv<fold>."""
    return f"cv{fold}" if split == "cv" else split


def save(store: results.Store, out: dict, summary_name: str | None = None, entry: dict | None = None) -> dict:
    """Save an evaluation to the results store: the whole document as a summary (`canonical_eval[_val|_cv<f>]` unless named),
    and, for every model, its numbers as the run's `canonical_<split>` section (prototype table in its own file).
    `entry` adds index fields (e.g. legacy) to all of them."""
    tag = tag_of(out["split"], out["fold"])
    registry = results.read_registry(store)
    context = {k: out[k] for k in ("protocol", "split", "fold", "seed", "n_boot", "baseline", "oracle", "n_clips", "n_clusters")}
    for name, e in out["models"].items():
        metrics = {**context, **{k: v for k, v in e.items() if k != "prototypes"}}
        headline = {k: e["point"][k] for k in ("sc_ap_primary", "sc_auc_primary", "sc_cmap_12", "f_ap_primary", "f_auc_primary", "n_fail", "median_delta")}
        if "gap_fraction" in e:
            headline["gap_fraction"] = e["gap_fraction"]["value"]
            headline["criteria_passed"] = sum(e["criteria"].values())
        results.save_results(store, name, f"canonical_{tag}", metrics, kind=registry.get(name, {}).get("kind", "reference"),
                             headline=headline, files={f"canonical_{tag}_prototypes.json": e["prototypes"]},
                             entry={"protocol": out["protocol"], **(entry or {})})
    name = summary_name or "canonical_eval" + ("" if tag == "test" else f"_{tag}")
    return results.save_summary(store, name, out, headline={"n_models": len(out["models"])}, entry={"protocol": out["protocol"], **(entry or {})})


def summary_table(out: dict, point: dict, boot: dict) -> str:
    """The table printed after an evaluation (one row per model, then the success criteria of non-baseline models)."""
    fmt = lambda n, k, d=3: f"{point[n][k]:.{d}f} [{ci(boot[n][k])[0]:.{d}f},{ci(boot[n][k])[1]:.{d}f}]"
    fold_txt = "" if out["fold"] is None else f" fold {out['fold']}"
    lines = [f"\nCanonical evaluation {out['protocol']} on the {out['split']}{fold_txt} split: primary = {out['species']['primary']}; "
             f"{len(out['species']['secondary'])} secondary; {out['n_boot']} site/recordist-cluster resamples",
             f"{'model':18} {'sc AP primary':>22} {'sc AUC primary':>22} {'cmAP-12':>22} {'focal AUC prim':>22}  fail  fail(birds)"]
    for n in out["models"]:
        lines.append(f"{n:18} {fmt(n, 'sc_ap_primary'):>22} {fmt(n, 'sc_auc_primary'):>22} {fmt(n, 'sc_cmap_12'):>22} {fmt(n, 'f_auc_primary'):>22}  "
                     f"{int(point[n]['n_fail']):4d}  {int(point[n]['n_fail_birds']):4d}")
    for n, e in out["models"].items():
        if "criteria" in e:
            lines.append(f"{n}: gap closed {e['gap_fraction']['value']:.2f} {e['gap_fraction']['ci95']}; criteria {e['criteria']}")
    return "\n".join(lines)
