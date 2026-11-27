"""Everything that goes into the dashboard's data bundle (dashboard/data.json, data.js), computed from the model's responses and the results store.

Used by scripts/report/build_dashboard_data.py. The bundle holds, per analysed species, the prototype statistics, activation CDFs, cross-species
confusion and top-activating spectrogram patches; the findings (text and evidence computed from the same numbers); and the per-model
comparison blocks. `build_bundle` assembles it; the pieces are separate functions so they can be tested on small inputs.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from matplotlib.image import imsave

from prototype_transfer.analysis.prototype_responses import (extract_prototype_responses, grid_cell_to_spec_slice,
                                                             grid_row_centers_khz)
from prototype_transfer.data.datasets import HSNInferenceDataset
from prototype_transfer.data.preprocessing import AudioPreprocessor
from prototype_transfer.evaluation import results
from prototype_transfer.evaluation.canonical import FAIL, IMPROVE
from prototype_transfer.models.classifier import MODEL_REVISION
from prototype_transfer.reporting.charts import score, signed

logger = logging.getLogger(__name__)

DOMAINS = ("focal", "soundscape")
QUANTILES = (10, 25, 50, 75, 90)
SILENT_SHARE = 0.25              # a prototype silent on more than this share of soundscape positives is reported as going silent
HIGH_KHZ, LOW_KHZ = 1.0, 0.5     # a peak that moves from above HIGH_KHZ on focal clips to below LOW_KHZ on soundscapes

# Observations from visually inspecting the patch galleries (not computable from the JSON). They describe the top-3 crops of
# the NON-TEST bundle; another set of clips needs another inspection. {auc3}-style fields are filled from the statistics.
CURATED_FINDINGS = [
    {
        "title": "Some prototypes match the same acoustic part in both domains",
        "text": "whcspa P3 (a pure-tone note) and P4 (a series of falling sweeps) mostly fire on the same kind of structure in "
                "focal and soundscape clips, with soundscape AUC {auc3:.0f} and {auc4:.0f} (out of 100). P5's best soundscape matches also "
                "resemble its focal ones, but it is usually silent: median soundscape activation {act5:.0f} on clips with the "
                "species, and AUC {auc5:.0f}.",
        "species": "whcspa", "prototype": 3,
        "evidence": {"kind": "patches", "species": "whcspa", "prototypes": [3, 4, 5],
                     "method": "Visual inspection of the top-3 activating patches per domain (crop = peak cell "
                               "± 2 grid cells), checked against each prototype's AUC and median activation "
                               "on all positive clips (table)."},
    },
    {
        "title": "Top matches can look different across domains",
        "text": "Two of whcspa P2's three top focal matches are sharp rising sweeps; all three top soundscape matches are a "
                "buzz-like band near 6 kHz, which resembles the buzz in the third focal match. The images cannot show whether "
                "these are song or noise. Its soundscape AUC is {auc2:.0f}.",
        "species": "whcspa", "prototype": 2,
        "evidence": {"kind": "patches", "species": "whcspa", "prototypes": [2],
                     "method": "Visual inspection of the top-3 activating patches per domain, without listening to the audio. "
                               "Three examples per domain say nothing about how common each kind of match is."},
    },
]


def fill_curated(curated: list[dict], species: list[dict]) -> list[dict]:
    """The visual findings with their {auc<k>} / {act<k>} fields taken from the inspected species' statistics."""
    out = []
    for f in curated:
        s = next(x for x in species if x["code"] == f["species"])
        values = {f"auc{p['k']}": 100 * p["soundscape"]["auc"] for p in s["prototypes"]}   # shown out of 100
        values |= {f"act{p['k']}": 100 * p["soundscape"]["act_q"][QUANTILES.index(50)] for p in s["prototypes"]}
        out.append({**f, "text": f["text"].format(**values)})
    return out


CAVEATS = [
    "Focal clips are Xeno-Canto recordings, which are part of the model's XCL training data: "
    "focal numbers are an in-domain reference, not a held-out score.",
    "Species with fewer than 30 soundscape positives have noisy soundscape statistics.",
    "Prototype AUC uses the prototype's max activation as a detector of its own species "
    "(focal negatives = other species' focal clips; soundscape negatives = all other soundscape clips).",
]


def r(x, nd=4):
    return None if x is None else round(float(x), nd)


def has_auc(p: dict) -> bool:
    return p["focal"]["auc"] is not None and p["soundscape"]["auc"] is not None


CDF_GRID = np.round(np.arange(0.0, 0.7 + 1e-9, 0.01), 4)


def cdf(vals: np.ndarray, grid: np.ndarray = CDF_GRID) -> list:
    return [r(v, 3) for v in np.searchsorted(np.sort(vals), grid, side="right") / max(1, len(vals))]


def species_activation_cdfs(data: dict, c: int) -> dict:
    """Per-species CDFs of max activation, clips with / without species c, both domains.
    Views: 'all' = 5 prototypes pooled, 'max' = strongest prototype per clip, '1'..'5' = single prototype."""
    out = {"series": {}, "medians": {}, "n": {}}
    for dom in DOMAINS:
        d = data[dom]
        y = d["targets"][:, c]
        act = d["proto_max"][:, c, :]  # (N, 5)
        for name, mask in (("pos", y), ("neg", ~y)):
            key = f"{dom}_{name}"
            a = act[mask]
            out["n"][key] = int(mask.sum())
            views = {"all": a.ravel(), "max": a.max(axis=1), **{str(k + 1): a[:, k] for k in range(a.shape[1])}}
            for v, vals in views.items():
                out["series"].setdefault(v, {})[key] = cdf(vals)
                out["medians"].setdefault(v, {})[key] = r(np.median(vals), 4) if len(vals) else None
    return out


CONFUSION_TOP_N = 20


def confusion_stats(data: dict, c: int, sp_list: list[str], top_n: int = CONFUSION_TOP_N) -> list[dict]:
    """For each prototype of species c: among its top-N activating soundscape clips WITHOUT c, how many
    contain another annotated species, vs the base rate over all soundscape clips without c."""
    d = data["soundscape"]
    T = d["targets"]                                    # (N, 21) bool, all HSN species
    neg = np.nonzero(~T[:, c])[0]
    base_any = float(T[neg].any(axis=1).mean())
    base_sp = T[neg].mean(axis=0)                       # per-species base rate among negatives
    out = []
    for k in range(d["proto_max"].shape[2]):
        top = neg[np.argsort(-d["proto_max"][neg, c, k])][:top_n]
        counts = T[top].sum(axis=0)
        j = int(np.argmax(counts))
        out.append({
            "n": int(len(top)),
            "frac_other": r(T[top].any(axis=1).mean(), 3), "base_other": r(base_any, 3),
            "top_species": sp_list[j] if counts[j] > 0 else None, "top_count": int(counts[j]),
            "top_base": r(base_sp[j], 3),
        })
    return out


def activation_cdfs(data: dict, species: list[dict], sp_list: list[str], step: float = 0.01) -> dict:
    """Pooled CDFs of per-prototype max activation over the analysed species' prototypes,
    on clips with (positives) and without (negatives) the species, in both domains."""
    grid = np.round(np.arange(0.0, 0.7 + 1e-9, step), 4)
    out = {"x": grid.tolist(), "series": {}, "medians": {}, "n": {}}
    cols = [sp_list.index(s["code"]) for s in species]
    for dom in DOMAINS:
        d = data[dom]
        y = d["targets"][:, cols]                 # (N, S)
        act = d["proto_max"][:, cols, :]          # (N, S, 5)
        for name, mask in (("pos", y), ("neg", ~y)):
            vals = act[mask].ravel()              # every (clip, species, prototype) entry
            key = f"{dom}_{name}"
            out["series"][key] = [r(v, 4) for v in np.searchsorted(np.sort(vals), grid, side="right") / len(vals)]
            out["medians"][key] = r(np.median(vals), 4)
            out["n"][key] = int(len(vals))
    return out


def computed_findings(species: list[dict], dists: dict, n_soundscape: int) -> list[dict]:
    """Findings whose text AND evidence are computed from the same numbers."""
    protos = [(s["code"], p) for s in species for p in s["prototypes"] if has_auc(p)]
    dauc = {(sp, p["k"]): p["soundscape"]["auc"] - p["focal"]["auc"] for sp, p in protos}
    out = []

    # 1. Species-level detection
    drops = sorted(((s["code"], s["class_auc"]["focal"] - s["class_auc"]["soundscape"]) for s in species),
                   key=lambda t: -t[1])
    worst_sp, worst_drop = drops[0]
    mean_f = np.mean([s["class_auc"]["focal"] for s in species])
    mean_s = np.mean([s["class_auc"]["soundscape"] for s in species])
    n_above_09 = sum(s["class_auc"]["soundscape"] >= 0.9 for s in species)
    out.append({
        "title": "Species-level detection mostly survives",
        "text": f"Mean class AUC {score(mean_f)} on focal vs {score(mean_s)} on soundscape (out of 100); {n_above_09} of "
                f"{len(species)} species keep AUC ≥ 90 on soundscapes. {worst_sp} collapses "
                f"(AUC drops by {score(worst_drop, 0)} points).",
        "species": worst_sp, "prototype": None,
        "evidence": {
            "kind": "class_auc_drop", "highlight": [worst_sp],
            "stats": [["Mean class AUC, focal", score(mean_f)], ["Mean class AUC, soundscape", score(mean_s)],
                      ["Species with soundscape AUC ≥ 90", f"{n_above_09} / {len(species)}"],
                      ["Largest drop (points)", f"{worst_sp}: {score(worst_drop)}"],
                      ["Median drop (points)", score(np.median([d for _, d in drops]))]],
            "method": "Class ROC-AUC of the model's sigmoid output for the species, on the focal clips "
                      f"(negatives = other species) and the {n_soundscape:,} soundscape clips.",
        },
    })

    # 2. Prototype-level unevenness
    n_drop = sum(v < -0.1 for v in dauc.values())
    n_gain = sum(v > 0.02 for v in dauc.values())
    ranges = {}
    for sp, p in protos:
        ranges.setdefault(sp, []).append(dauc[(sp, p["k"])])
    spread = {sp: max(v) - min(v) for sp, v in ranges.items()}
    n_spread = sum(v >= 0.15 for v in spread.values())
    out.append({
        "title": "Prototypes transfer unevenly",
        "text": f"{n_drop} of {len(protos)} prototypes lose more than 10 points of detection AUC on soundscapes while "
                f"{n_gain} improve; in {n_spread} of {len(spread)} species the best- and worst-transferring "
                f"prototypes differ by at least 15 points of AUC.",
        "species": None, "prototype": None,
        "evidence": {
            "kind": "delta_auc", "threshold": -0.1,
            "spread": {sp: r(v, 3) for sp, v in spread.items()},
            "stats": [["Prototypes with ΔAUC below −10 points", f"{n_drop} / {len(protos)}"],
                      ["Prototypes with ΔAUC above +2 points", f"{n_gain} / {len(protos)}"],
                      ["Median ΔAUC (points)", signed(np.median(list(dauc.values())))],
                      ["Species with within-species spread ≥ 15 points", f"{n_spread} / {len(spread)}"]],
            "method": "ΔAUC = soundscape − focal ROC-AUC of each prototype's max activation as a detector of its "
                      "own species. Spread = max − min ΔAUC among a species' prototypes.",
        },
    })

    # 3. Distributional: activations shrink, ranking survives
    m = dists["medians"]
    ratios = [p["soundscape"]["act_q"][2] / p["focal"]["act_q"][2] for _, p in protos
              if p["focal"]["act_q"][2] and p["focal"]["act_q"][2] > 0.05]
    ratio = float(np.median(ratios))
    med_sauc = float(np.median([p["soundscape"]["auc"] for _, p in protos]))
    x = np.array(dists["x"])
    frac_neg_low = dists["series"]["soundscape_neg"][int(np.searchsorted(x, 0.05))]
    frac_pos_low = dists["series"]["soundscape_pos"][int(np.searchsorted(x, 0.05))]
    out.append({
        "title": "Activations shrink, ranking survives",
        "text": f"On clips with the species, the median soundscape activation is {ratio:.0%} of the focal one "
                f"(pooled medians {score(m['focal_pos'], 0)} → {score(m['soundscape_pos'], 0)} out of 100). On clips without it, "
                f"activations stay near zero ({frac_neg_low:.0%} are 5 or less), so positives still rank above "
                f"negatives: median prototype AUC on soundscapes is {score(med_sauc, 0)}.",
        "species": None, "prototype": None,
        "evidence": {
            "kind": "activation_cdf", "cdf": dists,
            "stats": [["Median activation, focal positives", score(m['focal_pos'])],
                      ["Median activation, soundscape positives", score(m['soundscape_pos'])],
                      ["Median activation, focal negatives", score(m['focal_neg'])],
                      ["Median activation, soundscape negatives", score(m['soundscape_neg'])],
                      ["Soundscape negatives at 5 or less", f"{frac_neg_low:.0%}"],
                      ["Soundscape positives at 5 or less", f"{frac_pos_low:.0%}"],
                      ["Median per-prototype ratio (soundscape / focal median)", f"{ratio:.0%}"],
                      ["Median prototype AUC, soundscape", score(med_sauc)]],
            "method": "Each entry is one prototype's max activation on one clip, pooled over the 70 prototypes of the "
                      "analysed species. Positives = clips labelled with the prototype's species. "
                      f"n = {', '.join(f'{k} {v:,}' for k, v in dists['n'].items())}.",
        },
    })

    # 4. Evidence shares: broadly stable, but the top prototype often changes
    pairs = [(sp, p["k"], p["focal"]["logit_share"], p["soundscape"]["logit_share"]) for sp, p in protos]
    fs, ss = np.array([t[2] for t in pairs]), np.array([t[3] for t in pairs])
    corr = float(np.corrcoef(fs, ss)[0, 1])
    mad = float(np.mean(np.abs(ss - fs)))
    top_same = sum(
        max(s["prototypes"], key=lambda p: p["focal"]["logit_share"])["k"]
        == max(s["prototypes"], key=lambda p: p["soundscape"]["logit_share"])["k"] for s in species)
    shifts = sorted(pairs, key=lambda t: -abs(t[3] - t[2]))[:3]
    out.append({
        "title": "Evidence shares are stable, but the top prototype often changes",
        "text": f"Each prototype's share of the class evidence correlates across domains (r = {corr:.2f}, mean "
                f"change {mad * 100:.1f} points), yet the top prototype is the same in only {top_same} of "
                f"{len(species)} species. Largest shifts: "
                + ", ".join(f"{sp} P{k} {a:.0%}→{b:.0%}" for sp, k, a, b in shifts) + ".",
        "species": shifts[0][0], "prototype": shifts[0][1],
        "evidence": {
            "kind": "share_scatter", "highlight": [[sp, k] for sp, k, _, _ in shifts],
            "stats": [["Correlation of shares (focal vs soundscape)", f"{corr:.2f}"],
                      ["Mean |share change|", f"{mad * 100:.1f} points"],
                      ["Species with the same top prototype", f"{top_same} / {len(species)}"]],
            "method": "Share = mean over positive clips of w_k·a_k / Σ_j w_j·a_j (w = non-negative last-layer "
                      "weight, a = max activation): the fraction of the class logit (minus bias) from prototype k.",
        },
    })

    # 5. Silent prototypes
    silent = [(sp, p, p["soundscape"]["silent"]) for sp, p in protos if p["soundscape"]["silent"] > SILENT_SHARE]
    failing = {k for k, v in dauc.items() if v < FAIL}
    if silent and failing:
        n_fail_silent = sum((sp, p["k"]) in failing for sp, p, _ in silent)
        n_ok = len(protos) - len(failing)
        half = sum(p["soundscape"]["silent"] > 0.5 for sp, p in protos if (sp, p["k"]) in failing)
        med = {name: float(np.median([p["soundscape"]["silent"] for sp, p in protos if ((sp, p["k"]) in failing) == f]))
               for name, f in (("failing", True), ("other", False))}
        worst = sorted(silent, key=lambda t: -t[2])[:3]
        moved = [f"{sp} P{p['k']}" for sp, p in protos
                 if p["soundscape"]["peak_khz"] is not None and p["soundscape"]["peak_khz"] < LOW_KHZ
                 and p["focal"]["peak_khz"] is not None and p["focal"]["peak_khz"] > HIGH_KHZ]
        out.append({
            "title": "Some failing prototypes simply go silent",
            "text": f"{n_fail_silent} of {len(failing)} failing prototypes (ΔAUC below −10 points) have zero activation on more than "
                    f"{SILENT_SHARE:.0%} of the soundscape clips that contain their species, against "
                    f"{len(silent) - n_fail_silent} of the other {n_ok}. Only {half} failing prototypes are silent on more than "
                    "half, so silence accounts for part of the failures, not most. Counting only clips where a prototype fires, "
                    f"the number whose typical peak moves from above {HIGH_KHZ:g} kHz on focal clips to below {LOW_KHZ:g} kHz on "
                    f"soundscapes is {len(moved)} of {len(protos)}" + (f" ({', '.join(moved)})." if moved else "."),
            "species": worst[0][0], "prototype": worst[0][1]["k"],
            "evidence": {
                "kind": "silent_scatter", "highlight": [[sp, p["k"]] for sp, p, _ in worst],
                "stats": [[f"Failing prototypes silent on > {SILENT_SHARE:.0%} of soundscape positives", f"{n_fail_silent} / {len(failing)}"],
                          [f"Other prototypes silent on > {SILENT_SHARE:.0%}", f"{len(silent) - n_fail_silent} / {n_ok}"],
                          ["Median silent share, failing", f"{med['failing']:.0%}"],
                          ["Median silent share, others", f"{med['other']:.0%}"]]
                         + [[f"{sp} P{p['k']} silent on positives, focal → soundscape",
                             f"{p['focal']['silent']:.0%} → {v:.0%}"] for sp, p, v in worst],
                "method": "Silent = the prototype's max activation on a clip is exactly zero (activations pass through a ReLU). "
                          "Shares are over clips labelled with the species. The link with failing is partly built in: a silent "
                          "positive ties with every silent negative, which lowers AUC. Peak frequency is taken only over clips "
                          "where the prototype is active.",
            },
        })
    # 6. Cross-species confusion: top false matches are enriched for other (co-occurring) species
    rows = [(sp["code"], k + 1, cf) for sp in species for k, cf in enumerate(sp["confusion"])]
    enriched = [t for t in rows if t[2]["frac_other"] > t[2]["base_other"] + 0.1]
    pairs = [(sp, k, cf) for sp, k, cf in rows
             if cf["top_species"] and cf["top_count"] >= 0.3 * cf["n"] and cf["top_base"] > 0
             and (cf["top_count"] / cf["n"]) / cf["top_base"] >= 2]
    pairs.sort(key=lambda t: -(t[2]["top_count"] / t[2]["n"]) / t[2]["top_base"])
    med_top = float(np.median([cf["frac_other"] for _, _, cf in rows]))
    med_base = float(np.median([cf["base_other"] for _, _, cf in rows]))
    if pairs:
        sp0, k0, cf0 = pairs[0]
        out.append({
            "title": "False matches are often other birds, not noise",
            "text": f"Among each prototype's top {CONFUSION_TOP_N} soundscape false matches, a median of "
                    f"{med_top:.0%} contain another annotated species, vs {med_base:.0%} of all clips without the "
                    f"species; {len(enriched)} of {len(rows)} prototypes are enriched by > 10 points. "
                    f"{len(pairs)} prototypes concentrate on one other species, e.g. {sp0} P{k0} → "
                    f"{cf0['top_species']} ({cf0['top_count']}/{cf0['n']} top false matches vs a "
                    f"{cf0['top_base']:.0%} base rate).",
            "species": sp0, "prototype": k0,
            "evidence": {
                "kind": "confusion", "top_n": CONFUSION_TOP_N,
                "pairs": [{"species": sp, "proto": k, "other": cf["top_species"], "frac": r(cf["top_count"] / cf["n"], 3),
                           "base": cf["top_base"]} for sp, k, cf in pairs[:14]],
                "stats": [[f"Median share of top-{CONFUSION_TOP_N} false matches with another species", f"{med_top:.0%}"],
                          ["Median base rate (all clips without the species)", f"{med_base:.0%}"],
                          ["Prototypes enriched by > 10 points", f"{len(enriched)} / {len(rows)}"],
                          ["Prototypes concentrating on one species (≥ 30% of top matches, ≥ 2× base)", f"{len(pairs)} / {len(rows)}"]],
                "method": f"For each prototype, its {CONFUSION_TOP_N} highest-activating soundscape clips not labelled with "
                          "its species. 'Contains another species' uses the BirdSet HSN multi-label annotations (21 "
                          "species). Base rate = fraction of all soundscape clips without the species that contain the "
                          "other species (or any species). Caveats: some top 'false' matches may be unannotated calls of the "
                          "species itself, and any bird vocalisation resembles a prototype more than silence does, so "
                          "enrichment for 'any species' is partly expected; concentration on one specific species is "
                          "the stronger signal of confusion.",
            },
        })
    return out


# Patch groups: (key, domain, use clips WITH the species?)
PATCH_GROUPS = (("focal", "focal", True), ("soundscape", "soundscape", True), ("soundscape_neg", "soundscape", False))


def top_clips(d: dict, c: int, k: int, positive: bool, top: int) -> np.ndarray:
    """Indices of the `top` clips with (positive) or without species c on which its prototype k is most active."""
    idx = np.nonzero(d["targets"][:, c] if positive else ~d["targets"][:, c])[0]
    return idx[np.argsort(-d["proto_max"][idx, c, k])][:top]


def export_patches(data: dict, sp: str, c: int, out_dir: Path, pre: AudioPreprocessor,
                   cache: dict, top: int) -> dict:
    """Top-activating spectrogram crops per prototype for: focal and soundscape clips containing the
    species, and soundscape clips NOT containing it (what the prototype fires on by mistake)."""
    row_khz = grid_row_centers_khz()
    patches: dict[str, dict] = {}
    for k in range(5):
        patches[str(k + 1)] = {}
        for key, dom, positive in PATCH_GROUPS:
            d = data[dom]
            order = top_clips(d, c, k, positive, top)
            items = []
            for rank, i in enumerate(order):
                path = d["audio_path"][i]
                if path not in cache:
                    cache[path] = pre.process_file(path, is_soundscape=dom == "soundscape").squeeze().numpy()
                spec = cache[path]
                row, col = int(d["peak_freq"][i, c, k]), int(d["peak_time"][i, c, k])
                assert col >= 0, f"no peak position for {path}: run fill_peak_time first"
                fs, ts = grid_cell_to_spec_slice(row, col, n_frames=spec.shape[1])
                name = f"{sp}_P{k + 1}_{key}_{rank + 1}.png"
                imsave(out_dir / name, spec[fs, ts], cmap="magma", origin="lower")
                item = {"src": f"patches/{name}", "act": r(d["proto_max"][i, c, k], 3),
                        "khz": r(row_khz[row], 2), "clip": Path(path).name}
                if not positive:
                    item["labels"] = d["species"][i]  # species annotated in that clip ("background" if none)
                items.append(item)
            patches[str(k + 1)][key] = items
    return patches


# ------------------------------------------------------------------------------------------- clips outside the test split
MATCH_TOL = 1e-3   # largest difference in a max activation between two extractions of the same clip


def _stack(parts: list[dict], keys: tuple[str, ...]) -> dict:
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["audio_path"] = [q for p in parts for q in p["audio_path"]]
    return out


def _check_clips(got: list[str], splits_dir: Path, manifests: tuple[str, ...]) -> None:
    want = sorted(Path(q).name for m in manifests for q in pd.read_parquet(splits_dir / f"{m}.parquet")["audio_path"])
    assert sorted(Path(q).name for q in got) == want, f"cached responses are not exactly the clips of {manifests}"


def non_test_payload(store: results.Store, splits_dir: Path, n_folds: int = 3) -> dict:
    """The pretrained model's responses on every clip outside the test split, assembled from the cached extractions:
    soundscape = the held-out clips of the cross-validation folds (together the training and validation sites), focal =
    focal_train (diagnostics extraction) + focal_val. Stops unless the clips are exactly those manifests. The diagnostics
    extraction stores no peak time, so focal-train `peak_time` is -1 until `fill_peak_time`."""
    feat = store.features("pretrained-xcl")
    load = lambda name: torch.load(feat / name, map_location="cpu", weights_only=False)
    folds = [load(f"responses_cv{f}.pt") for f in range(n_folds)]
    diag = load("diag_responses.pt")
    assert np.allclose(diag["last_layer_weights"], folds[0]["last_layer_weights"]), "the two extractions use different weights"
    keys = ("targets", "probs", "proto_max", "peak_freq", "peak_time")
    soundscape = _stack([f["soundscape"] for f in folds], keys)
    soundscape["species"] = [x for f in folds for x in f["soundscape"]["species"]]
    train = dict(diag["focal_train"])
    train["peak_time"] = np.full(train["peak_freq"].shape, -1, dtype=np.int16)
    focal = _stack([train, folds[0]["focal"]], keys)
    _check_clips(soundscape["audio_path"], splits_dir, ("soundscape_train", "soundscape_val"))
    _check_clips(focal["audio_path"], splits_dir, ("focal_train", "focal_val"))
    return {"focal": focal, "soundscape": soundscape, "species_list": folds[0]["species_list"],
            "last_layer_weights": folds[0]["last_layer_weights"]}


def fill_peak_time(focal: dict, cols: list[int], splits_dir: Path, top: int, model) -> int:
    """Run `model` on the focal clips that will be cropped and have no peak time yet; their max activations and peak rows
    must agree with the cache. Returns the number of clips run."""
    need = sorted({int(i) for c in cols for k in range(5) for i in top_clips(focal, c, k, True, top)
                   if focal["peak_time"][i, 0, 0] < 0})
    if not need:
        return 0
    ds = HSNInferenceDataset(splits_dir / "focal_train.parquet", is_soundscape=False)
    row_of = {Path(q).name: j for j, q in enumerate(ds.df["audio_path"])}
    ds.df = ds.df.iloc[[row_of[Path(focal["audio_path"][i]).name] for i in need]].reset_index(drop=True)
    got = extract_prototype_responses(model, ds, batch_size=8, num_workers=1, desc="peak positions")
    assert np.abs(got["proto_max"] - focal["proto_max"][need]).max() < MATCH_TOL, "re-run activations differ from the cache"
    for j, i in enumerate(need):
        for c in cols:   # peak positions are only needed (and only compared) for the cropped species
            assert (got["peak_freq"][j, c] == focal["peak_freq"][i, c]).all(), f"peak rows differ: {focal['audio_path'][i]}"
        focal["peak_time"][i] = got["peak_time"][j]
    return len(need)


MODEL_LABELS = {"pretrained-xcl": "Pretrained (XCL)", "focal-only": "Focal-only", "soundscape-only": "Soundscape-only"}
COMPARISON_METRICS = ("cmAP", "macro_auc", "proto_auc_median", "proto_auc_frac_gt_0.8", "per_species_ap", "per_species_auc")


def model_comparison_block(path: Path) -> dict | None:
    """Compact, dashboard-ready view of compare_models.py output (None if it hasn't been run)."""
    if not path.exists():
        return None
    cmp = json.loads(path.read_text())
    rr = lambda v: [r(x) for x in v] if isinstance(v, list) else r(v)
    return {
        "split": "test",
        "species": cmp["species"],
        "n_pos": cmp["n_pos"],
        "min_pos": cmp["min_pos"],
        "models": [{"name": name, "label": MODEL_LABELS.get(name, name), "train_domain": m["train_domain"],
                    **{d: {k: rr(m[d][k]) for k in COMPARISON_METRICS} for d in DOMAINS}}
                   for name, m in cmp["models"].items()],
    }


DRIFT_FOCAL_KHZ, DRIFT_SOUNDSCAPE_KHZ = 1.0, 0.2  # "drifts to the noise floor": >1 kHz on focal, lowest band on soundscape


def model_prototypes_block(names: list[str], store: results.Store) -> dict | None:
    """Per-model, per-prototype transfer stats on the test split (same species filter as each results file)."""
    models = []
    for name in names:
        res_path = store.run_dir(name) / "prototype_transfer.json"
        if not res_path.exists():
            continue
        stats = json.loads(res_path.read_text())
        feat_path = store.features(name) / "responses.pt"
        payload = torch.load(feat_path, map_location="cpu", weights_only=False) if feat_path.exists() else None
        if payload is None:
            logger.warning(f"{feat_path} missing: no confusion stats for {name}")
        species, pooled = {}, []
        for sp, st in stats["species"].items():
            if not st["analysed"]:
                continue
            conf = confusion_stats(payload, payload["species_list"].index(sp), payload["species_list"]) if payload else None
            protos = []
            for k in range(5):
                e = {"k": k + 1, "weight": r(st["last_layer_weights"][k])}
                for dom in DOMAINS:
                    ps = st["domains"][dom]["prototypes"][k]
                    e[dom] = {"auc": r(ps["auc"]), "act_pos": r(ps["act_pos"]), "act_neg": r(ps["act_neg"]),
                              "share": r(ps["logit_share"]), "peak_khz": r(ps["peak_khz"], 2)}
                if conf:
                    e["confusion"] = conf[k]
                e["drift"] = bool(e["focal"]["peak_khz"] is not None and e["soundscape"]["peak_khz"] is not None
                                  and e["focal"]["peak_khz"] > DRIFT_FOCAL_KHZ and e["soundscape"]["peak_khz"] < DRIFT_SOUNDSCAPE_KHZ)
                protos.append(e)
                pooled.append(e)
            species[sp] = {"n": {d: st["domains"][d]["n_pos"] for d in DOMAINS},
                           "class_auc": {d: r(st["domains"][d]["class_auc"]) for d in DOMAINS}, "prototypes": protos}
        both = [e for e in pooled if e["focal"]["auc"] is not None and e["soundscape"]["auc"] is not None]
        med = lambda vals: r(float(np.median([v for v in vals if v is not None]))) if vals else None
        dauc = [e["soundscape"]["auc"] - e["focal"]["auc"] for e in both]
        models.append({
            "name": name, "label": MODEL_LABELS.get(name, name), "species": species,
            "summary": {
                "n_protos": len(both),
                "auc_median": {d: med([e[d]["auc"] for e in both]) for d in DOMAINS},
                "act_pos_median": {d: med([e[d]["act_pos"] for e in pooled]) for d in DOMAINS},
                "act_neg_median": {d: med([e[d]["act_neg"] for e in pooled]) for d in DOMAINS},
                "n_lose": sum(x < -0.1 for x in dauc), "n_improve": sum(x > IMPROVE for x in dauc),
                "n_drift": sum(e["drift"] for e in pooled),
                "frac_other_median": med([e["confusion"]["frac_other"] for e in pooled if "confusion" in e]),
                "base_other_median": med([e["confusion"]["base_other"] for e in pooled if "confusion" in e]),
            },
        })
    if not models:
        return None
    codes = sorted(set.intersection(*(set(m["species"]) for m in models)),
                   key=lambda sp: -models[0]["species"][sp]["n"]["soundscape"])
    return {"split": "test", "species": codes, "min_soundscape_pos": stats["min_soundscape_pos"],
            "drift_rule": f"peak > {DRIFT_FOCAL_KHZ:g} kHz on focal, < {DRIFT_SOUNDSCAPE_KHZ:g} kHz on soundscape",
            "confusion_top_n": CONFUSION_TOP_N, "models": models}


def species_entries(data: dict, stats: dict, sp_list: list[str], patch_dir: Path | None, top: int,
                    pre: AudioPreprocessor | None = None) -> list[dict]:
    """One dashboard entry per analysed species (sorted by soundscape positives): prototype statistics, activation CDFs, confusion and,
    unless `patch_dir` is None, patches (written to `patch_dir`, which needs the audio and `pre`)."""
    cache: dict = {}
    analysed = sorted([sp for sp, s in stats["species"].items() if s["analysed"]],
                      key=lambda sp: -stats["species"][sp]["domains"]["soundscape"]["n_pos"])
    species = []
    for sp in analysed:
        c = sp_list.index(sp)
        st = stats["species"][sp]
        protos = []
        for k in range(5):
            entry = {"k": k + 1, "weight": r(st["last_layer_weights"][k])}
            for dom in DOMAINS:
                d = data[dom]
                y = d["targets"][:, c]
                act = d["proto_max"][y, c, k]
                ps = st["domains"][dom]["prototypes"][k]
                entry[dom] = {
                    "auc": r(ps["auc"]),
                    "act_q": [r(v) for v in np.percentile(act, QUANTILES)],
                    "neg_median": r(ps["act_neg"]),
                    "silent": r((act <= 0).mean()) if len(act) else None,
                    "logit_share": r(ps["logit_share"]),
                    "peak_khz": r(ps["peak_khz"], 2),
                }
            protos.append(entry)
        species.append({
            "code": sp,
            "n": {dom: st["domains"][dom]["n_pos"] for dom in DOMAINS},
            "class_auc": {dom: r(st["domains"][dom]["class_auc"]) for dom in DOMAINS},
            "prototypes": protos,
            "act_cdf": species_activation_cdfs(data, c),
            "confusion": confusion_stats(data, c, sp_list),
            "patches": None if patch_dir is None else export_patches(data, sp, c, patch_dir, pre, cache, top),
        })
        logger.info(f"{sp}: exported")
    return species


def build_bundle(payload: dict, stats: dict, store: results.Store, comparison_path: Path | None, models: list[str],
                 patch_dir: Path | None, top: int = 3, curated: list[dict] = CURATED_FINDINGS) -> dict:
    """The whole data bundle from the pretrained model's responses `payload` and statistics `stats` (see extract_prototype_activations.py)."""
    sp_list = payload["species_list"]
    data = {d: payload[d] for d in DOMAINS}
    if patch_dir is not None:
        patch_dir.mkdir(parents=True, exist_ok=True)
    species = species_entries(data, stats, sp_list, patch_dir, top, AudioPreprocessor() if patch_dir is not None else None)
    return {
        "meta": {
            "model": "DBD-research-group/AudioProtoPNet-5-BirdSet-XCL",
            "revision": MODEL_REVISION,
            "dataset": "BirdSet HSN (High Sierra Nevada), 21 species",
            "n_clips": stats["n_clips"],
            "min_soundscape_pos": stats["min_soundscape_pos"],
            "activation_quantiles": list(QUANTILES),
            "cdf_x": CDF_GRID.tolist(),
            "generated": date.today().isoformat(),
        },
        "findings": computed_findings(species, activation_cdfs(data, species, sp_list), stats["n_clips"]["soundscape"])
                    + fill_curated(curated, species),
        "caveats": CAVEATS,
        "species": species,
        "model_comparison": None if comparison_path is None else model_comparison_block(comparison_path),
        "model_prototypes": model_prototypes_block(models, store),
    }


def write_bundle(out: dict, out_dir: Path) -> None:
    (out_dir / "data.json").write_text(json.dumps(out, indent=1))
    # data.js lets index.html work from file:// (browsers block fetch() of local JSON there).
    (out_dir / "data.js").write_text("window.DASHBOARD_DATA = " + json.dumps(out) + ";\n")
