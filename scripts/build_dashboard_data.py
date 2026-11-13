#!/usr/bin/env python3
"""Export the prototype-transfer results into a self-contained dashboard data bundle.

Reads outputs/features/prototype_responses.pt + outputs/results/prototype_transfer.json and writes
  dashboard/data.json, data.js     per-species / per-prototype stats + findings
  dashboard/patches/<sp>_P<k>_<domain>_<rank>.png   top-activating spectrogram crops

Patch export recomputes spectrograms from audio, so run it where data/ lives.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import MODEL_REVISION
from src.preprocessing import AudioPreprocessor
from src.prototype_analysis import grid_cell_to_spec_slice, grid_row_centers_khz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_dashboard_data")

DOMAINS = ("focal", "soundscape")
QUANTILES = (10, 25, 50, 75, 90)

# Observations from visually inspecting the patch galleries (not computable from the JSON).
CURATED_FINDINGS = [
    {
        "title": "Some prototypes match the same acoustic part in both domains",
        "text": "whcspa P3 (a pure-tone note) and P4 (a trill) fire on the same structures in focal and soundscape "
                "clips and keep soundscape AUC above 0.9. P5's best soundscape matches also resemble its focal "
                "ones, but they are rare: its median soundscape activation is ~0 and its AUC drops to 0.67.",
        "species": "whcspa", "prototype": 3,
        "evidence": {"kind": "patches", "species": "whcspa", "prototypes": [3, 4, 5],
                     "method": "Visual inspection of the top-3 activating patches per domain (crop = peak cell "
                               "± 2 grid cells), checked against each prototype's AUC and median activation "
                               "on all positive clips (table)."},
    },
    {
        "title": "Others latch onto noise in soundscapes",
        "text": "whcspa P2's top soundscape matches are broadband noise bursts around 6 kHz rather "
                "than song: high activation there is likely spurious.",
        "species": "whcspa", "prototype": 2,
        "evidence": {"kind": "patches", "species": "whcspa", "prototypes": [2],
                     "method": "Visual inspection of the top-3 activating patches per domain. Note P2 still has "
                               "soundscape AUC 0.91: the spurious matches are among its strongest activations, "
                               "not the majority of its detections."},
    },
]

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


def computed_findings(species: list[dict], dists: dict) -> list[dict]:
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
        "text": f"Mean class AUC {mean_f:.3f} on focal vs {mean_s:.3f} on soundscape; {n_above_09} of "
                f"{len(species)} species keep AUC ≥ 0.90 on soundscapes. {worst_sp} collapses "
                f"(AUC drops by {worst_drop:.2f}).",
        "species": worst_sp, "prototype": None,
        "evidence": {
            "kind": "class_auc_drop", "highlight": [worst_sp],
            "stats": [["Mean class AUC, focal", f"{mean_f:.3f}"], ["Mean class AUC, soundscape", f"{mean_s:.3f}"],
                      ["Species with soundscape AUC ≥ 0.90", f"{n_above_09} / {len(species)}"],
                      ["Largest drop", f"{worst_sp}: {worst_drop:.2f}"],
                      ["Median drop", f"{np.median([d for _, d in drops]):.3f}"]],
            "method": "Class ROC-AUC of the model's sigmoid output for the species, on all focal clips "
                      "(negatives = other species) and all 12,000 soundscape clips.",
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
        "text": f"{n_drop} of {len(protos)} prototypes lose more than 0.1 detection AUC on soundscapes while "
                f"{n_gain} improve; in {n_spread} of {len(spread)} species the best- and worst-transferring "
                f"prototypes differ by ≥ 0.15 AUC.",
        "species": None, "prototype": None,
        "evidence": {
            "kind": "delta_auc", "threshold": -0.1,
            "spread": {sp: r(v, 3) for sp, v in spread.items()},
            "stats": [["Prototypes with ΔAUC < −0.10", f"{n_drop} / {len(protos)}"],
                      ["Prototypes with ΔAUC > +0.02", f"{n_gain} / {len(protos)}"],
                      ["Median ΔAUC", f"{np.median(list(dauc.values())):+.3f}"],
                      ["Species with within-species spread ≥ 0.15", f"{n_spread} / {len(spread)}"]],
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
                f"(pooled medians {m['focal_pos']:.2f} → {m['soundscape_pos']:.2f}). On clips without it, "
                f"activations stay near zero ({frac_neg_low:.0%} ≤ 0.05), so positives still rank above "
                f"negatives: median prototype AUC on soundscapes is {med_sauc:.2f}.",
        "species": None, "prototype": None,
        "evidence": {
            "kind": "activation_cdf", "cdf": dists,
            "stats": [["Median activation, focal positives", f"{m['focal_pos']:.3f}"],
                      ["Median activation, soundscape positives", f"{m['soundscape_pos']:.3f}"],
                      ["Median activation, focal negatives", f"{m['focal_neg']:.3f}"],
                      ["Median activation, soundscape negatives", f"{m['soundscape_neg']:.3f}"],
                      ["Soundscape negatives ≤ 0.05", f"{frac_neg_low:.0%}"],
                      ["Soundscape positives ≤ 0.05", f"{frac_pos_low:.0%}"],
                      ["Median per-prototype ratio (soundscape / focal median)", f"{ratio:.0%}"],
                      ["Median prototype AUC, soundscape", f"{med_sauc:.3f}"]],
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

    # 5. Noise-floor drift
    low = [(sp, p) for sp, p in protos
           if p["soundscape"]["peak_khz"] is not None and p["soundscape"]["peak_khz"] < 0.5
           and p["focal"]["peak_khz"] is not None and p["focal"]["peak_khz"] > 1.0]
    if low:
        low_keys = {(sp, p["k"]) for sp, p in low}
        low_d = float(np.mean([dauc[k] for k in low_keys]))
        rest_d = float(np.mean([v for k, v in dauc.items() if k not in low_keys]))
        out.append({
            "title": "Failing prototypes drift to the noise floor",
            "text": f"{len(low)} prototypes that fire at bird-song frequencies (> 1 kHz) on focal clips peak in the "
                    f"lowest band (~0.1 kHz, wind/ambient) on soundscapes: "
                    + ", ".join(f"{sp} P{p['k']}" for sp, p in low)
                    + f". Their AUC changes by {low_d:+.2f} on average vs {rest_d:+.2f} for the rest.",
            "species": low[0][0], "prototype": low[0][1]["k"],
            "evidence": {
                "kind": "peak_scatter", "highlight": [[sp, p["k"]] for sp, p in low],
                "stats": [["Prototypes drifting to < 0.5 kHz", f"{len(low)} / {len(protos)}"],
                          ["Mean ΔAUC, drifting", f"{low_d:+.3f}"], ["Mean ΔAUC, others", f"{rest_d:+.3f}"]],
                "method": "Peak = median (over positive clips) of the frequency row where the prototype's activation "
                          "map is maximal; rows are 16 mel bins wide (lowest row centred at ~0.1 kHz).",
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
            mask = d["targets"][:, c] if positive else ~d["targets"][:, c]
            idx = np.nonzero(mask)[0]
            order = idx[np.argsort(-d["proto_max"][idx, c, k])][:top]
            items = []
            for rank, i in enumerate(order):
                path = d["audio_path"][i]
                if path not in cache:
                    cache[path] = pre.process_file(path, is_soundscape=dom == "soundscape").squeeze().numpy()
                spec = cache[path]
                row, col = int(d["peak_freq"][i, c, k]), int(d["peak_time"][i, c, k])
                fs, ts = grid_cell_to_spec_slice(row, col, n_frames=spec.shape[1])
                name = f"{sp}_P{k + 1}_{key}_{rank + 1}.png"
                plt.imsave(out_dir / name, spec[fs, ts], cmap="magma", origin="lower")
                item = {"src": f"patches/{name}", "act": r(d["proto_max"][i, c, k], 3),
                        "khz": r(row_khz[row], 2), "clip": Path(path).name}
                if not positive:
                    item["labels"] = d["species"][i]  # species annotated in that clip ("background" if none)
                items.append(item)
            patches[str(k + 1)][key] = items
    return patches


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-path", type=Path, default=Path("outputs/features/prototype_responses.pt"))
    p.add_argument("--results-path", type=Path, default=Path("outputs/results/prototype_transfer.json"))
    p.add_argument("--out-dir", type=Path, default=Path("dashboard"))
    p.add_argument("--top", type=int, default=3, help="Patches per prototype and domain")
    p.add_argument("--skip-patches", action="store_true")
    args = p.parse_args()

    payload = torch.load(args.features_path, map_location="cpu", weights_only=False)
    stats = json.loads(args.results_path.read_text())
    sp_list = payload["species_list"]
    data = {d: payload[d] for d in DOMAINS}
    patch_dir = args.out_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    pre, cache = AudioPreprocessor(), {}

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
            "patches": None if args.skip_patches else export_patches(data, sp, c, patch_dir, pre, cache, args.top),
        })
        logger.info(f"{sp}: exported")

    out = {
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
        "findings": computed_findings(species, activation_cdfs(data, species, sp_list)) + CURATED_FINDINGS,
        "caveats": CAVEATS,
        "species": species,
    }
    (args.out_dir / "data.json").write_text(json.dumps(out, indent=1))
    # data.js lets index.html work from file:// (browsers block fetch() of local JSON there).
    (args.out_dir / "data.js").write_text("window.DASHBOARD_DATA = " + json.dumps(out) + ";\n")
    logger.info(f"Wrote {args.out_dir / 'data.json'} ({len(species)} species)")


if __name__ == "__main__":
    main()
