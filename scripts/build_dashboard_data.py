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
        "text": "whcspa P3 and P5 fire on the same pure-tone note, and P4 on the same trill, "
                "in focal and soundscape clips: these prototypes transfer as intended.",
        "species": "whcspa", "prototype": 3,
    },
    {
        "title": "Others latch onto noise in soundscapes",
        "text": "whcspa P2's top soundscape matches are broadband noise bursts around 6 kHz rather "
                "than song: high activation there is likely spurious.",
        "species": "whcspa", "prototype": 2,
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


def computed_findings(species: list[dict]) -> list[dict]:
    drops = [(s["code"], s["class_auc"]["focal"] - s["class_auc"]["soundscape"]) for s in species]
    worst_sp, worst_drop = max(drops, key=lambda t: t[1])
    mean_f = np.mean([s["class_auc"]["focal"] for s in species])
    mean_s = np.mean([s["class_auc"]["soundscape"] for s in species])
    protos = [(s["code"], p) for s in species for p in s["prototypes"]
              if p["focal"]["auc"] is not None and p["soundscape"]["auc"] is not None]
    n_drop = sum(p["focal"]["auc"] - p["soundscape"]["auc"] > 0.1 for _, p in protos)
    n_gain = sum(p["soundscape"]["auc"] - p["focal"]["auc"] > 0.02 for _, p in protos)
    ratio = np.median([p["soundscape"]["act_q"][2] / p["focal"]["act_q"][2] for _, p in protos
                       if p["focal"]["act_q"][2] and p["focal"]["act_q"][2] > 0.05])
    shifts = []
    for s in species:
        for p in s["prototypes"]:
            fs, ss = p["focal"]["logit_share"], p["soundscape"]["logit_share"]
            if fs is not None and ss is not None:
                shifts.append((abs(ss - fs), s["code"], p["k"], fs, ss))
    shifts.sort(reverse=True)
    top_shift = ", ".join(f"{sp} P{k} {fs:.0%}→{ss:.0%}" for _, sp, k, fs, ss in shifts[:3])
    # Prototypes whose activation peak collapses to the lowest band (wind / ambient floor)
    # on soundscapes although they fire on bird-song frequencies in focal recordings.
    low = [(s["code"], p) for s in species for p in s["prototypes"]
           if p["soundscape"]["peak_khz"] is not None and p["soundscape"]["peak_khz"] < 0.5
           and p["focal"]["peak_khz"] is not None and p["focal"]["peak_khz"] > 1.0]
    low_txt = ", ".join(f"{sp} P{p['k']}" for sp, p in low)
    low_drop = np.mean([p["soundscape"]["auc"] - p["focal"]["auc"] for _, p in low if hasattr_auc(p)]) if low else 0.0
    rest_drop = np.mean([p["soundscape"]["auc"] - p["focal"]["auc"] for _, p in protos if (_, p) not in low])
    findings_low = [{
        "title": "Failing prototypes drift to the noise floor",
        "text": f"{len(low)} prototypes that fire at bird-song frequencies (> 1 kHz) on focal clips peak in the "
                f"lowest band (~0.1 kHz, wind/ambient) on soundscapes: {low_txt}. Their AUC changes by "
                f"{low_drop:+.2f} on average vs {rest_drop:+.2f} for the rest.",
        "species": low[0][0], "prototype": low[0][1]["k"],
    }] if low else []
    return [
        {"title": "Species-level detection mostly survives",
         "text": f"Mean class AUC {mean_f:.3f} on focal vs {mean_s:.3f} on soundscape across "
                 f"{len(species)} species; {worst_sp} collapses (AUC drops by {worst_drop:.2f}).",
         "species": worst_sp, "prototype": None},
        {"title": "Prototypes transfer unevenly",
         "text": f"{n_drop} of {len(protos)} prototypes lose more than 0.1 detection AUC on soundscapes, "
                 f"while {n_gain} improve; within a species, some prototypes hold up and others don't.",
         "species": None, "prototype": None},
        {"title": "Activations shrink, ranking survives",
         "text": f"On positive clips, the median soundscape activation is {ratio:.0%} of the focal one, "
                 "but activations on clips without the species stay near zero, so detection ranking "
                 "mostly holds.",
         "species": None, "prototype": None},
        {"title": "The decisive prototype mostly stays the same",
         "text": f"Shares of class evidence are similar across domains; largest shifts: {top_shift}.",
         "species": shifts[0][1], "prototype": shifts[0][2]},
    ] + findings_low


def hasattr_auc(p: dict) -> bool:
    return p["focal"]["auc"] is not None and p["soundscape"]["auc"] is not None


def export_patches(data: dict, sp: str, c: int, out_dir: Path, pre: AudioPreprocessor,
                   cache: dict, top: int) -> dict:
    row_khz = grid_row_centers_khz()
    patches: dict[str, dict] = {}
    for k in range(5):
        patches[str(k + 1)] = {}
        for dom in DOMAINS:
            d = data[dom]
            idx = np.nonzero(d["targets"][:, c])[0]
            order = idx[np.argsort(-d["proto_max"][idx, c, k])][:top]
            items = []
            for rank, i in enumerate(order):
                path = d["audio_path"][i]
                if path not in cache:
                    cache[path] = pre.process_file(path, is_soundscape=dom == "soundscape").squeeze().numpy()
                spec = cache[path]
                row, col = int(d["peak_freq"][i, c, k]), int(d["peak_time"][i, c, k])
                fs, ts = grid_cell_to_spec_slice(row, col, n_frames=spec.shape[1])
                name = f"{sp}_P{k + 1}_{dom}_{rank + 1}.png"
                plt.imsave(out_dir / name, spec[fs, ts], cmap="magma", origin="lower")
                items.append({"src": f"patches/{name}", "act": r(d["proto_max"][i, c, k], 3),
                              "khz": r(row_khz[row], 2), "clip": Path(path).name})
            patches[str(k + 1)][dom] = items
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
            "generated": date.today().isoformat(),
        },
        "findings": computed_findings(species) + CURATED_FINDINGS,
        "caveats": CAVEATS,
        "species": species,
    }
    (args.out_dir / "data.json").write_text(json.dumps(out, indent=1))
    # data.js lets index.html work from file:// (browsers block fetch() of local JSON there).
    (args.out_dir / "data.js").write_text("window.DASHBOARD_DATA = " + json.dumps(out) + ";\n")
    logger.info(f"Wrote {args.out_dir / 'data.json'} ({len(species)} species)")


if __name__ == "__main__":
    main()
