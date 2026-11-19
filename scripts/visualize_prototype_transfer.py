#!/usr/bin/env python3
"""Figures: how AudioProtoPNet's HSN prototypes transfer from focal to soundscape audio.

Reads outputs/features/prototype_responses.pt and outputs/results/prototype_transfer.json
(written by scripts/extract_prototype_activations.py) and writes to outputs/figures/:

  pt1_prototype_auc_transfer.png   per-prototype detection AUC, focal vs soundscape
  pt2_activation_by_prototype.png  max activation on positives per prototype, both domains
  pt3_logit_share.png              which prototype carries the class evidence in each domain
  pt4_peak_frequency.png           where (in frequency) each prototype fires, both domains
  pt5_patches_<species>.png        top-activating spectrogram patches per prototype & domain
  pt6_latent_map.png               PCA of latent peak patches vs prototype vectors

Only species with >= min_soundscape_pos soundscape positives are analysed (see JSON).
pt5 needs the audio (it recomputes spectrogram crops), so run it where data/ lives.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import AudioPreprocessor
from src.prototype_analysis import grid_cell_to_spec_slice, grid_row_centers_khz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("visualize_prototype_transfer")

# Validated categorical palette (dataviz reference, light mode), fixed order.
FOCAL, SOUNDSCAPE = "#2a78d6", "#eb6834"
PROTO_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e4e1"
DOMAIN_COLOR = {"focal": FOCAL, "soundscape": SOUNDSCAPE}
DOMAIN_LABEL = {"focal": "Focal (Xeno-Canto)", "soundscape": "Soundscape (PAM)"}

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {path}")


def panel_grid(n: int, ncols: int = 4, w: float = 3.2, h: float = 2.6):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat[:n]


# ----------------------------------------------------------------------------
def fig_auc_transfer(stats: dict, species: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    pts = []
    for sp in species:
        s = stats["species"][sp]
        w = np.array(s["last_layer_weights"])
        for k in range(5):
            fa = s["domains"]["focal"]["prototypes"][k]["auc"]
            sa = s["domains"]["soundscape"]["prototypes"][k]["auc"]
            if fa is not None and sa is not None:
                pts.append((fa, sa, w[k], f"{sp} P{k + 1}"))
    if not pts:
        logger.warning("pt1: no prototype has an AUC in both domains (too few clips?); skipping")
        plt.close(fig)
        return
    fa, sa, w, lab = zip(*pts)
    fa, sa, w = map(np.array, (fa, sa, w))
    size = 12 + 60 * w / max(w.max(), 1e-8)
    ax.plot([0.3, 1], [0.3, 1], color=MUTED, lw=1, ls="--", zorder=1)
    ax.scatter(fa, sa, s=size, color=FOCAL, alpha=0.75, edgecolor="white", linewidth=1.5, zorder=2)
    for i in np.argsort(sa - fa)[:6]:  # largest drops
        ax.annotate(lab[i], (fa[i], sa[i]), xytext=(4, -3), textcoords="offset points",
                    fontsize=7, color=MUTED)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("Prototype detection AUC: focal")
    ax.set_ylabel("Prototype detection AUC: soundscape")
    ax.set_title("Does each prototype still detect its species in soundscapes?\n"
                 "(one dot per prototype; size = last-layer weight; dashed = no change)", loc="left")
    save(fig, out)


def fig_activation_by_prototype(data: dict, species: list[str], sp_list: list[str], out: Path) -> None:
    fig, axes = panel_grid(len(species))
    for ax, sp in zip(axes, species):
        c = sp_list.index(sp)
        for d_i, dom in enumerate(("focal", "soundscape")):
            d = data[dom]
            y = d["targets"][:, c]
            acts = d["proto_max"][y, c, :]
            pos = np.arange(5) + (d_i - 0.5) * 0.36
            bp = ax.boxplot([acts[:, k] for k in range(5)], positions=pos, widths=0.3,
                            patch_artist=True, showfliers=False, manage_ticks=False)
            for b in bp["boxes"]:
                b.set(facecolor=DOMAIN_COLOR[dom], alpha=0.75, edgecolor="white", linewidth=1)
            for part in ("whiskers", "caps"):
                for l in bp[part]:
                    l.set(color=DOMAIN_COLOR[dom], linewidth=1)
            for l in bp["medians"]:
                l.set(color="white", linewidth=1.5)
        # Reference: soundscape clips WITHOUT the species
        neg = data["soundscape"]["proto_max"][~data["soundscape"]["targets"][:, c], c, :]
        ax.scatter(np.arange(5), np.median(neg, axis=0), marker="_", s=120, color=INK, lw=1.5, zorder=3)
        n_f = int(data["focal"]["targets"][:, c].sum())
        n_s = int(data["soundscape"]["targets"][:, c].sum())
        ax.set_title(f"{sp}  ·  n = {n_f} focal / {n_s} soundscape", loc="left", fontsize=8.5)
        ax.set_xticks(range(5), [f"P{k + 1}" for k in range(5)])
        ax.set_ylabel("Max activation")
    handles = [plt.Rectangle((0, 0), 1, 1, color=DOMAIN_COLOR[d], alpha=0.75) for d in ("focal", "soundscape")]
    handles.append(plt.Line2D([], [], color=INK, marker="_", ls="", markersize=10, mew=1.5))
    fig.legend(handles, [DOMAIN_LABEL["focal"] + " positives", DOMAIN_LABEL["soundscape"] + " positives",
                         "Soundscape negatives (median)"], loc="upper left", ncol=3, bbox_to_anchor=(0, 1.04))
    fig.suptitle("Prototype activation on clips containing the species", x=0.0, ha="left", y=1.08,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, out)


def fig_logit_share(stats: dict, species: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(species) * 2 + 1))
    yticks, ylabels = [], []
    y = 0.0
    for sp in species:
        for dom in ("focal", "soundscape"):
            shares = [p["logit_share"] or 0.0 for p in stats["species"][sp]["domains"][dom]["prototypes"]]
            left = 0.0
            for k, s in enumerate(shares):
                ax.barh(y, s, left=left, height=0.8, color=PROTO_COLORS[k], edgecolor="white", linewidth=2)
                if s >= 0.12:
                    ax.text(left + s / 2, y, f"P{k + 1} {s:.0%}", ha="center", va="center", fontsize=7, color=INK)
                left += s
            yticks.append(y)
            ylabels.append(f"{sp} · {'focal' if dom == 'focal' else 'soundscape'}")
            y += 1
        y += 0.6
    ax.set_yticks(yticks, ylabels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean share of class evidence  w_k · a_k / Σ_j w_j · a_j  on positive clips")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PROTO_COLORS]
    ax.legend(handles, [f"P{k + 1}" for k in range(5)], ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title("Which prototype carries the decision, focal vs soundscape", loc="left", pad=22,
                 fontweight="bold")
    save(fig, out)


def fig_peak_frequency(stats: dict, species: list[str], out: Path) -> None:
    rows = [(sp, k) for sp in species for k in range(5)]
    fig, ax = plt.subplots(figsize=(6.5, 0.22 * len(rows) + 1.2))
    for i, (sp, k) in enumerate(rows):
        f = stats["species"][sp]["domains"]["focal"]["prototypes"][k]["peak_khz"]
        s = stats["species"][sp]["domains"]["soundscape"]["prototypes"][k]["peak_khz"]
        if f is None or s is None:
            continue
        ax.plot([f, s], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter([f], [i], color=FOCAL, s=28, zorder=2, edgecolor="white", linewidth=1)
        ax.scatter([s], [i], color=SOUNDSCAPE, s=28, zorder=2, edgecolor="white", linewidth=1)
    ax.set_yticks(range(len(rows)), [f"{sp} P{k + 1}" for sp, k in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ticks = [0.5, 1, 2, 4, 8, 16]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlabel("Median frequency of activation peak (kHz, log scale)")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=DOMAIN_COLOR[d]) for d in ("focal", "soundscape")],
              labels=[DOMAIN_LABEL[d] for d in ("focal", "soundscape")], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_title("Where in frequency each prototype fires", loc="left", pad=20, fontweight="bold")
    save(fig, out)


def fig_patches(data: dict, sp: str, sp_list: list[str], out: Path, top: int = 3) -> None:
    c = sp_list.index(sp)
    pre = AudioPreprocessor()
    cache: dict[str, np.ndarray] = {}
    row_khz = grid_row_centers_khz()

    def spec_of(path: str, is_sc: bool) -> np.ndarray:
        if path not in cache:
            cache[path] = pre.process_file(path, is_soundscape=is_sc).squeeze().numpy()
        return cache[path]

    fig, axes = plt.subplots(5, 2 * top, figsize=(1.6 * 2 * top, 1.55 * 5), squeeze=False)
    for k in range(5):
        for d_i, dom in enumerate(("focal", "soundscape")):
            d = data[dom]
            idx = np.nonzero(d["targets"][:, c])[0]
            order = idx[np.argsort(-d["proto_max"][idx, c, k])][:top]
            for j in range(top):
                ax = axes[k, d_i * top + j]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for s in ax.spines.values():
                    s.set(visible=True, color=DOMAIN_COLOR[dom], linewidth=1.5)
                if j >= len(order):
                    continue
                i = order[j]
                r, t = int(d["peak_freq"][i, c, k]), int(d["peak_time"][i, c, k])
                spec = spec_of(d["audio_path"][i], dom == "soundscape")
                fs, ts = grid_cell_to_spec_slice(r, t, n_frames=spec.shape[1])
                ax.imshow(spec[fs, ts], origin="lower", aspect="auto", cmap="magma")
                ax.set_title(f"a={d['proto_max'][i, c, k]:.2f} · {row_khz[r]:.1f} kHz", fontsize=7, color=MUTED)
        axes[k, 0].set_ylabel(f"P{k + 1}", rotation=0, labelpad=14, fontsize=10)
    # Column headers via fig.text *after* layout so tight_layout doesn't shrink the axes.
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for d_i, dom in enumerate(("focal", "soundscape")):
        x0 = axes[0, d_i * top].get_position().x0
        fig.text(x0, 0.95, DOMAIN_LABEL[dom] + f": top {top} positive clips",
                 fontsize=9, fontweight="bold", color=DOMAIN_COLOR[dom])
    fig.text(0.0, 0.99, f"{sp}: most-activating patches per prototype "
             "(crop = peak cell ± 2 cells ≈ 0.65 s × 5 grid rows)", fontweight="bold", fontsize=10)
    save(fig, out)


def fig_latent_map(data: dict, payload: dict, species: list[str], sp_list: list[str], out: Path,
                   max_pts: int = 300, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = panel_grid(len(species))
    for ax, sp in zip(axes, species):
        c = sp_list.index(sp)
        protos = payload["prototype_vectors"][c]  # (5, D)
        pts = {}
        for dom in ("focal", "soundscape"):
            d = data[dom]
            sel = np.nonzero(d["pos_class"] == c)[0]
            if len(sel) > max_pts:
                sel = rng.choice(sel, max_pts, replace=False)
            clips = d["pos_clip"][sel]
            best_k = d["proto_max"][clips, c, :].argmax(axis=1)  # prototype that fires most
            pts[dom] = d["pos_peak_feats"][sel, best_k].astype(np.float32)
        pca = PCA(n_components=2).fit(np.concatenate([pts["focal"], pts["soundscape"], protos]))
        for dom in ("focal", "soundscape"):
            if not len(pts[dom]):
                continue
            z = pca.transform(pts[dom])
            ax.scatter(z[:, 0], z[:, 1], s=8, color=DOMAIN_COLOR[dom], alpha=0.5, linewidth=0)
        zp = pca.transform(protos)
        ax.scatter(zp[:, 0], zp[:, 1], marker="*", s=110, color=INK, edgecolor="white", linewidth=1, zorder=3)
        for k, (x, y) in enumerate(zp):
            ax.annotate(f"P{k + 1}", (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax.set_title(sp, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [plt.Line2D([], [], marker="o", ls="", color=DOMAIN_COLOR[d]) for d in ("focal", "soundscape")]
    handles.append(plt.Line2D([], [], marker="*", ls="", color=INK, markersize=10))
    fig.legend(handles, [DOMAIN_LABEL["focal"] + " patch", DOMAIN_LABEL["soundscape"] + " patch", "Prototype"],
               loc="upper left", ncol=3, bbox_to_anchor=(0, 1.04))
    fig.suptitle("Latent patch at the strongest prototype's peak (per-species PCA, cosine-normalised)",
                 x=0.0, ha="left", y=1.08, fontweight="bold")
    fig.tight_layout()
    save(fig, out)


# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-path", type=Path, default=Path("outputs/features/prototype_responses.pt"))
    p.add_argument("--results-path", type=Path, default=Path("outputs/results/prototype_transfer.json"))
    p.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))
    p.add_argument("--patch-species", nargs="*", default=None,
                   help="Species for pt5 patch galleries (default: top 3 by soundscape positives)")
    p.add_argument("--skip-patches", action="store_true", help="Skip pt5 (needs audio files)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.features_path, map_location="cpu", weights_only=False)
    stats = json.loads(args.results_path.read_text())
    sp_list = payload["species_list"]
    data = {d: payload[d] for d in ("focal", "soundscape")}

    species = sorted([sp for sp, s in stats["species"].items() if s["analysed"]],
                     key=lambda sp: -stats["species"][sp]["domains"]["soundscape"]["n_pos"])
    logger.info(f"Analysing {len(species)} species: {species}")
    out = args.figures_dir

    fig_auc_transfer(stats, species, out / "pt1_prototype_auc_transfer.png")
    fig_activation_by_prototype(data, species, sp_list, out / "pt2_activation_by_prototype.png")
    fig_logit_share(stats, species, out / "pt3_logit_share.png")
    fig_peak_frequency(stats, species, out / "pt4_peak_frequency.png")
    if not args.skip_patches:
        for sp in args.patch_species or species[:3]:
            fig_patches(data, sp, sp_list, out / f"pt5_patches_{sp}.png")
    fig_latent_map(data, payload, species, sp_list, out / "pt6_latent_map.png")


if __name__ == "__main__":
    main()
