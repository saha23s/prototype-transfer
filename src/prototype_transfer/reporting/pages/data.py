"""Data exploration: what differs between focal recordings and soundscapes."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from prototype_transfer.data.splits import site_of
from prototype_transfer.reporting import charts as c
from prototype_transfer.reporting.runs import Results
from prototype_transfer.reporting.site import Page

GALLERY = "figures/spectrogram_examples.png"
GALLERY_PER_DOMAIN, GALLERY_SEED = 2, 0


def positives(df, codes) -> dict[str, int]:
    labels = [set(x) for x in (df["hsn_classes"] if "hsn_classes" in df else df["ebird_code_multilabel"])]
    return {s: sum(s in x for x in labels) for s in codes}


def long_tail_chart(focal: dict[str, int], sc: dict[str, int]) -> str:
    order = sorted(sc, key=lambda s: -sc[s])
    x0, x1, row, top = 230, 760, 22, 28
    lo, hi = 0.7, 6000
    x = lambda v: x0 + (x1 - x0) * (math.log10(max(v, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    bottom = top + row * len(order)
    body = c.x_grid(x, (1, 10, 100, 1000), top - 8, bottom, lambda t: f"{t:,}")
    body += c.text((x0 + x1) / 2, bottom + 34, "Clips containing the species (log scale; 0 is drawn at the left edge)", "axis-title", "middle")
    body += c.text(x1 + 16, top - 12, "focal / soundscape")
    for i, s in enumerate(order):
        y = top + i * row + 6
        body += c.text(x0 - 12, y + 4, c.name_of(s), "row-label", "end")
        body += c.line(x(focal[s]), y, x(sc[s]), y, "connector")
        body += c.circle(x(focal[s]), y, 5, "dot focal", f"{c.name_of(s)}: {focal[s]:,} focal clips")
        body += c.circle(x(sc[s]), y, 5, "dot soundscape", f"{c.name_of(s)}: {sc[s]:,} soundscape clips")
        body += c.text(x1 + 16, y + 4, f"{focal[s]:,} / {sc[s]:,}")
    return c.svg(bottom + 42, "Clips per species in focal and soundscape data", body)


def density_chart(focal_counts: dict[int, int], sc_counts: dict[int, int]) -> str:
    ks = sorted(set(focal_counts) | set(sc_counts))
    x0, x1, row, top = 230, 760, 46, 10
    x = c.linear(0, 1, x0, x1)
    nf, ns = sum(focal_counts.values()), sum(sc_counts.values())
    bottom = top + row * len(ks)
    body = c.x_grid(x, (0, 0.25, 0.5, 0.75, 1), top, bottom, c.pct)
    body += c.text((x0 + x1) / 2, bottom + 34, "Share of clips", "axis-title", "middle")
    for i, k in enumerate(ks):
        y = top + i * row + 6
        body += c.text(x0 - 12, y + 20, f"{k} labeled species", "row-label", "end")
        for j, (dom, counts, total) in enumerate((("focal", focal_counts, nf), ("soundscape", sc_counts, ns))):
            v = counts.get(k, 0)
            yy = y + j * 16
            w = max(x(v / total) - x0, 0)
            if v:
                body += c.rect(x0, yy, max(w, 1.5), 13, f"seg f-{dom}", f"{dom}: {v:,} of {total:,} clips have {k} labeled species")
            body += c.text(x0 + w + 8, yy + 11, "0%" if not v else "<1%" if v / total < 0.005 else c.pct(v / total))
    return c.svg(bottom + 42, "Number of labeled species per clip", body)


def site_heatmap(sc_all, codes: list[str]) -> str:
    sites = site_of(sc_all["audio_path"].map(lambda q: Path(q).name))
    ids = sorted(sites.unique())
    per_site = sites.value_counts()
    split_of = dict(zip(sites, sc_all["split"]))
    labels = [set(x) for x in sc_all["hsn_classes"]]
    hit = {s: sites[[s in x for x in labels]].value_counts().to_dict() for s in codes}
    order = sorted(codes, key=lambda s: -sum(hit[s].values()))
    split_rank = {"train": 0, "val": 1, "test": 2}
    ids.sort(key=lambda i: (split_rank[split_of[i]], i))
    x0, cw, ch, top = 230, 6.0, 17, 38
    body = ""
    for part in ("train", "val", "test"):
        members = [j for j, i in enumerate(ids) if split_of[i] == part]
        a, b = x0 + members[0] * cw, x0 + (members[-1] + 1) * cw
        body += c.line(a + 1, top - 10, b - 2, top - 10, "bracket") + c.text((a + b) / 2, top - 16, f"{part} ({len(members)})", "", "middle")
    for r, s in enumerate(order):
        y = top + r * ch
        body += c.text(x0 - 12, y + 12, c.name_of(s), "row-label", "end")
        body += c.text(x0 + len(ids) * cw + 12, y + 12, f"{len(hit[s])} site{'s' if len(hit[s]) != 1 else ''}")
        for j, i in enumerate(ids):
            k = hit[s].get(i, 0)
            share = k / per_site[i]
            cls, op = ("seg f-soundscape", 0.2 + 0.8 * math.sqrt(share)) if k else ("empty-cell", None)
            body += c.rect(x0 + j * cw, y + 1, cw - 1, ch - 2, cls,
                           f"{c.name_of(s)}, {i} ({split_of[i]}): {k} of {per_site[i]} clips" if k else "", op, 1)
    height = top + len(order) * ch + 28
    body += c.text(x0 + len(ids) * cw / 2, height - 8, f"{len(ids)} recording sites (one column each), grouped by split", "axis-title", "middle")
    return c.svg(height, "Which species occur at which soundscape recording site", body)


def make_gallery(res: Results, species: list[str], out: Path) -> None:
    """Random (seeded, not hand-picked) training clips per species and domain, as the model sees them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchaudio.functional import melscale_fbanks

    from prototype_transfer.data.preprocessing import AudioPreprocessor

    pre = AudioPreprocessor()
    focal = res.manifest("focal").query("split == 'train'")
    sc = res.manifest("soundscape").query("split == 'train'")
    rng = np.random.default_rng(GALLERY_SEED)
    cols = 2 * GALLERY_PER_DOMAIN
    fig, axes = plt.subplots(len(species), cols, figsize=(11, 1.55 * len(species)), sharex=True, sharey=True)
    # mel row of each frequency tick: the filter whose response at that frequency is largest
    banks = melscale_fbanks(pre.n_stft, 0.0, pre.sample_rate / 2, pre.n_mels, pre.sample_rate).numpy()
    freqs = np.linspace(0, pre.sample_rate / 2, pre.n_stft)
    ticks_khz = (1, 4, 8)
    tick_rows = [int(banks[np.argmin(np.abs(freqs - 1000 * f))].argmax()) for f in ticks_khz]
    for i, s in enumerate(species):
        pools = {"focal": focal[[list(x) == [s] for x in focal["ebird_code_multilabel"]]],
                 "soundscape": sc[[list(x) == [s] for x in sc["hsn_classes"]]]}
        j = 0
        for dom, pool in pools.items():
            for idx in rng.choice(len(pool), GALLERY_PER_DOMAIN, replace=False):
                wav = pre.load_and_resample(pool.iloc[int(idx)]["audio_path"])
                wav = pre.extract_focal_window(wav) if dom == "focal" else pre.extract_soundscape_window(wav)
                spec = pre.transform_to_spectrogram(wav)[0].numpy()  # [n_mels, frames]
                ax = axes[i, j]
                ax.imshow(spec, origin="lower", aspect="auto", cmap="magma", extent=(0, 5, 0, pre.n_mels))
                ax.set_yticks(tick_rows, [str(f) for f in ticks_khz])
                ax.set_xticks([0, 2.5, 5])
                if i == 0:
                    ax.set_title(f"{'Focal' if dom == 'focal' else 'Soundscape'}", fontsize=10)
                if j == 0:
                    ax.set_ylabel(f"{c.name_of(s)}\nkHz (mel)", fontsize=8)
                if i == len(species) - 1:
                    ax.set_xlabel("Time (s)", fontsize=8)
                ax.tick_params(labelsize=7)
                j += 1
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)


def build(res: Results) -> Page:
    focal, sc = res.manifest("focal"), res.manifest("soundscape")
    codes = list(c.SPECIES_NAMES)
    pf, ps = positives(focal, codes), positives(sc, codes)
    focal_density = focal["ebird_code_multilabel"].map(len).value_counts().to_dict()
    sc_density = sc["num_hsn_classes"].value_counts().to_dict()
    n_sites = site_of(sc["audio_path"].map(lambda q: Path(q).name)).nunique()
    empty = sc_density.get(0, 0) / len(sc)
    multi = sum(v for k, v in sc_density.items() if k >= 2) / len(sc)
    top3 = sorted(ps.values(), reverse=True)[:3]
    top3_share = sum(top3) / sum(ps.values())
    length = float(focal["length"].median())
    tiles = c.tiles([(c.n(len(focal)), "focal clips (Xeno-Canto)"), (c.n(len(sc)), f"5 s soundscape clips from {n_sites} sites"),
                     (str(len(codes)), "bird species (BirdSet High Sierra Nevada)"), (c.pct(empty), "soundscape clips with none of the species")])
    take = c.takeaways([
        f"<b>Different class balance.</b> The three most frequent soundscape species account for {c.pct(top3_share)} of all "
        "soundscape species labels. Species that are common in focal data can be rare in the field.",
        f"<b>Different clip content.</b> Each focal clip carries one label. {c.pct(empty)} of soundscape clips have no target species "
        f"and {c.pct(multi)} have two or more.",
        "<b>Species are clustered by site.</b> Many species occur at only a few sites, so neighbouring clips are far from independent. "
        "This decides how the data must be split and how uncertainty must be measured.",
    ])
    comp = c.figure("data-species", "Clips per species", "All clips of both domains, ordered by soundscape frequency.",
                    long_tail_chart(pf, ps), c.legend([("focal", "focal"), ("soundscape", "soundscape")]))
    dens = c.figure("data-density", "Labeled species per clip", "Labels count the 21 target species only.",
                    density_chart(focal_density, sc_density), c.legend([("focal", "focal"), ("soundscape", "soundscape")]))
    heat = c.figure("data-sites", "Where each species occurs",
                    "Each site contributes the same number of soundscape clips. Darker cells: a larger share of the site's clips "
                    "contain the species.", site_heatmap(sc, codes))
    gallery = c.figure(
        "data-examples", "What the two domains sound like",
        f"Randomly drawn training clips (fixed seed, not hand-picked) with exactly one labeled species, shown as the model's input. "
        f"Focal recordings are longer (median {length:.0f} s), so the loudest 5 s window is used; soundscapes are already 5 s clips. "
        "The colour scale is set per panel, so compare shapes and signal against background, not loudness between panels.",
        f'<img class="wide" src="{GALLERY}" alt="Spectrograms of random focal and soundscape clips for the seven primary test species">')
    more = ('<p class="more">To see how individual prototypes respond to these two domains, open the '
            '<a href="explorer.html">species explorer</a>.</p>')
    return Page("data.html", "Data exploration", "Data exploration",
                "The two domains differ in class balance, clip content and recording conditions.",
                tiles + take + comp + dens + heat + gallery + more)
