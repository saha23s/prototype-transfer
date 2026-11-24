#!/usr/bin/env python3
"""Add the "Data splits" and "Evaluation protocol" sections to dashboard/index.html.

Both sections are rendered here as static HTML with inline SVG (no JavaScript) from the committed results, so the page shows exactly
what is on disk and the numbers cannot drift from the code:
  outputs/results/<dataset>/dataset/{splits,splits_cv}.json   what make_splits.py produced
  configs/splits/hsn.json                                     the pinned fingerprints
  outputs/results/<dataset>/summaries/canonical_eval.json     protocol v1 on the pretrained model (test split)
  prototype_transfer.evaluation.canonical                     the protocol constants (thresholds, species rules)
The sections are written between the markers <!-- protocol:start --> and <!-- protocol:end --> in index.html; run it again after
the splits or the baseline evaluation change.

  python scripts/report/build_dashboard_protocol.py [--dataset hsn] [--html dashboard/index.html] [--out-dir DIR_FOR_PREVIEW_SVGS]
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

from prototype_transfer.data.splits import SPLITS_LOCK
from prototype_transfer.evaluation import canonical, results

START, END = "<!-- protocol:start -->", "<!-- protocol:end -->"
SPLITS = ("train", "val", "test")
SHADE = {"train": 0.35, "val": 0.65, "test": 1.0}        # light -> dark: the held-out split is the darkest
FOCAL_TARGET, SC_TARGET = (0.7, 0.1, 0.2), (0.6, 0.1, 0.3)  # intended train / val / test shares (focal: clips; soundscape: sites)
W = 900                                                   # SVG width; CSS scales it


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def n(x: int) -> str:
    return f"{x:,}"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def svg(height: float, label: str, body: str) -> str:
    return (f'<svg viewBox="0 0 {W} {height:.0f}" role="img" aria-label="{esc(label)}" xmlns="http://www.w3.org/2000/svg">'
            f"{body}</svg>")


def text(x: float, y: float, s: str, cls: str = "", anchor: str = "start") -> str:
    klass = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{klass}>{esc(s)}</text>'


def stacked(x0: float, y: float, w: float, h: float, parts: list[float], dom: str, titles: list[str], gap: float = 2) -> str:
    """One horizontal bar split into train / val / test segments (a 2px surface gap between fills, rounded outer ends)."""
    total = sum(parts)
    out, x = [], x0
    for part, s, t in zip(parts, SPLITS, titles):
        if total == 0 or part == 0:
            continue
        seg = w * part / total
        out.append(f'<rect class="seg {dom}" x="{x:.1f}" y="{y:.1f}" width="{max(seg - gap, 0.5):.1f}" height="{h:.1f}" rx="3" '
                   f'fill-opacity="{SHADE[s]}"><title>{esc(t)}</title></rect>')
        x += seg
    return "".join(out)


# ------------------------------------------------------------------------------------------------------------ splits
def sizes_chart(sp: dict) -> str:
    rows = [("Focal (Xeno-Canto), grouped by recordist", "focal", [sp["focal"][s]["n_clips"] for s in SPLITS],
             [f'{sp["focal"][s]["recordists"]} recordists' for s in SPLITS]),
            ("Soundscape (field recordings), grouped by recording site", "soundscape", [sp["soundscape"][s]["n_clips"] for s in SPLITS],
             [f'{len(sp["soundscape"]["sites"][s])} sites' for s in SPLITS])]
    body, y = "", 8
    for title, dom, counts, groups in rows:
        total = sum(counts)
        body += text(0, y + 12, title, "row-label")
        bar_x, bar_w = 0, W
        titles = [f"{s}: {n(c)} clips ({pct(c / total)}), {g}" for s, c, g in zip(SPLITS, counts, groups)]
        body += stacked(bar_x, y + 22, bar_w, 26, counts, "f-" + dom, titles)
        for j, (s, c, g) in enumerate(zip(SPLITS, counts, groups)):
            x = j * W / 3
            body += (f'<rect class="seg f-{dom}" x="{x:.1f}" y="{y + 58}" width="12" height="12" rx="2" fill-opacity="{SHADE[s]}"/>'
                     + text(x + 20, y + 68, f"{s}: {n(c)} clips ({pct(c / total)})", "row-label") + text(x + 20, y + 84, g))
        y += 108
    return svg(y, "Clips per split for focal and soundscape data", body)


def species_order(sp: dict) -> list[str]:
    tot = {k: sum(sp["soundscape"][s]["positives"][k] for s in SPLITS) for k in sp["soundscape"]["train"]["positives"]}
    return sorted(tot, key=lambda k: -tot[k])


def species_chart(sp: dict, primary: set[str], secondary: set[str]) -> str:
    order = species_order(sp)
    row, top = 22, 54
    panels = [("focal", "Focal: share of each species' clips", 130, 250, FOCAL_TARGET),
              ("soundscape", "Soundscape: share of each species' positive clips", 500, 230, SC_TARGET)]
    body = text(0, 14, "Role: P = primary species, S = secondary, - = not scored. Dashed lines: intended train | val | test shares. Right: total positives (test).", "")
    for dom, title, x0, w, tgt in panels:
        body += text(x0, 36, title, "row-label")
        for frac in (tgt[0], tgt[0] + tgt[1]):
            xl = x0 + w * frac
            body += f'<line class="diag" x1="{xl:.1f}" y1="{top - 6}" x2="{xl:.1f}" y2="{top + row * len(order) - 4}"/>'
    for i, k in enumerate(order):
        y = top + i * row
        role = "P" if k in primary else ("S" if k in secondary else "-")
        body += text(0, y + 11, k, "row-label") + text(70, y + 11, role, "role-" + role.replace("-", "none"))
        for dom, _, x0, w, _ in panels:
            counts = [sp[dom][s]["positives"][k] for s in SPLITS]
            tot = sum(counts)
            if tot == 0:
                body += text(x0, y + 11, "no positives")
                continue
            titles = [f"{k} ({dom}) {s}: {n(c)} of {n(tot)} ({pct(c / tot)})" for s, c in zip(SPLITS, counts)]
            body += stacked(x0, y, w, 13, counts, "f-" + dom, titles)
            body += text(x0 + w + 8, y + 11, f"{n(tot)} ({n(counts[2])})")
    height = top + row * len(order) + 6
    return svg(height + 8, "Each species' positive clips split into train, validation and test, for focal and soundscape data", body)


def cv_block(cv: dict) -> str:
    folds = sorted(cv["sites"], key=int)
    species = sorted(cv["eligible_species"], key=lambda k: -sum(cv["positives"][f][k] for f in folds))
    mx = max(cv["positives"][f][k] for f in folds for k in species)
    cw, ch, x0, top = 120, 20, 70, 40
    body = text(x0, 14, "Positive clips per held-out fold", "row-label")
    for j, f in enumerate(folds):
        body += text(x0 + j * cw + cw / 2, top - 8, f"fold {f}", "", "middle")
    for i, k in enumerate(species):
        y = top + i * ch
        body += text(0, y + 14, k, "row-label")
        for j, f in enumerate(folds):
            c = cv["positives"][f][k]
            op = 0.08 + 0.52 * math.sqrt(c / mx) if c else 0
            body += (f'<rect class="seg f-soundscape" x="{x0 + j * cw + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="3" '
                     f'fill-opacity="{op:.2f}"><title>{esc(k)}, fold {f}: {c} positive clips</title></rect>'
                     f'{text(x0 + j * cw + cw / 2, y + 14, n(c), "cell", "middle")}')
    heat = svg(top + ch * len(species) + 4, "Positive clips per species in each cross-validation fold", body)
    rows = "".join(
        f"<tr><td>fold {f}</td><td>{len(cv['sites'][f])}</td><td>{n(cv['n_clips'][f]['heldout'])}</td><td>{n(cv['n_clips'][f]['train'])}</td></tr>"
        for f in folds)
    table = ('<table class="mini"><thead><tr><th>Fold</th><th>Held-out sites</th><th>Held-out clips</th><th>Train clips</th></tr></thead>'
             f"<tbody>{rows}</tbody></table>")
    return f'<div class="cv-grid"><div>{table}<p class="hint">A fold holds out whole sites and trains on the other folds\' sites.</p></div>{heat}</div>'


def fingerprint_table(lock: dict) -> str:
    rows = "".join(f"<tr><td>{esc(name)}</td><td>{n(v['n_clips'])}</td><td><code>{v['sha256'][:16]}</code></td></tr>"
                   for name, v in lock["files"].items())
    return ('<details><summary>Fingerprints of the 12 pinned split files</summary><div class="table-wrap"><table class="mini">'
            "<thead><tr><th>File</th><th>Clips</th><th>sha256 (first 16)</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div></details>")


def splits_section(sp: dict, cv: dict, lock: dict, primary: set[str], secondary: set[str]) -> str:
    for dom in ("focal", "soundscape"):   # the summary and the pinned fingerprints must describe the same splits
        for s in SPLITS:
            assert sp[dom][s]["n_clips"] == lock["files"][f"{dom}_{s}.parquet"]["n_clips"], f"{dom}_{s}: summary and fingerprint differ"
    nf, ns = (sum(sp[d][s]["n_clips"] for s in SPLITS) for d in ("focal", "soundscape"))
    rec = sum(sp["focal"][s]["recordists"] for s in SPLITS)
    sites = sp["soundscape"]["sites"]
    tiles = [(n(nf), f"focal clips ({n(rec)} recordists)"), (n(ns), f"soundscape clips ({sum(len(v) for v in sites.values())} sites)"),
             (f'{len(sites["train"])} / {len(sites["val"])} / {len(sites["test"])}', "soundscape sites: train / val / test"),
             (str(len(lock["files"])), "split files pinned by checksum")]
    tile_html = "".join(f'<div class="stat-tile"><span class="value">{v}</span><span class="label">{esc(l)}</span></div>' for v, l in tiles)
    bg = {s: sp["soundscape"][s]["background_clips"] / sp["soundscape"][s]["n_clips"] for s in SPLITS}
    return f"""
  <section id="splits" aria-labelledby="splits-title">
    <h2 id="splits-title">Data splits</h2>
    <p class="hint">Every model is trained and scored on the same pinned splits. Focal clips are split by recordist and soundscape clips by recording site,
      because clips of one recording are near-duplicates and would leak between splits. The test split is used only for final numbers.
      Lighter to darker = train, validation, test.</p>
    <div class="tiles">{tile_html}</div>
    <figure class="chart" id="splits-sizes"><figcaption><h3>Clips per split</h3>
      <p class="hint">{pct(bg['train'])} of soundscape clips contain none of the 21 target species (background).</p></figcaption>
      {sizes_chart(sp)}</figure>
    <figure class="chart" id="splits-species"><figcaption><h3>Positive clips of each species, per split</h3>
      <p class="hint">Soundscape sites are chosen so that every species found at three or more sites keeps positives in train and test, with test shares close to the target;
        rare, site-clustered species cannot be balanced exactly. Focal shares follow the recordist grouping.</p></figcaption>
      {species_chart(sp, primary, secondary)}</figure>
    <figure class="chart" id="splits-cv"><figcaption><h3>Cross-validation folds (3, site-grouped, over the 70 non-test sites)</h3>
      <p class="hint">Recipe choices are made on these folds, never on the test sites. {len(cv['eligible_species'])} species occur at three or more sites and drive the fold balance
        (objective {cv['objective']:.3f}, lower is more even).</p></figcaption>
      {cv_block(cv)}</figure>
    {fingerprint_table(lock)}
  </section>"""


# -------------------------------------------------------------------------------------------------------- protocol
def log_x(v: float, x0: float, w: float, lo: float, hi: float) -> float:
    return x0 + w * (math.log10(max(v, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))


def threshold_chart(sp: dict, primary: set[str], secondary: set[str]) -> str:
    order = sorted(sp["soundscape"]["test"]["positives"], key=lambda k: -sp["soundscape"]["test"]["positives"][k])
    x0, w, lo, hi, row, top = 150, 520, 0.7, 1000, 22, 56
    body = text(x0, 14, "Test-set positive clips per species (log scale; 0 is drawn at the left edge)", "row-label")
    for t in (1, 10, 100, 1000):
        xt = log_x(t, x0, w, lo, hi)
        body += f'<line class="gridline" x1="{xt:.1f}" y1="{top - 8}" x2="{xt:.1f}" y2="{top + row * len(order) - 6}"/>' + text(xt, top - 14, str(t), "", "middle")
    for t, lab, anchor in ((canonical.MIN_POS, f"secondary: ≥ {canonical.MIN_POS} in both ", "end"),
                           (canonical.PRIMARY_MIN_POS, f" primary: ≥ {canonical.PRIMARY_MIN_POS} soundscape", "start")):
        xt = log_x(t, x0, w, lo, hi)
        body += f'<line class="diag" x1="{xt:.1f}" y1="{top - 8}" x2="{xt:.1f}" y2="{top + row * len(order) - 6}"/>' + text(xt, top - 28, lab, "row-label", anchor)
    for i, k in enumerate(order):
        y = top + i * row + 8
        role = "primary" if k in primary else ("secondary" if k in secondary else "not scored")
        body += text(0, y + 4, k, "row-label" + (" sel" if k in primary else "")) + text(60, y + 4, role if role != "primary" else "primary")
        f, s = sp["focal"]["test"]["positives"][k], sp["soundscape"]["test"]["positives"][k]
        xf, xs = log_x(f, x0, w, lo, hi), log_x(s, x0, w, lo, hi)
        body += f'<line class="connector" x1="{min(xf, xs):.1f}" y1="{y}" x2="{max(xf, xs):.1f}" y2="{y}"/>'
        body += f'<circle class="dot focal" cx="{xf:.1f}" cy="{y}" r="5"><title>{esc(k)}: {f} focal test positives</title></circle>'
        body += f'<circle class="dot soundscape" cx="{xs:.1f}" cy="{y}" r="5"><title>{esc(k)}: {s} soundscape test positives</title></circle>'
        body += text(x0 + w + 24, y + 4, f"{n(f)} / {n(s)}")
    height = top + row * len(order)
    body += text(x0 + w + 24, top - 14, "focal / soundscape")
    return svg(height + 4, "Test positives per species with the thresholds that define primary and secondary species", body)


def baseline_chart(model: dict, name: str) -> str:
    rows = [("Soundscape AP (primary)", "sc_ap_primary", "soundscape"), ("Soundscape AUC (primary)", "sc_auc_primary", "soundscape"),
            ("Soundscape cmAP (12 species)", "sc_cmap_12", "soundscape"), ("Focal AP (primary)", "f_ap_primary", "focal"),
            ("Focal AUC (primary)", "f_auc_primary", "focal")]
    x0, w, lo, hi, row, top = 200, 520, 0.3, 1.0, 34, 40
    xs = lambda v: x0 + w * (v - lo) / (hi - lo)
    body = text(x0, 14, f"{name}, test split; dot = estimate, bar = 95% cluster-bootstrap interval", "row-label")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        body += f'<line class="gridline" x1="{xs(t):.1f}" y1="{top - 8}" x2="{xs(t):.1f}" y2="{top + row * len(rows) - 12}"/>' + text(xs(t), top - 14, f"{t:.1f}", "", "middle")
    for i, (label, key, dom) in enumerate(rows):
        y = top + i * row + 6
        p, (a, b) = model["point"][key], model["ci95"][key]
        body += text(0, y + 4, label, "row-label")
        body += f'<line class="whisker {dom}" x1="{xs(a):.1f}" y1="{y}" x2="{xs(b):.1f}" y2="{y}"/>'
        body += f'<circle class="dot {dom}" cx="{xs(p):.1f}" cy="{y}" r="5"><title>{esc(label)}: {p:.3f} [{a:.3f}, {b:.3f}]</title></circle>'
        body += text(x0 + w + 14, y + 4, f"{p:.3f}  [{a:.3f}, {b:.3f}]")
    return svg(top + row * len(rows) - 4, "Protocol metrics of the pretrained model with 95% intervals", body)


def protocol_section(out: dict, base: str) -> str:
    th, sp = out["thresholds"], out["species"]
    m = out["models"][base]
    cards = [
        ("1 · Species", f"<b>Secondary</b>: at least {canonical.MIN_POS} positives in both the focal and soundscape test sets ({len(sp['secondary'])} species). "
         f"<b>Primary</b>: at least {canonical.PRIMARY_MIN_POS} soundscape test positives ({len(sp['primary'])}: {', '.join(sp['primary'])})."),
        ("2 · Headline metric", "<b>Soundscape macro AP over the primary species.</b> Also reported: soundscape AUC, cmAP over the secondary species, "
         "and focal AUC / AP to check that focal performance is retained."),
        ("3 · Prototype transfer", f"For each prototype, Δ = AUC<sub>soundscape</sub> − AUC<sub>focal</sub> of its maximum activation as a detector of its species. "
         f"<b>Fails</b> if Δ &lt; {th['fail']:+.2f}; <b>improves</b> if Δ &gt; {th['improve']:+.2f}. {canonical.K} prototypes per species."),
        ("4 · Uncertainty", f"Cluster bootstrap, {out['n_boot']} resamples shared across models: soundscape clips by site ({out['n_clusters']['sites']} test sites), "
         f"focal clips by recordist ({out['n_clusters']['recordists']}). Intervals are 95% percentile intervals; differences between models are paired."),
        ("5 · Development", "Choices are made on the validation split or 3-fold site-grouped cross-validation (<code>--split val</code>, <code>--split cv --fold F</code>); "
         "the test sites are scored once, for final numbers."),
        ("6 · Honesty rules", "Soundscape labels are used to evaluate and, for selection, only through val or the folds. Adaptation may use training-site audio "
         "without labels, never held-out audio. Every model is scored on identical clips (checked)."),
    ]
    card_html = "".join(f'<div class="step"><h3>{t}</h3><p>{b}</p></div>' for t, b in cards)
    crit = [("AP gap closed", f"at least {pct(th['gap_fraction'])} of the gap between the baseline and the supervised soundscape-only model, "
             "and the 95% interval of the improvement over the baseline excludes 0"),
            ("AUC not below baseline", "soundscape AUC (primary) is at least the baseline's"),
            ("Failing prototypes halved", "at most half as many prototypes fail as for the baseline"),
            ("Focal retained", f"focal AUC (primary) within {th['focal_tolerance']:.2f} of the baseline's")]
    crit_rows = "".join(f"<tr><td>{i + 1}</td><td>{a}</td><td>{b}</td></tr>" for i, (a, b) in enumerate(crit))
    return f"""
  <section id="protocol" aria-labelledby="protocol-title">
    <h2 id="protocol-title">Evaluation protocol ({esc(out['protocol'])})</h2>
    <p class="hint">One fixed way to score every model; changing any part means a new protocol version and re-scoring all models.
      Details: <a href="../docs/PROTOCOL.md">docs/PROTOCOL.md</a>. Code: <code>src/prototype_transfer/evaluation/canonical.py</code>.</p>
    <div class="steps">{card_html}</div>
    <figure class="chart" id="protocol-species"><figcaption><h3>Which species are scored</h3>
      <div class="legend"><span><i class="swatch focal"></i>focal test positives</span><span><i class="swatch soundscape"></i>soundscape test positives</span></div>
      <p class="hint">Dashed lines mark the two thresholds. Species below them are still trained on but not scored.</p></figcaption>
      {{THRESHOLDS}}</figure>
    <figure class="chart" id="protocol-baseline"><figcaption><h3>What the protocol measures: the pretrained model</h3>
      <div class="legend"><span><i class="swatch focal"></i>focal</span><span><i class="swatch soundscape"></i>soundscape</span></div>
      <p class="hint">{m['point']['n_fail']} of {len(m['prototypes'])} prototypes fail (Δ &lt; {th['fail']:+.2f}); median Δ {m['point']['median_delta']:+.3f}.</p></figcaption>
      {baseline_chart(m, base)}</figure>
    <h3>Success criteria for a method</h3>
    <p class="hint">All four must hold, measured against the baseline and the soundscape-only oracle.</p>
    <div class="table-wrap"><table class="mini crit"><thead><tr><th>#</th><th>Criterion</th><th>Definition</th></tr></thead><tbody>{crit_rows}</tbody></table></div>
  </section>"""


# ------------------------------------------------------------------------------------------------------------- main
def build(store: results.Store, baseline: str = "pretrained-xcl") -> tuple[str, dict[str, str]]:
    sp = json.loads((store.dataset_dir / "splits.json").read_text())
    cv = json.loads((store.dataset_dir / "splits_cv.json").read_text())
    lock = json.loads(SPLITS_LOCK.read_text())
    out = json.loads(store.summary("canonical_eval").read_text())
    primary, secondary = set(out["species"]["primary"]), set(out["species"]["secondary"])
    parts = {"splits": splits_section(sp, cv, lock, primary, secondary), "protocol": protocol_section(out, baseline)}
    parts["protocol"] = parts["protocol"].replace("{THRESHOLDS}", threshold_chart(sp, primary, secondary))
    return START + parts["splits"] + parts["protocol"] + "\n  " + END, parts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=results.DEFAULT_DATASET)
    p.add_argument("--html", type=Path, default=Path("dashboard/index.html"))
    args = p.parse_args()
    block, _ = build(results.Store(args.dataset))
    page = args.html.read_text()
    if START in page and END in page:
        a, b = page.index(START), page.index(END) + len(END)
        page = page[:a] + block + page[b:]
    else:   # first run: place the sections after the summary tiles, before the findings
        anchor = '<section aria-labelledby="findings-title">'
        assert anchor in page, "index.html has no findings section to anchor on"
        page = page.replace(anchor, block + "\n\n  " + anchor, 1)
    args.html.write_text(page)
    print(f"updated {args.html}")


if __name__ == "__main__":
    main()
