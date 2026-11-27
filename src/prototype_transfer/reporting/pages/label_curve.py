"""Label curve: how many labeled soundscape clips does it take to close the gap?"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from prototype_transfer.reporting import charts as c
from prototype_transfer.reporting.runs import BASELINE, FOLDS, LABEL_BUDGETS, ORACLE, Results
from prototype_transfer.reporting.site import Page

KEYS = (BASELINE, *(f"lb{k}" for k in LABEL_BUDGETS), ORACLE)
TICKS = ("0", *(f"{k:,}" for k in LABEL_BUDGETS), "all")
NAMES = ("pretrained model (0 labels)", *(f"{k:,} labeled clips" for k in LABEL_BUDGETS), "all labeled clips")


def curve(per_x: Sequence[Sequence[float]], ticks: Sequence[str], names: Sequence[str], lo: float, hi: float,
          yticks: Sequence[float], *, width: float = c.W, height: float = 340, fmt=lambda v: c.score(v, 0),
          value=c.score, refs: Sequence[tuple[float, str, str]] = (), label_means: bool = True,
          folds: bool = True, title: str = "", xlabel: str = "Labeled soundscape clips used for training (not to scale)") -> str:
    """Per-fold thin lines and their mean over evenly spaced x positions. per_x[i] = values of the folds at position i."""
    left, right, top, bottom = 56, width - 14, 30 if title else 14, height - 46
    y = c.linear(lo, hi, bottom, top)
    xs = [left + 24 + i * (right - left - 48) / (len(ticks) - 1) for i in range(len(ticks))]
    body = c.text(left - 44, 14, title, "panel-title") if title else ""
    body += c.y_grid(y, yticks, left, right, fmt)
    for i, t in enumerate(ticks):
        body += c.text(xs[i], bottom + 17, t, "", "middle")
    body += c.text((left + right) / 2, height - 8, xlabel, "axis-title", "middle")
    for v, label, cls in refs:
        body += c.line(left, y(v), right, y(v), cls) + c.text(right, y(v) - 5, label, "ref-label", "end")
    if folds:
        for f in range(len(per_x[0])):
            pts = " ".join(f"{xs[i]:.1f},{y(per_x[i][f]):.1f}" for i in range(len(ticks)))
            body += f'<polyline class="fold-line" points="{pts}"/>'
            for i in range(len(ticks)):
                body += c.circle(xs[i], y(per_x[i][f]), 3.5, "dot soundscape fold", f"{names[i]}, fold {f}: {value(per_x[i][f])}")
    means = [float(np.mean(v)) for v in per_x]
    body += f'<polyline class="mean-line" points="{" ".join(f"{a:.1f},{y(m):.1f}" for a, m in zip(xs, means))}"/>'
    for i, m in enumerate(means):
        body += c.circle(xs[i], y(m), 5, "dot soundscape", f"{names[i]}: mean {value(m)}")
        if label_means:
            body += c.text(xs[i] - 6, y(m) - 10, value(m), "value", "end")
    return c.svg(height, title or xlabel, body, width)


def build(res: Results) -> Page:
    ap = [res.values(k, "sc_ap_primary") for k in KEYS]
    mean = [float(np.mean(v)) for v in ap]
    gamma = {k: res.cv[f"lb{k}"]["gap_fraction"] for k in LABEL_BUDGETS}
    half = next(k for k in LABEL_BUDGETS if gamma[k]["mean"] >= 0.5)
    better = {k: res.folds_better_worse(f"lb{k}")[0] for k in LABEL_BUDGETS}
    every = next(k for k in LABEL_BUDGETS if all(better[j] == len(FOLDS) for j in LABEL_BUDGETS if j >= k))
    focal_pass = {k: res.criteria(f"lb{k}")[3] for k in LABEL_BUDGETS}
    last = LABEL_BUDGETS[-1]
    sub = res.labeled_subsets()
    shared_gain = {k: float(np.mean(res.shared_deltas(f"lb{k}"))) for k in LABEL_BUDGETS}
    shared_full = float(np.mean(res.shared_deltas(ORACLE)))
    shared_gamma = {k: shared_gain[k] / shared_full for k in LABEL_BUDGETS}   # ratio of the mean gains
    fixed = res.fixed_species()
    fixed_full = float(np.mean(res.set_deltas(ORACLE, fixed)))
    fixed_gamma = {k: float(np.mean(res.set_deltas(f"lb{k}", fixed))) / fixed_full for k in LABEL_BUDGETS}
    for g in (shared_gamma, fixed_gamma):
        assert next(k for k in LABEL_BUDGETS if g[k] >= 0.5) == half, "the half-gap budget depends on the species set"
    bg = [x["background"] for k in LABEL_BUDGETS for x in sub[k]]
    absent = sorted({s for x in sub[LABEL_BUDGETS[0]] for s, v in x["positives"].items() if v == 0})

    full_clips = [res.splits_cv["n_clips"][str(f)]["train"] for f in FOLDS]
    for f, n_clips in zip(FOLDS, full_clips):
        cfg = json.loads((res.store.run_dir(f"{ORACLE}-cv{f}") / "config.json").read_text())
        assert cfg["effective"]["n_train_clips"] == n_clips and cfg["args"]["train_domain"] == "soundscape" \
            and cfg["args"]["train_labels"] is None, "the all-labels model no longer matches the explanation of “all”"
    meaning = c.takeaways([
        f"<b>“All”</b> is the pretrained model fine-tuned on <b>every labeled soundscape clip of that fold's training sites</b>: "
        f"{min(full_clips):,}–{max(full_clips):,} clips, empty ones included.",
        "It does <b>not</b> mean all the data. The held-out sites it is scored on and the test sites are never included, and it uses "
        "no focal training data.",
        f"The other points use a random N of those same clips ({LABEL_BUDGETS[0]:,} to {LABEL_BUDGETS[-1]:,}); <b>0</b> is the "
        "pretrained model, unchanged. “All” is the reference that counts as closing 100% of the gap. It is not a proven upper "
        "limit.",
    ], "What “all” means on this page")
    tiles = c.tiles([(f"{half:,}", "smallest tested label budget closing at least half of the gap on average"),
                     (c.pct(gamma[half]["mean"]), f"of the gap closed with {half:,} labels (folds: "
                      + ", ".join(c.pct(v) for v in gamma[half]["folds"]) + ")"),
                     (c.pct(gamma[last]["mean"]), f"closed with {last:,} labels"),
                     (c.score(mean[-1] - mean[0]), "total gap in soundscape AP, in points: all labels minus pretrained")])
    take = c.takeaways([
        f"With {LABEL_BUDGETS[0]} labeled clips the effect is small and not reliable ({better[LABEL_BUDGETS[0]]} of 3 folds reliably better; "
        f"mean gap closed {c.pct(gamma[LABEL_BUDGETS[0]]['mean'])}).",
        f"At {half:,} labels the average gap closed passes one half, but the improvement is reliable on only {better[half]} of 3 folds. "
        f"From {every:,} labels it is reliable on every fold.",
        f"The {half:,}-label threshold holds when every fold scores the same species: {half:,} labels close "
        f"{c.pct(shared_gamma[half])} on the {len(res.shared_species())} primary on all folds and {c.pct(fixed_gamma[half])} on the "
        f"{len(fixed)} eligible on all folds. These budgets are for full fine-tuning; a more "
        "label-efficient method might need fewer, so this is not yet the bar a label-free method must beat.",
        f"The curve flattens: {last:,} labels average {c.score(mean[-2])}, all labels {c.score(mean[-1])}.",
        "<b>The cost:</b> training on soundscapes lowers focal AUC. The focal-retention criterion is met on "
        + ", ".join(f"{focal_pass[k]}/3 folds at {k:,}" for k in LABEL_BUDGETS) + " labels. The absolute drop is small, and these focal "
        "clips were part of the pretraining data, which favours the pretrained model.",
    ])
    refs = [(mean[0], "pretrained model", "ref-base"), ((mean[0] + mean[-1]) / 2, "halfway to all labels", "ref-half")]
    main_fig = c.figure("lc-ap", "Soundscape AP on held-out sites vs number of labeled clips",
                        "Thin lines: the 3 cross-validation folds. Thick line: their mean. Folds differ in sites and scored species, so "
                        "compare points along a line rather than between lines.",
                        curve(ap, TICKS, NAMES, 0.65, 0.92, (0.65, 0.7, 0.75, 0.8, 0.85, 0.9), refs=refs, height=380),
                        c.legend([("soundscape", "mean of folds"), ("soundscape faded", "single fold")]))
    g_vals = [gamma[k]["folds"] for k in LABEL_BUDGETS]
    gap_fig = c.figure("lc-gap", "Share of the gap closed (γ)",
                       "0% = pretrained model, 100% = trained on all labeled clips; computed within each fold. Values below 0% mean worse "
                       "than the pretrained model.",
                       curve(g_vals, TICKS[1:-1], NAMES[1:-1], -0.25, 1.2, (-0.25, 0, 0.25, 0.5, 0.75, 1.0), height=300,
                             fmt=c.pct, value=c.pct, refs=[(0.5, "target: half the gap", "ref-half"), (0.0, "pretrained", "ref-base")]))
    panel_w = 360
    base_focal = float(np.mean(res.values(BASELINE, "f_auc_primary")))
    tol = res.test["thresholds"]["focal_tolerance"]
    sc_auc = curve([res.values(k, "sc_auc_primary") for k in KEYS], TICKS, NAMES, 0.90, 1.0, (0.9, 0.92, 0.94, 0.96, 0.98, 1.0),
                   width=panel_w, height=280, label_means=False, title="Soundscape AUC (higher is better)", xlabel="Labeled clips")
    f_auc = curve([res.values(k, "f_auc_primary") for k in KEYS], TICKS, NAMES, 0.96, 1.0, (0.96, 0.97, 0.98, 0.99, 1.0),
                  width=panel_w, height=280, label_means=False, title="Focal AUC (dotted: retention limit)", xlabel="Labeled clips",
                  refs=[(base_focal - tol, "", "ref-half")])
    fail = curve([res.fail_share(k) for k in KEYS], TICKS, NAMES, 0, 0.6, (0, 0.2, 0.4, 0.6), width=panel_w, height=280,
                 label_means=False, title="Failing prototypes (lower is better)", xlabel="Labeled clips", fmt=c.pct, value=c.pct)
    side = c.figure("lc-other", "The other success criteria", "",
                    f'<div class="panels">{sc_auc}{f_auc}{fail}</div>',
                    c.legend([("soundscape", "mean of folds"), ("soundscape faded", "single fold")]))
    panels = ""
    for s in res.primary_union():
        vals = [[res.species_mean(k, s)[0]] for k in KEYS]
        nf = res.species_mean(BASELINE, s)[1]
        panels += curve(vals, ("0", "100", "250", "500", "1k", "2k", "all"), NAMES, 0.2, 1.0, (0.2, 0.4, 0.6, 0.8, 1.0), width=280,
                        height=210, label_means=False, folds=False, title=f"{c.name_of(s)} ({nf} fold{'s' if nf > 1 else ''})",
                        xlabel="Labeled clips", refs=[(res.species_mean(BASELINE, s)[0], "", "ref-base")])
    species = c.figure("lc-species", "Species by species",
                       "Soundscape AP, averaged over the folds on which the species is primary (number in the title). Dashed: pretrained "
                       "model. Species do not respond alike, and species scored on one fold are especially noisy.",
                       f'<div class="panels small">{panels}</div>')

    rows = []
    for k in LABEL_BUDGETS:
        cells = [f"{k:,}", f"{c.pct(min(x['background'] for x in sub[k]))}–{c.pct(max(x['background'] for x in sub[k]))}",
                 f"{min(x['sites'] for x in sub[k])}–{max(x['sites'] for x in sub[k])}"]
        for s in res.primary_union():
            v = [x["positives"][s] for x in sub[k] if s in x["positives"]]
            cells.append(f"{min(v)}–{max(v)}" if min(v) != max(v) else str(v[0]))
        rows.append(cells)
    content = c.section("lc-content", "What the labeled clips contain",
                        c.table(["Labeled clips", "Empty clips", "Sites", *(c.name_of(s) for s in res.primary_union())], rows,
                                "mini wrap-head"),
                        f"Range over folds; species columns count positive clips (folds on which the species is primary). Clips are drawn "
                        f"uniformly at random, so {c.pct(min(bg))}–{c.pct(max(bg))} contain no target species, and at {LABEL_BUDGETS[0]} "
                        f"clips some folds have no positive at all for {', '.join(c.name_of(s) for s in absent)}. "
                        "Smaller sets are contained in larger ones.")
    nrows = []
    for key, name in zip(KEYS, NAMES):
        cells = [name]
        v = res.values(key, "sc_ap_primary")
        cells += [c.score(x) for x in v] + [c.score(np.mean(v))]
        if key.startswith("lb"):
            g = res.cv[key]["gap_fraction"]
            cells += [c.pct(g["mean"]), c.pct(shared_gamma[int(key[2:])]), c.pct(fixed_gamma[int(key[2:])]),
                      f"{res.folds_better_worse(key)[0]}/3"]
        else:
            full = "0%" if key == BASELINE else "100%"
            cells += [full, full, full, "—" if key == BASELINE else f"{res.folds_better_worse(key)[0]}/3"]
        nrows.append(cells)
    numbers = c.section("lc-numbers", "Numbers",
                        c.table(["Training data", "Fold 0", "Fold 1", "Fold 2", "Mean AP", "Mean γ", f"γ, same {len(res.shared_species())}", f"γ, same {len(fixed)}", "Folds reliably better"], nrows))
    limits = c.section("lc-limits", "How to read this curve", """<ul class="prose">
    <li>Each budget is <b>one random draw</b> with one training seed. The 3 folds give only a rough idea of the variation.</li>
    <li>Labeled clips are spread over most training sites. Annotating a few sites thoroughly could give a different curve.</li>
    <li>All budgets use the same number of optimiser steps, so small sets are repeated many times. The last epoch is used; no labeled
    validation set selects the model. A budget therefore counts all soundscape labels the training uses.</li>
    <li>Every point fine-tunes the whole network. A label-efficient method (frozen backbone, or reweighting only) may need fewer labels;
    that has not been tested.</li></ul>""")
    return Page("label-curve.html", "Label curve", "Label curve",
                "How many labeled soundscape clips does full fine-tuning need to close the gap?",
                meaning + tiles + take + main_fig + gap_fig + side + species + content + numbers + limits)
