"""Splits: how clips are divided so that no recording leaks between train and evaluation."""

from __future__ import annotations

import math

from prototype_transfer.reporting import charts as c
from prototype_transfer.reporting.runs import FOLDS, Results
from prototype_transfer.reporting.site import Page

SPLITS = ("train", "val", "test")
SHADE = {"train": 0.35, "val": 0.65, "test": 1.0}        # light -> dark: the held-out split is the darkest
FOCAL_TARGET, SC_TARGET = (0.7, 0.1, 0.2), (0.6, 0.1, 0.3)  # intended train / val / test shares (focal: clips; soundscape: sites)


def stacked(x0: float, y: float, w: float, h: float, parts: list[float], dom: str, titles: list[str], gap: float = 2) -> str:
    """One horizontal bar split into train / val / test segments with a surface gap between fills."""
    total = sum(parts)
    out, x = "", x0
    for part, s, t in zip(parts, SPLITS, titles):
        if total == 0 or part == 0:
            continue
        seg = w * part / total
        out += c.rect(x, y, max(seg - gap, 0.5), h, f"seg f-{dom}", t, SHADE[s])
        x += seg
    return out


def sizes_chart(sp: dict) -> str:
    rows = [("Focal (Xeno-Canto), grouped by recordist", "focal", [sp["focal"][s]["n_clips"] for s in SPLITS],
             [f'{sp["focal"][s]["recordists"]} recordists' for s in SPLITS]),
            ("Soundscape (field recordings), grouped by recording site", "soundscape", [sp["soundscape"][s]["n_clips"] for s in SPLITS],
             [f'{len(sp["soundscape"]["sites"][s])} sites' for s in SPLITS])]
    body, y = "", 8
    for title, dom, counts, groups in rows:
        total = sum(counts)
        body += c.text(0, y + 12, title, "row-label")
        titles = [f"{s}: {c.n(k)} clips ({c.pct(k / total)}), {g}" for s, k, g in zip(SPLITS, counts, groups)]
        body += stacked(0, y + 22, c.W, 26, counts, dom, titles)
        for j, (s, k, g) in enumerate(zip(SPLITS, counts, groups)):
            x = j * c.W / 3
            body += (c.rect(x, y + 58, 12, 12, f"seg f-{dom}", "", SHADE[s], 2)
                     + c.text(x + 20, y + 68, f"{s}: {c.n(k)} clips ({c.pct(k / total)})", "row-label") + c.text(x + 20, y + 84, g))
        y += 108
    return c.svg(y, "Clips per split for focal and soundscape data", body)


def species_chart(sp: dict, primary: set[str], secondary: set[str]) -> str:
    tot = {k: sum(sp["soundscape"][s]["positives"][k] for s in SPLITS) for k in sp["soundscape"]["train"]["positives"]}
    order = sorted(tot, key=lambda k: -tot[k])
    row, top = 22, 54
    panels = [("focal", "Focal: share of each species' clips", 200, 220, FOCAL_TARGET),
              ("soundscape", "Soundscape: share of each species' positive clips", 530, 220, SC_TARGET)]
    body = c.text(0, 14, "P = primary species, S = secondary, – = not scored on test. Dashed lines: intended train | val | test shares. "
                         "Right of each bar: total (test).", "")
    for _, title, x0, w, tgt in panels:
        body += c.text(x0, 36, title, "row-label")
        for frac in (tgt[0], tgt[0] + tgt[1]):
            body += c.line(x0 + w * frac, top - 6, x0 + w * frac, top + row * len(order) - 4, "diag")
    for i, k in enumerate(order):
        y = top + i * row
        role = "P" if k in primary else ("S" if k in secondary else "–")
        body += c.text(0, y + 11, c.name_of(k), "row-label")
        body += c.text(176, y + 11, role, "role-" + {"P": "P", "S": "S"}.get(role, "none"))
        for dom, _, x0, w, _t in panels:
            counts = [sp[dom][s]["positives"][k] for s in SPLITS]
            total = sum(counts)
            if total == 0:
                body += c.text(x0, y + 11, "no positives")
                continue
            titles = [f"{c.name_of(k)} ({dom}) {s}: {c.n(v)} of {c.n(total)} ({c.pct(v / total)})" for s, v in zip(SPLITS, counts)]
            body += stacked(x0, y, w, 13, counts, dom, titles)
            body += c.text(x0 + w + 8, y + 11, f"{c.n(total)} ({c.n(counts[2])})")
    return c.svg(top + row * len(order) + 14, "Each species' positive clips split into train, validation and test", body)


def cv_heatmap(cv: dict, primary_by_fold: list[list[str]]) -> str:
    folds = sorted(cv["sites"], key=int)
    species = sorted(cv["eligible_species"], key=lambda k: -sum(cv["positives"][f][k] for f in folds))
    mx = max(cv["positives"][f][k] for f in folds for k in species)
    cw, ch, x0, top = 150, 21, 230, 40
    body = c.text(x0, 14, "Positive clips per held-out fold (outlined: primary on that fold)", "row-label")
    for j, f in enumerate(folds):
        body += c.text(x0 + j * cw + cw / 2, top - 8, f"fold {f}", "", "middle")
    for i, k in enumerate(species):
        y = top + i * ch
        body += c.text(x0 - 12, y + 14, c.name_of(k), "row-label", "end")
        for j, f in enumerate(folds):
            v = cv["positives"][f][k]
            op = 0.08 + 0.52 * math.sqrt(v / mx) if v else 0
            cls = "seg f-soundscape" + (" outlined" if k in primary_by_fold[int(f)] else "")
            body += c.rect(x0 + j * cw + 2, y + 2, cw - 4, ch - 4, cls, f"{c.name_of(k)}, fold {f}: {v} positive clips", op)
            body += c.text(x0 + j * cw + cw / 2, y + 14, c.n(v), "cell", "middle")
    return c.svg(top + ch * len(species) + 6, "Positive clips per species in each cross-validation fold", body)


def build(res: Results) -> Page:
    sp, cv = res.splits, res.splits_cv
    nf, ns = (sum(sp[d][s]["n_clips"] for s in SPLITS) for d in ("focal", "soundscape"))
    rec = sum(sp["focal"][s]["recordists"] for s in SPLITS)
    sites = sp["soundscape"]["sites"]
    primary, secondary = set(res.test["species"]["primary"]), set(res.test["species"]["secondary"])
    tiles = c.tiles([(c.n(nf), f"focal clips from {c.n(rec)} recordists"),
                     (c.n(ns), f"soundscape clips from {sum(len(v) for v in sites.values())} sites"),
                     (f'{len(sites["train"])} / {len(sites["val"])} / {len(sites["test"])}', "soundscape sites: train / validation / test"),
                     (str(len(cv["sites"])), "site-grouped cross-validation folds over the non-test sites")])
    rules = c.takeaways([
        "<b>Focal clips are split by recordist.</b> One person's recordings share equipment and place.",
        "<b>Soundscape clips are split by recording site.</b> Neighbouring 5 s clips are near-duplicates, and species cluster by site.",
        "<b>Test sites are reserved for final numbers.</b> Methods are compared by cross-validation over the other sites.",
        "<b>Splits are fixed.</b> Every model is trained and scored on the same ones.",
    ], "Splitting rules")
    sizes = c.figure("splits-sizes", "Clips per split", f"{c.pct(sp['soundscape']['train']['background_clips'] / sp['soundscape']['train']['n_clips'])} "
                     "of soundscape training clips contain none of the target species. Lighter to darker: train, validation, test.", sizes_chart(sp))
    species = c.figure("splits-species", "Positive clips of each species, per split",
                       "Sites are assigned so that every species found at three or more sites keeps positives in both train and test, "
                       "with shares close to the target. Rare, site-clustered species cannot be balanced exactly.",
                       species_chart(sp, primary, secondary))
    rows = [[f"fold {f}", str(len(cv["sites"][str(f)])), c.n(cv["n_clips"][str(f)]["heldout"]), c.n(cv["n_clips"][str(f)]["train"]),
             str(len(res.species[f]["primary"])), str(len(res.species[f]["secondary"]))] for f in FOLDS]
    folds = c.figure("splits-cv", "Cross-validation folds",
                     "Each fold holds out whole sites. A model is trained on the other folds' sites and scored on the held-out ones. "
                     "Focal training data is the same for every fold, and focal retention is measured on the focal validation clips. "
                     "<b>Which species qualify for scoring differs between folds</b>, so fold averages are not over one fixed species set.",
                     c.table(["Fold", "Held-out sites", "Held-out clips", "Soundscape training clips", "Primary species", "Secondary species"], rows)
                     + '<div class="cv-heat">' + cv_heatmap(cv, [s["primary"] for s in res.species]) + "</div>")
    return Page("splits.html", "Splits", "Data splits", "Splits are built so that no recording, recordist or site leaks into evaluation.",
                tiles + rules + sizes + species + folds)
