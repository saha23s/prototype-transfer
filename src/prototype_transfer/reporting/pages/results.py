"""Initial results: every method against the pretrained model, by cross-validation."""

from __future__ import annotations

import math

import numpy as np

from prototype_transfer.evaluation import results
from prototype_transfer.reporting import charts as c
from prototype_transfer.reporting.runs import (BASELINE, CONTRASTS, FOLDS, GROUPS, LABEL_FREE, METHODS, ORACLE,
                                                TEST_RUNS, Results)
from prototype_transfer.reporting.site import Page

SPECIES_TABLE_KEYS = ((BASELINE, "Pretrained"), ("r0x-nopush", "Frozen, no projection"), ("focal-only", "Full fine-tuning"),
                      ("r3", "Background + overlap"), ("lb250", "250 labels"), ("lb1000", "1,000 labels"), (ORACLE, "All labels"))
HEATMAP_ROWS = ("tf-quantile", "tf-reweight", "focal-only", "r0", "r3", "r4", "r0x-nopush", "lb250", "lb1000", ORACLE)


def dot_rows(res: Results, per_method, lo: float, hi: float, ticks, fmt, xlabel: str, ref: float, ref_label: str,
             tip, solid, right_label, right_head: str) -> str:
    """One row per method, grouped: a dot per fold and a tick at the mean."""
    x0, x1, row, head, top = 300, 800, 25, 30, 34
    x = c.linear(lo, hi, x0, x1)
    height = top + sum(head + row * len(ms) for _, _, ms in GROUPS)
    body = c.x_grid(x, ticks, top, height, fmt)
    body += c.line(x(ref), top, x(ref), height, "ref-base") + c.text(x(ref), 14, ref_label, "ref-label", "middle")
    body += c.text(x1 + 18, 28, right_head, "")
    y = top
    for gkey, gtitle, methods in GROUPS:
        body += c.text(0, y + 21, gtitle, "group-label")
        y += head
        for m in methods:
            vals = per_method(m.key)
            mean = float(np.mean(vals))
            body += c.text(x0 - 14, y + 16, m.label, "row-label", "end")
            body += c.line(x(min(vals)), y + 12, x(max(vals)), y + 12, "connector")
            for f, v in enumerate(vals):
                cls = "dot soundscape fold" if solid(m.key, f) else "dot hollow"
                body += c.circle(x(v), y + 12, 4.5, cls, tip(m, f, v))
            body += c.line(x(mean), y + 3, x(mean), y + 21, "mean-tick")
            body += c.text(x1 + 18, y + 16, right_label(mean), "cell")
            y += row
    body += c.text((x0 + x1) / 2, height + 36, xlabel, "axis-title", "middle")
    return c.svg(height + 44, xlabel, body)


def delta_chart(res: Results) -> str:
    def tip(m, f, v):
        lo, hi = res.delta_ci(m.key, f)
        return f"{m.label}, fold {f}: {c.signed(v)} (95% interval {c.signed(lo)} to {c.signed(hi)})"

    def solid(key, f):
        lo, hi = res.delta_ci(key, f)
        return lo > 0 or hi < 0

    return dot_rows(res, res.deltas, -0.20, 0.20, (-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2),
                    lambda t: "0" if t == 0 else c.signed(t, 2),
                    "Change in soundscape AP vs the pretrained model, same held-out sites", 0.0, "pretrained model", tip, solid,
                    c.signed, "mean change")


def species_sets(res: Results) -> tuple[tuple[str, str], ...]:
    """(key in cv_contrasts, label) of the three species sets a contrast is shown on."""
    return (("primary", "scored species (change by fold)"), ("shared", f"same {len(res.shared_species())} on every fold"),
            ("fixed", f"same {len(res.fixed_species())} on every fold"))


def contrast_chart(res: Results) -> str:
    """One block per ingredient, one row per species set: the 3-fold mean of variant - control with its paired interval."""
    x0, x1, row, head, top = 330, 640, 17, 22, 30
    sets = species_sets(res)
    x = c.linear(-0.14, 0.10, x0, x1)
    block = head + row * len(sets)
    bottom = top + block * len(CONTRASTS)
    body = c.x_grid(x, (-0.1, -0.05, 0, 0.05, 0.1), top - 6, bottom, lambda t: "0" if t == 0 else c.signed(t, 2))
    body += c.line(x(0), top - 6, x(0), bottom, "ref-base") + c.text(x(0), 14, "same as control", "ref-label", "middle")
    body += c.text(x1 + 24, 14, "mean [95% interval]", "") + c.text(c.W, 14, "folds up", "", "end")
    for i, (key, control, what) in enumerate(CONTRASTS):
        y0 = top + i * block
        body += c.text(0, y0 + 16, what, "group-label")
        for j, (name, label) in enumerate(sets):
            y = y0 + head + j * row + 7
            e = res.contrast_interval(key, control, name)
            mean, (lo, hi) = e["mean"]["diff"], e["mean"]["ci95"]
            assert -0.14 < lo and hi < 0.10, f"interval of {key} - {control} is outside the axis"
            up = sum(f["diff"] > 0 for f in e["folds"])
            tip = (f"{METHODS[key].label} minus {METHODS[control].label}, {label}: mean {c.signed(mean)} "
                   f"(95% interval {c.signed(lo)} to {c.signed(hi)}); folds " + ", ".join(c.signed(f["diff"]) for f in e["folds"]))
            body += c.text(x0 - 14, y + 4, label, "row-label", "end")
            body += c.line(x(lo), y, x(hi), y, "mean-tick")
            body += c.circle(x(mean), y, 4.5, "dot soundscape" if lo > 0 or hi < 0 else "dot hollow", tip)
            body += c.text(x1 + 24, y + 4, f"{c.signed(mean)} [{c.signed(lo)}, {c.signed(hi)}]", "cell")
            body += c.text(c.W, y + 4, f"{up}/{len(FOLDS)}", "cell", "end")
    body += c.text((x0 + x1) / 2, bottom + 36, "Change in soundscape AP: variant minus its control, mean of 3 folds", "axis-title", "middle")
    return c.svg(bottom + 44, "Effect of single ingredients against the matching control, on three species sets", body)


def proto_auc_chart(res: Results) -> str:
    """The two terms of delta: mean prototype detection AUC on focal and on soundscape clips."""
    x0, x1, row, head, top = 300, 720, 22, 28, 18
    lo = 0.74
    assert min(np.mean(res.proto_auc(k, "matched")) for k in (BASELINE, *METHODS)) > lo + 0.005, "prototype AUC below the axis"
    x = c.linear(lo, 1.0, x0, x1)
    rows = [(None, [BASELINE])] + [(title, [m.key for m in ms]) for _, title, ms in GROUPS]
    height = top + sum((head if t else 0) + row * len(ks) for t, ks in rows)
    base_s = float(np.mean(res.proto_auc(BASELINE, "soundscape")))
    body = c.x_grid(x, (0.8, 0.85, 0.9, 0.95, 1.0), top, height, lambda t: f"{t:.2f}")
    body += c.line(x(base_s), top, x(base_s), height, "ref-base") + c.text(x(base_s), 12, "pretrained, soundscape", "ref-label", "middle")
    body += c.text(c.W, 12, "focal / sc / sc matched", "", "end")
    y = top
    for title, keys in rows:
        if title:
            body += c.text(0, y + 20, title, "group-label")
            y += head
        for key in keys:
            label = "Pretrained (no adaptation)" if key == BASELINE else METHODS[key].label
            f, s, mt = (float(np.mean(res.proto_auc(key, d))) for d in ("focal", "soundscape", "matched"))
            body += c.text(x0 - 14, y + 15, label, "row-label", "end")
            body += c.line(x(min(s, mt)), y + 11, x(f), y + 11, "connector")
            body += c.circle(x(mt), y + 11, 4.2, "dot hollow", f"{label}: mean prototype AUC on soundscapes, other-bird negatives {mt:.3f}")
            body += c.circle(x(f), y + 11, 4.5, "dot focal", f"{label}: mean prototype AUC on focal clips {f:.3f}")
            body += c.circle(x(s), y + 11, 4.5, "dot soundscape", f"{label}: mean prototype AUC on soundscapes {s:.3f}")
            body += c.text(c.W, y + 15, f"{f:.3f} / {s:.3f} / {mt:.3f}", "cell", "end")
            y += row
    body += c.text((x0 + x1) / 2, height + 36, "Mean prototype detection AUC (axis does not start at 0)", "axis-title", "middle")
    return c.svg(height + 44, "Mean prototype detection AUC on focal and soundscape clips by method", body)


def fail_table(res: Results) -> str:
    head = ["Method", "Failing share<br>(protocol: all soundscape negatives)", "Failing share<br>(matched: other-bird negatives)"]
    cells = lambda k: [c.pct(float(np.mean(res.fail_share(k)))), c.pct(float(np.mean(res.fail_share_matched(k))))]
    rows, classes = [["<b>Pretrained (no adaptation)</b>", *cells(BASELINE)]], [""]
    for _, title, methods in GROUPS:
        rows.append([title, "", ""])
        classes.append("group")
        for m in methods:
            rows.append([m.label, *cells(m.key)])
            classes.append("")
    return c.table(head, rows, "mini results", classes)


def species_heatmap(res: Results) -> str:
    species = res.primary_union()
    x0, cw, ch, top = 300, 74, 28, 66
    body = ""
    for j, s in enumerate(species):
        x = x0 + j * cw + cw / 2
        words = c.name_of(s).split(" ", 1)
        for i, w in enumerate(words):
            body += c.text(x, 14 + 14 * i, w, "col-label", "middle")
        base, nf = res.species_mean(BASELINE, s)
        body += c.text(x, 46, f"{base:.2f}", "cell", "middle") + c.text(x, 59, f"{nf} fold{'s' if nf > 1 else ''}", "", "middle")
    body += c.text(x0 - 14, 46, "pretrained AP →", "row-label", "end")
    for r, key in enumerate(HEATMAP_ROWS):
        y = top + r * ch
        body += c.text(x0 - 14, y + 18, METHODS[key].label, "row-label", "end")
        for j, s in enumerate(species):
            d = res.species_mean(key, s)[0] - res.species_mean(BASELINE, s)[0]
            op = 0.06 + 0.54 * math.sqrt(min(abs(d) / 0.3, 1.0))
            cls = "heat-up" if d > 0 else "heat-down"
            body += c.rect(x0 + j * cw + 1, y + 1, cw - 2, ch - 2, cls, f"{METHODS[key].label}, {c.name_of(s)}: {c.signed(d)}", op)
            body += c.text(x0 + j * cw + cw / 2, y + 18, c.signed(d, 2), "cell", "middle")
    return c.svg(top + ch * len(HEATMAP_ROWS) + 6, "Change in soundscape AP per species and method", body)


def results_table(res: Results) -> str:
    head = ["Method", "AP<br>mean ± sd", "AP<br>change", f"Change,<br>same {len(res.shared_species())}", f"Change,<br>same {len(res.fixed_species())}", "Folds<br>better /<br>worse", "AUC", "cmAP", "Focal<br>AUC",
            "Criteria met (folds of 3)<br>gap · AUC · protos · focal"]
    ms = lambda key, metric: float(np.mean(res.values(key, metric)))
    sd = lambda key: float(np.std(res.values(key, "sc_ap_primary"), ddof=1))

    def row(key: str, label: str) -> list[str]:
        base = key == BASELINE
        crit = None if base else res.criteria(key)
        bw = "—" if base else "{} / {}".format(*res.folds_better_worse(key))
        return [label, f"{ms(key, 'sc_ap_primary'):.3f} ± {sd(key):.3f}", "—" if base else c.signed(float(np.mean(res.deltas(key)))),
                "—" if base else c.signed(float(np.mean(res.shared_deltas(key)))),
                "—" if base else c.signed(float(np.mean(res.set_deltas(key, res.fixed_species())))), bw,
                f"{ms(key, 'sc_auc_primary'):.3f}", f"{ms(key, 'sc_cmap_12'):.3f}", f"{ms(key, 'f_auc_primary'):.3f}",
                "reference" if crit is None else " · ".join(str(v) for v in crit)]

    rows, classes = [row(BASELINE, "<b>Pretrained (no adaptation)</b>")], [""]
    for _, title, methods in GROUPS:
        rows.append([title] + [""] * (len(head) - 1))
        classes.append("group")
        for m in methods:
            rows.append(row(m.key, m.label))
            classes.append("")
    return c.table(head, rows, "mini results", classes)


def group_table(res: Results) -> str:
    """Soundscape AP of every method on the 7 fixed species, the other 5 and all 12."""
    seven, five = res.fixed_species(), res.other_test_species()
    groups = (seven, five, seven + five)
    head = ["Method"] + [f"{name}<br>{col}" for name in (f"Main {len(seven)}", f"Other {len(five)}", f"All {len(seven) + len(five)}")
                         for col in ("AP", "change")]

    def row(key: str, label: str) -> list[str]:
        cells = [label]
        for codes in groups:
            cells.append(f"{np.mean(res.set_ap(key, codes)):.3f}")
            cells.append("—" if key == BASELINE else c.signed(float(np.mean(res.set_deltas(key, codes)))))
        return cells

    rows, classes = [row(BASELINE, "<b>Pretrained (no adaptation)</b>")], [""]
    for _, title, methods in GROUPS:
        rows.append([title] + [""] * (len(head) - 1))
        classes.append("group")
        for m in methods:
            rows.append(row(m.key, m.label))
            classes.append("")
    return c.table(head, rows, "mini results", classes)


def per_species_table(res: Results) -> str:
    """AP of selected methods on each of the 12 species (mean over all 3 folds), with the held-out positives of each fold."""
    seven, five = res.fixed_species(), res.other_test_species()
    positives = res.heldout_positives()
    head = ["Species", "Held-out positive<br>clips, by fold", *(label for _, label in SPECIES_TABLE_KEYS)]
    rows, classes = [], []
    for title, codes in ((f"Main {len(seven)}", seven), (f"Other {len(five)}", five)):
        rows.append([title] + [""] * (len(head) - 1))
        classes.append("group")
        for s in codes:
            rows.append([c.name_of(s), " · ".join(str(n) for n in positives[s]),
                         *(f"{np.mean([res.species_ap(k, f)[s] for f in FOLDS]):.3f}" for k, _ in SPECIES_TABLE_KEYS)])
            classes.append("")
    return c.table(head, rows, "mini results wrap-head", classes)


TEST_ROWS = ((BASELINE, "Pretrained (no adaptation)"), ("focal-only", "Full fine-tuning on focal labels"),
             (ORACLE, "All labeled soundscape clips (upper reference)"))


def test_section(res: Results) -> str:
    """The three models scored on the test split: group means with the protocol's intervals, then species by species."""
    main, five = res.final["species"]["primary"], res.other_test_species()
    assert main == [s for s in res.final["species"]["secondary"] if s in main] and sorted(main) == sorted(res.fixed_species()), \
        "the test split's primary species are no longer the 7 fixed species"
    scores = {k: res.test_scores(k) for k, _ in TEST_ROWS}
    mean = lambda k, codes, m: float(np.mean([scores[k][s][m] for s in codes]))
    ci = lambda v: f"{v[0]:.3f} to {v[1]:.3f}"
    rows = []
    for key, label in TEST_ROWS:
        e = res.final["models"][key]
        cells = [label]
        for metric in ("sc_ap_primary", "sc_cmap_12"):
            cells.append(f"{e['point'][metric]:.3f}<br><small>{ci(e['ci95'][metric])}</small>")
            if key == BASELINE:
                cells.append("—")
            else:
                d = e["vs_baseline"][metric]
                cells.append(f"{c.signed(d['diff'])}<br><small>{c.signed(d['ci95'][0])} to {c.signed(d['ci95'][1])}</small>")
        cells += [f"{mean(key, five, 'ap'):.3f}", f"{mean(key, five, 'auc'):.3f}"]
        rows.append(cells)
    groups = c.table(["Model", f"Main {len(main)}: AP", "change", f"All {len(main) + len(five)}: AP", "change",
                      f"Other {len(five)}: AP", f"Other {len(five)}: AUC"], rows, "mini results")

    # claims in the text
    d = {k: res.final["models"][k]["vs_baseline"] for k in TEST_RUNS}
    assert all(d["focal-only"][m]["ci95"][1] < 0 and d[ORACLE][m]["ci95"][0] > 0 for m in ("sc_ap_primary", "sc_cmap_12")), \
        "the direction of a final baseline changed"
    gain = {s: scores[ORACLE][s]["ap"] - scores[BASELINE][s]["ap"] for s in five}
    top = sorted(gain, key=gain.get, reverse=True)[:2]
    cv_five = float(np.mean(res.set_deltas(ORACLE, five)))
    test_five = mean(ORACLE, five, "ap") - mean(BASELINE, five, "ap")
    assert sum(gain[s] for s in top) > 0.6 * sum(gain.values()) and test_five > 0.1 and cv_five < 0.02, "the other-species result changed"

    positives = res.test["species"]["n_pos_soundscape"]
    srows, classes = [], []
    for title, codes in ((f"Main {len(main)}", main), (f"Other {len(five)}", five)):
        srows.append([title] + [""] * 4)
        classes.append("group")
        for s in codes:
            srows.append([c.name_of(s), str(positives[s]), *(f"{scores[k][s]['ap']:.3f}" for k, _ in TEST_ROWS)])
            classes.append("")
    species = c.table(["Species", "Positive test clips", "Pretrained", "Focal fine-tuning", "All labels"], srows, "mini results", classes)
    return (c.section("res-test", f"On the test split: the main {len(main)}, the other {len(five)} and all {len(main) + len(five)}", groups,
                      "The only models scored on the test split so far. Soundscape AP; the small figures are the protocol's 95% intervals "
                      "(sites resampled), for changes paired with the pretrained model. They agree in direction with cross-validation: "
                      "fine-tuning on focal labels lowers AP, training on soundscape labels raises it. The test split's primary species "
                      f"are the main {len(main)}. On the other {len(five)}, the model trained on all labels gains {c.signed(test_five)} here "
                      f"but {c.signed(cv_five)} in cross-validation; most of the test gain comes from {' and '.join(c.name_of(s) for s in top)}. "
                      "These species are too rare for either number to be relied on.")
            + c.section("res-test-species", "Test split, species by species", species, "Soundscape AP of each model."))


def build(res: Results) -> Page:
    avg = lambda v: float(np.mean(v))
    mean_d = {k: avg(res.deltas(k)) for k in METHODS}
    ap = lambda k: avg(res.values(k, "sc_ap_primary"))
    share = lambda k: avg(res.fail_share(k))
    matched = lambda k: avg(res.fail_share_matched(k))
    proto = lambda k, d: avg(res.proto_auc(k, d))
    best = max(LABEL_FREE, key=mean_d.get)
    fixed = res.fixed_species()
    best_shared = max(avg(res.shared_deltas(k)) for k in LABEL_FREE)
    best_fixed = max(avg(res.set_deltas(k, fixed)) for k in LABEL_FREE)
    half_gap = (ap(ORACLE) - ap(BASELINE)) / 2
    # precision: on which folds does the interval of the gap fraction leave room for closing half of the gap?
    tight = ("tf-quantile", "tf-reweight", "r0x-nopush", "r0x-nopush-paperortho")
    tight_max = max(res.gamma_upper(k, f) for k in tight for f in FOLDS)
    pairs = [(k, f) for k in LABEL_FREE for f in FOLDS]
    open_pairs = sum(res.gamma_upper(k, f) >= 0.5 for k, f in pairs)
    sets = [name for name, _ in species_sets(res)]
    mean_of = lambda k, ctl, name: res.contrast_interval(k, ctl, name)["mean"]
    excludes_zero = lambda k, ctl, name: mean_of(k, ctl, name)["ci95"][0] > 0 or mean_of(k, ctl, name)["ci95"][1] < 0
    interval = lambda e: f"{c.signed(e['diff'])}, 95% interval {c.signed(e['ci95'][0])} to {c.signed(e['ci95'][1])}"
    clear = [(k, ctl) for k, ctl, _ in CONTRASTS if all(excludes_zero(k, ctl, name) for name in sets)]
    # the sentences below are claims about the data; the build fails if new results make one false
    assert all(res.folds_better_worse(k)[0] == 0 for k in LABEL_FREE), "a label-free method is now reliably better on a fold"
    assert all(res.criteria(k)[0] == 0 and res.criteria(k)[2] == 0 for k in LABEL_FREE), "a label-free method now meets criterion 1 or 3"
    assert max(mean_d[best], best_shared, best_fixed) < 0.2 * half_gap, "a label-free method now gains a sizeable part of the gap"
    assert tight_max < 0.5, "the narrow-interval methods no longer rule out closing half the gap"
    assert clear == [("r4", "r3")] and all(mean_of("r4", "r3", name)["ci95"][1] < 0 for name in sets), \
        f"the contrasts that are clear on every species set changed: {clear}"
    assert not excludes_zero("r3", "focal-only", "primary") and mean_of("r3", "focal-only", "shared")["ci95"][0] > 0 \
        and abs(mean_of("r3", "focal-only", "fixed")["diff"]) < 0.005, "the background + overlap result changed"
    assert not any(excludes_zero("r0", "focal-only", name) for name in sets), "freezing the backbone now has a clear effect"
    assert matched("r1b") - matched(BASELINE) < 0.02 < 0.08 < share("r1b") - share(BASELINE), "negative-set sensitivity changed"
    for key, domain in (("r0", "soundscape"), ("focal-only", "matched")):   # prototypes better, class AP not
        assert proto(key, domain) > proto(BASELINE, domain) + 0.01 and ap(key) <= ap(BASELINE)
        assert avg(res.shared_deltas(key)) <= 0
    assert proto(ORACLE, "matched") > proto(BASELINE, "matched") + 0.05
    assert matched(ORACLE) < 0.5 * matched(BASELINE) and proto(ORACLE, "soundscape") > proto(BASELINE, "soundscape") + 0.05
    assert abs(proto(ORACLE, "focal") - proto(BASELINE, "focal")) < 0.01 and res.folds_better_worse(ORACLE) == (3, 0)
    on_test = sorted(r for r, e in results.read_registry(res.store).items()
                     if "canonical_test" in e.get("sections", []) and r != BASELINE and not e.get("legacy"))
    assert on_test == sorted(TEST_RUNS), f"the models with test-split results are now {on_test}: revise the page's test section"

    tiles = c.tiles([
        (c.signed(mean_d[best]), f"largest mean AP change without soundscape labels, among {len(LABEL_FREE)} methods "
         f"(half of the gap would be {c.signed(half_gap)})"),
        (f"0 of {len(LABEL_FREE)}", "of those methods reliably better than the pretrained model on any fold"),
        (c.signed(mean_d["lb250"]), "mean AP change with 250 labeled soundscape clips"),
        (c.signed(mean_d[ORACLE]), "mean AP change with all labeled soundscape clips"),
    ])
    take = c.takeaways([
        f"<b>No method without soundscape labels shows a meaningful gain.</b> The best mean change is "
        f"{c.signed(mean_d[best])} on the scored species. When every fold scores the same species it is {c.signed(best_shared)} "
        f"({len(res.shared_species())} species) or {c.signed(best_fixed)} ({len(fixed)} species).",
        f"<b>How firmly this is ruled out varies.</b> For both training-free methods and both no-projection variants, the 95% interval "
        f"of the gap closed ends below {c.pct(tight_max)} on every fold. Other fine-tuned models have wider intervals: in {open_pairs} "
        f"of all {len(pairs)} method–fold pairs, closing half the gap is still inside the interval.",
        f"<b>Among the single ingredients, only one effect is clear on all three species sets: adding pseudo-labels lowers AP</b> "
        f"(scored species: {interval(mean_of('r4', 'r3', 'primary'))}; same {len(fixed)}: {interval(mean_of('r4', 'r3', 'fixed'))}).",
        f"<b>Real background with species overlap is not a dependable gain.</b> Against plain focal fine-tuning it is "
        f"{c.signed(mean_of('r3', 'focal-only', 'primary')['diff'])} on the scored species (interval includes zero) and "
        f"{c.signed(mean_of('r3', 'focal-only', 'shared')['diff'])} on the {len(res.shared_species())} shared species (interval above "
        f"zero), but {c.signed(mean_of('r3', 'focal-only', 'fixed')['diff'])} on the {len(fixed)} species eligible on every fold. "
        "The effect of freezing the backbone is not established: its interval includes zero on all three sets.",
        f"<b>With labels, prototypes detect better on soundscapes</b> (mean AUC {proto(BASELINE, 'soundscape'):.3f} → "
        f"{proto(ORACLE, 'soundscape'):.3f}; with matched negatives {proto(BASELINE, 'matched'):.3f} → {proto(ORACLE, 'matched'):.3f}; "
        f"focal barely moves). <b>But better prototypes have also occurred without better AP:</b> a frozen backbone "
        f"({proto(BASELINE, 'soundscape'):.3f} → {proto('r0', 'soundscape'):.3f}) and, with matched negatives, plain focal fine-tuning "
        f"({proto(BASELINE, 'matched'):.3f} → {proto('focal-only', 'matched'):.3f}) do not raise class AP.",
        f"<b>Read failing-prototype counts with care.</b> They depend on which soundscape clips serve as negatives. With real "
        f"background mixing the failing share rises from {c.pct(share(BASELINE))} to {c.pct(share('r1b'))} under the protocol, but with "
        f"negatives matched to focal it is {c.pct(matched('r1b'))}, against {c.pct(matched(BASELINE))} for the pretrained model.",
    ])
    contrasts = c.figure("res-contrasts", "What each ingredient changes, against the matching control",
                         "Comparing with the pretrained model mixes the effect of an ingredient with the effect of fine-tuning itself. "
                         "Here each variant is compared with the run that differs only in that ingredient, on three sets of species. "
                         "Dot: mean of 3 folds; bar: paired 95% interval from resampling held-out sites. These intervals are "
                         "<b>exploratory</b>: they are not part of the protocol, they treat the trained models as fixed (one seed, so "
                         f"seed-to-seed variation is missing), and the {len(CONTRASTS) * 3} of them are not corrected for multiple "
                         "comparisons, so an interval that only just excludes zero is weak evidence. Hover for the folds.",
                         contrast_chart(res), c.legend([("soundscape", "interval excludes zero"), ("hollow-key", "interval includes zero")]))
    delta = c.figure("res-delta", "Soundscape average precision: change from the pretrained model",
                     "Each dot is one cross-validation fold; the bar is the mean of the three. A filled dot means the fold's paired 95% "
                     "interval excludes zero; hover for the interval.", delta_chart(res),
                     c.legend([("soundscape", "reliably different on this fold"), ("hollow-key", "not reliably different")]))
    five = res.other_test_species()
    five_pos = [n for s in five for n in res.heldout_positives()[s]]
    five_auc = {k: avg(res.set_auc(k, five)) for k in (BASELINE, ORACLE)}
    oracle_fold = res.set_deltas(ORACLE, five)
    assert avg(res.set_deltas(ORACLE, fixed)) > 0.05 and avg(oracle_fold) < 0.02 and min(oracle_fold) < 0 < max(oracle_fold) \
        and five_auc[ORACLE] > five_auc[BASELINE] + 0.03, "the contrast between the main and the other species changed"
    twelve = c.section("res-twelve", f"The main {len(fixed)} species and the other {len(five)}", group_table(res),
                       f"The test split scores {len(fixed) + len(five)} species: the main {len(fixed)} "
                       f"({', '.join(c.name_of(s) for s in fixed)}) and {len(five)} rarer ones "
                       f"({', '.join(c.name_of(s) for s in five)}). Means over 3 folds, each fold averaging the same species. "
                       f"<b>Do not rank methods on the other {len(five)}:</b> a species has only {min(five_pos)} to {max(five_pos)} "
                       "positive clips per held-out fold, often from one site, so one clip can move its AP a lot. For the model trained "
                       f"on all labels the fold changes are {', '.join(c.signed(v) for v in oracle_fold)}, although its AUC on these "
                       f"species rises from {five_auc[BASELINE]:.3f} to {five_auc[ORACLE]:.3f}. No intervals are given for this group "
                       "because resampling sites often leaves a species with no positive clip.")
    by_species = c.section("res-twelve-species", "Species by species, all 12", per_species_table(res),
                           "Soundscape AP, mean over 3 folds; a selection of methods. The positive counts show how much each value "
                           "rests on.")
    shared = ", ".join(c.name_of(s) for s in res.shared_species())
    extra = ", ".join(c.name_of(s) for s in fixed if s not in res.shared_species())
    tab = c.section("res-table", "All metrics", results_table(res),
                    "Means over 3 folds. AP, AUC and cmAP are soundscape metrics; AP and AUC use the primary species, cmAP the secondary "
                    f"species. Because the primary species differ between folds, two further columns repeat the AP change on the same "
                    f"species every fold: the {len(res.shared_species())} that are primary on all folds ({shared}), and the {len(fixed)} "
                    f"eligible on all folds (adding {extra}). The order of methods without labels changes between these columns. "
                    'Criteria are defined under <a href="evaluation.html#criteria">evaluation</a>.')
    defs = "".join(f"<tr><td>{m.label}</td><td>{m.uses}</td><td>{m.how}</td></tr>" for _, _, ms in GROUPS for m in ms)
    methods = (f'<details class="methods"><summary>What each method does</summary><div class="table-wrap"><table class="mini crit">'
               f"<thead><tr><th>Method</th><th>Soundscape information</th><th>Description</th></tr></thead><tbody>{defs}</tbody></table></div>"
               "<p class=\"hint\">All fine-tuning starts from the pretrained model: 2 warm-up epochs (prototypes and last layer), 12 epochs "
               "of the whole network (skipped when frozen), prototype projection onto training patches, then 4 last-layer epochs. "
               "The last epoch is used. Mixing is applied with probability 0.5 (background at −5 to 20 dB SNR; the overlaid species 0 to "
               "10 dB below the original).</p></details>")
    protos = (c.figure("prototypes", "Did the prototypes get better on soundscapes, or only change relative to focal?",
                       "A prototype “fails” when its soundscape AUC is more than 0.10 below its focal AUC, so the count can change because "
                       "either side moved. Plain focal fine-tuning raises the focal side. Whether a focal-only recipe also improves the "
                       "soundscape side depends on the negatives (filled vs hollow orange); training with labels improves it under both. "
                       "Reweighting changes only class weights, so its prototypes equal the pretrained ones.", proto_auc_chart(res),
                       c.legend([("focal", "focal clips"), ("soundscape", "soundscape, all other clips as negatives (protocol)"),
                                 ("hollow-key", "soundscape, other-bird negatives (matched to focal)")]))
              + '<details class="methods"><summary>Failing-prototype share under both choices of negatives</summary>'
              + fail_table(res) + "</details>")
    heat = c.figure("res-species", "Which species gain or lose",
                    "Change in soundscape AP vs the pretrained model, averaged over the folds on which the species is primary. Blue: better; "
                    "red: worse. A selection of methods. Species scored on a single fold are especially noisy.", species_heatmap(res),
                    c.legend([("up-key", "better than pretrained"), ("down-key", "worse than pretrained")]))
    limits = c.section("res-status", "Status and limitations", f"""<ul class="prose">
    <li><b>Comparisons between methods are cross-validation results.</b> On the test split only the pretrained model and two
    baselines have been scored (<a href="#res-test">above</a>); the test sites are kept for final comparisons.</li>
    <li><b>One training seed per method.</b> Differences of a few hundredths between trained methods are within what seeds or folds
    could produce.</li>
    <li>The frozen-backbone ablations were added after the other runs had been launched. Treat them as exploratory.</li>
    <li>Comparisons on this page are not corrected for the number of methods and folds.</li>
    <li>The two fixed species sets and the intervals between adapted models were added after the first results were seen. The
    protocol's headline is still the change on each fold's primary species.</li>
    <li>All fine-tuned focal models use one fixed schedule and mixing strength; no hyperparameters were tuned for them.</li>
    <li>The training-free baselines are simple ones. Their result does not rule out better label-free adaptation.</li>
    <li>With 3 folds of about 23 sites, a missing “reliably better” does not show that a method has no effect.</li></ul>""")
    return Page("results.html", "Initial results", "Initial results",
                "Methods grouped by how much soundscape information they use, all scored on the same held-out sites.",
                tiles + take + delta + contrasts + tab + methods + twelve + by_species + test_section(res) + protos + heat + limits)
