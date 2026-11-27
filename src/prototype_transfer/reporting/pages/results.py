"""Initial results, laid out as a paper's results section: setup, numbered tables and figures, analysis, limitations, appendix."""

from __future__ import annotations

import math

import numpy as np

from prototype_transfer.evaluation import canonical, results
from prototype_transfer.reporting import charts as c
from prototype_transfer.reporting.runs import (BASELINE, CONTRASTS, FOLDS, GROUPS, LABEL_FREE, METHODS, ORACLE,
                                                PUBLISHED, TEST_RUNS, Results)
from prototype_transfer.reporting.site import Page

SPECIES_TABLE_KEYS = ((BASELINE, "Pretrained"), ("sfda-dropout_student", "Dropout Student"), ("sfda-notela", "NOTELA"),
                      ("r0x-nopush", "Frozen, no projection"), ("focal-only", "Full fine-tuning"),
                      ("r3", "Background + overlap"), ("lb250", "250 labels"), ("lb1000", "1,000 labels"), (ORACLE, "All labels"))
HEATMAP_ROWS = ("tf-quantile", "tf-reweight", "sfda-pseudo_label", "sfda-dropout_student", "sfda-notela", "focal-only", "r0", "r3", "r4", "r0x-nopush", "lb250", "lb1000", ORACLE)


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


SOURCE_LABEL = "Source (pretrained AudioProtoPNet)"
# rows of the main tables; every group but the last uses no soundscape labels and is ranked
PAPER_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("No adaptation", (BASELINE,)),
    ("Focal fine-tuning, no soundscape audio", ("focal-only", "r0")),
    ("Focal fine-tuning with unlabeled soundscape audio", ("r1b", "r3", "r4")),
    ("Training-free adaptation, unlabeled soundscape audio", ("tf-quantile", "tf-reweight")),
    ("Source-free adaptation, unlabeled soundscape audio", PUBLISHED),
    ("With soundscape labels (references, not ranked)", ("lb250", "lb1000", ORACLE)),
)
UNRANKED = ("lb250", "lb1000", ORACLE)
RANKED = tuple(k for _, keys in PAPER_GROUPS for k in keys if k not in UNRANKED)
TEST_MODELS = (BASELINE, *TEST_RUNS)
DASH = "–"


def label_of(key: str) -> str:
    if key == BASELINE:
        return SOURCE_LABEL
    return "All labeled clips (upper reference)" if key == ORACLE else METHODS[key].label


def ranked(values: dict[str, float], among, fmt: str, higher: bool = True) -> dict[str, str]:
    """Format every value; bold the best and underline the second-best distinct printed value among the keys `among`."""
    shown = {k: format(v, fmt) for k, v in values.items()}
    order = sorted({float(shown[k]) for k in among if k in shown}, reverse=higher)
    mark = dict(zip(order[:2], ("b", "u")))
    return {k: (f"<{mark[float(t)]}>{t}</{mark[float(t)]}>" if k in among and float(t) in mark else t) for k, t in shown.items()}


def interval(ci) -> str:
    return f"[{c.signed(ci[0])}, {c.signed(ci[1])}]"


def pct_signed(x: float) -> str:
    return f"{round(100 * x)}%".replace("-", "−")


def excludes(ci) -> bool:
    return ci[0] > 0 or ci[1] < 0


def paper_groups(row) -> list[tuple[str, list[list[str]]]]:
    return [(title, [row(k) for k in keys]) for title, keys in PAPER_GROUPS]


def main_table(res: Results) -> str:
    avg = lambda v: float(np.mean(v))
    keys = [k for _, ks in PAPER_GROUPS for k in ks]
    cv_ap = ranked({k: avg(res.values(k, "sc_ap_primary")) for k in keys}, RANKED, ".3f")
    point = {k: res.final["models"][k]["point"] for k in TEST_MODELS}
    test_ap = ranked({k: point[k]["sc_ap_primary"] for k in TEST_MODELS}, [k for k in RANKED if k in TEST_MODELS], ".3f")
    full = avg(res.deltas(ORACLE))

    def row(k: str) -> list[str]:
        cells = [label_of(k), f"{cv_ap[k]} <small>± {np.std(res.values(k, 'sc_ap_primary'), ddof=1):.3f}</small>"]
        if k == BASELINE:
            cells += [DASH] * 3
        else:
            d = res.deltas(k)
            better, worse = res.folds_better_worse(k)
            cells += [f"{c.signed(avg(d))} <small>± {np.std(d, ddof=1):.3f}</small>", f"{better} / {worse}", f"{round(100 * avg(d) / full)}".replace("-", "−")]
        if k not in TEST_MODELS:
            return cells + [DASH, DASH]
        e = res.final["models"][k]
        cells.append(test_ap[k])
        if k == BASELINE:
            return cells + [DASH]
        v = e["vs_baseline"]["sc_ap_primary"]
        return cells + [f"{c.signed(v['diff'])}{'*' if excludes(v['ci95']) else ''} <small>{interval(v['ci95'])}</small>"]

    return c.paper_table(
        "res-table", "Table 1", "<b>Soundscape macro average precision on primary species.</b> Cross-validation (3 site-grouped folds) "
        "and the held-out test sites. ΔAP is the paired difference from the source model; γ is the mean ΔAP as a share of the mean "
        "ΔAP of the model trained on all labeled clips.",
        [["", ("Cross-validation", 4), ("Test", 2)], ["Method", "AP", "ΔAP", "↑ / ↓", "γ (%)", "AP", "ΔAP [95% CI]"]],
        paper_groups(row),
        "How to read: AP is 1 when every clip with a species is ranked above every other clip "
        '(<a href="index.html#headline">worked example</a>). '
        f"A ΔAP of {c.signed(avg(res.deltas('sfda-dropout_student')))} means the mean AP moved from "
        f"{avg(res.values(BASELINE, 'sc_ap_primary')):.3f} to {avg(res.values('sfda-dropout_student', 'sc_ap_primary')):.3f}. "
        "<b>Bold</b> / <u>underline</u>: best / second best among methods that use no soundscape labels; in the test columns, among "
        "those scored on test. They do not indicate significance. ±: standard deviation over the 3 folds (for AP it mostly reflects "
        "how hard each fold is; one training seed). ↑ / ↓: folds whose paired 95% interval lies above / below zero. "
        "*: test interval excludes zero. " + DASH + ": not scored on the test split.")


def secondary_table(res: Results) -> str:
    avg = lambda v: float(np.mean(v))
    keys = [k for _, ks in PAPER_GROUPS for k in ks]
    on_test = [k for k in RANKED if k in TEST_MODELS]
    cv = [({k: avg(res.values(k, m)) for k in keys}, RANKED, ".3f", True) for m in ("sc_auc_primary", "sc_cmap_12")]
    cols = [ranked(*a) for a in cv]
    cols.append({k: f"{avg(res.values(k, 'f_auc_primary')):.3f}" for k in keys})
    cols += [ranked({k: 100 * avg(f(k)) for k in keys}, RANKED, ".0f", False) for f in (res.fail_share, res.fail_share_matched)]
    point = {k: res.final["models"][k]["point"] for k in TEST_MODELS}
    test = [ranked({k: point[k][m] for k in TEST_MODELS}, on_test, ".3f") for m in ("sc_auc_primary", "sc_cmap_12")]
    test.append({k: f"{point[k]['f_auc_primary']:.3f}" for k in TEST_MODELS})
    test += [ranked({k: float(point[k][m]) for k in TEST_MODELS}, on_test, ".0f", False) for m in ("n_fail", "n_fail_birds")]
    n_proto = len(res.final["models"][BASELINE]["prototypes"])
    row = lambda k: [label_of(k), *(col[k] for col in cols), *(col.get(k, DASH) for col in test)]
    heads = ["AUC", "cmAP", "Focal<br>AUC", "Fail<br>(%) ↓", "Fail<br>matched (%) ↓"]
    test_heads = ["AUC", "cmAP", "Focal<br>AUC", f"Fail<br>/{n_proto} ↓", f"Fail matched<br>/{n_proto} ↓"]
    return c.paper_table(
        "res-secondary", "Table 2", "<b>Secondary metrics.</b> Soundscape ROC-AUC (primary species) and cmAP (secondary species), "
        "focal ROC-AUC as a retention check, and failing prototypes (soundscape AUC more than 0.10 below focal AUC) with all other "
        "soundscape clips as negatives, or only other-bird clips (matched to the focal negatives).",
        [["", ("Cross-validation", 5), ("Test", 5)], ["Method", *heads, *test_heads]], paper_groups(row),
        "Ranking marks as in Table 1; lower is better where marked ↓. Focal AUC is not ranked: focal recordings are part of the "
        "source model's pretraining data.")


TEST_SPECIES_MODELS = (BASELINE, *PUBLISHED, ORACLE)


def species_test_table(res: Results, scores: dict) -> str:
    main, five = res.final["species"]["primary"], res.other_test_species()
    positives = res.test["species"]["n_pos_soundscape"]
    among = (BASELINE, *PUBLISHED)

    def line(name: str, n: str, values: dict[str, float]) -> list[str]:
        shown = ranked(values, among, ".3f")
        return [name, n, *(shown[k] for k in TEST_SPECIES_MODELS)]

    one = lambda s: line(c.name_of(s), str(positives[s]), {k: scores[k][s]["ap"] for k in TEST_SPECIES_MODELS})
    mean = lambda name, codes: line(f"<i>{name}</i>", "", {k: float(np.mean([scores[k][s]["ap"] for s in codes]))
                                                           for k in TEST_SPECIES_MODELS})
    groups = [(f"Primary species ({len(main)})", [*(one(s) for s in main), mean("Mean", main)]),
              (f"Other secondary species ({len(five)}; few positives)", [*(one(s) for s in five), mean("Mean", five)]),
              ("All", [mean(f"Mean of {len(main) + len(five)}", main + five)])]
    names = {BASELINE: "Source", ORACLE: "All labels<br>(ref.)", "sfda-pseudo_label": "Pseudo-<br>labelling",
             "sfda-dropout_student": "Dropout<br>Student"}
    return c.paper_table(
        "res-test-species", "Table 3", "<b>Per-species soundscape AP on the test split</b>, source model and source-free adaptation.",
        [["Species", "Positive<br>clips", *(names.get(k, label_of(k)) for k in TEST_SPECIES_MODELS)]], groups,
        "<b>Bold</b> / <u>underline</u>: best / second best of the source model and the five adaptation methods. No per-species "
        "intervals are given.")


def ablation_table(res: Results) -> str:
    sets = species_sets(res)

    def row(key: str, control: str, what: str) -> list[str]:
        cells = [f"{what}<br><small>{METHODS[key].label} vs {METHODS[control].label}</small>"]
        up = None
        for name, _ in sets:
            e = res.contrast_interval(key, control, name)
            v = e["mean"]
            cells.append(f"{c.signed(v['diff'])}{'*' if excludes(v['ci95']) else ''}<br><small>{interval(v['ci95'])}</small>")
            if name == "primary":
                up = sum(f["diff"] > 0 for f in e["folds"])
        return cells + [f"{up} / {len(FOLDS)}"]

    return c.paper_table(
        "res-contrasts", "Table 4", "<b>Ablation of focal fine-tuning.</b> Each variant against the run that differs in one ingredient: "
        "mean cross-validation difference in soundscape AP with a paired 95% interval, on three species sets.",
        [["Ingredient<br>(variant vs control)", "Scored species<br>(change by fold)", f"Same {len(res.shared_species())}<br>on every fold",
          f"Same {len(res.fixed_species())}<br>on every fold", "Folds up<br>(scored)"]],
        [("", [row(*x) for x in CONTRASTS])],
        f"*: interval excludes zero. These intervals are exploratory: not part of the protocol, one seed, trained models treated as "
        f"fixed, and the {len(CONTRASTS) * len(sets)} of them are not corrected for multiple comparisons.")


def method_table(res: Results) -> str:
    rows = [[m.label, m.uses, m.how] for _, _, ms in GROUPS for m in ms]
    return c.paper_table("res-methods", "Table A1", "<b>Compared methods.</b>", [["Method", "Soundscape information", "Description"]],
                         [("", rows)]).replace('<table class="paper">', '<table class="paper text">')


def appendix_table(tid: str, label: str, caption: str, table_html: str) -> str:
    """An existing wide table under a numbered caption."""
    return f'<div class="ptable" id="{tid}"><p class="pcaption"><b>{label}.</b> {caption}</p>{table_html}</div>'


def sec(sid: str, title: str, *parts: str) -> str:
    return c.section(sid, title, "".join(parts))


def par(head: str, text: str) -> str:
    return f'<p class="par"><b>{head}</b> {text}</p>'


def build(res: Results) -> Page:
    avg = lambda v: float(np.mean(v))
    mean_d = {k: avg(res.deltas(k)) for k in METHODS}
    ap = lambda k: avg(res.values(k, "sc_ap_primary"))
    share = lambda k: avg(res.fail_share(k))
    matched = lambda k: avg(res.fail_share_matched(k))
    proto = lambda k, d: avg(res.proto_auc(k, d))
    best = max(LABEL_FREE, key=mean_d.get)
    fixed = res.fixed_species()
    half_gap = (ap(ORACLE) - ap(BASELINE)) / 2
    # precision: on which folds does the interval of the gap fraction leave room for closing half of the gap?
    tight = ("tf-quantile", "tf-reweight", "r0x-nopush", "r0x-nopush-paperortho")
    tight_max = max(res.gamma_upper(k, f) for k in tight for f in FOLDS)
    sets = [name for name, _ in species_sets(res)]
    mean_of = lambda k, ctl, name: res.contrast_interval(k, ctl, name)["mean"]
    excludes_zero = lambda k, ctl, name: mean_of(k, ctl, name)["ci95"][0] > 0 or mean_of(k, ctl, name)["ci95"][1] < 0
    clear = [(k, ctl) for k, ctl, _ in CONTRASTS if all(excludes_zero(k, ctl, name) for name in sets)]
    # the sentences below are claims about the data; the build fails if new results make one false
    published = [m.key for g, _, ms in GROUPS if g == "sfda" for m in ms]
    earlier = [k for k in LABEL_FREE if k not in published]
    student, smoothed, small = "sfda-dropout_student", "sfda-notela", ("sfda-pseudo_label", "sfda-shot")
    assert all(res.folds_better_worse(k)[0] == 0 for k in earlier), "an earlier label-free method is now reliably better on a fold"
    assert all(res.criteria(k)[0] == 0 and res.criteria(k)[2] == 0 for k in earlier), "an earlier label-free method now meets criterion 1 or 3"
    assert max(max(mean_d[k], avg(res.shared_deltas(k)), avg(res.set_deltas(k, fixed))) for k in earlier) < 0.2 * half_gap, \
        "an earlier label-free method now gains a sizeable part of the gap"
    assert best == student and res.folds_better_worse(student) == (2, 0) and res.criteria(student)[0] == 1, "the Dropout Student result changed"
    assert all(res.criteria(k)[0] == 0 for k in published if k != student) and all(res.criteria(k)[2] == 0 for k in published), \
        "another published method now meets criterion 1 or 3"
    assert all(res.folds_better_worse(k) == (3, 0) and 0.005 < mean_d[k] < 0.02 for k in small), "the pseudo-labelling / SHOT result changed"
    assert res.folds_better_worse("sfda-tent")[0] == 0 and mean_d["sfda-tent"] < 0, "the Tent result changed"
    assert res.folds_better_worse(smoothed) == (0, 0) and abs(avg(res.set_deltas(smoothed, fixed))) < 0.005 \
        and mean_d[student] > mean_d[smoothed] + 0.03, "the NOTELA result changed"
    assert min(avg(res.shared_deltas(student)), avg(res.set_deltas(student, fixed))) > 0.05, "Dropout Student's gain depends on the species set"
    gains = {sp: avg(res.set_deltas(student, [sp])) for sp in fixed}
    top_sp, low_sp = max(gains, key=gains.get), min(gains, key=gains.get)
    assert gains[top_sp] > 0.4 * sum(gains.values()) and gains[low_sp] < -0.03, "the spread of Dropout Student's gain over species changed"
    other_gain = avg(res.set_deltas(student, res.other_test_species()))
    assert abs(other_gain) < 0.02, "Dropout Student's change on the other species is no longer small"
    assert proto(student, "soundscape") > proto(BASELINE, "soundscape") + 0.02 and proto(student, "matched") > proto(BASELINE, "matched") + 0.02 \
        and matched(student) < matched(BASELINE), "Dropout Student's prototype result changed"
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

    # ------------------------------------------------------------------------------------------- test-split claims
    main, five = res.final["species"]["primary"], res.other_test_species()
    assert main == [s for s in res.final["species"]["secondary"] if s in main] and sorted(main) == sorted(res.fixed_species()), \
        "the test split's primary species are no longer the 7 fixed species"
    scores = {k: res.test_scores(k) for k in TEST_MODELS}
    d = {k: res.final["models"][k]["vs_baseline"] for k in TEST_RUNS}
    assert all(d["focal-only"][m]["ci95"][1] < 0 and d[ORACLE][m]["ci95"][0] > 0 for m in ("sc_ap_primary", "sc_cmap_12")), \
        "the direction of a final baseline changed"
    assert all(d[k][m]["ci95"][0] < 0 < d[k][m]["ci95"][1] for k in PUBLISHED for m in ("sc_ap_primary", "sc_cmap_12")), \
        "a published method now differs reliably from the pretrained model on the test split"
    assert all(max(PUBLISHED, key=lambda k: d[k][m]["diff"]) == student for m in ("sc_ap_primary", "sc_cmap_12")), \
        "Dropout Student no longer has the largest test gain among the published methods"
    cv_student = mean_d[student]
    assert 0 < d[student]["sc_ap_primary"]["diff"] < 0.5 * cv_student, "the test and cross-validation gains of Dropout Student now agree"
    point = {k: res.final["models"][k]["point"] for k in TEST_MODELS}
    assert min(PUBLISHED, key=lambda k: point[k]["n_fail"]) == student and point[student]["n_fail"] < point[BASELINE]["n_fail"]
    lost_focal = [k for k in PUBLISHED if not res.final["models"][k]["criteria"]["4_focal_retained"]]
    assert lost_focal == [smoothed], f"the methods losing focal AUC on test changed: {lost_focal}"
    gap = res.final["models"][student]["gap_fraction"]
    gain = {s: scores[ORACLE][s]["ap"] - scores[BASELINE][s]["ap"] for s in five}
    top = sorted(gain, key=gain.get, reverse=True)[:2]
    cv_five = avg(res.set_deltas(ORACLE, five))
    test_five = float(np.mean([gain[s] for s in five]))
    assert sum(gain[s] for s in top) > 0.6 * sum(gain.values()) and test_five > 0.1 and cv_five < 0.02, "the other-species result changed"
    test_best = d[best]["sc_ap_primary"]
    # per species, the CV leader's change on test (stated in the text, so computed rather than typed)
    test_gain = {s: scores[student][s]["ap"] - scores[BASELINE][s]["ap"] for s in main}
    t_top, t_low = max(test_gain, key=test_gain.get), min(test_gain, key=test_gain.get)
    n_down = sum(v < 0 for v in test_gain.values())
    five_pos = [n for s in five for n in res.heldout_positives()[s]]
    five_auc = {k: avg(res.set_auc(k, five)) for k in (BASELINE, ORACLE)}
    oracle_fold = res.set_deltas(ORACLE, five)
    assert avg(res.set_deltas(ORACLE, fixed)) > 0.05 and avg(oracle_fold) < 0.02 and min(oracle_fold) < 0 < max(oracle_fold) \
        and five_auc[ORACLE] > five_auc[BASELINE] + 0.03, "the contrast between the main and the other species changed"
    assert all(mean_d["lb250"] > mean_d[k] for k in LABEL_FREE), "a label-free method now beats 250 labeled clips"

    sp = res.splits
    heldout = [len(res.splits_cv["sites"][str(f)]) for f in FOLDS]
    n_test_sites = res.test["n_clusters"]["sites"]
    primary_by_fold = [len(x["primary"]) for x in res.species]
    S, V = canonical.PRIMARY_MIN_POS, canonical.MIN_POS
    name = lambda k: METHODS[k].label
    sg = c.signed

    setup = sec(
        "res-setup", "1&ensp;Experimental setup",
        par("Task.", "Multi-label recognition of 21 High Sierra Nevada bird species. The source model is AudioProtoPNet, pretrained "
            "on Xeno-Canto focal recordings; a species score is a non-negative weighted sum of the activations of its five prototypes "
            '(<a href="formulation.html">problem formulation</a>).'),
        par("Data.", f"Focal clips are split by recordist into {sp['focal']['train']['n_clips']:,} / {sp['focal']['val']['n_clips']:,} "
            f"/ {sp['focal']['test']['n_clips']:,} train / validation / test clips. Soundscapes are 5&nbsp;s passive-recording clips, "
            f"split by site: {sum(heldout)} development sites and {n_test_sites} test sites "
            f"({sp['soundscape']['test']['n_clips']:,} clips). Development uses 3-fold site-grouped cross-validation (CV), holding out "
            f"{' / '.join(map(str, heldout))} sites. Adaptation is <i>inductive</i>: a method may use unlabeled audio of training "
            "sites, is then frozen, and is scored on sites it has not heard."),
        par("Metrics.", f"The headline metric is soundscape macro AP over <i>primary</i> species, those with at least {S} positive clips "
            f"in the scored split ({', '.join(map(str, primary_by_fold))} species on the three folds; {len(main)} on test). We also "
            f"report ROC-AUC, cmAP over <i>secondary</i> species (at least {V} positives in both domains), focal ROC-AUC, and the number "
            "of prototypes that fail to transfer. Intervals are paired 95% cluster bootstraps (soundscape clips resampled by site, "
            "focal clips by recordist; 1,000 resamples)."),
        par("Test evaluations.", "The test split was scored for the source model, the five source-free adaptation methods, full focal "
            "fine-tuning and the model trained on all labels. The settings of the five adaptation methods were fixed before any of "
            f"their runs; each adapted on the {sp['soundscape']['train']['n_clips']:,} clips of the training sites."),
        par("Baselines.", "Source-free adaptation follows the audio code and settings of Boudiaf et al. (2023): Tent, pseudo-labelling, "
            "Dropout Student, NOTELA and SHOT, each species treated as a binary problem. Because this backbone has no batch "
            "normalisation, the layer-normalisation weights are adapted; prototypes and class weights stay fixed. We also compare two "
            "training-free methods, focal fine-tuning with and without unlabeled soundscape audio, and, as references, training on "
            'labeled soundscape clips (<a href="label-curve.html">label curve</a>). Details are in '
            '<a href="#res-methods">Table A1</a>.'))

    fold_fig = c.paper_figure(
        "res-delta", "Figure 1", "<b>Change in soundscape AP from the source model, by cross-validation fold.</b> Dots are folds and "
        "the bar is their mean; a filled dot means the fold's paired 95% interval excludes zero.", delta_chart(res),
        c.legend([("soundscape", "interval excludes zero"), ("hollow-key", "interval includes zero")]))
    results_sec = sec(
        "res-main", "2&ensp;Adaptation without soundscape labels",
        main_table(res),
        par("Cross-validation.", f"{name(student)} has the highest mean AP among methods without soundscape labels, "
            f"{sg(mean_d[student])} over the source model (γ = {100 * mean_d[student] / mean_d[ORACLE]:.0f}%), and is reliably better "
            f"on 2 of 3 folds. The gain persists when every fold scores the same {len(res.shared_species())} or {len(fixed)} species "
            f"({sg(avg(res.shared_deltas(student)))} and {sg(avg(res.set_deltas(student, fixed)))}). It still misses our target, closing "
            f"half the gap on 1 of 3 folds. {name(small[0])} ({sg(mean_d[small[0]])}) and {name(small[1])} ({sg(mean_d[small[1]])}) are "
            f"reliably better on all folds, by small margins; {name('sfda-tent')} is not better ({sg(mean_d['sfda-tent'])}). None of "
            f"the {len(earlier)} training-free and fine-tuning methods is reliably better on any fold. For both training-free methods "
            f"and both frozen, no-projection variants, the interval of the share of the gap closed ends below {c.pct(tight_max)} on "
            "every fold."),
        par("Held-out test sites.", f"<b>The cross-validation advantage is not confirmed on test.</b> None of the five source-free "
            f"methods differs reliably from the source model. {name(student)} keeps the largest point estimate, but its gain is "
            f"{sg(test_best['diff'])} {interval(test_best['ci95'])}, against {sg(cv_student)} in cross-validation. Its share of the gap, "
            f"{pct_signed(gap['value'])}, is poorly determined (95% interval {pct_signed(gap['ci95'][0])} to {pct_signed(gap['ci95'][1])}). "
            f"The split can still separate some differences: full focal fine-tuning is reliably worse "
            f"({sg(d['focal-only']['sc_ap_primary']['diff'])}) and training on all labels reliably better "
            f"({sg(d[ORACLE]['sc_ap_primary']['diff'])})."),
        par("Labels as a reference.", f"With 250 labeled soundscape clips the mean CV gain is {sg(mean_d['lb250'])}, above every method "
            f"without labels; with all labeled clips it is {sg(mean_d[ORACLE])}."),
        par("NOTELA versus Dropout Student.", f"NOTELA adds neighbour smoothing to Dropout Student's targets. Here it averages "
            f"{sg(mean_d[smoothed])} with no fold reliably better, and {sg(avg(res.set_deltas(smoothed, fixed)))} on the same "
            f"{len(fixed)} species, whereas its authors report it ahead of Dropout Student on all six of their test sets by sample-wise "
            "mAP, with a different model. In the binary form used for multi-label data, the smoothing term lowers a species' target "
            "unless similar clips score it above 0.5 on average. We have not tested whether this explains the difference."),
        fold_fig)

    proto_fig = c.paper_figure(
        "prototypes", "Figure 2", "<b>Mean prototype detection AUC on focal and soundscape clips</b> (cross-validation; the axis does "
        "not start at 0). A prototype's AUC uses its maximum activation as a detector of its species.", proto_auc_chart(res),
        c.legend([("focal", "focal clips"), ("soundscape", "soundscape, all other clips as negatives (protocol)"),
                  ("hollow-key", "soundscape, other-bird negatives (matched to focal)")]))
    proto_sec = sec(
        "res-prototypes", "3&ensp;Do the prototypes transfer better?", secondary_table(res),
        par("Prototype detection.", f"Training on labels raises mean soundscape prototype AUC from {proto(BASELINE, 'soundscape'):.3f} "
            f"to {proto(ORACLE, 'soundscape'):.3f} (matched negatives: {proto(BASELINE, 'matched'):.3f} to "
            f"{proto(ORACLE, 'matched'):.3f}) while focal AUC barely moves. {name(student)} raises it without labels, to "
            f"{proto(student, 'soundscape'):.3f} (matched: {proto(student, 'matched'):.3f}), with prototype vectors and class weights "
            f"unchanged; on test its failing prototypes fall from {point[BASELINE]['n_fail']} to {point[student]['n_fail']}."),
        par("Better prototypes are not sufficient.", f"A frozen-backbone fine-tune raises soundscape prototype AUC "
            f"({proto(BASELINE, 'soundscape'):.3f} to {proto('r0', 'soundscape'):.3f}), as does full focal fine-tuning with matched "
            f"negatives ({proto(BASELINE, 'matched'):.3f} to {proto('focal-only', 'matched'):.3f}); neither improves class AP. These "
            "comparisons are observational and do not show that prototype quality causes AP."),
        par("The failure count depends on the negatives.", f"With real background mixing, the failing share rises from "
            f"{c.pct(share(BASELINE))} to {c.pct(share('r1b'))} under the protocol, but with matched negatives it is "
            f"{c.pct(matched('r1b'))}, against {c.pct(matched(BASELINE))} for the source model."),
        par("Focal retention.", f"On test, {name(smoothed)} is the only source-free method that loses more than "
            f"{res.test['thresholds']['focal_tolerance']:.2f} of focal AUC."),
        proto_fig)

    heat_fig = c.paper_figure(
        "res-species", "Figure 3", "<b>Per-species change in soundscape AP from the source model</b> (cross-validation, mean over the "
        "folds on which the species is primary), for a selection of methods. Blue: better; red: worse.", species_heatmap(res),
        c.legend([("up-key", "better than source"), ("down-key", "worse than source")]))
    species_sec = sec(
        "res-test", "4&ensp;Per-species behaviour", species_test_table(res, scores),
        par("The gain is concentrated.", f"In cross-validation, over the {len(fixed)} species eligible on every fold, "
            f"{c.name_of(top_sp)} gains {sg(gains[top_sp])} with {name(student)}, {c.pct(gains[top_sp] / sum(gains.values()))} of the "
            f"summed gain, while {c.name_of(low_sp)} changes by {sg(gains[low_sp])}. On test, its changes on the primary species range "
            f"from {sg(test_gain[t_low])} ({c.name_of(t_low)}) to {sg(test_gain[t_top])} ({c.name_of(t_top)}), and {n_down} of "
            f"{len(main)} species go down."),
        par("Rare species are not informative.", f"The other {len(five)} species have {min(five_pos)} to {max(five_pos)} positive "
            f"clips per held-out fold, often from one site. For {name(student)} their mean CV change is {sg(other_gain)}. Even the model "
            f"trained on all labels gains {sg(test_five)} on them on test but {sg(cv_five)} in cross-validation (folds "
            f"{', '.join(sg(v) for v in oracle_fold)}), mostly through {' and '.join(c.name_of(s) for s in top)} on test, although its "
            f"CV AUC on them rises from {five_auc[BASELINE]:.3f} to {five_auc[ORACLE]:.3f}. We do not rank methods on them."),
        heat_fig)

    r4_all, r4_fixed = mean_of("r4", "r3", "primary"), mean_of("r4", "r3", "fixed")
    ablation_sec = sec(
        "res-ablation", "5&ensp;Which ingredients of focal fine-tuning matter?", ablation_table(res),
        par("Pseudo-labels hurt.", "The only effect that is clear on all three species sets is negative: adding the source model's soft "
            f"predictions as a training target lowers AP ({sg(r4_all['diff'])} {interval(r4_all['ci95'])} on the scored species; "
            f"{sg(r4_fixed['diff'])} {interval(r4_fixed['ci95'])} on the same {len(fixed)}). This is a training loss during focal "
            "fine-tuning, distinct from the source-free pseudo-labelling baseline."),
        par("Mixing and freezing are inconclusive.", f"Real background with species overlap, against plain focal fine-tuning, is "
            f"{sg(mean_of('r3', 'focal-only', 'primary')['diff'])} on the scored species (interval includes zero), "
            f"{sg(mean_of('r3', 'focal-only', 'shared')['diff'])} on the {len(res.shared_species())} shared species (interval above "
            f"zero) and {sg(mean_of('r3', 'focal-only', 'fixed')['diff'])} on the {len(fixed)} fixed species. The interval for "
            "freezing the backbone includes zero on all three sets."))

    limits = sec("res-status", "6&ensp;Limitations", """<ul class="prose">
    <li><b>One training seed per method.</b> Seed-to-seed variation has not been measured for any method. It may matter for Dropout
    Student and NOTELA, which rely on random noise during adaptation.</li>
    <li><b>Few sites.</b> Each fold holds out about 23 sites and test has 30, so intervals are wide. A missing reliable difference does
    not show that a method has no effect.</li>
    <li><b>No correction for multiple comparisons</b> across methods, folds or species.</li>
    <li><b>Inherited hyperparameters.</b> The source-free methods use their authors' settings without tuning here. The authors selected
    hyperparameters with labels on a High Sierras soundscape set, with a different model; we have not checked whether its recordings
    overlap ours. The results are not the best each method could reach.</li>
    <li><b>Adapted implementations.</b> Layer-normalisation weights replace batch-normalisation weights. Student noise is random
    skipping of network blocks (probability 0.2); the authors' additional dropout on a pooled feature has no counterpart here.
    Test-time methods that adapt on the scored audio are outside this protocol, and the training-free baselines are simple ones.</li>
    <li><b>The test split has now been seen</b> for these methods. A method changed after this can no longer claim a clean held-out
    score.</li>
    <li><b>Post-hoc analyses.</b> The frozen-backbone ablations, the fixed species sets and the intervals between adapted models were
    added after first results were seen. The protocol's headline remains the change on each fold's primary species.</li>
    <li><b>Focal scores are in-domain</b> for the source model, which was pretrained on Xeno-Canto.</li></ul>""")

    shared = ", ".join(c.name_of(s) for s in res.shared_species())
    extra = ", ".join(c.name_of(s) for s in fixed if s not in res.shared_species())
    appendix = sec(
        "res-appendix", "Appendix",
        method_table(res),
        par("Training details.", "All fine-tuning starts from the source model: 2 warm-up epochs (prototypes and last layer), 12 epochs "
            "of the whole network (skipped when frozen), prototype projection onto training patches, then 4 last-layer epochs; the last "
            "epoch is used. Mixing is applied with probability 0.5 (background at −5 to 20 dB SNR; the overlaid species 0 to 10 dB "
            "below the original). Source-free adaptation makes 10 passes over the unlabeled clips in batches of 64 with no checkpoint "
            "selection."),
        appendix_table("res-all", "Table A2", f"<b>All cross-validated methods and metrics</b> (means over folds). Two columns repeat "
                       f"the AP change on fixed species: the {len(res.shared_species())} primary on all folds ({shared}), and the "
                       f"{len(fixed)} eligible on all folds (adding {extra}). Criteria are defined under "
                       '<a href="evaluation.html#criteria">evaluation</a>.', results_table(res)),
        appendix_table("res-twelve", "Table A3", f"<b>Cross-validation soundscape AP on the {len(fixed)} fixed species, the other "
                       f"{len(five)} ({', '.join(c.name_of(s) for s in five)}) and all {len(fixed) + len(five)}.</b>", group_table(res)),
        appendix_table("res-twelve-species", "Table A4", "<b>Cross-validation per-species soundscape AP</b> (mean over 3 folds), a "
                       "selection of methods.", per_species_table(res)),
        appendix_table("res-fail", "Table A5", "<b>Failing-prototype share under both choices of negatives</b> (cross-validation).",
                       fail_table(res)))

    body = '<div class="paper-body">' + setup + results_sec + proto_sec + species_sec + ablation_sec + limits + appendix + "</div>"
    return Page("results.html", "Initial results", "Initial results",
                "Can unlabeled soundscape audio close the focal-to-soundscape gap of a prototype-based bird classifier?", body)
