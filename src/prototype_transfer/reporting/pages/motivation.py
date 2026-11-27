"""Motivation: the focal-to-soundscape gap, and what the prototypes reveal about it."""

from __future__ import annotations

import numpy as np

from prototype_transfer.evaluation import canonical
from prototype_transfer.reporting import charts as c
from prototype_transfer.reporting.runs import BASELINE, LABEL_FREE, ORACLE, Results
from prototype_transfer.reporting.site import Page


def auc_chart(auc: dict[str, dict], primary: list[str]) -> str:
    order = sorted(auc, key=lambda s: auc[s]["soundscape"] - auc[s]["focal"])
    x0, x1, row, top = 230, 860, 26, 12
    lo = min(0.7, min(min(d["focal"], d["soundscape"]) for d in auc.values()) - 0.02)
    lo = int(lo * 10) / 10
    x = c.linear(lo, 1.0, x0, x1)
    bottom = top + row * len(order)
    body = c.x_grid(x, [lo + 0.1 * i for i in range(int(round((1 - lo) * 10)) + 1)], top - 6, bottom, lambda t: c.score(t, 0))
    body += c.text((x0 + x1) / 2, bottom + 34, "ROC-AUC (test split; axis does not start at 0)", "axis-title", "middle")
    for i, s in enumerate(order):
        y, d = top + i * row + 10, auc[s]
        body += c.text(x0 - 12, y + 4, c.name_of(s), "row-label" + (" sel" if s in primary else ""), "end")
        body += c.line(x(d["focal"]), y, x(d["soundscape"]), y, "connector")
        for dom in ("focal", "soundscape"):
            body += c.circle(x(d[dom]), y, 5.5, f"dot {dom}", f"{c.name_of(s)}: {dom} AUC {c.score(d[dom])}")
    return c.svg(bottom + 42, "Pretrained model: ROC-AUC per species on focal and soundscape test clips", body)


def headroom_chart(res: Results) -> str:
    """Per species, the pretrained and the soundscape-trained model on the same held-out clips."""
    species = res.primary_union()
    vals = {s: (res.species_mean(BASELINE, s)[0], res.species_mean(ORACLE, s)[0], res.species_mean(BASELINE, s)[1]) for s in species}
    order = sorted(species, key=lambda s: vals[s][0] - vals[s][1])
    x0, x1, row, top = 230, 760, 28, 22
    x = c.linear(0.4, 1.0, x0, x1)
    bottom = top + row * len(order)
    body = c.x_grid(x, (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), top - 6, bottom, lambda t: c.score(t, 0))
    body += c.text((x0 + x1) / 2, bottom + 34, "Soundscape average precision, same held-out clips (axis does not start at 0)",
                   "axis-title", "middle")
    body += c.text(c.W, top - 10, "change · folds", "", "end")
    for i, s in enumerate(order):
        y, (base, top_ap, k) = top + i * row + 10, vals[s]
        body += c.text(x0 - 12, y + 4, c.name_of(s), "row-label", "end")
        body += c.line(x(base), y, x(top_ap), y, "connector")
        body += c.circle(x(base), y, 5.5, "dot neutral", f"{c.name_of(s)}: pretrained AP {c.score(base)}")
        body += c.circle(x(top_ap), y, 5.5, "dot soundscape", f"{c.name_of(s)}: trained on labeled soundscapes, AP {c.score(top_ap)}")
        body += c.text(c.W, y + 4, f"{c.signed(top_ap - base)} · {k}", "cell", "end")
    return c.svg(bottom + 42, "Headroom per species: pretrained model vs soundscape-trained model", body)


def closable_chart(res: Results) -> str:
    """Same held-out soundscape clips: pretrained model vs the same architecture trained with soundscape labels."""
    rows = [("Pretrained model", BASELINE, "dot neutral"), ("Trained on labeled soundscapes", ORACLE, "dot soundscape")]
    x0, x1, row, top = 230, 760, 34, 12
    x = c.linear(0.5, 1.0, x0, x1)
    bottom = top + row * len(rows)
    body = c.x_grid(x, (0.5, 0.6, 0.7, 0.8, 0.9, 1.0), top, bottom, lambda t: c.score(t, 0))
    body += c.text((x0 + x1) / 2, bottom + 34, "Soundscape average precision on held-out sites (primary species)", "axis-title", "middle")
    for i, (label, key, cls) in enumerate(rows):
        y = top + i * row + 16
        vals = res.values(key, "sc_ap_primary")
        mean = sum(vals) / len(vals)
        body += c.text(x0 - 12, y + 4, label, "row-label", "end")
        for f, v in enumerate(vals):
            body += c.circle(x(v), y, 4, cls + " fold", f"{label}, fold {f}: {c.score(v)}")
        body += c.line(x(mean), y - 11, x(mean), y + 11, "mean-tick")
        body += c.text(x1 + 16, y + 4, f"mean {c.score(mean)}")
    return c.svg(bottom + 42, "Pretrained model and soundscape-trained model on the same held-out sites", body)


EXAMPLE_SPECIES, EXAMPLE_FOLD, EXAMPLE_TOP = "clanut", 0, 100   # the worked example of the headline metric
STUDENT = "sfda-dropout_student"


def headline_section(res: Results) -> str:
    """The headline metric in plain terms: one worked example on held-out clips, then the few numbers to remember."""
    def example(key: str) -> tuple[int, int, int, float]:
        m = canonical.Model(key, res.responses(key, EXAMPLE_FOLD))
        j = m.species.index(EXAMPLE_SPECIES)
        order = np.argsort(-m.p_s[:, j], kind="stable")[:EXAMPLE_TOP]
        return int(m.y_s[order, j].sum()), int(m.y_s[:, j].sum()), len(m.y_s), res.species_ap(key, EXAMPLE_FOLD)[EXAMPLE_SPECIES]

    hit0, positives, clips, ap0 = example(BASELINE)
    hit1, _, _, ap1 = example(ORACLE)
    assert hit1 > hit0 and ap1 > ap0, "the worked example no longer shows a gap"
    assert max(LABEL_FREE, key=lambda k: sum(res.deltas(k))) == STUDENT, "the best method without labels changed"
    mean = lambda k: float(np.mean(res.values(k, "sc_ap_primary")))
    test = lambda k: res.final["models"][k]["point"]["sc_ap_primary"]
    held = res.final["models"][STUDENT]["vs_baseline"]["sc_ap_primary"]
    assert held["ci95"][0] < 0 < held["ci95"][1], "the headline table says the test gain could be zero"
    name = c.name_of(EXAMPLE_SPECIES)

    worked = c.table(
        [f"{name}: {positives} of {clips:,} held-out clips contain it",
         f"Of the {EXAMPLE_TOP} top-scoring clips,<br>how many contain it?", "Average precision<br>for this species"],
        [["Pretrained model", f"<b>{hit0}</b>", c.score(ap0)], ["Same model trained on soundscape labels", f"<b>{hit1}</b>", c.score(ap1)]],
        "mini wrap-head")

    def cell(k: str, value, note: str = "") -> str:
        return f"<b>{c.score(value(k))}</b><br><small>{c.signed(value(k) - value(BASELINE))} points vs pretrained{note}</small>"

    zero = f"<br>could be zero (95% interval {c.signed(held['ci95'][0])} to {c.signed(held['ci95'][1])} points)"
    numbers = c.table(
        ["Headline: soundscape average precision,<br>averaged over the main species",
         "Development sites<br><small>cross-validation, mean of 3</small>", "Test sites<br><small>final check</small>"],
        [["Pretrained model (starting point)", f"<b>{c.score(mean(BASELINE))}</b>", f"<b>{c.score(test(BASELINE))}</b>"],
         ["Best method so far without soundscape labels (Dropout Student)", cell(STUDENT, mean), cell(STUDENT, test, zero)],
         ["Trained with soundscape labels (what labels can buy)", cell(ORACLE, mean), cell(ORACLE, test)]],
        "mini wrap-head headline")

    return c.section("headline", "The headline number, in plain terms", f"""
    <div class="prose">
    <p>For one species, the model gives every 5-second clip a score. Sort the clips from highest to lowest. <b>Average precision
    (AP)</b> measures how well the clips that really contain the species are gathered at the top. We show it <b>out of 100</b>: it is
    <b>100</b> when every such clip is ranked above every other clip. A random ordering scores about the share of clips with the
    species, here about {c.score(positives / clips)}. Higher is better.</p></div>""" + worked + """
    <div class="prose"><p>This species was picked because its gap is easy to see; others differ less (chart further down). The
    <b>headline number</b> is AP averaged over the species with enough examples to score.</p></div>""" + numbers + f"""
    <div class="prose"><ul>
    <li><b>The gap we want to close</b> is the last row minus the first: {c.score(mean(ORACLE) - mean(BASELINE))} points on
    development sites.</li>
    <li>A change of <b>{c.signed(mean(STUDENT) - mean(BASELINE))} points</b> means the score went from {c.score(mean(BASELINE))} to
    {c.score(mean(STUDENT))}. It does not mean {c.score(mean(STUDENT) - mean(BASELINE))}% more clips are correct, and it is not a relative
    improvement (relative to the starting score it is {(mean(STUDENT) - mean(BASELINE)) / mean(BASELINE):.0%}).</li>
    <li>Compare numbers <b>down a column</b>, not across: the two columns use different sites.</li>
    <li>Changes are computed before rounding, so they can differ by 0.1 from the rounded numbers.</li></ul></div>""")


def build(res: Results) -> Page:
    point = res.test["models"][BASELINE]["point"]
    n_proto = len(res.test["models"][BASELINE]["prototypes"])
    base = res.cv["r0"]["baseline_metrics"]["sc_ap_primary"]["mean"]
    top = res.cv[ORACLE]["run"]["sc_ap_primary"]["mean"]
    assert res.folds_better_worse(ORACLE) == (3, 0), "the page says labels help reliably on every fold"
    worse = [s for s in res.primary_union() if res.species_mean(ORACLE, s)[0] < res.species_mean(BASELINE, s)[0]]
    why = c.section("why", "Why bird recognisers struggle in the field", f"""
    <div class="prose">
    <p>Passive acoustic monitoring records whole landscapes for weeks. Most labeled bird audio, in contrast, is <b>focal</b>:
    a person points a microphone at one bird. A model learned from focal recordings must then work on <b>soundscapes</b> with
    distance, wind, overlapping species and long silences. Labeling soundscapes needs experts, so the target domain is where
    labels are scarcest.</p>
    <p>We study <b>AudioProtoPNet</b>, a bird-sound classifier whose decision is a weighted sum of similarities to learned
    sound patterns (<i>prototypes</i>, five per species). Because each prototype can be inspected, we can ask <i>which parts of
    the model</i> stop working in the field, not only how much accuracy is lost.</p></div>""")
    exception = (f" Not every species gains: {', '.join(c.name_of(s) for s in worse)} does worse "
                 "(see the fold count; one fold is weak evidence)." if worse else "")
    closable = c.figure("gap-closable", "How much can be gained on soundscapes: same clips, different training data",
                        "Both rows are scored on identical held-out soundscape clips. Dots are the 3 cross-validation folds; the bar is "
                        "their mean. Training with soundscape labels is better on every fold (paired 95% interval above zero). It is an "
                        "empirical reference from one training recipe, not a proven ceiling.", closable_chart(res))
    headroom = c.figure("gap-headroom", "Where the headroom is, species by species",
                        "Cross-validation, averaged over the folds on which each species is primary." + exception, headroom_chart(res),
                        c.legend([("neutral-key", "pretrained model"), ("soundscape", "trained on labeled soundscapes")]))
    gap = c.figure("gap-species", "Focal vs soundscape, pretrained model, test split",
                   "Bold names are the 7 primary species. ROC-AUC is shown because average precision depends on how common a species is, "
                   f"which differs between the two test sets (macro AP: {c.score(point['f_ap_primary'])} focal, {c.score(point['sc_ap_primary'])} "
                   "soundscape, not directly comparable). The negatives also differ: soundscapes include many clips with no target bird. "
                   "Focal recordings come from Xeno-Canto, which the model was pretrained on, so focal scores are an in-domain reference.",
                   auc_chart(res.test_species_auc(), res.test["species"]["primary"]),
                   c.legend([("focal", "focal test clips"), ("soundscape", "soundscape test clips")]))
    findings = c.section("findings", "What the prototypes reveal", """
    <div id="findings-list" class="findings"></div>
    <div id="evidence" class="evidence" aria-live="polite"></div>""",
        "Descriptive analysis of the pretrained model on clips <b>outside the test split</b> (training and validation sites and "
        "recordists; 14 species with at least 10 soundscape positives, 70 prototypes), so counts differ from the test-split numbers "
        "above. It uses soundscape labels, and its sites are the cross-validation sites: a method designed from these findings would "
        "have an optimistic cross-validation score. Treat them as hypotheses. Click a finding to see its data.")
    take = c.takeaways([
        f"On identical held-out soundscape clips, training on soundscape labels raises AP from {c.score(base)} to {c.score(top)}. This is the "
        "gap we want to close.",
        f"The pretrained model's test AUC (the chance that a clip with a bird is scored above a clip without it) is "
        f"{c.score(point['sc_auc_primary'])} on soundscapes and {c.score(point['f_auc_primary'])} on focal clips (out of 100). "
        "Part of this difference may reflect that focal data is in-domain for it and that the negatives differ.",
        f"Prototype detection degrades unevenly, even within one species: on test sites {point['n_fail']} of {n_proto} prototypes are "
        f"clearly worse at detecting their bird on soundscapes than on focal clips ({point['n_fail_birds']} of {n_proto} when both are "
        "compared against clips of other birds). This is a hypothesis for where a method could act, not evidence that fixing "
        "prototypes will raise accuracy.",
    ], "In short")
    return Page("index.html", "Motivation", "Motivation",
                "Bird-sound models are learned from clean recordings of single birds, but are needed on noisy field recordings.",
                headline_section(res) + take + why + closable + headroom + gap + findings, ("data.js", "app.js"))
