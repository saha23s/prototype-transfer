"""Evaluation: the one protocol every model is scored with."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score

from prototype_transfer.evaluation import canonical
from prototype_transfer.reporting import charts as c
from prototype_transfer.reporting.runs import BASELINE, ORACLE, Results
from prototype_transfer.reporting.site import Page


def threshold_chart(sp: dict, primary: set[str], secondary: set[str]) -> str:
    order = sorted(sp["soundscape"]["test"]["positives"], key=lambda k: -sp["soundscape"]["test"]["positives"][k])
    x0, w, lo, hi, row, top = 250, 480, 0.7, 1000, 22, 56
    log_x = lambda v: x0 + w * (math.log10(max(v, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    body = c.text(x0, 14, "Test-set positive clips per species (log scale; 0 is drawn at the left edge)", "row-label")
    for t in (1, 10, 100, 1000):
        body += c.line(log_x(t), top - 8, log_x(t), top + row * len(order) - 6, "gridline") + c.text(log_x(t), top - 14, str(t), "", "middle")
    for t, lab, anchor in ((canonical.MIN_POS, f"secondary: ≥ {canonical.MIN_POS} in both ", "end"),
                           (canonical.PRIMARY_MIN_POS, f" primary: ≥ {canonical.PRIMARY_MIN_POS} soundscape", "start")):
        body += c.line(log_x(t), top - 8, log_x(t), top + row * len(order) - 6, "diag") + c.text(log_x(t), top - 28, lab, "row-label", anchor)
    for i, k in enumerate(order):
        y = top + i * row + 8
        role = "primary" if k in primary else ("secondary" if k in secondary else "not scored")
        body += c.text(0, y + 4, c.name_of(k), "row-label" + (" sel" if k in primary else "")) + c.text(170, y + 4, role)
        f, s = sp["focal"]["test"]["positives"][k], sp["soundscape"]["test"]["positives"][k]
        body += c.line(min(log_x(f), log_x(s)), y, max(log_x(f), log_x(s)), y, "connector")
        body += c.circle(log_x(f), y, 5, "dot focal", f"{c.name_of(k)}: {f} focal test positives")
        body += c.circle(log_x(s), y, 5, "dot soundscape", f"{c.name_of(k)}: {s} soundscape test positives")
        body += c.text(x0 + w + 24, y + 4, f"{c.n(f)} / {c.n(s)}")
    body += c.text(x0 + w + 24, top - 14, "focal / soundscape")
    return c.svg(top + row * len(order) + 4, "Test positives per species with the thresholds for primary and secondary species", body)


def baseline_chart(model: dict) -> str:
    rows = [("Soundscape AP (primary) — headline", "sc_ap_primary", "soundscape"), ("Soundscape AUC (primary)", "sc_auc_primary", "soundscape"),
            ("Soundscape cmAP (secondary)", "sc_cmap_12", "soundscape"), ("Focal AP (primary)", "f_ap_primary", "focal"),
            ("Focal AUC (primary)", "f_auc_primary", "focal")]
    x0, w, lo, hi, row, top = 240, 500, 0.3, 1.0, 34, 24
    x = c.linear(lo, hi, x0, x0 + w)
    body = ""
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        body += c.line(x(t), top - 6, x(t), top + row * len(rows) - 12, "gridline") + c.text(x(t), top - 12, c.score(t, 0), "", "middle")
    for i, (label, key, dom) in enumerate(rows):
        y = top + i * row + 6
        p, (a, b) = model["point"][key], model["ci95"][key]
        body += c.text(0, y + 4, label, "row-label") + c.line(x(a), y, x(b), y, f"whisker {dom}")
        body += c.circle(x(p), y, 5, f"dot {dom}", f"{label}: {c.score(p)} [{c.score(a)}, {c.score(b)}]")
        body += c.text(x0 + w + 14, y + 4, f"{c.score(p)}  [{c.score(a)}, {c.score(b)}]")
    return c.svg(top + row * len(rows) - 4, "Pretrained model on the test split with 95% intervals", body)


STUDENT = "sfda-dropout_student"
# where the 2 clips with the bird land in a ranking of 5 clips
AP_EXAMPLES = (("Both at the top", (1, 1, 0, 0, 0)), ("One slips to third place", (1, 0, 1, 0, 0)),
               ("Both at the bottom", (0, 0, 0, 1, 1)))


def ap_example() -> str:
    rows = []
    for name, has_bird in AP_EXAMPLES:
        ranks = [i + 1 for i, y in enumerate(has_bird) if y]
        shares = [(n + 1) / rank for n, rank in enumerate(ranks)]
        value = float(np.mean(shares))
        assert abs(value - average_precision_score(has_bird, np.arange(len(has_bird), 0, -1))) < 1e-9
        marks = " ".join("●" if y else "○" for y in has_bird)
        work = " and ".join(f"{n + 1}/{rank}" for n, rank in enumerate(ranks))
        rows.append([name, marks, f"average of {work}", f"<b>{c.score(value)}</b>"])
    return c.table(["Ranking of 5 clips", "Highest score → lowest<br>(● has the bird)", "Calculation", "AP"], rows, "mini wrap-head words")


def metrics_section(res: Results) -> str:
    """Every metric on the site in plain terms, and the order in which to read a result."""
    avg = lambda v: float(np.mean(v))
    gain, gap = avg(res.deltas(STUDENT)), avg(res.deltas(ORACLE))
    better, worse = res.folds_better_worse(STUDENT)
    e, point = res.final["models"][STUDENT], res.final["models"][BASELINE]["point"]
    held = e["vs_baseline"]["sc_ap_primary"]
    focal = e["point"]["f_auc_primary"]
    assert worse == 0 and 0 < better < len(res.deltas(STUDENT)) and held["ci95"][0] < 0 < held["ci95"][1] \
        and e["criteria"]["4_focal_retained"], "the worked reading of Dropout Student's result is no longer true"
    tolerance = c.score(res.test["thresholds"]["focal_tolerance"], 0)

    example = ('<h3>Average precision (AP), the main metric</h3><div class="prose"><p>Take one bird and sort all clips by the '
               "model's score for it. Go down the list. Each time you reach a clip that has the bird, note what share of the clips "
               "so far have it. <b>AP is the average of those shares.</b> It is 100 when every clip with the bird comes first, and "
               "mistakes near the top cost the most. A random order scores about the share of clips that have the bird, so rarer "
               "birds start lower and AP should not be compared between sets of clips in which birds are not equally common.</p></div>"
               + ap_example())

    metric_rows = [
        ["<b>Soundscape AP</b><br><small>primary species</small>", "Are the field clips with each bird near the top of its ranking?",
         f"AP for each species, then the plain average over primary species (at least {canonical.PRIMARY_MIN_POS} positive clips).",
         "<b>The main result.</b> Higher is better."],
        ["Soundscape ROC-AUC", "Take one clip with the bird and one without. How often does the first score higher?",
         "100 is always, 50 is a coin flip. Averaged over species.",
         f"Supporting. Higher is better. It can look high while AP is much lower (pretrained model on test: AUC "
         f"{c.score(point['sc_auc_primary'])}, AP {c.score(point['sc_ap_primary'])}): when a bird is rare, a small share of the many "
         "clips without it ranking high lowers AP a lot but AUC little."],
        ["Soundscape cmAP", "Does the result hold when less common species are included?",
         f"The same AP, averaged over secondary species (at least {canonical.MIN_POS} positive clips in both the focal and "
         "soundscape sets). It includes the primary species.", "Supporting. Higher is better; species with few clips make it noisy."],
        ["Focal AUC and AP", "Does the model still work on clean single-bird recordings?",
         "The same calculations on focal clips.", f"A safety check: focal AUC should stay within {tolerance} point of the "
         "pretrained model."],
        ["Failing prototypes", "How many learned sound patterns detect their bird clearly worse on soundscapes?",
         "Use one prototype's strongest match in a clip as a bird detector. It fails if its soundscape AUC is more than "
         f"{c.score(-canonical.FAIL, 0)} points below its focal AUC. <i>Matched</i>: only soundscape clips with another bird count "
         "as negatives, as for focal clips.", "A diagnosis of where the model struggles, not a measure of accuracy. Lower is better."]]
    metrics = ("<h3>All metrics</h3>"
               + c.table(["Metric", "Question it answers", "How it is computed", "How to treat it"], metric_rows, "mini words"))

    attached = [
        ["<b>Change</b>, in points", f"Method minus pretrained model on the same clips. {c.signed(gain)} points means the score rose by "
         "that much on the 0–100 scale; it is not a percentage of clips or a relative improvement."],
        ["<b>Share of the gap closed</b> (γ)", f"The method's change divided by the change from training on all soundscape labels "
         f"(development: {c.score(gain)} / {c.score(gap)} = {gain / gap:.0%}). 0% closes nothing, 100% matches that reference, and a "
         "negative value is worse than the pretrained model."],
        ["<b>95% interval</b>", "How much the change moves when recording sites are resampled. If the whole range is above zero the "
         "method is <i>reliably better</i>; below zero, <i>reliably worse</i>. If it includes zero we cannot tell, which is not "
         "evidence of no effect."],
        ["<b>Folds ↑ / ↓</b>", "Development uses 3 groups of held-out sites. These count the groups where the method is reliably "
         "better / reliably worse."],
        ["<b>± value</b>", "How much the score varies between the 3 folds. It is a spread, not a confidence interval."],
        ["<b>Bold, underline, *</b>", "Bold and underline mark the best and second-best scores; they say nothing about reliability. "
         "* means a test interval excludes zero."]]
    numbers = "<h3>Numbers attached to a result</h3>" + c.table(["Term", "Meaning"], attached, "mini words")

    steps = [
        ["1. How big is it?", "Compare the change in soundscape AP with the gap.",
         f"{c.signed(gain)} of a {c.score(gap)}-point gap ({gain / gap:.0%})."],
        ["2. Is it consistent?", "Check the folds ↑ / ↓.",
         f"Reliably better on {better} of {len(res.deltas(STUDENT))} folds, reliably worse on {worse}."],
        ["3. Does it hold on test?", "Check the test change and whether its interval includes zero.",
         f"{c.signed(held['diff'])} points, {c.signed(held['ci95'][0])} to {c.signed(held['ci95'][1])}: includes zero."],
        ["4. Did anything break?", "Check focal AUC.",
         f"Focal AUC on test {c.score(focal)}, pretrained {c.score(point['f_auc_primary'])}: retained."]]
    reading = ("<h3>How to read a result, in this order</h3>"
               + c.table(["Question", "What to check", "Example: Dropout Student"], steps, "mini words")
               + '<div class="prose"><p><b>Verdict for the example:</b> promising in development, not confirmed on test. The stricter '
               '<a href="#criteria">success criteria</a> are listed below.</p></div>')
    return c.section("metrics", "Metrics and how to read them", example + reading + metrics + numbers,
                     "All scores are shown out of 100. If you read only one thing: soundscape AP is the main metric; everything else "
                     "supports or explains it.")


def build(res: Results) -> Page:
    out = res.test
    th, sp = out["thresholds"], out["species"]
    m = out["models"][BASELINE]
    cards = [
        ("1 · Species", f"<b>Secondary</b>: at least {canonical.MIN_POS} positives in both the focal and soundscape evaluation sets "
         f"({len(sp['secondary'])} species on test). <b>Primary</b>: also at least {canonical.PRIMARY_MIN_POS} soundscape positives "
         f"({len(sp['primary'])} on test)."),
        ("2 · Headline metric", "<b>Soundscape macro average precision over the primary species.</b> Also reported: soundscape ROC-AUC, "
         "cmAP over the secondary species, and focal AUC / AP to check that focal performance is retained. Focal recordings are part of the "
         "model's pretraining data (Xeno-Canto), so focal scores are an in-domain reference. <b>All scores on this site are shown out of "
         "100, and changes in points</b> (for example 71.0 to 77.8 is +6.7 points)."),
        ("3 · Prototype transfer", "For each prototype, Δ = AUC<sub>soundscape</sub> − AUC<sub>focal</sub> of its maximum activation as a detector "
         f"of its species. <b>Fails</b> if Δ &lt; {c.signed(th['fail'], 0)} points; <b>improves</b> if Δ &gt; {c.signed(th['improve'], 0)} points. "
         f"{canonical.K} prototypes per species. Soundscape negatives are all other clips; as a sensitivity check they are also "
         "restricted to clips with another target bird, which matches the focal negatives."),
        ("4 · Uncertainty", f"Cluster bootstrap with {c.n(out['n_boot'])} resamples shared across models, so differences are paired. "
         f"Soundscape clips are resampled by site ({out['n_clusters']['sites']} test sites), focal clips by recordist "
         f"({out['n_clusters']['recordists']}). Intervals are 95% percentile intervals."),
        ("5 · Development", "Method choices are made on the validation split or 3-fold site-grouped cross-validation. "
         "The test sites are kept for final numbers."),
        ("6 · Use of labels and audio", "Soundscape labels are used to evaluate. Methods that adapt may use unlabeled audio of training "
         "sites, never audio of held-out sites. All models are scored on identical clips."),
    ]
    steps = '<div class="steps">' + "".join(f'<div class="step"><h3>{t}</h3><p>{b}</p></div>' for t, b in cards) + "</div>"
    which = c.figure("protocol-species", "Which species are scored (test split)",
                     "Dashed lines mark the two thresholds. Species below them are still trained on but not scored.",
                     threshold_chart(res.splits, set(sp["primary"]), set(sp["secondary"])),
                     c.legend([("focal", "focal test positives"), ("soundscape", "soundscape test positives")]))
    base = c.figure("protocol-baseline", "Reference point: the pretrained model on the test split",
                    f"Dot: estimate; bar: 95% interval. With {out['n_clusters']['sites']} test sites the soundscape intervals are wide. "
                    f"{m['point']['n_fail']} of {len(m['prototypes'])} prototypes fail; median Δ {c.signed(m['point']['median_delta'])} points.",
                    baseline_chart(m), c.legend([("focal", "focal"), ("soundscape", "soundscape")]))
    crit = [("AP gap closed", f"at least {c.pct(th['gap_fraction'])} of the gap to the model trained on labeled soundscapes is closed, and the "
             "95% interval of the improvement over the pretrained model excludes 0"),
            ("AUC not below baseline", "soundscape AUC (primary) is at least the pretrained model's"),
            ("Failing prototypes halved", "at most half as many prototypes fail as for the pretrained model"),
            ("Focal retained", f"focal AUC (primary) is within {c.score(th['focal_tolerance'], 0)} point of the pretrained model's")]
    criteria = c.section("criteria", "Success criteria for a method",
                         c.table(["#", "Criterion", "Definition"], [[str(i + 1), a, b] for i, (a, b) in enumerate(crit)], "mini crit"),
                         "All four must hold for a method to count as closing the gap.")
    cv_note = c.section("cv-reading", "Reading cross-validation numbers", """<ul class="prose">
    <li>Each fold has about 23 held-out sites. Results are shown <b>per fold</b>; a “mean ± sd” describes the spread over 3 folds and is not a confidence interval.</li>
    <li>“Reliably better” means the fold's paired 95% interval of the difference to the pretrained model lies above zero (and “reliably worse”, below).</li>
    <li>The scored species differ between folds, so failing prototypes are reported as a share of the prototypes scored on each fold,
    and method changes are also reported on the same species every fold (those primary on all folds, and those eligible on all folds).</li>
    <li>The protocol's intervals compare each model with the pretrained model. The intervals between two adapted models on the
    results page use the same site resampling but are exploratory additions. Neither includes variation between training seeds.</li>
    <li>Many methods are compared on the same folds without correction for multiple comparisons.</li></ul>""")
    return Page("evaluation.html", "Evaluation", "Evaluation protocol",
                "One fixed way to score every model. Changing any part means re-scoring all models.", metrics_section(res) + steps + which + base + criteria + cv_note)
