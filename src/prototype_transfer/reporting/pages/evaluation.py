"""Evaluation: the one protocol every model is scored with."""

from __future__ import annotations

import math

from prototype_transfer.evaluation import canonical
from prototype_transfer.reporting import charts as c
from prototype_transfer.reporting.runs import BASELINE, Results
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
        body += c.line(x(t), top - 6, x(t), top + row * len(rows) - 12, "gridline") + c.text(x(t), top - 12, f"{t:.1f}", "", "middle")
    for i, (label, key, dom) in enumerate(rows):
        y = top + i * row + 6
        p, (a, b) = model["point"][key], model["ci95"][key]
        body += c.text(0, y + 4, label, "row-label") + c.line(x(a), y, x(b), y, f"whisker {dom}")
        body += c.circle(x(p), y, 5, f"dot {dom}", f"{label}: {p:.3f} [{a:.3f}, {b:.3f}]")
        body += c.text(x0 + w + 14, y + 4, f"{p:.3f}  [{a:.3f}, {b:.3f}]")
    return c.svg(top + row * len(rows) - 4, "Pretrained model on the test split with 95% intervals", body)


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
         "model's pretraining data (Xeno-Canto), so focal scores are an in-domain reference."),
        ("3 · Prototype transfer", "For each prototype, Δ = AUC<sub>soundscape</sub> − AUC<sub>focal</sub> of its maximum activation as a detector "
         f"of its species. <b>Fails</b> if Δ &lt; {th['fail']:+.2f}; <b>improves</b> if Δ &gt; {th['improve']:+.2f}. "
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
                    f"{m['point']['n_fail']} of {len(m['prototypes'])} prototypes fail; median Δ {m['point']['median_delta']:+.3f}.",
                    baseline_chart(m), c.legend([("focal", "focal"), ("soundscape", "soundscape")]))
    crit = [("AP gap closed", f"at least {c.pct(th['gap_fraction'])} of the gap to the model trained on labeled soundscapes is closed, and the "
             "95% interval of the improvement over the pretrained model excludes 0"),
            ("AUC not below baseline", "soundscape AUC (primary) is at least the pretrained model's"),
            ("Failing prototypes halved", "at most half as many prototypes fail as for the pretrained model"),
            ("Focal retained", f"focal AUC (primary) is within {th['focal_tolerance']:.2f} of the pretrained model's")]
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
                "One fixed way to score every model. Changing any part means re-scoring all models.", steps + which + base + criteria + cv_note)
