"""Motivation: the focal-to-soundscape gap, and what the prototypes reveal about it."""

from __future__ import annotations

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
    body = c.x_grid(x, [lo + 0.1 * i for i in range(int(round((1 - lo) * 10)) + 1)], top - 6, bottom, lambda t: f"{t:.1f}")
    body += c.text((x0 + x1) / 2, bottom + 34, "ROC-AUC (test split; axis does not start at 0)", "axis-title", "middle")
    for i, s in enumerate(order):
        y, d = top + i * row + 10, auc[s]
        body += c.text(x0 - 12, y + 4, c.name_of(s), "row-label" + (" sel" if s in primary else ""), "end")
        body += c.line(x(d["focal"]), y, x(d["soundscape"]), y, "connector")
        for dom in ("focal", "soundscape"):
            body += c.circle(x(d[dom]), y, 5.5, f"dot {dom}", f"{c.name_of(s)}: {dom} AUC {d[dom]:.3f}")
    return c.svg(bottom + 42, "Pretrained model: ROC-AUC per species on focal and soundscape test clips", body)


def headroom_chart(res: Results) -> str:
    """Per species, the pretrained and the soundscape-trained model on the same held-out clips."""
    species = res.primary_union()
    vals = {s: (res.species_mean(BASELINE, s)[0], res.species_mean(ORACLE, s)[0], res.species_mean(BASELINE, s)[1]) for s in species}
    order = sorted(species, key=lambda s: vals[s][0] - vals[s][1])
    x0, x1, row, top = 230, 760, 28, 22
    x = c.linear(0.4, 1.0, x0, x1)
    bottom = top + row * len(order)
    body = c.x_grid(x, (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), top - 6, bottom, lambda t: f"{t:.1f}")
    body += c.text((x0 + x1) / 2, bottom + 34, "Soundscape average precision, same held-out clips (axis does not start at 0)",
                   "axis-title", "middle")
    body += c.text(c.W, top - 10, "change · folds", "", "end")
    for i, s in enumerate(order):
        y, (base, top_ap, k) = top + i * row + 10, vals[s]
        body += c.text(x0 - 12, y + 4, c.name_of(s), "row-label", "end")
        body += c.line(x(base), y, x(top_ap), y, "connector")
        body += c.circle(x(base), y, 5.5, "dot neutral", f"{c.name_of(s)}: pretrained AP {base:.3f}")
        body += c.circle(x(top_ap), y, 5.5, "dot soundscape", f"{c.name_of(s)}: trained on labeled soundscapes, AP {top_ap:.3f}")
        body += c.text(c.W, y + 4, f"{c.signed(top_ap - base, 2)} · {k}", "cell", "end")
    return c.svg(bottom + 42, "Headroom per species: pretrained model vs soundscape-trained model", body)


def closable_chart(res: Results) -> str:
    """Same held-out soundscape clips: pretrained model vs the same architecture trained with soundscape labels."""
    rows = [("Pretrained model", BASELINE, "dot neutral"), ("Trained on labeled soundscapes", ORACLE, "dot soundscape")]
    x0, x1, row, top = 230, 760, 34, 12
    x = c.linear(0.5, 1.0, x0, x1)
    bottom = top + row * len(rows)
    body = c.x_grid(x, (0.5, 0.6, 0.7, 0.8, 0.9, 1.0), top, bottom, lambda t: f"{t:.1f}")
    body += c.text((x0 + x1) / 2, bottom + 34, "Soundscape average precision on held-out sites (primary species)", "axis-title", "middle")
    for i, (label, key, cls) in enumerate(rows):
        y = top + i * row + 16
        vals = res.values(key, "sc_ap_primary")
        mean = sum(vals) / len(vals)
        body += c.text(x0 - 12, y + 4, label, "row-label", "end")
        for f, v in enumerate(vals):
            body += c.circle(x(v), y, 4, cls + " fold", f"{label}, fold {f}: {v:.3f}")
        body += c.line(x(mean), y - 11, x(mean), y + 11, "mean-tick")
        body += c.text(x1 + 16, y + 4, f"mean {mean:.3f}")
    return c.svg(bottom + 42, "Pretrained model and soundscape-trained model on the same held-out sites", body)


def build(res: Results) -> Page:
    point = res.test["models"][BASELINE]["point"]
    n_proto = len(res.test["models"][BASELINE]["prototypes"])
    base = res.cv["r0"]["baseline_metrics"]["sc_ap_primary"]["mean"]
    top = res.cv[ORACLE]["run"]["sc_ap_primary"]["mean"]
    assert res.folds_better_worse(ORACLE) == (3, 0), "the page says labels help reliably on every fold"
    free = max(sum(res.deltas(k)) / 3 for k in LABEL_FREE)
    worse = [s for s in res.primary_union() if res.species_mean(ORACLE, s)[0] < res.species_mean(BASELINE, s)[0]]
    tiles = c.tiles([
        (f"{base:.3f} → {top:.3f}", "soundscape average precision on the same held-out clips: pretrained → trained on soundscape labels"),
        (c.signed(free), "best mean change reached so far without any soundscape label (cross-validation)"),
        (f"{point['f_auc_primary']:.3f} → {point['sc_auc_primary']:.3f}", "pretrained ROC-AUC, focal → soundscape (test)"),
        (f"{point['n_fail']} of {n_proto}", f"pretrained prototypes losing more than 0.1 AUC on soundscapes (test); "
         f"{point['n_fail_birds']} of {n_proto} when both domains use other-bird clips as negatives"),
    ])
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
                   f"which differs between the two test sets (macro AP: {point['f_ap_primary']:.3f} focal, {point['sc_ap_primary']:.3f} "
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
    codes = "".join(f"<span><code>{k}</code> {c.esc(v)}</span>" for k, v in c.SPECIES_NAMES.items())
    key = f'<details class="codes"><summary>Species codes used in the evidence charts</summary><div class="code-grid">{codes}</div></details>'
    take = c.takeaways([
        f"On identical held-out soundscape clips, training on soundscape labels raises AP from {base:.3f} to {top:.3f}. This is the "
        "gap we want to close.",
        f"The pretrained model's test AUC is {point['sc_auc_primary']:.3f} on soundscapes and {point['f_auc_primary']:.3f} on focal clips. "
        "Part of this difference may reflect that focal data is in-domain for it and that the negatives differ.",
        "Prototype detection degrades unevenly, even within one species. This is a hypothesis for where a method could act, not "
        "evidence that fixing prototypes will raise accuracy.",
    ], "In short")
    return Page("index.html", "Motivation", "Motivation",
                "Bird-sound models are learned from clean recordings of single birds, but are needed on noisy field recordings.",
                tiles + take + why + closable + headroom + gap + findings + key, ("data.js", "app.js"))
