"""Problem formulation."""

from __future__ import annotations

from prototype_transfer.reporting import charts as c
from prototype_transfer.reporting.runs import Results
from prototype_transfer.reporting.site import Page


def setting_diagram() -> str:
    def box(x, y, w, h, title, sub, cls="box-node"):
        return (c.rect(x, y, w, h, cls, rx=8) + c.text(x + w / 2, y + 24, title, "node-title", "middle")
                + "".join(c.text(x + w / 2, y + 44 + 16 * i, s, "", "middle") for i, s in enumerate(sub)))

    def arrow(x1, y1, x2, y2):
        return f'<line class="arrow" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#head)"/>'

    body = ('<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
            '<path class="arrow-head" d="M0,0L10,5L0,10z"/></marker></defs>')
    body += box(0, 6, 250, 82, "Pretrained AudioProtoPNet", ["trained on Xeno-Canto", "(focal recordings)"])
    body += box(0, 108, 250, 82, "Unlabeled soundscapes", ["audio from training sites", "no annotations"], "box-node soundscape-edge")
    body += box(0, 210, 250, 82, "Soundscape labels", ["N clips from training sites", "N = 0 is the target setting"], "box-node dashed-edge")
    body += box(345, 108, 210, 82, "Adaptation", ["changes weights, prototypes", "or the whole network"])
    body += box(650, 108, 250, 82, "Held-out sites", ["never seen in adaptation", "scored with the protocol"], "box-node soundscape-edge")
    body += arrow(250, 47, 343, 125) + arrow(250, 149, 343, 149) + arrow(250, 251, 343, 173) + arrow(555, 149, 648, 149)
    return c.svg(300, "Pretrained model and training-site data feed an adaptation step that is scored on held-out sites", body)


def build(res: Results) -> Page:
    th = res.test["thresholds"]
    model = c.section("model", "Model", """
    <div class="prose">
    <p>A backbone maps the spectrogram <i>x</i> of a 5 s clip to patch embeddings z<sub>t,f</sub>(x). Species <i>s</i> has K = 5
    prototypes p<sub>sk</sub>. Each prototype's activation is its best match anywhere in the clip, and the species score is a
    non-negative weighted sum:</p>
    <div class="math">a<sub>sk</sub>(x) = max<sub>t,f</sub> ReLU( cos( z<sub>t,f</sub>(x), p<sub>sk</sub> ) )<br>
    ŷ<sub>s</sub>(x) = σ( Σ<sub>k</sub> w<sub>sk</sub> a<sub>sk</sub>(x) + b<sub>s</sub> ),&nbsp;&nbsp; w<sub>sk</sub> ≥ 0</div>
    <p>An error can therefore come from the <b>activations</b> a<sub>sk</sub> (a prototype no longer matches, or matches the wrong sound),
    from how the <b>weights</b> w<sub>sk</sub> combine them, or from both.</p></div>""")
    data = c.section("setting", "Setting", c.figure("setting-diagram", "What a method may use", "", setting_diagram()) + """
    <div class="prose">
    <p>The source domain F (focal) has one labeled species per clip. The target domain S (soundscapes) is multi-label, includes many
    clips with no target species, and is grouped by recording site. Sites are divided into training sites and held-out sites.
    The task is multi-label recognition over 21 species on held-out soundscape sites.</p></div>""" + c.table(
        ["Setting", "Focal labels", "Unlabeled soundscape audio", "Soundscape labels", "Role"], [
            ["Zero-shot", "pretraining only", "—", "—", "baseline"],
            ["Focal fine-tuning", "yes", "—", "—", "control"],
            ["Unsupervised adaptation", "optional", "training sites", "—", "<b>target setting</b>"],
            ["Few labels", "pretraining only", "—", "N clips", "label curve"],
            ["All labels", "pretraining only", "—", "all training-site clips", "upper reference"],
        ], "mini crit"))
    goal = c.section("objective", "Objective", f"""
    <div class="prose">
    <p>Let M(h) be the soundscape macro average precision of model h over the primary species P, on held-out sites:</p>
    <div class="math">M(h) = (1 / |P|) Σ<sub>s∈P</sub> AP<sub>s</sub>(h)</div>
    <p>With h<sub>0</sub> the pretrained model and h<sub>★</sub> the model trained on all labeled training-site soundscapes, the share of
    the gap closed by an adapted model h is</p>
    <div class="math">γ(h) = ( M(h) − M(h<sub>0</sub>) ) / ( M(h<sub>★</sub>) − M(h<sub>0</sub>) )</div>
    <p>We look for an adaptation that uses <b>no soundscape labels</b> and reaches γ ≥ {th['gap_fraction']:.1f} while keeping focal AUC within
    {th['focal_tolerance']:.2f} of h<sub>0</sub> (full list under <a href="evaluation.html#criteria">success criteria</a>).
    h<sub>★</sub> is an empirical reference, not a proven upper bound: a method or a single species can exceed it.</p>
    <p>Prototype transfer is measured by Δ<sub>sk</sub> = AUC<sub>S</sub>(a<sub>sk</sub>) − AUC<sub>F</sub>(a<sub>sk</sub>), a diagnostic of
    <i>where</i> the model breaks.</p></div>""")
    questions = c.section("questions", "Research questions", """<ol class="prose">
    <li><b>How much supervision does the gap cost?</b> How many labeled soundscape clips are needed to close half of it?
    → <a href="label-curve.html">Label curve</a></li>
    <li><b>Can it be closed without labels?</b> By adapting activations or weights with unlabeled audio, or by making focal training
    look more like soundscapes? → <a href="results.html">Initial results</a></li>
    <li><b>Is prototype transfer the lever?</b> When prototypes detect better on soundscapes, does average precision follow?
    (Observational comparisons across methods cannot show cause; that needs an intervention on prototypes alone.)
    → <a href="results.html#prototypes">Initial results</a></li></ol>""")
    return Page("formulation.html", "Problem formulation", "Problem formulation",
                "Adapt an interpretable bird-sound classifier from focal recordings to soundscapes, ideally without soundscape labels.",
                model + data + goal + questions)
