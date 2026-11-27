"""Problem formulation: the task, which data every model learns from and is scored on, and the objective."""

from __future__ import annotations

import json

from prototype_transfer.reporting import charts as c
from prototype_transfer.reporting.runs import BASELINE, FOLDS, GROUPS, LABEL_BUDGETS, METHODS, ORACLE, PUBLISHED, TEST_RUNS, Results
from prototype_transfer.reporting.site import Page

SPLITS = ("train", "val", "test")
YES, NO = "<b>✓</b>", "—"


def check_data_use(res: Results) -> None:
    """Fail the build if a saved run no longer matches what this page says about who learns from which data."""
    def config(run: str) -> dict:
        return json.loads((res.store.run_dir(run) / "config.json").read_text())

    def check(run: str, group: str, fold: int | None) -> None:
        cfg = config(run)
        if group == "free":     # training-free: fitted on activations only
            assert cfg["uses_labels"] is False, run
        elif group == "sfda":
            want = "soundscape_train" if fold is None else f"soundscape_cv{fold}_train"
            assert cfg["effective"]["uses_labels"] is False and cfg["effective"]["adapt_manifest"].endswith(f"/{want}.parquet"), run
        else:
            args = cfg["args"]
            assert args["select"] == "last" and args["cv_fold"] == fold, f"{run}: the page says no label picks a saved model"
            assert args["train_domain"] == ("soundscape" if group == "labels" else "focal"), run
            real, pseudo = args["bg_manifest"], args["pseudo_manifest"]
            assert (real is not None or pseudo is not None) == (group == "mix"), f"{run}: use of soundscape audio changed"
            for path in (real, pseudo):
                assert path is None or path.endswith(f"_cv{fold}.parquet"), f"{run}: soundscape audio from outside the training sites"

    for gkey, _, methods in GROUPS:
        group = "free" if methods[0].key.startswith("tf-") else gkey
        for m in methods:
            for f in FOLDS:
                check(f"{m.key}-cv{f}", group, f)
    for run in TEST_RUNS:
        check(run, "sfda" if run in PUBLISHED else ("labels" if run == ORACLE else "focal"), None)

    dev = [set(res.splits_cv["sites"][str(f)]) for f in FOLDS]
    sites = {s: set(res.splits["soundscape"]["sites"][s]) for s in SPLITS}
    assert sum(len(d) for d in dev) == len(set().union(*dev)), "a site is in two folds"
    assert set().union(*dev) == sites["train"] | sites["val"] and not set().union(*dev) & sites["test"], \
        "the folds are no longer the non-test sites"


def usage_diagram(res: Results) -> str:
    """Which soundscape sites and focal clips are learned from and scored, in development and in the final test."""
    heldout = [len(res.splits_cv["sites"][str(f)]) for f in FOLDS]
    sites = {s: len(res.splits["soundscape"]["sites"][s]) for s in SPLITS}
    x0, width, h, row = 160, c.W - 160, 30, 40
    total = sum(sites.values())

    def bar(y: float, parts: list[tuple[float, str, str]], whole: float) -> str:
        out, x = "", x0
        for size, cls, label in parts:
            w = width * size / whole
            out += c.rect(x, y, w - 3, h, cls, label, rx=4)
            out += c.text(x + (w - 3) / 2, y + 19, label, "on-solid" if cls.startswith("use-score") else "seg-label", "middle")
            x += w
        return out

    body = c.text(0, 14, f"Soundscape: {total} recording sites (box width = number of sites)", "panel-title")
    y = 24
    for f in FOLDS:
        parts = [(n, "use-score-sc" if g == f else "use-learn-sc", f"score · {n} sites" if g == f else f"train · {n} sites")
                 for g, n in enumerate(heldout)]
        body += c.text(x0 - 10, y + 19, f"Development, fold {f}", "row-label", "end")
        body += bar(y, parts + [(sites["test"], "use-off", f"{sites['test']} test sites: not used")], total)
        y += row
    y += 8
    body += c.text(x0 - 10, y + 19, "Final test", "row-label sel", "end")
    body += bar(y, [(sites["train"], "use-learn-sc", f"train · {sites['train']} sites"),
                    (sites["val"], "use-off", f"{sites['val']}: unused"),
                    (sites["test"], "use-score-sc", f"score · {sites['test']} test sites")], total)
    y += row + 24
    clips = {s: res.splits["focal"][s]["n_clips"] for s in SPLITS}
    body += c.text(0, y, "Focal: clips split by recordist (box width = number of clips)", "panel-title")
    y += 10
    body += c.text(x0 - 10, y + 19, "All stages", "row-label", "end")
    body += bar(y, [(clips["train"], "use-learn-focal", f"train · {clips['train']:,} clips"),
                    (clips["val"], "use-score-focal", f"val · {clips['val']:,}"),
                    (clips["test"], "use-score-focal", f"test · {clips['test']:,}")], sum(clips.values()))
    y += h + 16
    body += c.text(x0 + width * clips["train"] / sum(clips.values()) / 2, y, "focal fine-tuning learns here", "", "middle")
    start = x0 + width * clips["train"] / sum(clips.values())
    body += c.text(start + width * clips["val"] / sum(clips.values()) / 2, y, "dev. check", "", "middle")
    body += c.text(start + width * (clips["val"] + clips["test"] / 2) / sum(clips.values()), y, "final check", "", "middle")
    return c.svg(y + 10, "Soundscape sites and focal clips used for training and for scoring, in development and final test", body)


def split_table(res: Results) -> str:
    sp = res.splits
    sites = {s: len(sp["soundscape"]["sites"][s]) for s in SPLITS}
    size = {s: f"{sp['focal'][s]['n_clips']:,} clips<br><small>{sp['focal'][s]['recordists']} recordists</small>" for s in SPLITS}
    sc = {s: f"{sp['soundscape'][s]['n_clips']:,} clips<br><small>{sites[s]} sites</small>" for s in SPLITS}
    dev = sites["train"] + sites["val"]
    pooled = (f"The train and val sites are <b>pooled into {dev} development sites</b> and re-cut into {len(FOLDS)} folds. In each "
              "fold, one group is scored; methods may learn from the other two.")
    rows = [
        ["<b>Focal train</b>", size["train"], "Focal fine-tuning learns from <b>audio and labels</b>. Training-free methods use "
         "its <b>audio only</b>. Same clips in every fold.", "Full focal fine-tuning learns from audio and labels."],
        ["<b>Focal val</b>", size["val"], "<b>Labels score</b> the focal check (does the model still work on focal audio?).",
         "Not used for any reported number."],
        ["<b>Focal test</b>", size["test"], "Not used.", "<b>Labels score</b> the focal check."],
        ["<b>Soundscape train</b>", sc["train"], pooled, "Adaptation uses the <b>audio</b>. The all-labels reference uses audio "
         "<b>and labels</b>."],
        ["<b>Soundscape val</b>", sc["val"], f"Pooled with soundscape train (above).", "Not learned from. Scores are logged during "
         "training but change nothing."],
        ["<b>Soundscape test</b>", sc["test"], "Not used to train or compare methods.", "<b>Labels score</b> the headline result. "
         "No model learns from this audio or these labels."]]
    return c.table(["Split", "Size", "Development (cross-validation)", "Final test"], rows, "mini words")


def permission_table(res: Results) -> str:
    keys = {g: [m.key for m in ms] for g, _, ms in GROUPS}
    free = [k for k in METHODS if k.startswith("tf-")]
    focal_only = keys["focal"] + keys["ablation"]
    budgets = f"{min(LABEL_BUDGETS):,}–{max(LABEL_BUDGETS):,}"
    rows = [
        ["<b>Starting model</b><br><small>pretrained AudioProtoPNet, unchanged</small>", NO, NO, NO, NO, "yes", "baseline"],
        [f"<b>Source-free adaptation</b><br><small>{len(PUBLISHED)} published methods</small>", NO, NO, YES, NO, "all",
         "<b>target setting</b>"],
        [f"<b>Training-free adaptation</b><br><small>{len(free)} simple methods</small>", YES, NO, YES, NO, "no",
         "<b>target setting</b>"],
        [f"<b>Focal fine-tuning + soundscape audio</b><br><small>{len(keys['mix'])} variants</small>", YES, YES, YES, NO, "no",
         "<b>target setting</b>"],
        [f"<b>Focal fine-tuning only</b><br><small>{len(focal_only)} variants</small>", YES, YES, NO, NO, "full fine-tuning only",
         "control: what does more focal training do?"],
        [f"<b>Label curve</b><br><small>N = {budgets} clips</small>", NO, NO, "N clips", "<b>N clips</b>", "no",
         "reference: what do a few labels buy?"],
        ["<b>All labels</b><br><small>every training-site clip</small>", NO, NO, "all", "<b>all</b>", "yes",
         "reference: defines the gap"]]
    return c.table(["Method family", "Focal train<br>audio", "Focal train<br>labels", "Soundscape audio<br>(training sites)",
                    "Soundscape labels<br>(training sites)", "Scored on<br>test sites?", "Role"], rows, "mini words use")


def label_uses() -> str:
    rows = [
        ["<b>Building the splits</b>", "Species name is used to balance the split (clips grouped by recordist).",
         "Labels of all sites, <i>including test sites</i>, are used to spread each species across splits and folds. Done before any "
         "model was run."],
        ["<b>Learning</b>", "Focal fine-tuning (focal train only).",
         "Only the label curve and the all-labels reference, and only on training sites. <b>No method in the target setting reads "
         "them.</b>"],
        ["<b>Choosing which saved model to keep</b>", "Never.", "Never. Every trained run keeps its last epoch. Validation and "
         "held-out scores are only logged."],
        ["<b>Deciding which species are scored</b>", "Positive counts in the focal set being scored.",
         "Positive counts in the sites being scored."],
        ["<b>Scoring</b>", "Focal val (development), focal test (final).", "Held-out fold (development), test sites (final)."],
        ["<b>Comparing methods</b>", "Checks that focal performance is kept.", "We pick the best method from development scores, so "
         "its development score is on the favourable side. That is why test matters."],
        ["<b>Describing the model</b>", "Compared with soundscapes in the prototype analyses.",
         "The prototype findings use development-site labels; some charts show the pretrained model on test sites. A method designed "
         "from either would not get a fully independent score on those sites."]]
    return c.table(["Use", "Focal labels", "Soundscape labels"], rows, "mini words")


def build(res: Results) -> Page:
    check_data_use(res)
    th = res.test["thresholds"]
    sp = res.splits
    dev = sum(len(sp["soundscape"]["sites"][s]) for s in ("train", "val"))

    task = '<div class="summary"><p class="lead">Start with a bird-sound model trained on <b>focal</b> recordings. Using ' \
           "<b>unlabeled</b> soundscape audio from some recording sites, adapt it. Then freeze it and measure how well it recognises " \
           "birds on soundscapes from <b>other sites</b>.</p></div>"
    domains = c.section("domains", "The two kinds of data", c.table(
        ["", "Focal (source)", "Soundscape (target)"], [
            ["What it is", "A person points a microphone at one bird (Xeno-Canto).", "A fixed recorder captures everything at a site."],
            ["Labels", "One species per clip.", "Any number of the 21 species per 5 s clip, often none."],
            ["Grouped by", "Recordist.", "Recording site."],
            ["Relation to the starting model", "The model was pretrained on Xeno-Canto, including this data, so focal scores are "
             "<b>not</b> a test on unseen data.", "The target the model was not trained on."]], "mini words"))
    usage = c.section("data-use", "Which data is used where", c.figure(
        "usage-diagram", "Training and scoring, stage by stage",
        f"<b>Development</b> compares methods by {len(FOLDS)}-fold cross-validation on {dev} sites and reports the mean of the folds. "
        "<b>Final test</b> trains or adapts again on the original training sites and scores the test sites. In every row, no site is "
        "both trained on and scored. In the final-test row, widths show how many sites, not which development folds they came from.",
        usage_diagram(res),
        c.legend([("use-learn-sc-key", "soundscape: may train on"), ("soundscape", "soundscape: scored"),
                  ("use-learn-focal-key", "focal: may train on"), ("focal", "focal: scored"), ("use-off-key", "not used")]))
        + "<h3>What happens to each split</h3>" + split_table(res))
    allowed = c.section("allowed", "What each method is allowed to learn from", permission_table(res)
                        + '<div class="prose"><p>Every method <b>starts from the pretrained model</b>. Methods trained on soundscape '
                        "labels fine-tune it on soundscape clips only, with no focal training data. “Training sites” means the "
                        "development fold's training sites, or the original training sites in the final test.</p></div>")
    labels = c.section("labels", "Where labels are used", label_uses())

    model = c.section("model", "Model", """
    <div class="prose">
    <p>A backbone maps the spectrogram <i>x</i> of a 5 s clip to patch embeddings z<sub>t,f</sub>(x). Species <i>s</i> has K = 5
    prototypes p<sub>sk</sub>. Each prototype's activation is its best match anywhere in the clip, and the species score is a
    non-negative weighted sum:</p>
    <div class="math">a<sub>sk</sub>(x) = max<sub>t,f</sub> ReLU( cos( z<sub>t,f</sub>(x), p<sub>sk</sub> ) )<br>
    ŷ<sub>s</sub>(x) = σ( Σ<sub>k</sub> w<sub>sk</sub> a<sub>sk</sub>(x) + b<sub>s</sub> ),&nbsp;&nbsp; w<sub>sk</sub> ≥ 0</div>
    <p>An error can therefore come from the <b>activations</b> a<sub>sk</sub> (a prototype no longer matches, or matches the wrong sound),
    from how the <b>weights</b> w<sub>sk</sub> combine them, or from both.</p></div>""")
    goal = c.section("objective", "Formal problem and objective", f"""
    <div class="prose">
    <p><b>Given</b> the pretrained model h<sub>0</sub>; optionally the labeled focal training clips D<sub>F</sub>; and soundscape sites
    split into training sites T and held-out sites H with T ∩ H = ∅. U<sub>T</sub> is the audio of T and Y<sub>T</sub> its labels.</p>
    <p><b>Target setting</b> (inductive, no soundscape labels): find a procedure that returns</p>
    <div class="math">h = Adapt( h<sub>0</sub>, U<sub>T</sub> [, D<sub>F</sub>] )</div>
    <p>and freeze h before it sees any audio from H. Methods that adapt on the audio being scored are outside this problem.</p>
    <p><b>Score</b> on H: soundscape average precision, averaged over the primary species P<sub>H</sub>
    (<a href="evaluation.html#metrics">what the metric means</a>):</p>
    <div class="math">M<sub>H</sub>(h) = (1 / |P<sub>H</sub>|) Σ<sub>s∈P<sub>H</sub></sub> AP<sub>s</sub>(h; H)</div>
    <p><b>Reference</b>: h<sub>★</sub> is the pretrained model fine-tuned on U<sub>T</sub> with Y<sub>T</sub>. The share of the gap
    closed is</p>
    <div class="math">γ<sub>H</sub>(h) = ( M<sub>H</sub>(h) − M<sub>H</sub>(h<sub>0</sub>) ) / ( M<sub>H</sub>(h<sub>★</sub>) − M<sub>H</sub>(h<sub>0</sub>) )</div>
    <p><b>Goal</b>: γ ≥ {c.pct(th['gap_fraction'])} with the gain reliably above zero, while keeping focal AUC within
    {c.score(th['focal_tolerance'], 0)} point (out of 100) of h<sub>0</sub>. The full list is under
    <a href="evaluation.html#criteria">success criteria</a>. h<sub>★</sub> is a reference from one training recipe, not a proven upper
    bound: a method or a single species can exceed it.</p>
    <p>In development, H is one fold and T the other two; we report means over the folds, and γ as the mean gain divided by the mean
    gain of h<sub>★</sub>. In the final test, T is the original training sites and H the test sites.</p>
    <p>Prototype transfer is measured by Δ<sub>sk</sub> = AUC<sub>soundscape</sub>(a<sub>sk</sub>) − AUC<sub>focal</sub>(a<sub>sk</sub>),
    a diagnostic of <i>where</i> the model breaks, not the objective.</p></div>""")
    questions = c.section("questions", "Research questions", """<ol class="prose">
    <li><b>How much supervision does the gap cost?</b> How many labeled soundscape clips are needed to close half of it?
    → <a href="label-curve.html">Label curve</a></li>
    <li><b>Can it be closed without labels?</b> By adapting activations or weights with unlabeled audio, or by making focal training
    look more like soundscapes? → <a href="results.html">Initial results</a></li>
    <li><b>Is prototype transfer the lever?</b> When prototypes detect better on soundscapes, does average precision follow?
    (Observational comparisons across methods cannot show cause; that needs an intervention on prototypes alone.)
    → <a href="results.html#prototypes">Initial results</a></li></ol>""")
    assert BASELINE not in METHODS   # the starting model is listed separately in the permission table
    return Page("formulation.html", "Problem formulation", "Problem formulation",
                "Adapt an interpretable bird-sound classifier from focal recordings to soundscapes without soundscape labels.",
                task + domains + usage + allowed + labels + model + goal + questions)
