"""Every symbol, abbreviation and mark on the dashboard, defined once. Each page ends with the entries it uses."""

from __future__ import annotations

from prototype_transfer.evaluation import canonical
from prototype_transfer.reporting import charts as c

MIN_POS, PRIMARY, K = canonical.MIN_POS, canonical.PRIMARY_MIN_POS, canonical.K
FAIL_POINTS = c.score(-canonical.FAIL, 0)

# key -> (what appears on the page, what it means)
TERMS: dict[str, tuple[str, str]] = {
    # ---------------------------------------------------------------- data and splits
    "focal": ("Focal", "A recording aimed at one bird (here from Xeno-Canto). The source domain. Shown in orange."),
    "soundscape": ("Soundscape", "A recording from a fixed field recorder (passive acoustic monitoring, “PAM”). The target domain. "
                   "Shown in blue."),
    "clip": ("Clip", "A 5-second piece of audio: the unit the model scores."),
    "site": ("Site, recordist", "Where a soundscape was recorded; who made a focal recording. Splits keep each whole."),
    "splits": ("Train / val / test", "Parts of the data. Who learns from and is scored on each is set out under "
               '<a href="formulation.html#data-use">problem formulation</a>.'),
    "fold": ("Fold, cross-validation (CV)", "The development sites are cut into 3 groups. Each fold scores one group; training uses "
             "the other two."),
    "devtest": ("Development / test sites", "Development: the 70 non-test sites, used to compare methods. Test: 30 sites kept for "
                "final numbers."),
    "heldout": ("Held-out sites", "Sites whose audio and labels a model never learned from; it is scored on them."),
    "roles": ("P / S / –", "In the split charts: primary species, secondary species, not scored on test."),
    "pos": ("Positive / negative clip", "A clip that does / does not contain the species in question."),
    "empty": ("Empty clip", "A soundscape clip labeled with none of the 21 species."),
    "khz": ("kHz, s, mel", "Kilohertz (pitch), seconds, and the mel frequency scale of the spectrogram, which gives low pitches more "
            "room."),
    "codes": ("Species codes", "Six-letter eBird codes such as <code>amepip</code> (American Pipit); the full list is below."),
    # ---------------------------------------------------------------- metrics
    "scale": ("Scores and “points”", "AP, AUC, cmAP and activations are shown out of 100. A change in points is the difference of two "
              "such numbers: 71.0 to 77.8 is +6.7 points (not 6.7% more correct clips)."),
    "ap": ("AP", "Average precision: how near the top of the model's ranking the clips with a species are; 100 = all first "
           '(<a href="evaluation.html#metrics">worked example</a>).'),
    "headline": ("Soundscape AP (primary species)", "The main metric: AP on held-out soundscape clips, averaged over primary species."),
    "auc": ("AUC, ROC-AUC", "How often a clip with the species scores above a clip without it. 100 = always, 50 = chance."),
    "cmap": ("cmAP", "Class-mean average precision: AP averaged over species; here over the secondary species."),
    "map": ("Sample-wise mAP", "Average precision computed per clip over species, then averaged over clips. Not the metric of this "
            "study; quoted from other authors."),
    "primary": ("Primary species", f"Secondary species with at least {PRIMARY} positive clips in the soundscape sites being scored."),
    "secondary": ("Secondary species", f"Species with at least {MIN_POS} positive clips in both the focal and soundscape sets being "
                  "scored."),
    "dap": ("ΔAP, “change”", "A method's AP minus the pretrained (source) model's on the same clips. Positive is better."),
    "dauc": ("Δ, ΔAUC (prototypes)", "A prototype's AUC on soundscape clips minus its AUC on focal clips. Negative means it detects "
             "its bird worse in the field. Not a comparison between methods."),
    "fail": ("Failing prototype", f"A prototype whose ΔAUC is below −{FAIL_POINTS} points."),
    "matched": ("Matched negatives", "Only soundscape clips containing another target bird count as negatives, as for focal clips. "
                "“sc” stands for soundscape and “sc matched” for this version."),
    "gap": ("Gap", "AP of the model trained on all labeled training-site soundscapes minus AP of the pretrained model."),
    "gamma": ("γ, share of the gap closed", "A method's gain divided by the gap. 0% = no better than the pretrained model, 100% = "
              "equal to the all-labels model, below 0% = worse, above 100% = better than it."),
    "gamma_ratio": ("How γ is averaged here", "Mean gain over the folds divided by the mean gap over the folds. The label curve page "
                    "computes γ in each fold and then averages, so it can differ slightly (59% there, 60% here, for 250 labels)."),
    "gamma_folds": ("How γ is averaged here", "γ is computed in each fold and then averaged (“Mean γ”). The results pages divide "
                    "mean gain by mean gap instead, so the two differ slightly (59% here and 60% there for 250 labels). "
                    "“γ, same 5” and “γ, same 7” use mean gain over mean gap on a fixed species set."),
    "ci": ("95% interval, [a, b]", "The range the number covers when recording sites (focal: recordists) are resampled 1,000 times. "
           "A change whose range includes zero cannot be told apart from no change; this is not proof of no effect."),
    "reliable": ("Reliably better / worse", "The 95% interval of the change lies entirely above / below zero."),
    "updown": ("↑ / ↓ (folds)", "Number of folds on which the method is reliably better / worse."),
    "lower": ("↓ in a column title", "In the secondary-metrics table, ↓ after a column name means lower is better."),
    "pm": ("± value", "Standard deviation over the 3 folds: how much they differ. Not a confidence interval."),
    "bold": ("<b>Bold</b>, <u>underline</u>", "Best and second-best printed value among the methods compared. Says nothing about "
             "reliability."),
    "star": ("*", "The test interval of the change excludes zero."),
    "criteria": ("Criteria met: gap · AUC · protos · focal", "On how many of 3 folds each success criterion is met: gap closed, "
                 "AUC not below, failing prototypes halved, focal retained."),
    "dash": ("– or —", "Not run, not scored or not applicable."),
    "check": ("✓ and —", "In the method table: the method uses this data / does not."),
    "nclips": ("N", "A number of labeled soundscape clips."),
    "all_labels": ("“All” / all labeled clips", "The pretrained model fine-tuned on <b>every labeled soundscape clip of the training "
                   "sites</b>, never on held-out or test sites, with no focal training data. It defines γ = 100%."),
    "zero_labels": ("0 labeled clips", "The pretrained model, unchanged."),
    "folds_n": ("2/3, “2 folds”", "On how many of the 3 folds something holds (or a species was scored)."),
    # ---------------------------------------------------------------- model and problem
    "k": ("K, Pk", f"Each species has K = {K} prototypes (learned sound patterns); P1 to P{K} name them, e.g. “whcspa P3”."),
    "x_s": ("x, s", "A clip's spectrogram; a species."),
    "z": ("z<sub>t,f</sub>(x)", "The backbone's description of the patch of x at time t and frequency f."),
    "p": ("p<sub>sk</sub>", "Prototype k of species s."),
    "a": ("a<sub>sk</sub>(x), activation", "How strongly prototype k matches its best location in clip x: from 0 (no match) to 1 "
          "(shown out of 100 in charts)."),
    "ops": ("cos, max<sub>t,f</sub>, ReLU, σ", "Cosine similarity; the largest value over all locations; set negative values to 0; the "
            "logistic function, which turns a number into a score between 0 and 1."),
    "yhat": ("ŷ<sub>s</sub>(x)", "The model's score for species s on clip x."),
    "w": ("w<sub>sk</sub>, b<sub>s</sub>", "The non-negative weight of prototype k in species s's score; a per-species offset."),
    "h": ("h<sub>0</sub>, h, h<sub>★</sub>", "The pretrained model; an adapted model; the reference model trained on all labeled "
          "training-site soundscapes."),
    "adapt": ("Adapt( )", "The method: the procedure that turns h<sub>0</sub> and the allowed data into h. Square brackets mark "
              "optional input."),
    "sets": ("T, H, T ∩ H = ∅", "Training sites and held-out sites; no site is in both."),
    "data_sym": ("D<sub>F</sub>, U<sub>T</sub>, Y<sub>T</sub>", "Labeled focal training clips; unlabeled audio of the training sites; "
                 "the labels of that audio."),
    "mh": ("M<sub>H</sub>(h), AP<sub>s</sub>(h; H)", "Model h's headline score on the sites H; its AP for species s there."),
    "ph": ("P<sub>H</sub>, |P<sub>H</sub>|, Σ", "The primary species on H; how many there are; a sum over them."),
    "gh": ("γ<sub>H</sub>(h)", "γ measured on the sites H."),
    # ---------------------------------------------------------------- explorer and findings
    "share": ("Share of class evidence", "The part of a species' score (before the offset) that comes from one prototype, averaged over "
              "the clips with the species."),
    "weight": ("Weight", "The prototype's last-layer weight w<sub>sk</sub>; larger dots mean larger weights."),
    "silent": ("Silent", "The prototype's activation is exactly zero on a clip."),
    "peak": ("Peak frequency", "The frequency at which an active prototype matches, as a median over clips with the species."),
    "base": ("Base rate, “base”", "How often the same thing happens among all soundscape clips without the species: the comparison "
             "for a prototype's false matches."),
    "falsematch": ("False match", "One of a prototype's strongest clips that is not labeled with its species."),
    "quant": ("Median, q10–q90, q25–q75", "The middle value; the ranges holding the central 80% and 50% of values."),
    "corr": ("r = …", "Correlation between two sets of numbers, from −1 to 1."),
    "n_eq": ("n", "A count: the number of clips or items behind a figure."),
    "match": ("Match", "The one spot on the spectrogram (time and pitch) where a prototype responds most strongly in a clip."),
    "contested": ("Contested", "At a match, a prototype of another of the 21 species responds at least as strongly."),
    "margin": ("Margin, rival", "The rival is the strongest other-species prototype at the match; the margin is its activation minus "
               "the prototype's own."),
    "matchkinds": ("False / correct / unclear match", "Species absent from the clip / spot on that species' call / species in the "
                   "clip but spot elsewhere."),
    "strat_auc": ("AUC of a signal (here)", "How often the signal ranks a false match as more suspect than a correct one, comparing "
                  "matches of the same prototype with similar own scores. Out of 100; 50 = coin flip."),
    "rr": ("Reliability r", "Correlation, from −1 to 1, between a prototype's activation and its species being present. Not shown out "
           "of 100."),
    "order": ("Order of the 5", "Rank correlation (−1 to 1) with the true order of a species' prototypes; 1 = same order."),
    "dagger": ("†", "Too few false or correct matches to count in the rule."),
    "crules": ("C0 to C4", "The five preset rules of the competition test."),
    "snr": ("SNR, dB", "Signal-to-noise ratio in decibels: how loud the bird is relative to the added background."),
}

_METRICS = ("scale", "ap", "headline", "auc", "primary", "gap", "ci", "reliable")
PAGES: dict[str, tuple[str, ...]] = {
    "index.html": ("focal", "soundscape", "clip", "devtest", "fold", *_METRICS, "secondary", "dauc", "fail", "matched", "k", "a",
                   "share", "silent", "peak", "falsematch", "base", "pos", "quant", "corr", "n_eq", "khz", "codes"),
    "data.html": ("focal", "soundscape", "clip", "site", "splits", "empty", "primary", "quant", "khz"),
    "splits.html": ("focal", "soundscape", "clip", "site", "splits", "fold", "devtest", "heldout", "primary", "secondary", "roles",
                    "pos", "empty"),
    "evaluation.html": ("focal", "soundscape", "clip", "devtest", "fold", "pos", *_METRICS, "cmap", "secondary", "dap", "dauc",
                        "fail", "matched", "k", "gamma", "gamma_ratio", "updown", "pm", "bold", "star", "n_eq"),
    "formulation.html": ("focal", "soundscape", "clip", "site", "splits", "fold", "devtest", "heldout", "check", "nclips",
                         "all_labels", "scale", "ap", "headline", "auc", "primary", "gap", "gamma", "gamma_ratio", "dauc", "x_s",
                         "k", "z", "p", "a", "ops", "yhat", "w", "h", "adapt", "sets", "data_sym", "mh", "ph", "gh"),
    "label-curve.html": ("focal", "soundscape", "clip", "fold", "heldout", "nclips", "zero_labels", "all_labels", *_METRICS,
                         "secondary", "gamma", "gamma_folds", "folds_n", "fail", "dauc", "empty", "pos", "dash"),
    "results.html": ("focal", "soundscape", "clip", "devtest", "fold", "heldout", "all_labels", "nclips", *_METRICS, "secondary",
                     "cmap", "map", "dap", "dauc", "fail", "matched", "k", "gamma", "gamma_ratio", "updown", "lower", "pm", "bold",
                     "star", "folds_n", "criteria", "dash", "pos", "snr"),
    "trust.html": ("soundscape", "clip", "site", "devtest", "fold", "pos", "k", "a", "scale", "ap", "auc", "ci", "match", "contested",
                   "margin", "matchkinds", "strat_auc", "crules", "dagger", "rr", "order", "weight"),
    "explorer.html": ("focal", "soundscape", "clip", "scale", "auc", "dauc", "k", "a", "weight", "share", "silent", "peak",
                      "falsematch", "base", "pos", "quant", "n_eq", "khz", "codes"),
}


def section(file: str) -> str:
    """The notation table that closes a page."""
    keys = PAGES[file]
    assert len(set(keys)) == len(keys), f"{file}: a notation entry is listed twice"
    rows = "".join(f'<tr data-term="{k}"><td>{TERMS[k][0]}</td><td>{TERMS[k][1]}</td></tr>' for k in keys)
    codes = ""
    if "codes" in keys:
        codes = ('<div class="code-grid">' + "".join(f"<span><code>{k}</code> {c.esc(v)}</span>" for k, v in c.SPECIES_NAMES.items())
                 + "</div>")
    return ('<section id="notation" class="notation" aria-labelledby="notation-title">'
            '<h2 id="notation-title">Notation used on this page</h2><div class="table-wrap"><table class="notation-table"><tbody>'
            f"{rows}</tbody></table></div>{codes}</section>")
