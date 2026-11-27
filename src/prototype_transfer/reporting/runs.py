"""What the dashboard pages read: the method registry, cross-validation summaries from the results store, and per-species
breakdowns of the canonical soundscape AP (recomputed from the cached responses and checked against the saved values)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from prototype_transfer.data.splits import site_of, split_manifest
from prototype_transfer.evaluation import canonical, results
from prototype_transfer.evaluation.metrics import ap_cols, auc_cols

FOLDS = (0, 1, 2)
BASELINE, ORACLE = "pretrained-xcl", "soundscape-only"
LABEL_BUDGETS = (100, 250, 500, 1000, 2000)
PUBLISHED = ("sfda-tent", "sfda-pseudo_label", "sfda-dropout_student", "sfda-notela", "sfda-shot")   # source-free adaptation
TEST_RUNS = (*PUBLISHED, "focal-only", ORACLE)   # besides the baseline, the models scored on the test split
CHECK_TOL = 1e-6


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    uses: str          # the soundscape information the method sees
    how: str


GROUPS: tuple[tuple[str, str, tuple[Method, ...]], ...] = (
    ("free", "No training: adapt the pretrained model with unlabeled soundscape audio", (
        Method("tf-quantile", "Quantile mapping", "unlabeled audio",
               "Each prototype's activation distribution on unlabeled soundscapes is mapped onto its focal one (zeros stay zero); the mapping "
               "is then applied to every evaluated clip."),
        Method("tf-reweight", "Prototype reweighting", "unlabeled audio",
               "Each prototype's class weight is scaled by how much of its contrast (top 5% mean − median) survives on soundscapes; "
               "each species keeps its total weight. A heuristic introduced here."),
    )),
    ("sfda", "Published source-free adaptation: update normalisation weights on unlabeled soundscape audio", (
        Method("sfda-tent", "Tent", "unlabeled audio",
               "Minimises the entropy of the predictions (Wang et al., 2021)."),
        Method("sfda-pseudo_label", "Pseudo-labelling", "unlabeled audio",
               "Pushes up the species the model currently scores above 0.5 on a clip (after Lee, 2013); no pseudo-negatives."),
        Method("sfda-dropout_student", "Dropout Student", "unlabeled audio",
               "A noisy copy of the model matches the soft predictions of the noise-free model, renewed every epoch."),
        Method("sfda-notela", "NOTELA", "unlabeled audio",
               "Dropout Student whose targets are also pulled towards those of similar clips (mutual nearest neighbours; "
               "Boudiaf et al., 2023)."),
        Method("sfda-shot", "SHOT", "unlabeled audio",
               "Makes predictions confident per clip while keeping them varied across a batch (Liang et al., 2020). With the "
               "setting used here, its cluster pseudo-labels have zero weight."),
    )),
    ("focal", "Fine-tuning on focal labels only", (
        Method("focal-only", "Full fine-tuning", "none", "Backbone, prototypes and last layer are trained."),
        Method("r0", "Frozen backbone", "none", "Only prototypes and last layer are trained."),
        Method("r1a", "Full + synthetic noise", "none", "Full fine-tuning; coloured noise is mixed into half of the focal clips."),
        Method("r2", "Full + species overlap", "none", "Full fine-tuning; a second focal species is overlaid on half of the clips."),
    )),
    ("mix", "Fine-tuning on focal labels plus unlabeled soundscape audio", (
        Method("r1b", "Full + real background", "unlabeled audio",
               "Quiet soundscape clips (lowest 30% of the pretrained model's top score) are mixed into focal clips."),
        Method("r3", "Full + background + overlap", "unlabeled audio", "Real background mixing and species overlap together."),
        Method("r4", "Full + background + overlap + pseudo-labels", "unlabeled audio",
               "As above, plus a loss toward the pretrained model's soft predictions on soundscape clips (weight 0.2)."),
    )),
    ("ablation", "Exploratory: frozen-backbone ablations (added after the runs above)", (
        Method("r0x-nopush", "Frozen, no prototype projection", "none",
               "Prototypes are not projected onto training patches after training."),
        Method("r0x-paperortho", "Frozen, alternative orthogonality loss", "none",
               "The orthogonality term of the later AudioProtoPNet paper version, with weight 1.0."),
        Method("r0x-nopush-paperortho", "Frozen, both changes", "none", "Both changes together."),
    )),
    ("labels", "With soundscape labels (full fine-tuning on soundscape clips)", tuple(
        Method(f"lb{k}", f"{k:,} labeled clips", f"{k:,} labels",
               "A random subset of training-site clips with labels; same number of optimiser steps as with all clips.")
        for k in LABEL_BUDGETS) + (
        Method(ORACLE, "All labeled clips (upper reference)", "all labels", "All 5,520–5,640 training-site clips with labels."),)),
)

# (variant, control, what differs): each pair isolates one ingredient, unlike a comparison with the pretrained model
CONTRASTS = (
    ("r1a", "focal-only", "Synthetic noise"),
    ("r2", "focal-only", "Species overlap"),
    ("r1b", "focal-only", "Real soundscape background"),
    ("r3", "focal-only", "Real background + overlap"),
    ("r1b", "r1a", "Real background instead of noise"),
    ("r4", "r3", "Pseudo-labels (added to background + overlap)"),
    ("r0", "focal-only", "Freezing the backbone"),
    ("r0x-nopush", "r0", "Dropping prototype projection (frozen)"),
    ("r0x-paperortho", "r0", "Other orthogonality loss (frozen)"),
)

METHODS = {m.key: m for _, _, ms in GROUPS for m in ms}
LABEL_FREE = tuple(m.key for g, _, ms in GROUPS if g != "labels" for m in ms)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


class Results:
    """Everything the pages read, loaded once. Needs the committed results store; per-species numbers also need the
    cached responses under outputs/features/, and the data pages need the split manifests under data/."""

    def __init__(self, store: results.Store, splits_dir: Path) -> None:
        self.store, self.splits_dir = store, splits_dir
        self.splits = _json(store.dataset_dir / "splits.json")
        self.splits_cv = _json(store.dataset_dir / "splits_cv.json")
        self.test = _json(store.summary("canonical_eval"))
        # one evaluation holding every model scored on test, on the same resamples
        self.final = _json(store.summary("canonical_eval_test_baselines"))
        assert self.final["split"] == "test" and set(self.final["models"]) == {BASELINE, *TEST_RUNS}, "unexpected final evaluation"
        self.cv = {k: _json(store.summary(f"cv_{k}")) for k in METHODS}
        self.folds = {(k, f): _json(store.summary(f"canonical_eval_{k}-cv{f}")) for k in METHODS for f in FOLDS}
        self.species = [self.folds[("r0", f)]["species"] for f in FOLDS]
        for (k, f), e in self.folds.items():   # every method of a fold is scored on the same species
            assert e["species"]["primary"] == self.species[f]["primary"], (k, f)
            assert e["species"]["secondary"] == self.species[f]["secondary"], (k, f)
        self._ap: dict[tuple[str, int], dict[str, float]] = {}
        self.contrasts = _json(store.summary("cv_contrasts"))   # scripts/eval/paired_contrasts.py

    # ------------------------------------------------------------------------------------------- cross-validation
    def model(self, key: str, fold: int) -> dict:
        """The canonical_eval entry of one method on one fold (key may be the baseline)."""
        if key == BASELINE:
            return self.folds[("r0", fold)]["models"][BASELINE]
        return self.folds[(key, fold)]["models"][f"{key}-cv{fold}"]

    def values(self, key: str, metric: str) -> list[float]:
        return [self.model(key, f)["point"][metric] for f in FOLDS]

    def deltas(self, key: str, metric: str = "sc_ap_primary") -> list[float]:
        return [a - b for a, b in zip(self.values(key, metric), self.values(BASELINE, metric))]

    def delta_ci(self, key: str, fold: int, metric: str = "sc_ap_primary") -> tuple[float, float]:
        lo, hi = self.model(key, fold)["vs_baseline"][metric]["ci95"]
        return lo, hi

    def folds_better_worse(self, key: str, metric: str = "sc_ap_primary") -> tuple[int, int]:
        """Folds whose paired 95% interval against the pretrained model lies entirely above / below zero."""
        cis = [self.delta_ci(key, f, metric) for f in FOLDS]
        return sum(lo > 0 for lo, _ in cis), sum(hi < 0 for _, hi in cis)

    def fail_share(self, key: str) -> list[float]:
        """Failing prototypes / scored prototypes per fold (the number of scored species differs between folds)."""
        return [self.model(key, f)["point"]["n_fail"] / (canonical.K * len(self.species[f]["secondary"])) for f in FOLDS]

    def contrast(self, key: str, control: str) -> list[float]:
        """Per-fold soundscape AP of one adapted model minus another's (no interval: the protocol pairs only with the baseline)."""
        return [a - b for a, b in zip(self.values(key, "sc_ap_primary"), self.values(control, "sc_ap_primary"))]

    def contrast_interval(self, key: str, control: str, species_set: str) -> dict:
        """{"folds": [{diff, ci95}], "mean": {diff, ci95}} of key - control for a species set (primary | shared | fixed);
        the point differences must agree with the ones computed here from the same runs."""
        entry = next(e for e in self.contrasts["pairs"] if (e["variant"], e["control"]) == (key, control))[species_set]
        codes = {"primary": None, "shared": self.shared_species(), "fixed": self.fixed_species()}[species_set]
        here = self.contrast(key, control) if codes is None else np.subtract(self.set_deltas(key, codes), self.set_deltas(control, codes))
        assert np.allclose([f["diff"] for f in entry["folds"]], here, rtol=0, atol=CHECK_TOL), \
            f"cv_contrasts is stale for {key} - {control}: run scripts/eval/paired_contrasts.py"
        return entry

    def gamma_upper(self, key: str, fold: int) -> float:
        """Upper end of the 95% interval of the share of the gap closed."""
        return self.model(key, fold)["gap_fraction"]["ci95"][1]

    def fail_share_matched(self, key: str) -> list[float]:
        """As fail_share, with soundscape negatives restricted to clips containing another target species (like focal negatives)."""
        return [self.model(key, f)["point"]["n_fail_birds"] / (canonical.K * len(self.species[f]["secondary"])) for f in FOLDS]

    def proto_auc(self, key: str, domain: str) -> list[float]:
        """Per fold, the mean detection AUC of the scored prototypes. domain: focal or soundscape (the two terms of delta), or
        matched (soundscape with negatives restricted to other-bird clips; saved as delta_birds = that AUC - focal AUC)."""
        value = ((lambda p: p["delta_birds"] + p["auc_focal"]) if domain == "matched" else (lambda p: p[f"auc_{domain}"]))
        return [float(np.mean([value(p) for p in self.model(key, f)["prototypes"]])) for f in FOLDS]

    def shared_species(self) -> list[str]:
        """Species that are primary on every fold."""
        return [s for s in self.primary_union() if all(s in sp["primary"] for sp in self.species)]

    def fixed_species(self) -> list[str]:
        """Species eligible (protocol "secondary") on every fold: the largest set that can be scored the same way on all folds."""
        return [s for s in self.species[0]["secondary"] if all(s in sp["secondary"] for sp in self.species)]

    def other_test_species(self) -> list[str]:
        """The test split's secondary species beyond the fixed set: too rare to be eligible on every cross-validation fold."""
        return [s for s in self.test["species"]["secondary"] if s not in self.fixed_species()]

    def set_ap(self, key: str, codes: list[str]) -> list[float]:
        """Per fold, soundscape AP averaged over the same species `codes`."""
        return [float(np.mean([self.species_ap(key, f)[s] for s in codes])) for f in FOLDS]

    def set_auc(self, key: str, codes: list[str]) -> list[float]:
        """Per fold, soundscape ROC-AUC of the class scores averaged over the same species `codes`."""
        out = []
        for f in FOLDS:
            m = canonical.Model(key, self.responses(key, f))
            cols = [m.species.index(s) for s in codes]
            out.append(float(np.mean(auc_cols(m.p_s[:, cols], m.y_s[:, cols]))))
        return out

    def heldout_positives(self) -> dict[str, list[int]]:
        """Species -> soundscape clips containing it on each fold's held-out sites."""
        labels = [pd.read_parquet(split_manifest(self.splits_dir, "soundscape", "val", f))["hsn_classes"] for f in FOLDS]
        return {s: [int(sum(s in list(x) for x in col)) for col in labels] for s in self.test["species"]["secondary"]}

    def set_deltas(self, key: str, codes: list[str]) -> list[float]:
        """Per-fold change in soundscape AP vs the pretrained model, averaging the same species `codes` on every fold."""
        return [float(np.mean([self.species_ap(key, f)[s] - self.species_ap(BASELINE, f)[s] for s in codes])) for f in FOLDS]

    def shared_deltas(self, key: str) -> list[float]:
        """set_deltas over the shared species."""
        return self.set_deltas(key, self.shared_species())

    def criteria(self, key: str) -> list[int] | None:
        c = self.cv[key].get("criteria_folds_passed")
        return None if c is None else [c[k] for k in sorted(c)]

    # ------------------------------------------------------------------------------------------------ per species
    def responses(self, key: str, fold: int) -> Path:
        if key == BASELINE:
            return self.store.features(BASELINE) / f"responses_cv{fold}.pt"
        return self.store.features(f"{key}-cv{fold}") / "responses.pt"

    def species_ap(self, key: str, fold: int) -> dict[str, float]:
        """Soundscape AP of each eligible (secondary) species of the fold, plus every species scored on the test split (they need
        only one held-out soundscape positive to have an AP). The means over the fold's primary and secondary species must equal
        the two saved canonical values."""
        if (key, fold) not in self._ap:
            m = canonical.Model(key, self.responses(key, fold))
            sec, prim = self.species[fold]["secondary"], self.species[fold]["primary"]
            codes_all = list(dict.fromkeys([*sec, *self.test["species"]["secondary"]]))
            cols = [m.species.index(s) for s in codes_all]
            ap = {s: float(v) for s, v in zip(codes_all, ap_cols(m.p_s[:, cols], m.y_s[:, cols]))}
            assert not any(np.isnan(v) for v in ap.values()), f"fold {fold}: a species has no held-out soundscape positive"
            for codes, metric in ((prim, "sc_ap_primary"), (sec, "sc_cmap_12")):
                want = self.model(key, fold)["point"][metric]
                assert abs(float(np.mean([ap[s] for s in codes])) - want) < CHECK_TOL, \
                    f"{key} fold {fold}: responses disagree with the saved {metric}"
            self._ap[(key, fold)] = ap
        return self._ap[(key, fold)]

    def primary_union(self) -> list[str]:
        """Species that are primary on any fold, by total held-out soundscape positives."""
        every = {s for sp in self.species for s in sp["primary"]}
        pos = {s: sum(sp["n_pos_soundscape"].get(s, 0) for sp in self.species) for s in every}
        return sorted(every, key=lambda s: -pos[s])

    def species_mean(self, key: str, code: str) -> tuple[float, int]:
        """Mean AP of one species over the folds on which it is primary, and the number of such folds."""
        vals = [self.species_ap(key, f)[code] for f in FOLDS if code in self.species[f]["primary"]]
        return float(np.mean(vals)), len(vals)

    def test_species_auc(self) -> dict[str, dict]:
        """Pretrained model on the test split: ROC-AUC per scored species in each domain."""
        # the saved test evaluation was computed from the diagnostics extraction when present (a later re-extraction goes to
        # responses.pt and need not match to the last digit); whichever file is read is checked against the saved values below
        path = self.store.features(BASELINE) / "diag_responses.pt"
        m = canonical.Model(BASELINE, path if path.exists() else path.with_name("responses.pt"))
        codes = self.test["species"]["secondary"]
        cols = [m.species.index(s) for s in codes]
        auc_f, auc_s = auc_cols(m.p_f[:, cols], m.y_f[:, cols]), auc_cols(m.p_s[:, cols], m.y_s[:, cols])
        point = self.test["models"][BASELINE]["point"]
        pri = [codes.index(s) for s in self.test["species"]["primary"]]
        for auc, key in ((auc_f, "f_auc_primary"), (auc_s, "sc_auc_primary")):
            assert abs(float(np.mean(auc[pri])) - point[key]) < CHECK_TOL, f"{key}: test responses disagree with the saved evaluation"
        return {s: {"focal": float(auc_f[i]), "soundscape": float(auc_s[i])} for i, s in enumerate(codes)}

    def test_scores(self, key: str) -> dict[str, dict]:
        """One of the models scored on the test split: soundscape AP and AUC of each of the 12 species. The mean AP over the
        primary and over all 12 species must equal the two saved values."""
        m = canonical.Model(key, self.store.features(key) / "responses.pt")
        codes = self.final["species"]["secondary"]
        cols = [m.species.index(s) for s in codes]
        ap, auc = ap_cols(m.p_s[:, cols], m.y_s[:, cols]), auc_cols(m.p_s[:, cols], m.y_s[:, cols])
        out = {s: {"ap": float(a), "auc": float(u)} for s, a, u in zip(codes, ap, auc)}
        point = self.final["models"][key]["point"]
        for names, metric in ((self.final["species"]["primary"], "sc_ap_primary"), (codes, "sc_cmap_12")):
            assert abs(float(np.mean([out[s]["ap"] for s in names])) - point[metric]) < CHECK_TOL, \
                f"{key}: test responses disagree with the saved {metric}"
        return out

    # --------------------------------------------------------------------------------------------------- manifests
    def manifest(self, domain: str) -> pd.DataFrame:
        """All clips of one domain, with the split each belongs to."""
        return pd.concat([pd.read_parquet(self.splits_dir / f"{domain}_{s}.parquet").assign(split=s)
                          for s in ("train", "val", "test")], ignore_index=True)

    def labeled_subsets(self) -> dict[int, list[dict]]:
        """What each label budget contained, per fold. Reproduces the trainer's draw (sample(n, random_state=seed 0))."""
        out: dict[int, list[dict]] = {k: [] for k in LABEL_BUDGETS}
        for f in FOLDS:
            df = pd.read_parquet(split_manifest(self.splits_dir, "soundscape", "train", f))
            for k in LABEL_BUDGETS:
                s = df.sample(n=k, random_state=0)
                sites = site_of(s["audio_path"].map(lambda q: Path(q).name))
                out[k].append({"background": float((s["num_hsn_classes"] == 0).mean()), "sites": int(sites.nunique()),
                               "positives": {c: int(sum(c in list(x) for x in s["hsn_classes"]))
                                             for c in self.species[f]["primary"]}})
        return out
