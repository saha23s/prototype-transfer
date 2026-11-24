# Evaluation protocol (v1)

This is the one way models are scored in this repository. The code is `src/prototype_transfer/evaluation/canonical.py`
(entry point `scripts/eval/canonical_eval.py`); this page explains the decisions. Changing anything below means a new protocol
version (`PROTOCOL` in `canonical.py`) and re-scoring every model.

## Data and splits

- **Focal** clips (Xeno-Canto) are split by *recordist*, stratified by species, because one person's recordings share equipment and
  place and would leak between splits.
- **Soundscape** clips (passive acoustic monitoring) are split by *recording site*: neighbouring 5 s clips of one recording are
  near-duplicates, and species are strongly clustered by site. 60 / 10 / 30 sites for train / val / test, chosen as the random site
  partition whose per-species positive fractions are closest to the targets.
- **Test** is touched only for final numbers. **Val** (10 sites, with `focal_val`) is the split for development
  (`canonical_eval.py --split val`). **Cross-validation** is 3 site-grouped folds over the 70 non-test sites
  (`--split cv --fold F`); recipe choices are made on it. A run that is cross-validated is trained without fold F's sites and scored on them.
- The splits are pinned: `configs/splits/hsn.json` holds a fingerprint of all 12 manifests, `scripts/data/check_splits.py` verifies them,
  and training and scoring refuse a different split. Scores on different splits are not comparable.

## What is scored

- **Species.** *Secondary* = species with at least 5 positives in both the focal and soundscape test sets (12). *Primary* = at least
  30 soundscape test positives (7: whcspa, amepip, rocwre, daejun, clanut, herthr, amerob).
- **Headline: soundscape macro AP over the primary species.** Also reported: soundscape macro AUC (primary and the 12), cmAP over the 12,
  and focal macro AUC / AP (retention only: focal data is part of the pretrained model's training set, so it is an in-domain reference).
- **Per-prototype transfer.** For prototype k of species s, `delta_k = AUC_soundscape - AUC_focal` of its maximum activation as a detector
  of s. A prototype *fails* if `delta_k < -0.10` and *improves* if `delta_k > 0.02`. `delta` uses every other soundscape clip as negatives;
  `delta_birds` uses only clips with another target species as negatives, which matches the focal negatives (other species' clips) and
  removes empty-background clips from the comparison.

## Uncertainty

Cluster bootstrap with B = 1000 resamples shared across models (so differences are paired): soundscape clips are resampled by site (30 clusters),
focal clips by recordist (110). Intervals are 95% percentile intervals. With few sites the intervals are wide; read them before reading
a difference.

## Comparison criteria

For a model against a baseline (the pretrained model) and an oracle (the supervised soundscape-only model), all four must hold for the model to count as
closing the gap:

1. **AP gap closed:** at least half of the gap between baseline and oracle in soundscape AP (primary), and the 95% interval of the improvement over the baseline excludes 0.
2. **AUC not below baseline** (soundscape AUC, primary).
3. **Failing prototypes at most halved** relative to the baseline.
4. **Focal retained:** focal AUC (primary) within 0.01 of the baseline.

## Rules that keep the numbers honest

- Soundscape labels are used only to evaluate, and for model selection only through the validation split or the CV folds.
  Methods that adapt to the target domain may use soundscape *audio* of the training sites without labels, never audio of held-out sites
  (`train_protopnet.py` refuses manifests that contain held-out clips).
- All models in one evaluation must have been run on identical clips (checked), and exactly on the clips of the requested split (checked).
- Diagnostics (`scripts/analysis/`) were run on the test split. Before publishing a claim derived from them, repeat them on val / CV.
