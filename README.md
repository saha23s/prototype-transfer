# prototype-transfer

**Goal:** close the gap between how well a bird-sound classifier works on clean *focal* recordings (Xeno-Canto) and how well it
works on field *soundscapes* (passive acoustic monitoring), using prototype-based models.

The model is [AudioProtoPNet-5-BirdSet-XCL](https://huggingface.co/DBD-research-group/AudioProtoPNet-5-BirdSet-XCL) on the
21-species High Sierra Nevada (HSN) subset of [BirdSet](https://huggingface.co/datasets/DBD-research-group/BirdSet).

## The gap (what we measured)

On the held-out test sites, for the pretrained model on the 7 primary species:

| | focal | soundscape |
|---|---|---|
| macro AP | 0.947 | **0.644** |
| macro AUC | 0.988 | 0.941 |

Ranking still works on soundscapes (AUC drops little); the scores are what collapse. Prototype activations shrink to about 30% of their
focal level, and 26 of the 60 scored prototypes (12 species x 5) lose more than 0.1 AUC. What we found about why (diagnostics in
`scripts/analysis/`, results in `outputs/results/hsn/summaries/diagnostics_stage2*.json`): it is not input normalisation; generic noise
does not reproduce the shift; about half of the prototype-level drop comes from how focal clips are cropped (the loudest 5 s);
false matches are mostly other birds. These diagnostics were run on the test split and should be repeated on validation / CV before
publication.

## How results are judged (the protocol)

One fixed protocol, v1: see [docs/PROTOCOL.md](docs/PROTOCOL.md). In short: headline metric is soundscape macro AP on the 7 primary species,
uncertainty is a cluster bootstrap over sites and recordists, development uses the validation split or 3-fold site-grouped
cross-validation and never the test sites, and the splits themselves are pinned by checksum so everyone scores the same clips.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and ~15 GB of free disk. Run on a compute node (a GPU is used if available).

```bash
scripts/setup_storage.sh /path/to/big/disk/prototype-transfer   # once: data/models/venv on a big disk, then `uv sync --locked`
source .venv/bin/activate
scripts/run_pipeline.sh --smoke     # 16 clips per domain, checks the plumbing; writes only under outputs/results/smoke/
scripts/run_pipeline.sh             # download -> manifests -> canonical splits -> pretrained responses -> canonical evaluation
```

## Use it

```bash
# 1. canonical splits (the pipeline does this too); must print "splits are canonical"
python scripts/data/make_splits.py && python scripts/data/check_splits.py

# 2. train one model on one domain, starting from the pretrained model
python scripts/train/train_protopnet.py --train-domain focal --name my-run --rung R0 --anchored
#    on SLURM, train + extract + score in one job (add --array=0-2 for 3-fold cross-validation):
sbatch scripts/slurm/run_recipe.sbatch my-run R0 --train-domain focal --anchored

# 3. score it with the canonical evaluation (CPU); use --split val while developing, test only for final numbers
python scripts/analysis/extract_prototype_activations.py my-run \
    --focal-manifest data/HSN/splits/focal_test.parquet --soundscape-manifest data/HSN/splits/soundscape_test.parquet
python scripts/eval/canonical_eval.py --model pretrained-xcl --model my-run --baseline pretrained-xcl

# 4. look at results
python scripts/report/results.py list
python scripts/report/results.py show my-run
```

Every number is saved in the results store and found with `scripts/report/results.py`; see [docs/RESULTS.md](docs/RESULTS.md).

### Training options

`scripts/train/train_protopnet.py` trains in four phases (warm, joint, push, last layer) and has these options, all off by default so
the earlier focal-only and soundscape-only runs reproduce. `--rung` is a free label recorded with the run (e.g. to group runs of one
recipe in `results.py list --rung`).

| Option | Effect |
|---|---|
| `--anchored` | skip the joint phase: backbone stays frozen |
| `--bg-kind noise` / `real --bg-manifest M` | mix generic noise / real soundscape background into focal clips (SNR U[-5, 20] dB, probability 0.5) |
| `--overlap-prob P` | overlay a second focal species (labels = union) |
| `--pseudo-manifest M --pseudo-weight W` | soft pseudo-label loss on unlabeled soundscape-train clips (teacher = pretrained model) |
| `--train-labels N` | use only N labeled clips at constant optimiser steps (label budget; needs `--train-domain soundscape --select last`) |
| `--select best` / `last` | keep the epoch with the best in-domain val cmAP / the last epoch of each phase (no validation labels) |
| `--cv-fold F` | cross-validation fold F: train without its sites; it is scored with `canonical_eval.py --split cv --fold F` |

Background and pseudo-label manifests are built without any annotation by `scripts/method/make_background_manifest.py` from the pretrained
responses of `scripts/analysis/diagnose_shift.py --task extract`. Cross-validation folds are combined with
`scripts/eval/cv_aggregate.py NAME` (mean ± sd over folds next to the pretrained baseline on the same sites). Exploratory paired
intervals between two cross-validated runs: `scripts/eval/paired_contrasts.py` (CPU). The next planned reruns (more seeds, a
frozen-backbone label curve) are listed by `scripts/slurm/submit_replications.sh`, which only prints unless given `--submit`.

### Published adaptation baselines

`scripts/method/adapt_sfda.py` runs five source-free domain adaptation methods (`tent`, `pseudo_label`, `dropout_student`,
`notela`, `shot`) on unlabeled soundscape audio of the training sites; the model is then frozen and scored on held-out sites
(inductive, as protocol v1 requires). The multi-label math is a port of the audio code of Boudiaf et al. (ICML 2023,
[sfda-codebase-snapshot](https://github.com/google-research/perch/tree/sfda-codebase-snapshot/chirp/projects/sfda), commit 7412704),
with their audio settings, untuned. When ported, its labels, neighbour graphs, losses, loss gradients and optimiser steps matched the
authors' JAX functions to floating-point precision on random inputs; `tests/test_sfda.py` covers the math and the loop without JAX.
Because this model is a ConvNeXt, the LayerNorm scale and shift are trained (there is no BatchNorm) and the student noise is
stochastic depth 0.2 (details in `src/prototype_transfer/training/adapt.py`). Test-time methods that need the evaluated audio (e.g.
LAME) are not run, and T3A, DUST and NRC are not implemented.

```bash
sbatch scripts/slurm/smoke_sfda.sbatch                          # GPU check of all five (a few batches; peak memory in the log)
TRAIN_SCRIPT=scripts/method/adapt_sfda.py sbatch --array=0-2 --constraint="48gb|80gb" --time=03:00:00 \
    scripts/slurm/run_recipe.sbatch sfda-notela sfda-notela --method notela   # adapt + extract + score, 3 folds
python scripts/eval/cv_aggregate.py sfda-notela
```

## Layout

```
src/prototype_transfer/   the package (editable install): data, models, method, training, evaluation, analysis, reporting
scripts/                  thin command-line entry points: data/ train/ eval/ analysis/ method/ report/ slurm/
configs/splits/hsn.json   checksums of the canonical splits
docs/                     PROTOCOL.md (how models are scored), RESULTS.md (where numbers live)
tests/                    plain-assert tests, no pytest needed: for t in tests/test_*.py; do python $t; done
dashboard/                static multi-page report (see below)
outputs/results/<dataset>/ committed results; outputs/features and outputs/checkpoints are not committed
```

| Package | What |
|---|---|
| `data/` | loading, preprocessing (32 kHz, 5 s, 256-mel log-spectrogram, model's mean/std), splits and their checksums |
| `models/` | the 21-class ProtoPNet and its checkpointing / prototype push |
| `training/` | the four-phase trainer and losses; the source-free adaptation loop |
| `method/` | background mixing, species overlap, pseudo-label dataset, label-free background / pseudo-label manifests (the recipe components); training-free and source-free adaptation baselines |
| `evaluation/` | the canonical evaluation, CV aggregation, intervals between runs, the results store |
| `analysis/` | per-prototype responses and the domain-shift diagnostics |
| `reporting/` | the dashboard: page builders (`pages/`), SVG charts, and the interactive data bundle |

### Dashboard

`dashboard/index.html` is a static, offline site (open it in a browser). Pages: motivation, data exploration, splits, evaluation,
problem formulation, label curve, initial results, and an interactive species explorer. Rebuild the pages after results or splits
change with `python scripts/report/build_dashboard.py` (CPU; reads the results store, `data/HSN/splits` and the cached responses in
`outputs/features/`; `--gallery` also redraws the example spectrograms from the audio). A new cross-validated method appears once it
is added to the registry in `src/prototype_transfer/reporting/runs.py`; the build stops if new results contradict a stated
conclusion. The findings and the explorer read `dashboard/data.js`, rebuilt with `python scripts/report/build_dashboard_data.py`.
By default it leaves out the test split and works from cached responses (the pretrained model's `responses_cv{0,1,2}.pt` and
`diag_responses.pt`) plus the audio; `--scope all` includes every clip and needs `extract_prototype_activations.py pretrained-xcl
--tag full`. The two findings from visual inspection describe the non-test crops: look again if the crops change. Checks: `python tests/test_dashboard_pages.py`, and for the interactive parts
`npm i jsdom@24 && node dashboard/tests/check.mjs dashboard`.

## Environment (uv)

Dependencies are managed with uv and locked in `uv.lock`; always install with `uv sync --locked` (done by `setup_storage.sh`).
Python is pinned by `.python-version`.

- **CUDA build.** On Linux, `torch`/`torchaudio` come from the PyTorch **cu121** index (`[tool.uv.sources]` in `pyproject.toml`), because
  the GPU machines run NVIDIA driver 535 (CUDA 12.2), which cannot run the default PyPI wheels (CUDA 13). macOS uses the PyPI wheels.
  `setup_storage.sh` fails loudly if a GPU is present but torch cannot use it.
- **Changing dependencies.** `uv add <pkg>` (or edit `pyproject.toml` then `uv lock`), commit `pyproject.toml` + `uv.lock` together. To bump
  torch, change the pin *and* the index URL together, after checking `nvidia-smi` on the target machine.

## Notes

- `transformers` must stay `<5.0`: the model's remote code is incompatible with 5.x (pinned in `pyproject.toml`).
- The model is loaded with `trust_remote_code=True` from the Hugging Face Hub at a pinned commit (`MODEL_REVISION` in
  `src/prototype_transfer/models/classifier.py`), so weights and modeling code can't change under you; the HF cache is kept in `models/hf`.
- Focal recordings are Xeno-Canto, which is part of the model's XCL training data, so focal numbers are an in-domain reference, not a held-out score.
- `data/`, `models/` and `.venv` are symlinks created by `setup_storage.sh` and are git-ignored.
