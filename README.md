# prototype-transfer

Runs [AudioProtoPNet-5-BirdSet-XCL](https://huggingface.co/DBD-research-group/AudioProtoPNet-5-BirdSet-XCL)
on the 21-species High Sierra Nevada (HSN) subset of
[BirdSet](https://huggingface.co/datasets/DBD-research-group/BirdSet), and compares behaviour on
**focal** (Xeno-Canto) vs **soundscape** (passive acoustic monitoring) recordings
(cmAP, ROC-AUC, top-1, prototype activation maps).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and ~15 GB of free disk (full run).

```bash
# 1. One-time: put data/models/venv on a big disk, symlink into the repo, install deps
scripts/setup_storage.sh /path/to/big/disk/prototype-transfer

# 2. Quick check (16 clips per type; writes to outputs/smoke/)
scripts/run_pipeline.sh --smoke

# 3. Full run (all focal shards, up to 1050 focal + 1000 soundscape clips; writes to outputs/)
scripts/run_pipeline.sh
# or on SLURM:
sbatch scripts/slurm/pipeline.sbatch
```

Run on a compute node (a GPU is used automatically if available; CPU works but is slow:
~40 s per batch of 16 clips on 4 CPUs).

## Pipeline

| Step | Script | Output |
|---|---|---|
| Download HSN (metadata, audio) | `scripts/download_hsn.py` | `data/HSN/raw/` |
| Build manifests | `scripts/preprocess_hsn.py` | `data/HSN/preprocessed/*.parquet` |
| Inference + metrics | `scripts/run_inference.py` | `outputs/results/*.json`, `eval_cache.pt` |
| Figures | `scripts/visualize_analysis.py` | `outputs/figures/fig*.png` |
| Full manifests (no caps) | `scripts/preprocess_hsn.py --full` | `data/HSN/preprocessed/*_full.parquet` |
| Per-prototype responses | `scripts/extract_prototype_activations.py` | `outputs/features/prototype_responses.pt`, `outputs/results/prototype_transfer.json` |
| Prototype-transfer figures | `scripts/visualize_prototype_transfer.py` | `outputs/figures/pt*.png` |
| Dashboard data | `scripts/build_dashboard_data.py` | `dashboard/data.js`, `dashboard/patches/` |
| Canonical evaluation (protocol v1, CPU) | `scripts/canonical_eval.py` | `outputs/results/canonical_eval.json` |
| Domain-shift diagnostics (GPU, pretrained model) | `scripts/slurm/diagnose_shift.sbatch`, `diagnose_shift_v2.sbatch` (run `scripts/diagnose_shift.py`, `scripts/analyze_diagnostics.py`) | `outputs/results/diagnostics_stage2*.json` |

### Prototype transfer (focal -> soundscape)

`extract_prototype_activations.py` records, for every clip, the max activation and peak
(frequency row, time column) of all 105 HSN prototypes (21 species x 5), plus full heatmaps
and the latent patch at each peak for clips containing the species. `prototype_transfer.json`
reports, per prototype and domain: median activation on positives/negatives, detection
ROC-AUC, share of the class evidence (`w_k * a_k / sum_j w_j * a_j`), and peak frequency.
Only species with >= 10 soundscape positives are plotted. Note that the focal recordings are
Xeno-Canto, i.e. part of the model's XCL training data, so focal numbers are an in-domain
reference rather than a held-out score.

Spectrograms are computed on the fly from the audio (`src/preprocessing.py`: 32 kHz, 5 s, 256-mel
log-spectrogram, normalised with the model's training mean/std). Pass `--cache-tensors` to
`preprocess_hsn.py` to pre-compute them to disk.

## Splits and training recipe

Splits (`scripts/make_splits.py`, seed 0): focal grouped by recordist, soundscape grouped by site (60 / 10 / 30 sites for
train / val / test). Soundscape val is monitoring-only during training and the split to use whenever soundscape labels are
needed during development: `scripts/canonical_eval.py --split val` and `scripts/diagnose_shift.py` /
`analyze_diagnostics.py --eval-split val` score val (with `focal_val`) instead of test. The split is recorded in the output JSON.

**Cross-validation (part of the protocol).** Recipe choices (which rung, epoch rule, hyperparameters) are made by 3-fold
cross-validation over the 70 non-test soundscape sites (train + val), never on test. `make_splits.py --cv-folds 3` (default)
divides those sites into 3 site-grouped folds balanced in species coverage (`outputs/results/splits_cv.json`) and writes, for
fold F, `soundscape_cv{F}_train.parquet` (the other folds: the only soundscape audio / labels the run may use) and
`soundscape_cv{F}_heldout.parquet`. Each fold is a separate training run that holds fold F out
(`train_protopnet.py --cv-fold F`, which refuses background / pseudo-label manifests containing held-out clips, and for a
soundscape-trained run requires `--select last` so the held-out fold never picks a checkpoint), scored with
`canonical_eval.py --split cv --fold F`; `scripts/cv_aggregate.py NAME` reports mean ± sd over the folds next to the pretrained
baseline on the same held-out sites. Label-free rungs need per-fold manifests (`make_background_manifest.py --fold F`, which also
needs the val responses from `diagnose_shift.py --task extract --eval-split val`). The chosen configuration is then retrained on
all non-test data and scored once on test. The whole ladder is 3 folds x 7 rungs of about 15-25 GPU-minutes each.
`sbatch --array=0-2 scripts/slurm/run_recipe.sbatch NAME RUNG [train args]` runs the three folds.

`scripts/train_protopnet.py` takes the recipe options below. Everything defaults to off, so the earlier focal-only and
soundscape-only runs reproduce. `src/mixing.py` holds the (unit-tested) waveform mixing.

| Rung | Flags (with `--train-domain focal`) |
|---|---|
| R0 anchored | `--anchored` (joint phase skipped, backbone frozen) |
| R1a generic noise | `--bg-kind noise` |
| R1b target background | `--bg-kind real --bg-manifest data/HSN/splits/soundscape_background.parquet` |
| R2 overlap | `--overlap-prob 0.5` |
| R3 both | R1 flags + `--overlap-prob 0.5` |
| R4 pseudo-labels | R3 flags + `--pseudo-manifest data/HSN/splits/soundscape_pseudo.parquet --pseudo-weight 0.2` |
| label budget (oracle, uses labels) | `--train-domain soundscape --train-labels N --select last` |

Defaults: background probability 0.5 at SNR U[-5, 20] dB, overlap probability from `--overlap-prob` at U[-10, 0] dB relative level.
`--select last` keeps the last epoch of each phase instead of the best by in-domain val cmAP; both are reported in
`outputs/results/train_<name>.json`. The background and pseudo-label manifests are built without any annotation by
`scripts/make_background_manifest.py` from the pretrained responses of `scripts/diagnose_shift.py --task extract`.
One run end to end (train, extract responses, canonical eval): `sbatch scripts/slurm/run_recipe.sbatch NAME RUNG [train args]` (add `--array=0-2` for cross-validation).
Runs are recorded in `outputs/runs/registry.json` (name, rung, seed, commit, settings).

## Layout

```
configs/   dataset_hsn.yaml (audio/STFT/species), model_protonet.yaml (device, batch size, output dirs)
src/       dataset, preprocessing, model wrapper, metrics, splits, mixing, plotting
scripts/   entry points + setup_storage.sh, run_pipeline.sh, slurm/*.sbatch
tests/     plain-assert tests (no pytest needed): python tests/test_<name>.py (canonical_eval, splits, mixing, cv_aggregate)
```

`run_inference.py` reads its defaults (device, batch size, output dir) from
`configs/model_protonet.yaml`; any of them can be overridden on the command line.

### Dashboard

`dashboard/index.html` is a static, offline dashboard of the main findings: summary stats, findings,
class- and prototype-level AUC charts, per-species prototype cards with top-activating patches,
and a data table. Open it directly in a browser (no server needed). Rebuild its data after
re-running the analysis (needs the audio for the patch crops):

```bash
python scripts/build_dashboard_data.py   # -> dashboard/data.js, data.json, patches/
```

Every finding is generated together with its evidence (chart data, key numbers, method note), so the
text and the numbers come from the same computation; open `dashboard/index.html#finding-N` to link
to a finding's evidence. Acceptance test (renders the page in jsdom and checks structure, evidence
for every finding, interactions and patch zoom): `npm i jsdom@24 && node dashboard/tests/check.mjs dashboard`.

## Environment (uv)

Dependencies are managed with uv and locked in `uv.lock`; always install with
`uv sync --locked` (done by `setup_storage.sh`) so the environment matches the lock exactly.
Python is pinned by `.python-version`.

- **CUDA build.** On Linux, `torch`/`torchaudio` come from the PyTorch **cu121** index
  (`[tool.uv.sources]` in `pyproject.toml`), because the GPU machines run NVIDIA driver 535
  (CUDA 12.2), which cannot run the default PyPI wheels (CUDA 13). macOS uses the PyPI wheels.
  `setup_storage.sh` fails loudly if a GPU is present but torch cannot use it.
- **Changing dependencies.** `uv add <pkg>` (or edit `pyproject.toml` then `uv lock`), commit
  `pyproject.toml` + `uv.lock` together. To bump torch, change the pin *and* the index URL
  together, after checking `nvidia-smi` on the target machine.

## Notes

- `transformers` must stay `<5.0`: the model's remote code is incompatible with 5.x (pinned in `pyproject.toml`).
- The model is loaded with `trust_remote_code=True` from the Hugging Face Hub at a pinned commit (`MODEL_REVISION` in `src/model.py`), so weights and modeling code can't change under you; the HF cache is kept in `models/hf`.
- `data/`, `models/` and `.venv` are symlinks created by `setup_storage.sh` and are git-ignored.
