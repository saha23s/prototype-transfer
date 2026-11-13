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

## Layout

```
configs/   dataset_hsn.yaml (audio/STFT/species), model_protonet.yaml (device, batch size, output dirs)
src/       dataset, preprocessing, model wrapper, metrics, plotting
scripts/   entry points + setup_storage.sh, run_pipeline.sh, slurm/pipeline.sbatch
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
