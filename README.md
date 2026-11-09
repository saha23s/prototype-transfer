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

## Notes

- `transformers` must stay `<5.0`: the model's remote code is incompatible with 5.x (pinned in `pyproject.toml`).
- The model is loaded with `trust_remote_code=True` from the Hugging Face Hub; the HF cache is kept in `models/hf`.
- `data/`, `models/` and `.venv` are symlinks created by `setup_storage.sh` and are git-ignored.
