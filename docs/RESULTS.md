# Results store

Every number this repository produces is saved through one writer
(`prototype_transfer.evaluation.results.save_results`) into one layout, and indexed in one file.
If you are looking for a number, start with the CLI below; you should not need to grep.

## Find a result (people and agents)

```bash
source .venv/bin/activate
python scripts/report/results.py list                       # every run: id, kind, rung, status, headline numbers
python scripts/report/results.py list --no-legacy --kind train --metric train.val_soundscape_cmAP
python scripts/report/results.py show focal-only            # index entry, headline numbers, file locations
python scripts/report/results.py get focal-only train.selected_val.soundscape.cmAP   # one value
python scripts/report/results.py compare focal-only soundscape-only --metric train.val_soundscape_cmAP
python scripts/report/results.py paths focal-only           # config, metrics, log, features, checkpoint
python scripts/report/results.py --json list                # machine-readable
```

`--dataset` (default `hsn`) selects the dataset; `--root` the outputs directory. A wrong run id lists the known ones;
a wrong metric path lists what is available at that level.

## Layout

```
outputs/results/<dataset>/            the store (git-ignored, like everything under outputs/)
    registry.jsonl                    the index: one JSON line per save; the LAST line of a run_id is current
    runs/<run_id>/
        config.json                   how the run was made: all arguments, effective settings, git commit + dirty flag, versions
        metrics.json                  {"sections": {<section>: {...}}}: every metric of the run
        train_log.csv                 per-epoch log of a training run (phase, epoch, loss, val metrics for both domains)
    summaries/<name>.json             results spanning several runs, e.g. model_comparison
    dataset/                          description of the data: splits.json
outputs/features/<dataset>/<run_id>/  responses.pt (per-clip probabilities, prototype activations), eval_cache.pt
outputs/checkpoints/<dataset>/<run_id>/  best.pt
```

A **run** is a model: one we trained (`kind: train`) or only evaluate (`kind: reference`, e.g. `pretrained-xcl`).
**Summaries** have the id `summary:<name>`.

## Metric sections of a run (`metrics.json`)

| Section | Written by | Contents |
|---|---|---|
| `train` | `scripts/train/train_protopnet.py` | `selected_val` (cmAP, macro AUC on the focal and soundscape validation sets of the selected checkpoint), `train_domain` |
| `prototype_transfer[_tag]` | `scripts/analysis/extract_prototype_activations.py` | small header; the per-prototype statistics are in `prototype_transfer[_tag].json` |
| `inference` | `scripts/eval/run_inference.py` | focal / soundscape cmAP, macro AUC, top-1 of the pretrained model; per-class metrics in `{focal,soundscape}_metrics.json` |

`get` takes `SECTION.KEY.KEY...`, for example `train.selected_val.soundscape.cmAP`.
For a summary, the path indexes the document directly: `get summary:model_comparison models.focal-only.gap_cmAP`.

## The index entry (`registry.jsonl`)

```json
{"schema": 1, "run_id": "focal-only", "dataset": "hsn", "kind": "train", "seed": 0,
 "status": "done", "started": "...", "finished": "...", "commit": "<git sha or null>", "dirty": false, "legacy": false,
 "note": "...", "sections": ["prototype_transfer", "train"],
 "headline": {"train.val_soundscape_cmAP": 0.487},
 "paths": {"metrics": "...", "config": "...", "train_log": "...", "files": ["..."], "features": "...", "checkpoints": "..."},
 "updated": "..."}
```

`headline` keys are `<section>.<name>`; they are the numbers people look up first, so `list` answers most questions
without opening any other file. `legacy: true` marks results copied from before the store existed: no commit was recorded
for them, so they cannot be reproduced from a known state, and `--no-legacy` hides them.

## Where results live

Nothing under `outputs/` is committed: results stay on the machine where the run ran (a workstation, an sbatch job's node).
The store is plain files, so it can be copied or synced like any directory. `registry.jsonl` is append-only and written under a
file lock, so concurrent jobs writing to the same store are safe. Use unique run ids (for example `<initials>-<idea>-<variant>`)
so runs from different people or machines never clash if stores are combined. `config.json` records each run's git commit.

## Writing results

```python
from prototype_transfer.evaluation import results

store = results.Store("hsn")
results.save_results(store, run_id, "my_eval", metrics_dict,   # saving a section again replaces it
                     kind="train", headline={"cmAP": 0.64}, entry={"seed": 0})
results.save_summary(store, "model_comparison", summary_dict)
```

`save_results` also takes `config=` (config.json), `log=` (train_log.csv) and `files=` (extra `<name>.json` files in the run directory). Values are
checked (JSON-serialisable, headline numeric, unknown index fields rejected), files are replaced atomically and the registry
line is appended under a lock, so concurrent SLURM jobs do not corrupt anything. Do not write results with `json.dump`
elsewhere. Reading helpers: `read_registry`, `list_runs`, `load_metrics`, `get_metric`. Tests: `tests/test_results.py`.
