# Results store

Every number this repository produces is saved through one writer
(`prototype_transfer.evaluation.results.save_results`) into one layout, and indexed in one file.
If you are looking for a number, start with the CLI below; you should not need to grep.

## Find a result (people and agents)

```bash
source .venv/bin/activate
python scripts/report/results.py list                       # every run: id, kind, rung, status, headline numbers
python scripts/report/results.py list --no-legacy --kind train --metric canonical_test.sc_ap_primary
python scripts/report/results.py show focal-only            # index entry, headline numbers, file locations
python scripts/report/results.py get pretrained-xcl canonical_test.point.sc_ap_primary   # one value
python scripts/report/results.py compare focal-only soundscape-only --metric train.val_soundscape_cmAP
python scripts/report/results.py paths focal-only           # config, metrics, log, features, checkpoint
python scripts/report/results.py --json list                # machine-readable
```

`--dataset` (default `hsn`) selects the dataset; `--root` the outputs directory. A wrong run id lists the known ones;
a wrong metric path lists what is available at that level.

## Layout

```
outputs/results/<dataset>/            committed (small)
    registry.jsonl                    the index: one JSON line per save; a run_id's current entry combines its lines
    runs/<run_id>/
        config.json                   how the run was made: all arguments, effective settings, git commit + dirty flag (uncommitted changes to tracked files outside outputs/results/), versions
        metrics.json                  {"sections": {<section>: {...}}}: every metric of the run
        train_log.csv                 per-epoch log of a training run (phase, epoch, loss, val metrics for both domains)
        push.json, prototype_transfer*.json, canonical_*_prototypes.json   other small per-run files
    summaries/<name>.json             results spanning several runs: canonical_eval*, cv_<name>, model_comparison, diagnostics_stage2*
    dataset/                          description of the data: splits.json, splits_cv.json
outputs/features/<dataset>/<run_id>/  NOT committed: responses.pt (per-clip probabilities, prototype activations), diag_*.pt
outputs/checkpoints/<dataset>/<run_id>/  NOT committed: best.pt
```

A **run** is a model: one we trained (`kind: train`) or only evaluate (`kind: reference`, e.g. `pretrained-xcl`).
Cross-validation folds are separate runs (`<name>-cv0`, `<name>-cv1`, `<name>-cv2`). **Summaries** have the id `summary:<name>`.

## Metric sections of a run (`metrics.json`)

| Section | Written by | Contents |
|---|---|---|
| `train` | `scripts/train/train_protopnet.py` | `selected_val` and `last_epoch_val` (cmAP, macro AUC on the focal and soundscape validation sets), `select`, `train_domain` |
| `canonical_test` / `canonical_val` / `canonical_cv<f>` | `scripts/eval/canonical_eval.py` | protocol v1: `point`, `ci95` (cluster bootstrap), `vs_baseline`, `gap_fraction`, `criteria`, plus the split, seed, sample sizes. The per-prototype table is in `canonical_<tag>_prototypes.json` |
| `prototype_transfer[_tag]` | `scripts/analysis/extract_prototype_activations.py` | small header; the per-prototype statistics are in `prototype_transfer[_tag].json` |

`get` takes `SECTION.KEY.KEY...`, for example `canonical_test.point.sc_ap_primary` (headline soundscape AP on the primary
species), `canonical_test.ci95.sc_ap_primary` (its 95% interval), `canonical_test.gap_fraction.value`.
For a summary, the path indexes the document directly: `get summary:model_comparison models.focal-only.gap_cmAP`.
Metric definitions and the evaluation protocol: the docstring of `prototype_transfer/evaluation/canonical.py`.

## The index entry (`registry.jsonl`)

```json
{"schema": 1, "run_id": "focal-only", "dataset": "hsn", "kind": "train", "rung": "baseline", "seed": 0, "cv_fold": null,
 "status": "done", "started": "...", "finished": "...", "commit": "<git sha or null>", "dirty": false, "legacy": false,
 "protocol": "v1", "note": "...", "sections": ["canonical_test", "train"],
 "headline": {"train.val_soundscape_cmAP": 0.487, "canonical_test.sc_ap_primary": 0.7},
 "section_saved": {"canonical_test": "...", "train": "..."},
 "paths": {"metrics": "...", "config": "...", "train_log": "...", "files": ["..."], "features": "...", "checkpoints": "..."},
 "updated": "..."}
```

`headline` keys are `<section>.<name>`; they are the numbers people look up first, so `list` answers most questions
without opening any other file. `legacy: true` marks results copied from before the store existed: no commit was recorded
for them, so they cannot be reproduced from a known state, and `--no-legacy` hides them.

## Sharing results through git

Commit `outputs/results/<dataset>/` (runs, summaries, `registry.jsonl`); features and checkpoints are git-ignored. `registry.jsonl` is append-only, so
`.gitattributes` merges it with `merge=union`: both branches' lines are kept. When both saved to the same run (for example, two evaluations
of `pretrained-xcl`), `read_registry` keeps the sections of both; a section saved on both keeps the most recently saved version, and other
index fields come from the most recently updated line that has them. Always read the index with `read_registry` or the CLI, not by taking a
run's last line. The union merge covers the index only: if git reports a conflict in a run's `metrics.json`, keep both branches' sections.
Use unique run ids (for example `<initials>-<idea>-<variant>`) so two people never write the same id.

## Writing results

```python
from prototype_transfer.evaluation import results

store = results.Store("hsn")
results.save_results(store, run_id, "canonical_test", metrics_dict,   # saving a section again replaces it
                     kind="train", headline={"sc_ap_primary": 0.64}, entry={"rung": "R0", "seed": 0})
results.save_summary(store, "canonical_eval", summary_dict)
```

`save_results` also takes `config=` (config.json), `log=` (train_log.csv) and `files=` (extra `<name>.json` files). Values are
checked (JSON-serialisable, headline numeric, unknown index fields rejected), files are replaced atomically and the registry
line is appended under a lock, so concurrent SLURM jobs do not corrupt anything. Do not write results with `json.dump`
elsewhere. Reading helpers: `read_registry`, `list_runs`, `load_metrics`, `get_metric`. Tests: `tests/test_results.py`.
