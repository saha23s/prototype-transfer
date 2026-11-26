# How the committed HSN cross-validation summaries were made

A record, not part of the pipeline (`scripts/slurm/run_recipe.sbatch` now passes each fold's oracle itself).
For each fold, a `rescore_*.sh` script scored many models together against `soundscape-only-cv<fold>`; its `split_*.py`
then wrote one `canonical_eval_<name>-cv<fold>` summary per method, and `cv_aggregate.py` combined the folds.

| Scripts | Joint summary | Methods |
|---|---|---|
| `rescore_oracle.sh`, `split_joint.py` | `canonical_eval_all-cv<fold>` | soundscape-only, focal-only, r0-r4, tf-* |
| `rescore_curve.sh`, `split_curve.py` | `canonical_eval_curve-cv<fold>` | lb*, r0x-* |

Usage: `bash scripts/eval/cv_rescore_record/rescore_oracle.sh LOG_DIR`. **Running it overwrites the committed results.**
