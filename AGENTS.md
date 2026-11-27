# Agent Instructions
1. Reusable code goes in `src/prototype_transfer/`; runnable scripts in `scripts/<stage>/`.
2. Features and checkpoints go in `outputs/features/<dataset>/<run>/` and `outputs/checkpoints/<dataset>/<run>/`.
3. Write results only through `prototype_transfer.evaluation.results.save_results`; find them with
   `python scripts/report/results.py list|show|get` (docs/RESULTS.md). Do not grep `outputs/` for numbers.
4. Score models only with `scripts/eval/canonical_eval.py` (docs/PROTOCOL.md); splits must pass `scripts/data/check_splits.py`.
