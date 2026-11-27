# Agent Instructions
1. Put all reusable dataset and model code in the `src/prototype_transfer/` package (import it as `prototype_transfer.*`).
2. Put runnable experiment scripts in `scripts/<stage>/` (data, analysis, eval, report, train).
3. Features and checkpoints go in `outputs/features/<dataset>/<run>/` and `outputs/checkpoints/<dataset>/<run>/`.
4. Write results only through `prototype_transfer.evaluation.results.save_results` (or `save_summary`); find them with
   `python scripts/report/results.py list|show|get` (docs/RESULTS.md). Do not grep `outputs/` for numbers.
