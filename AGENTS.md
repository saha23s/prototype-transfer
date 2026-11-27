# Agent Instructions
1. Put all reusable dataset and model code in the `src/prototype_transfer/` package (import it as `prototype_transfer.*`).
2. Put runnable experiment scripts in `scripts/<stage>/` (data, analysis, eval, report, train).
3. Save pre-computed ConvNeXt features in `outputs/features/`.
4. Write evaluation outputs (cmAP, TAR) to JSON files in `outputs/results/`.