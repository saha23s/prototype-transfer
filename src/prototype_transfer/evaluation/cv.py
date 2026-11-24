"""Combine the per-fold canonical evaluations of one cross-validated run (CPU only).

For fold f, scripts/slurm/run_recipe.sbatch (array job) trains NAME-cv<f> without fold f's sites and writes
the summary canonical_eval_NAME-cv<f> (scripts/eval/canonical_eval.py --split cv --fold f --summary-name ...). The aggregation reads those
files and reports, per metric, the value on every fold, the mean and the sd across folds (ddof = 1), for the run, the
baseline (pretrained, evaluated on the same held-out sites) and their per-fold difference, plus how many folds meet each
success criterion. The CLI is scripts/eval/cv_aggregate.py (saves the summary cv_NAME). The folds are only a
handful of sites each, so treat the sd as a rough spread, not a confidence interval.
"""

from __future__ import annotations

import numpy as np

KEYS = ("sc_ap_primary", "sc_auc_primary", "sc_cmap_12", "f_auc_primary", "n_fail", "n_fail_birds")


def spread(x: list[float]) -> dict:
    a = np.asarray(x, dtype=float)
    return {"folds": [float(v) for v in a], "mean": float(np.nanmean(a)), "sd": float(np.nanstd(a, ddof=1)) if len(a) > 1 else float("nan")}


def aggregate(evals: list[dict], name: str, baseline: str = "pretrained-xcl", oracle: str | None = None) -> dict:
    """evals[f] = canonical_eval.json of fold f (model `NAME-cv<f>` and the baseline are in it)."""
    for f, e in enumerate(evals):
        assert e.get("split") == "cv" and e.get("fold") == f, f"fold {f}: canonical_eval was run with split={e.get('split')} fold={e.get('fold')}"
    run = [e["models"][f"{name}-cv{f}"] for f, e in enumerate(evals)]
    base = [e["models"][baseline] for e in evals]
    out: dict = {"name": name, "n_folds": len(evals), "baseline": baseline, "run": {}, "baseline_metrics": {}, "difference": {}}
    for k in KEYS:
        out["run"][k] = spread([m["point"][k] for m in run])
        out["baseline_metrics"][k] = spread([m["point"][k] for m in base])
        out["difference"][k] = spread([r["point"][k] - b["point"][k] for r, b in zip(run, base)])
    if all("gap_fraction" in m for m in run):
        out["gap_fraction"] = spread([m["gap_fraction"]["value"] for m in run])
    crit = [m.get("criteria") for m in run]
    if all(crit):
        out["criteria_folds_passed"] = {c: int(sum(bool(x[c]) for x in crit)) for c in crit[0]}
    return out


def summary_table(res: dict) -> str:
    """The table printed after aggregating (run vs baseline vs per-fold difference, then criteria met per fold)."""
    baseline = res["baseline"]
    f = lambda d, k: f"{d[k]['mean']:.3f} ± {d[k]['sd']:.3f}"
    lines = [f"{res['name']}: {res['n_folds']}-fold cross-validation (mean ± sd over folds; difference = run - {baseline}, per fold)",
             f"{'metric':16} {'run':>16} {baseline:>18} {'difference':>16}"]
    for k in KEYS:
        lines.append(f"{k:16} {f(res['run'], k):>16} {f(res['baseline_metrics'], k):>18} {f(res['difference'], k):>16}")
    if "criteria_folds_passed" in res:
        lines.append(f"criteria met on (folds): {res['criteria_folds_passed']}")
    return "\n".join(lines)
