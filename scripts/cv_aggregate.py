#!/usr/bin/env python3
"""Combine the per-fold canonical evaluations of one cross-validated run (CPU only).

For fold f, scripts/slurm/run_recipe.sbatch (array job) trains NAME-cv<f> without fold f's sites and writes
outputs/results/canonical_eval_NAME-cv<f>.json (scripts/canonical_eval.py --split cv --fold f). This script reads those
files and reports, per metric, the value on every fold, the mean and the sd across folds (ddof = 1), for the run, the
baseline (pretrained, evaluated on the same held-out sites) and their per-fold difference, plus how many folds meet each
success criterion. Writes outputs/results/cv_NAME.json. The folds are only a handful of sites each, so treat the sd as a
rough spread, not a confidence interval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="run name without the -cv<fold> suffix")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--baseline", default="pretrained-xcl")
    p.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    p.add_argument("--out", type=Path, default=None, help="default outputs/results/cv_<name>.json")
    args = p.parse_args()
    paths = [args.results_dir / f"canonical_eval_{args.name}-cv{f}.json" for f in range(args.folds)]
    missing = [str(q) for q in paths if not q.exists()]
    if missing:
        sys.exit(f"missing per-fold evaluations: {missing}")
    res = aggregate([json.loads(q.read_text()) for q in paths], args.name, args.baseline)
    out = args.out or args.results_dir / f"cv_{args.name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"{args.name}: {args.folds}-fold cross-validation (mean ± sd over folds; difference = run - {args.baseline}, per fold)")
    print(f"{'metric':16} {'run':>16} {args.baseline:>18} {'difference':>16}")
    f = lambda d, k: f"{d[k]['mean']:.3f} ± {d[k]['sd']:.3f}"
    for k in KEYS:
        print(f"{k:16} {f(res['run'], k):>16} {f(res['baseline_metrics'], k):>18} {f(res['difference'], k):>16}")
    if "criteria_folds_passed" in res:
        print(f"criteria met on (folds): {res['criteria_folds_passed']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
