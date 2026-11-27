#!/usr/bin/env python3
"""Query the results store (outputs/results/<dataset>/, see docs/RESULTS.md). Text tables by default, --json for machines.

  results.py list [--kind train|reference|summary|analysis] [--rung R0] [--no-legacy] [--metric canonical_test.sc_ap_primary ...]
  results.py show RUN_ID [--section canonical_test]      index entry, file locations and the run's headline numbers
  results.py get RUN_ID SECTION.PATH                     one value, e.g. r0 canonical_test.point.sc_ap_primary
  results.py compare RUN_ID [RUN_ID ...] --metric M ...  runs side by side on chosen headline metrics
  results.py paths RUN_ID                                where the run's config, metrics, log, features and checkpoint are
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prototype_transfer.evaluation import results as R


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def table(rows: list[list], header: list[str]) -> str:
    cols = [header] + [[fmt(c) for c in r] for r in rows]
    w = [max(len(r[i]) for r in cols) for i in range(len(header))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip() for r in cols)


def headline_value(entry: dict, metric: str):
    h = entry.get("headline", {})
    if metric in h:
        return h[metric]
    hits = [k for k in h if k.endswith("." + metric)]   # allow the bare name when it is unambiguous
    return h[hits[0]] if len(hits) == 1 else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=R.DEFAULT_DATASET)
    p.add_argument("--root", type=Path, default=Path("outputs"), help="outputs directory")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list", help="the registry, one row per run")
    s.add_argument("--kind", choices=R.KINDS)
    s.add_argument("--rung")
    s.add_argument("--status")
    s.add_argument("--section", help="only runs that have this metrics section")
    s.add_argument("--no-legacy", action="store_true", help="hide results backfilled from before the store existed")
    s.add_argument("--metric", action="append", default=[], help="headline metric column to show (repeatable)")
    s = sub.add_parser("show")
    s.add_argument("run_id")
    s.add_argument("--section", help="print this section of metrics.json in full")
    s = sub.add_parser("get")
    s.add_argument("run_id")
    s.add_argument("path", help="SECTION.KEY.KEY... into metrics.json (for summary:<name>: KEY.KEY... into the document)")
    s = sub.add_parser("compare")
    s.add_argument("run_ids", nargs="+")
    s.add_argument("--metric", action="append", required=True)
    s = sub.add_parser("paths")
    s.add_argument("run_id")
    a = p.parse_args()
    store = R.Store(a.dataset, a.root)
    reg = R.read_registry(store)

    def entry(run_id: str) -> dict:
        if run_id not in reg:
            sys.exit(f"unknown run {run_id!r} in dataset {a.dataset!r}; known: {', '.join(sorted(reg)) or '(none)'}")
        return reg[run_id]

    if a.cmd == "list":
        rows = list(R.list_runs(store, kind=a.kind, rung=a.rung, status=a.status, section=a.section, legacy=False if a.no_legacy else None))
        if a.json:
            print(json.dumps(rows, indent=1))
            return
        if not rows:
            print(f"no runs in {store.registry}")
            return
        metrics = a.metric or ["train.val_focal_cmAP", "train.val_soundscape_cmAP"]
        print(table([[e["run_id"], e.get("kind"), e.get("rung"), e.get("status"), "legacy" if e.get("legacy") else "",
                      *[headline_value(e, m) for m in metrics]] for e in rows],
                    ["run_id", "kind", "rung", "status", "", *[m.split(".")[-1] if a.metric == [] else m for m in metrics]]))
    elif a.cmd == "show":
        e = entry(a.run_id)
        if a.section:
            print(json.dumps(R.load_metrics(store, a.run_id)["sections"][a.section] if not a.run_id.startswith(R.SUMMARY_PREFIX)
                             else R.load_metrics(store, a.run_id), indent=1))
        elif a.json:
            print(json.dumps(e, indent=1))
        else:
            for k in ("run_id", "kind", "rung", "status", "seed", "cv_fold", "commit", "dirty", "legacy", "note", "updated", "sections"):
                if e.get(k) not in (None, [], ""):
                    print(f"{k:10} {e[k]}")
            print("headline:")
            for k, v in sorted(e.get("headline", {}).items()):
                print(f"  {k:48} {fmt(v)}")
            print("files:")
            for k, v in e.get("paths", {}).items():
                print(f"  {k:10} {v}")
    elif a.cmd == "get":
        entry(a.run_id)
        try:
            v = R.get_metric(store, a.run_id, a.path)
        except KeyError as err:
            sys.exit(str(err.args[0]))
        print(json.dumps(v) if a.json or isinstance(v, (dict, list)) else v)
    elif a.cmd == "compare":
        rows = [[rid, *[headline_value(entry(rid), m) for m in a.metric]] for rid in a.run_ids]
        print(json.dumps(rows) if a.json else table(rows, ["run_id", *a.metric]))
    elif a.cmd == "paths":
        paths = entry(a.run_id).get("paths", {})
        print(json.dumps(paths, indent=1) if a.json else "\n".join(f"{k:10} {v}" for k, v in paths.items()))


if __name__ == "__main__":
    main()
