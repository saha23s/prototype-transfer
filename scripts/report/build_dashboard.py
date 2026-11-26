#!/usr/bin/env python3
"""Build the multi-page dashboard in dashboard/ (static HTML with inline SVG; see prototype_transfer.reporting.pages).

Reads the results store (outputs/results/<dataset>/), the split manifests (data/HSN/splits) and the cached responses
(outputs/features/<dataset>/: per-species breakdowns, checked against the saved canonical values), so run it where those
live (CPU is enough). Numbers in the text are computed from these inputs, and the build fails if a stated conclusion no
longer holds. The interactive findings and species explorer use dashboard/data.js (scripts/report/build_dashboard_data.py).

  python scripts/report/build_dashboard.py [--dataset hsn] [--out-dir dashboard] [--gallery]
--gallery redraws the example spectrograms (needs the audio); it also happens when the image is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prototype_transfer.evaluation import results
from prototype_transfer.reporting import charts, site
from prototype_transfer.reporting.pages import (data, evaluation, explorer, formulation, label_curve, motivation,
                                                results as results_page, splits)
from prototype_transfer.reporting.runs import Results

BUILDERS = (motivation, data, splits, evaluation, formulation, label_curve, results_page, explorer)


def build(res: Results) -> dict[str, str]:
    """file name -> contents of every generated page, in navigation order, plus names.js."""
    pages = [b.build(res) for b in BUILDERS]
    assert [p.file for p in pages] == [f for f, _ in site.PAGE_ORDER], "page order and navigation differ"
    out = {p.file: site.render(p) for p in pages}
    out["names.js"] = "window.SPECIES_NAMES = " + json.dumps(charts.SPECIES_NAMES) + ";\n"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=results.DEFAULT_DATASET)
    p.add_argument("--splits-dir", type=Path, default=Path("data/HSN/splits"))
    p.add_argument("--out-dir", type=Path, default=Path("dashboard"))
    p.add_argument("--gallery", action="store_true")
    args = p.parse_args()
    res = Results(results.Store(args.dataset), args.splits_dir)
    files = build(res)
    if args.gallery or not (args.out_dir / data.GALLERY).exists():
        data.make_gallery(res, res.test["species"]["primary"], args.out_dir / data.GALLERY)
    for name, content in files.items():
        (args.out_dir / name).write_text(content)
    print(f"wrote {len(files)} files to {args.out_dir}")


if __name__ == "__main__":
    main()
