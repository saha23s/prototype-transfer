"""The multi-page dashboard shows what is on disk and reads as a report. Run: python tests/test_dashboard_pages.py
Rebuilding needs the committed results plus the split manifests and cached responses; without them only the checks on the
committed pages run."""
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd

from prototype_transfer.evaluation import results
from prototype_transfer.reporting import charts, site
from prototype_transfer.reporting.runs import BASELINE, FOLDS, METHODS, Results

OUT, SPLITS_DIR = Path("dashboard"), Path("data/HSN/splits")
spec = importlib.util.spec_from_file_location("build_dashboard", Path("scripts/report/build_dashboard.py"))
bd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd)


def page(name: str) -> str:
    return (OUT / name).read_text()


def test_pages_exist_with_one_current_nav_entry_and_working_links():
    for name, label in site.PAGE_ORDER:
        text = page(name)
        assert text.count('aria-current="page"') == 1, name
        assert re.search(rf'<a href="{name}" aria-current="page"[^>]*>{label}</a>', text), name
        for target in re.findall(r'(?:href|src)="([^"#]+)(?:#[^"]*)?"', text):
            assert (OUT / target).exists(), (name, target)


def test_pages_have_no_bookkeeping_text():
    banned = (".parquet", "outputs/", "scripts/", "src/", "sha256", "registry", "--split", "r0x-", "tf-quantile", "-cv0", "commit")
    for name, _ in site.PAGE_ORDER:
        visible = re.sub(r"<[^>]+>", " ", page(name))
        assert not [b for b in banned if b in visible], (name, [b for b in banned if b in visible])


def test_pages_carry_the_saved_numbers():
    store = results.Store("hsn")
    out = json.loads(store.summary("canonical_eval").read_text())
    assert f"{out['models'][BASELINE]['point']['sc_ap_primary']:.3f}" in page("evaluation.html")
    sp = json.loads((store.dataset_dir / "splits.json").read_text())
    for dom in ("focal", "soundscape"):
        for s in ("train", "val", "test"):
            assert f"{sp[dom][s]['n_clips']:,}" in page("splits.html"), (dom, s)
    for key in METHODS:   # every cross-validated method is on the results page, with its mean soundscape AP
        cv = json.loads(store.summary(f"cv_{key}").read_text())
        assert METHODS[key].label in page("results.html"), key
        assert f"{cv['run']['sc_ap_primary']['mean']:.3f}" in page("results.html"), key
        if key.startswith("lb"):
            for v in cv["run"]["sc_ap_primary"]["folds"]:
                assert f"{v:.3f}" in page("label-curve.html"), (key, v)


def test_committed_pages_match_a_fresh_build():
    store = results.Store("hsn")
    if not (SPLITS_DIR.exists() and all((store.features(BASELINE) / f"responses_cv{f}.pt").exists() for f in FOLDS)):
        print("skipped the rebuild check: split manifests or cached responses are not here")
        return
    for name, content in bd.build(Results(store, SPLITS_DIR)).items():
        assert page(name) == content, f"dashboard/{name} is out of date: run scripts/report/build_dashboard.py"


def test_findings_bundle_leaves_out_the_test_split():
    bundle = json.loads((OUT / "data.json").read_text())
    assert bundle["meta"]["scope"] == "non-test"
    assert f"{len(bundle['species'])} species with at least 10 soundscape positives" in page("index.html")
    if not SPLITS_DIR.exists():
        print("skipped the clip check: split manifests are not here")
        return
    clips = {Path(q).name for d in ("focal", "soundscape") for q in pd.read_parquet(SPLITS_DIR / f"{d}_test.parquet")["audio_path"]}
    shown = {it["clip"] for s in bundle["species"] for by_domain in s["patches"].values() for items in by_domain.values() for it in items}
    assert shown and not shown & clips, f"{len(shown & clips)} test clips are shown as examples"
    counts = {d: sum(len(pd.read_parquet(SPLITS_DIR / f"{d}_{s}.parquet")) for s in ("train", "val")) for d in ("focal", "soundscape")}
    assert bundle["meta"]["n_clips"] == counts, (bundle["meta"]["n_clips"], counts)


def test_signed_has_no_negative_zero():
    assert charts.signed(-0.0001) == "±0.000" and charts.signed(-0.002) == "−0.002" and charts.signed(0.15, 2) == "+0.15"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests OK")
