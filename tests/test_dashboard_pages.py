"""The multi-page dashboard shows what is on disk and reads as a report. Run: python tests/test_dashboard_pages.py
Rebuilding needs the committed results plus the split manifests and cached responses; without them only the checks on the
committed pages run."""
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd

from prototype_transfer.evaluation import results
from prototype_transfer.reporting import charts, notation, site
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
    assert charts.score(out['models'][BASELINE]['point']['sc_ap_primary']) in page("evaluation.html")
    sp = json.loads((store.dataset_dir / "splits.json").read_text())
    for dom in ("focal", "soundscape"):
        for s in ("train", "val", "test"):
            assert f"{sp[dom][s]['n_clips']:,}" in page("splits.html"), (dom, s)
    for key in METHODS:   # every cross-validated method is on the results page, with its mean soundscape AP
        cv = json.loads(store.summary(f"cv_{key}").read_text())
        assert METHODS[key].label in page("results.html"), key
        assert charts.score(cv['run']['sc_ap_primary']['mean']) in page("results.html"), key
        if key.startswith("lb"):
            for v in cv["run"]["sc_ap_primary"]["folds"]:
                assert charts.score(v) in page("label-curve.html"), (key, v)


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


# what a reader can meet on a page -> a notation entry (any one of them) that has to explain it
SYMBOLS = {r"γ": ("gamma", "gh"), r"\bAP\b": ("ap",), r"AUC": ("auc",), r"cmAP": ("cmap",), r"ΔAP": ("dap", "gap"),
           r"ΔAUC": ("dauc",), r"Δ(?!AP)": ("dauc", "dap"), r"±": ("pm",), r"↑ / ↓": ("updown",), r"✓": ("check",),
           r"[Ff]ail": ("fail",), r"95% interval": ("ci",), r"\bcross-validation\b|\bfold": ("fold",), r"\bSNR\b": ("snr",),
           r"\bkHz\b": ("khz",), r"h★": ("h",), r"\bK = ": ("k",), r"\br = ": ("corr",), r"\bq25\b": ("quant",),
           r"sample-wise": ("map",)}
SYMBOL_EXEMPT = {"splits.html": {r"AUC"}}


def visible_text(html_text: str) -> str:
    text = re.sub(r"<title>.*?</title>|<nav.*?</nav>|<footer.*?</footer>", " ", html_text, flags=re.S)
    text = re.sub(r"<section id=\"notation\".*?</section>", " ", text, flags=re.S)
    return re.sub(r"<[^>]+>", "", text)


def test_every_page_ends_with_notation_that_covers_its_symbols():
    bundle = (OUT / "data.json").read_text()
    findings = " ".join(f["title"] + " " + f["text"] for f in json.loads(bundle)["findings"])
    for name, _ in site.PAGE_ORDER:
        text = page(name)
        assert text.count('id="notation"') == 1 and text.index('id="notation"') < text.index("</main>"), name
        assert not text[text.index('id="notation"'):text.index("</main>")].count("<section"), f"{name}: notation is not last"
        shown = visible_text(text) + (" " + findings if name == "index.html" else "")
        missing = [pattern for pattern, terms in SYMBOLS.items() if re.search(pattern, shown)
                   and not set(terms) & set(notation.PAGES[name]) and pattern not in SYMBOL_EXEMPT.get(name, ())]
        assert not missing, (name, missing)


def test_signed_has_no_negative_zero():
    assert charts.signed(-0.0001) == "±0.0" and charts.signed(-0.002) == "−0.2" and charts.signed(0.15, 0) == "+15"
    assert charts.score(0.7104) == "71.0"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests OK")
