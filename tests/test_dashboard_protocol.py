"""The dashboard's splits and protocol sections show what is on disk. Run: python tests/test_dashboard_protocol.py"""
import importlib.util
import json
import re
from pathlib import Path

from prototype_transfer.evaluation import results

spec = importlib.util.spec_from_file_location("build_dashboard_protocol", Path("scripts/report/build_dashboard_protocol.py"))
bdp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bdp)


def test_sections_carry_the_committed_numbers():
    store = results.Store("hsn")
    block, _ = bdp.build(store)
    sp = json.loads((store.dataset_dir / "splits.json").read_text())
    out = json.loads(store.summary("canonical_eval").read_text())
    for dom in ("focal", "soundscape"):
        for s in ("train", "val", "test"):
            assert f"{sp[dom][s]['n_clips']:,}" in block, (dom, s)
    for k in out["species"]["primary"]:
        assert k in block
    ap = out["models"]["pretrained-xcl"]["point"]["sc_ap_primary"]
    assert f"{ap:.3f}" in block
    assert block.count("<svg") == 5 and block.startswith(bdp.START) and block.endswith(bdp.END)


def test_page_has_both_sections_once_and_the_rebuild_is_stable():
    page = Path("dashboard/index.html").read_text()
    assert page.count(bdp.START) == 1 and page.count(bdp.END) == 1
    assert len(re.findall(r'id="splits"', page)) == 1 and len(re.findall(r'id="protocol"', page)) == 1
    block, _ = bdp.build(results.Store("hsn"))
    assert block in page, "dashboard/index.html is out of date: run scripts/report/build_dashboard_protocol.py"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests OK")
