"""Tests of the results store (prototype_transfer.evaluation.results); run as `python tests/test_results.py`."""

import json
import multiprocessing as mp
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from prototype_transfer.evaluation import results as R


def _store(tmp: str) -> R.Store:
    return R.Store("toy", Path(tmp))


def test_dirty_flags_code_changes_but_not_rewritten_results():
    with tempfile.TemporaryDirectory() as tmp:
        git = lambda *a: subprocess.run(["git", "-C", tmp, "-c", "user.name=t", "-c", "user.email=t@example.com",
                                         "-c", "commit.gpgsign=false", *a], capture_output=True, check=True)
        code, res = Path(tmp) / "src" / "model.py", Path(tmp) / "outputs" / "results" / "hsn" / "registry.jsonl"
        for path in (code, res):
            path.parent.mkdir(parents=True)
            path.write_text("1\n")
        git("init", "-q"); git("add", "-A"); git("commit", "-q", "-m", "init")
        cwd = os.getcwd()
        try:
            for where in (Path(tmp), code.parent):                     # the answer must not depend on the directory
                os.chdir(where)
                assert R.provenance()["dirty"] is False
                res.write_text("2\n")
                assert R.provenance()["dirty"] is False, "a rewritten result file is not a code change"
                git("add", "-A")                                       # ... staged or not
                assert R.provenance()["dirty"] is False
                code.write_text("2\n")
                assert R.provenance()["dirty"] is True, "an edited source file is"
                git("reset", "-q", "--hard")
        finally:
            os.chdir(cwd)


def test_roundtrip_and_merge():
    with tempfile.TemporaryDirectory() as tmp:
        s = _store(tmp)
        e = R.save_results(s, "r0", "train", {"val": {"cmAP": 0.5}}, headline={"val_cmAP": 0.5}, config={"args": {"seed": 0}},
                           log=[{"epoch": 1, "loss": 1.0}, {"epoch": 2, "loss": 0.5, "extra": 3}], files={"push.json": {"n": 1}},
                           entry={"rung": "R0", "seed": 0, "status": "done"})
        assert e["headline"] == {"train.val_cmAP": 0.5} and e["paths"]["train_log"].endswith("train_log.csv")
        R.save_results(s, "r0", "canonical_test", {"point": {"sc_ap_primary": 0.7}}, headline={"sc_ap_primary": 0.7})
        cur = R.read_registry(s)["r0"]
        assert cur["sections"] == ["canonical_test", "train"], cur
        assert cur["rung"] == "R0" and cur["headline"] == {"train.val_cmAP": 0.5, "canonical_test.sc_ap_primary": 0.7}
        assert R.get_metric(s, "r0", "canonical_test.point.sc_ap_primary") == 0.7
        assert R.get_metric(s, "r0", "train.val.cmAP") == 0.5
        assert (s.run_dir("r0") / "train_log.csv").read_text().splitlines()[0] == "epoch,loss,extra"
        assert json.loads((s.run_dir("r0") / "config.json").read_text())["args"] == {"seed": 0}


def test_resave_section_replaces_and_registry_last_wins():
    with tempfile.TemporaryDirectory() as tmp:
        s = _store(tmp)
        R.save_results(s, "a", "canonical_test", {"x": 1}, headline={"x": 1})
        R.save_results(s, "a", "canonical_test", {"x": 2}, headline={"x": 2})
        assert R.get_metric(s, "a", "canonical_test.x") == 2
        assert R.read_registry(s)["a"]["headline"] == {"canonical_test.x": 2}
        assert len(s.registry.read_text().splitlines()) == 2   # append-only log


def test_summary_and_filters():
    with tempfile.TemporaryDirectory() as tmp:
        s = _store(tmp)
        R.save_results(s, "r0", "train", {}, entry={"rung": "R0", "legacy": True})
        R.save_results(s, "pretrained-xcl", "canonical_test", {}, kind="reference")
        R.save_summary(s, "canonical_eval", {"protocol": "v1", "models": {"r0": {"a": 1}}}, headline={"n": 2})
        assert [e["run_id"] for e in R.list_runs(s, kind="summary")] == ["summary:canonical_eval"]
        assert [e["run_id"] for e in R.list_runs(s, legacy=True)] == ["r0"]
        assert [e["run_id"] for e in R.list_runs(s, legacy=False)] == ["pretrained-xcl", "summary:canonical_eval"]
        assert R.get_metric(s, "summary:canonical_eval", "models.r0.a") == 1


def test_bad_inputs_fail_loudly():
    with tempfile.TemporaryDirectory() as tmp:
        s = _store(tmp)
        for bad in (lambda: R.save_results(s, "../x", "train", {}),
                    lambda: R.save_results(s, "r", "bad section", {}),
                    lambda: R.save_results(s, "r", "train", {}, kind="summary"),
                    lambda: R.save_results(s, "r", "train", {}, entry={"colour": "red"}),
                    lambda: R.save_results(s, "r", "train", {}, files={"metrics.json": {}}),
                    lambda: R.save_results(s, "r", "train", {}, headline={"x": "high"}),
                    lambda: R.save_results(s, "r", "train", {"a": object()})):
            try:
                bad()
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError("expected an error")
        assert not s.registry.exists() or "r" not in R.read_registry(s)   # nothing half-written is indexed
        R.save_results(s, "r", "train", {})
        try:
            R.save_results(s, "r", "x", {}, kind="reference")
        except ValueError:
            pass
        else:
            raise AssertionError("kind change must fail")
        try:
            R.get_metric(s, "r", "train.nope")
        except KeyError as e:
            assert "available" in str(e)
        else:
            raise AssertionError("expected KeyError")


def test_numpy_values_and_nan():
    with tempfile.TemporaryDirectory() as tmp:
        s = _store(tmp)
        R.save_results(s, "n", "train", {"a": np.arange(3), "b": np.float32(0.5), "c": float("nan")},
                       headline={"b": np.float32(0.5), "c": float("nan")})
        assert R.get_metric(s, "n", "train.a") == [0, 1, 2]
        assert R.read_registry(s)["n"]["headline"]["train.b"] == 0.5


def _writer(args):
    root, i = args
    s = R.Store("toy", Path(root))
    for j in range(5):
        R.save_results(s, f"run{i}", f"s{j}", {"i": i, "j": j}, headline={"v": j})


def test_concurrent_writers():
    with tempfile.TemporaryDirectory() as tmp:
        with mp.get_context("spawn").Pool(4) as pool:
            pool.map(_writer, [(tmp, i) for i in range(4)])
        s = _store(tmp)
        reg = R.read_registry(s)   # raises if any line is torn
        assert sorted(reg) == [f"run{i}" for i in range(4)]
        assert all(len(e["sections"]) == 5 for e in reg.values())
        assert len(s.registry.read_text().splitlines()) == 20


def test_corrupt_registry_is_reported():
    with tempfile.TemporaryDirectory() as tmp:
        s = _store(tmp)
        R.save_results(s, "r", "train", {})
        with open(s.registry, "a") as f:
            f.write("{not json\n")
        try:
            R.read_registry(s)
        except ValueError as e:
            assert "registry.jsonl:2" in str(e)
        else:
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests OK")
