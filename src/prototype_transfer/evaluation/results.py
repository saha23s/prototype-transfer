"""The results store: one place, one writer, one index, so a person or an agent can find any run's numbers.

Layout, per dataset (everything under `outputs/`, none of it committed to git)::

    outputs/results/<dataset>/
        registry.jsonl              index, one line per save; the LAST line of a run_id is its current entry
        runs/<run_id>/
            config.json             how the run was made: arguments, effective settings, git commit, environment
            metrics.json            {"sections": {<section>: {...}}}: every metric of the run, grouped by section
            train_log.csv           per-epoch training log (training runs only)
            <name>.json             other small per-run files (e.g. push.json, prototypes.json)
        summaries/<name>.json       results that span several runs (e.g. the model comparison)
        dataset/                    description of the data (split summaries)
    outputs/features/<dataset>/<run_id>/    heavy tensors (responses.pt, ...), not committed
    outputs/checkpoints/<dataset>/<run_id>/ model checkpoints (best.pt, ...), not committed

Every writer goes through `save_results` (or `save_summary` for cross-run results). Files are replaced atomically
(write to a temp file, then rename), so a reader never sees a half-written file, and the registry line is appended under a
file lock. The index entry holds the run's headline numbers, so `scripts/report/results.py list` answers most questions
without opening any other file. See docs/RESULTS.md for the schema and the query CLI.
"""

from __future__ import annotations

import csv
import io
import json
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None

SCHEMA = 1
DEFAULT_DATASET = "hsn"
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SUMMARY_PREFIX = "summary:"
KINDS = ("train", "reference", "summary", "analysis")
# index fields a caller may set through `entry=`; everything else is derived by the store
ENTRY_FIELDS = ("rung", "seed", "cv_fold", "status", "started", "finished", "commit", "dirty", "legacy", "note", "protocol")


@dataclass(frozen=True)
class Store:
    """Locations of everything a dataset's runs produce. Paths are relative to the working directory (the repo root)."""

    dataset: str = DEFAULT_DATASET
    root: Path = Path("outputs")

    @property
    def results(self) -> Path:
        return self.root / "results" / self.dataset

    @property
    def registry(self) -> Path:
        return self.results / "registry.jsonl"

    @property
    def summaries(self) -> Path:
        return self.results / "summaries"

    @property
    def dataset_dir(self) -> Path:
        return self.results / "dataset"

    def run_dir(self, run_id: str) -> Path:
        return self.results / "runs" / check_run_id(run_id)

    def features(self, run_id: str) -> Path:
        return self.root / "features" / self.dataset / check_run_id(run_id)

    def checkpoints(self, run_id: str) -> Path:
        return self.root / "checkpoints" / self.dataset / check_run_id(run_id)

    def summary(self, name: str) -> Path:
        return self.summaries / f"{name}.json"


def check_run_id(run_id: str) -> str:
    if not RUN_ID.match(run_id):
        raise ValueError(f"invalid run id {run_id!r}: use letters, digits, '.', '_' and '-' (no slashes, no leading punctuation)")
    return run_id


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def provenance() -> dict:
    """What produced a result: the git commit (and whether the tree had uncommitted changes) and library versions."""
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
    try:
        git = {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        git = {"commit": None, "dirty": None}
    env = {"python": platform.python_version()}
    try:
        import torch
        env["torch"] = torch.__version__
    except ImportError:
        pass
    return {**git, "env": env}


# ---------------------------------------------------------------------------------------------------------- file I/O
def atomic_write(path: Path, text: str) -> None:
    """Write `text` to `path` so that readers see either the old file or the whole new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=1, default=_json_default, allow_nan=True)


def _json_default(o: Any) -> Any:
    """Numpy / torch scalars and arrays, Paths: anything else is an error rather than a silent str()."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


class _Lock:
    """Advisory lock on `<results>/.lock`; best effort (a filesystem without flock support just skips it)."""

    def __init__(self, store: Store):
        self.path = store.results / ".lock"
        self.f = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a")
        if fcntl is not None:
            try:
                fcntl.flock(self.f, fcntl.LOCK_EX)
            except OSError:
                pass
        return self

    def __exit__(self, *exc):
        if self.f is not None:
            if fcntl is not None:
                try:
                    fcntl.flock(self.f, fcntl.LOCK_UN)
                except OSError:
                    pass
            self.f.close()


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from e


# ----------------------------------------------------------------------------------------------------------- registry
def read_registry(store: Store) -> dict[str, dict]:
    """run_id -> current index entry (the last line written for it). Empty if nothing was saved yet."""
    if not store.registry.exists():
        return {}
    out: dict[str, dict] = {}
    for i, line in enumerate(store.registry.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError as err:
            raise ValueError(f"{store.registry}:{i} is not valid JSON ({err}); fix or delete that line") from err
        out[e["run_id"]] = e
    return out


def _append_registry(store: Store, entry: dict) -> None:
    store.registry.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, default=_json_default, sort_keys=True, allow_nan=True) + "\n"
    fd = os.open(store.registry, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
    try:
        os.write(fd, line.encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def _num(v: Any) -> float | int | None:
    """Headline values must be plain numbers (NaN allowed); None passes through."""
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    if hasattr(v, "item"):
        return v.item()
    raise TypeError(f"headline values must be numbers, got {type(v).__name__}")


def log_csv(rows: list[dict]) -> str:
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


# -------------------------------------------------------------------------------------------------------------- write
def write_train_log(store: Store, run_id: str, rows: list[dict]) -> None:
    """Refresh train_log.csv during training (so a running job can be watched); save_results writes the final one."""
    atomic_write(store.run_dir(run_id) / "train_log.csv", log_csv(rows))


def save_results(store: Store, run_id: str, section: str, metrics: dict, *, kind: str = "train",
                 headline: dict[str, Any] | None = None, config: dict | None = None, log: list[dict] | None = None,
                 files: dict[str, Any] | None = None, entry: dict | None = None) -> dict:
    """Write one section of a run's results and update its registry entry. Returns the new entry.

    Args:
        store: where results live.
        run_id: the run (a trained model, or a reference such as `pretrained-xcl`); the same id across calls adds sections.
        section: name of this group of metrics in metrics.json, e.g. `train`, `canonical_test`, `canonical_cv0`.
            Saving a section again replaces it.
        metrics: JSON-serialisable dict, the full numbers of the section.
        kind: `train` (a model we trained) or `reference` (a model we only evaluate); `analysis` for diagnostics of a run.
        headline: flat {name: number} copied into the index as `<section>.<name>`; keep it to the numbers people look up first.
        config: how the run was made (written to config.json, replacing it).
        log: per-epoch records, written as train_log.csv (replacing it).
        files: {"push.json": obj, ...}, extra small JSON files written to the run directory.
        entry: index fields, any of ENTRY_FIELDS (rung, seed, cv_fold, status, commit, dirty, legacy, note, ...).
    """
    if kind not in KINDS or kind == "summary":
        raise ValueError(f"kind must be one of {[k for k in KINDS if k != 'summary']} (summaries use save_summary), got {kind!r}")
    if not re.match(r"^[A-Za-z0-9_]+$", section):
        raise ValueError(f"invalid section name {section!r}: letters, digits and '_' only")
    bad = set(entry or {}) - set(ENTRY_FIELDS)
    if bad:
        raise ValueError(f"unknown index fields {sorted(bad)}; allowed: {ENTRY_FIELDS}")
    rdir = store.run_dir(run_id)
    with _Lock(store):
        mpath = rdir / "metrics.json"
        doc = _read_json(mpath) if mpath.exists() else {"schema": SCHEMA, "run_id": run_id, "dataset": store.dataset, "sections": {}}
        doc["sections"][section] = metrics
        atomic_write(mpath, _dump(doc))
        if config is not None:
            atomic_write(rdir / "config.json", _dump({"schema": SCHEMA, "run_id": run_id, "dataset": store.dataset, **config}))
        if log is not None:
            atomic_write(rdir / "train_log.csv", log_csv(log))
        for fname, obj in (files or {}).items():
            if not fname.endswith(".json") or "/" in fname or fname in ("config.json", "metrics.json"):
                raise ValueError(f"extra run file must be a plain <name>.json (not config/metrics), got {fname!r}")
            atomic_write(rdir / fname, _dump(obj))
        old = read_registry(store).get(run_id, {})
        if old and old.get("kind") != kind:
            raise ValueError(f"run {run_id!r} is already registered as kind {old.get('kind')!r}, cannot save as {kind!r}")
        head = {k: v for k, v in old.get("headline", {}).items() if not k.startswith(f"{section}.")}
        head.update({f"{section}.{k}": _num(v) for k, v in (headline or {}).items()})
        paths = {"metrics": str(mpath)}
        for key, fname in (("config", "config.json"), ("train_log", "train_log.csv")):
            if (rdir / fname).exists():
                paths[key] = str(rdir / fname)
        extra = sorted(p.name for p in rdir.glob("*.json") if p.name not in ("config.json", "metrics.json"))
        if extra:
            paths["files"] = [str(rdir / n) for n in extra]
        for key, d in (("features", store.features(run_id)), ("checkpoints", store.checkpoints(run_id))):
            if d.exists():
                paths[key] = str(d)
        new = {**old, **{k: v for k, v in (entry or {}).items()}, "schema": SCHEMA, "run_id": run_id, "dataset": store.dataset,
               "kind": kind, "sections": sorted(set(old.get("sections", [])) | {section}), "headline": head, "paths": paths,
               "updated": now()}
        _append_registry(store, new)
    return new


def save_summary(store: Store, name: str, obj: dict, *, headline: dict[str, Any] | None = None, entry: dict | None = None) -> dict:
    """Write a result that spans several runs to summaries/<name>.json and index it as `summary:<name>`."""
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", name):
        raise ValueError(f"invalid summary name {name!r}")
    bad = set(entry or {}) - set(ENTRY_FIELDS)
    if bad:
        raise ValueError(f"unknown index fields {sorted(bad)}; allowed: {ENTRY_FIELDS}")
    path = store.summary(name)
    rid = SUMMARY_PREFIX + name
    with _Lock(store):
        atomic_write(path, _dump(obj))
        old = read_registry(store).get(rid, {})
        new = {**old, **(entry or {}), "schema": SCHEMA, "run_id": rid, "dataset": store.dataset, "kind": "summary",
               "sections": [], "headline": {k: _num(v) for k, v in (headline or {}).items()}, "paths": {"summary": str(path)},
               "updated": now()}
        _append_registry(store, new)
    return new


# --------------------------------------------------------------------------------------------------------------- read
def load_metrics(store: Store, run_id: str) -> dict:
    """metrics.json of a run (or the document of `summary:<name>`)."""
    if run_id.startswith(SUMMARY_PREFIX):
        return _read_json(store.summary(run_id[len(SUMMARY_PREFIX):]))
    path = store.run_dir(run_id) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"no metrics for run {run_id!r} in {store.results} (known runs: `python scripts/report/results.py list`)")
    return _read_json(path)


def get_metric(store: Store, run_id: str, path: str) -> Any:
    """One value by dotted path, e.g. get_metric(store, "r0", "canonical_test.point.sc_ap_primary").
    For a run the first component is the section; for a summary it indexes the summary document directly."""
    doc = load_metrics(store, run_id)
    node: Any = doc if run_id.startswith(SUMMARY_PREFIX) else doc["sections"]
    for i, part in enumerate(path.split(".")):
        try:
            node = node[int(part)] if isinstance(node, list) else node[part]
        except (KeyError, IndexError, ValueError):
            where = ".".join(path.split(".")[:i]) or "<top>"
            options = list(node)[:20] if isinstance(node, dict) else f"(a {type(node).__name__})"
            raise KeyError(f"{run_id}: no {part!r} under {where}; available: {options}") from None
    return node


def list_runs(store: Store, *, kind: str | None = None, rung: str | None = None, status: str | None = None,
              legacy: bool | None = None, section: str | None = None) -> Iterator[dict]:
    """Index entries, oldest saved first, optionally filtered. `legacy=False` hides backfilled pre-store results."""
    for e in sorted(read_registry(store).values(), key=lambda e: e.get("updated", "")):
        if kind and e.get("kind") != kind:
            continue
        if rung and e.get("rung") != rung:
            continue
        if status and e.get("status") != status:
            continue
        if legacy is not None and bool(e.get("legacy")) != legacy:
            continue
        if section and section not in e.get("sections", []):
            continue
        yield e
