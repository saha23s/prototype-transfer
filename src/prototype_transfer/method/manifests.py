"""Label-free manifests for shift-matched training, built from the pretrained model's responses on soundscape-train clips.

Only each clip's audio path and the teacher's class probabilities are read; no soundscape annotation is. From them:
  background manifest  the clips whose max species probability is in the lowest `quantile` (the model finds them empty);
                       columns audio_path, max_prob. Used by train_protopnet.py --bg-kind real.
  pseudo manifest      every clip with the teacher's probabilities, columns audio_path, teacher_probs (21 floats).
                       Used by train_protopnet.py --pseudo-manifest.
The selection rule is `mixing.select_background`. For cross-validation, `restrict_to_fold` keeps only the clips of the fold's training
sites, so no held-out audio is used for adaptation. The CLI is scripts/method/make_background_manifest.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from prototype_transfer.method.mixing import select_background


def load_responses(sources: list[str]) -> tuple[np.ndarray, list[str]]:
    """Concatenate the probabilities and audio paths of `FILE:KEY` entries of response files (as written by diagnose_shift.py).

    Raises:
        AssertionError: for a test-split entry (never used for adaptation), duplicate clips or misaligned arrays.
    """
    probs_l, paths = [], []
    for src in sources:
        file, key = src.rsplit(":", 1)
        assert not key.startswith("soundscape_test"), "the test split is never used for adaptation"
        entry = torch.load(file, map_location="cpu", weights_only=False)[key]
        probs_l.append(np.asarray(entry["probs"])); paths += list(entry["audio_path"])      # only probs and paths: annotations are never read
    probs = np.concatenate(probs_l)                                                          # [N, 21]
    assert len(probs) == len(paths) and len(set(paths)) == len(paths), "duplicate or misaligned clips"
    return probs, paths


def restrict_to_fold(probs: np.ndarray, paths: list[str], keep_paths: set[str], fold: int) -> tuple[np.ndarray, list[str]]:
    """Keep the clips of `keep_paths` (the manifest of the fold's training sites); every one of them must have responses."""
    missing = keep_paths - set(paths)
    assert not missing, f"{len(missing)} clips of soundscape_cv{fold}_train have no responses; add the val responses with --source"
    sel = np.array([q in keep_paths for q in paths])
    return probs[sel], [q for q, k in zip(paths, sel) if k]


def build_manifests(probs: np.ndarray, paths: list[str], quantile: float) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """(background manifest, pseudo-label manifest, the max-probability cut-off of the background clips)."""
    max_prob = probs.max(axis=1)
    keep = select_background(max_prob, quantile)
    bg = pd.DataFrame({"audio_path": np.array(paths)[keep], "max_prob": max_prob[keep]})
    pseudo = pd.DataFrame({"audio_path": paths, "teacher_probs": list(probs.astype(np.float32))})
    return bg, pseudo, float(np.quantile(max_prob, quantile))
