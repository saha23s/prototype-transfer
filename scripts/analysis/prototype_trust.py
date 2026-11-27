#!/usr/bin/env python3
"""Prototype-trust diagnostics on the pretrained model and the non-test soundscape clips (docs/TRUST_DIAGNOSTIC.md).

  agreement     CPU, seconds: label-free reliability from agreement among a species' prototypes (cached responses).
  extract       model forward (CPU or GPU) on the clips that hold each prototype's top matches; for every prototype peak,
                the activations of all 105 prototypes at that grid cell -> outputs/features/<dataset>/pretrained-xcl/
                trust_matches.pt. Resumes from its partial file.
  competition   applies the pre-set criteria to those matches, using the call boxes only to evaluate.
  controls      exploratory checks added after that result: does the rival need to be at the same location? (does not
                change the decision)

Results are saved to the results store as summaries trust_agreement, trust_competition and trust_controls.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from prototype_transfer.analysis import trust
from prototype_transfer.data.datasets import HSNInferenceDataset
from prototype_transfer.data.splits import site_of, split_manifest
from prototype_transfer.evaluation import results
from prototype_transfer.models.classifier import HSN_SPECIES_LIST, HSNProtoPNetClassifier

logger = logging.getLogger("prototype_trust")
CLIP_NAME = re.compile(r"^(.*)_(\d{3})_(\d{3})\.ogg$")   # <recording>_<start s>_<end s>.ogg
PEAK_TOL = 2e-3   # recomputed maxima must match the cached ones (CPU fp32 vs the cached run)


def pooled_responses(store: results.Store) -> dict:
    """The pretrained model's cached soundscape responses on all non-test clips (the three held-out folds)."""
    folds = [torch.load(store.features("pretrained-xcl") / f"responses_cv{f}.pt", map_location="cpu", weights_only=False)
             for f in range(3)]
    out = {k: np.concatenate([f["soundscape"][k] for f in folds]) for k in ("proto_max", "targets")}
    out["audio_path"] = [p for f in folds for p in f["soundscape"]["audio_path"]]
    out["weights"], out["species"] = np.asarray(folds[0]["last_layer_weights"]), folds[0]["species_list"]
    assert out["species"] == HSN_SPECIES_LIST
    assert len({Path(p).name for p in out["audio_path"]}) == len(out["audio_path"]), "a clip appears twice"
    return out


def non_test_manifest(splits_dir: Path) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(split_manifest(splits_dir, "soundscape", part)) for part in ("train", "val")],
                   ignore_index=True)
    return df.set_index(df.audio_path.map(lambda q: Path(q).name), drop=False)


def top_matches(acts: np.ndarray, targets: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    """(species, prototype, indices of its TOP_MATCHES highest-scoring clips) for species with enough positives."""
    return [(s, k, np.argsort(-acts[:, s, k], kind="stable")[:trust.TOP_MATCHES])
            for s in range(acts.shape[1]) if targets[:, s].sum() >= trust.MIN_POSITIVES for k in range(acts.shape[2])]


def extract(args: argparse.Namespace, store: results.Store) -> None:
    pooled = pooled_responses(store)
    manifest = non_test_manifest(args.splits_dir)
    names = [Path(p).name for p in pooled["audio_path"]]
    assert set(names) == set(manifest.index), "cached clips are not the non-test soundscape clips"
    need = sorted({int(i) for _, _, idx in top_matches(pooled["proto_max"], pooled["targets"]) for i in idx})
    if args.max_clips:
        need = need[:args.max_clips]
    out_path = store.features("pretrained-xcl") / "trust_matches.pt"
    done = torch.load(out_path, weights_only=False) if out_path.exists() and not args.max_clips else {}
    todo = [i for i in need if names[i] not in done]
    logger.info(f"{len(need)} clips hold the top matches; {len(todo)} still to run")
    tmp = manifest.loc[[names[i] for i in todo]].reset_index(drop=True)
    list_path = out_path.with_name("trust_clips.parquet")
    tmp.to_parquet(list_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HSNProtoPNetClassifier(device=device)
    loader = DataLoader(HSNInferenceDataset(list_path, is_soundscape=True), batch_size=args.batch_size,
                        num_workers=args.num_workers)
    seen, worst = 0, 0.0
    for b, batch in enumerate(loader):
        r = model.prototype_responses(batch["spectrogram"])
        heat = r["heatmaps"].flatten(1, 2).cpu()                         # [B, 105, H, W]
        B, P, H, W = heat.shape
        flat = heat.reshape(B, P, H * W)
        peak, where = flat.max(dim=-1)                                   # [B, 105]
        # at the peak cell of prototype p (rows), the activation of every prototype q (columns): [B, 105, 105]
        at_peak = torch.gather(flat, 2, where[:, None, :].expand(B, P, P)).transpose(1, 2)
        for j in range(B):
            i = todo[seen + j]
            worst = max(worst, float(np.abs(peak[j].numpy() - pooled["proto_max"][i].reshape(-1)).max()))
            done[names[i]] = {"cell": where[j].numpy().astype(np.int16), "at_peak": at_peak[j].numpy().astype(np.float16)}
        assert worst < PEAK_TOL, f"recomputed activations differ from the cached ones by {worst:.2e}"
        seen += B
        if not args.max_clips and (b % 25 == 24 or seen == len(todo)):
            torch.save(done, out_path)
            logger.info(f"{seen}/{len(todo)} clips (max difference from cached maxima {worst:.1e})")
    logger.info(f"done: {len(done)} clips -> {out_path}; max difference from cached maxima {worst:.1e}")


def boxes_by_clip(events: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, dict[int, np.ndarray]]:
    """clip name -> species index -> [n, 4] boxes (t_low, t_high clip-relative s, f_low, f_high Hz)."""
    rec = {r: g for r, g in events.groupby(events.index.map(lambda f: Path(str(f)).name))}
    out: dict[str, dict[int, np.ndarray]] = {}
    for name in manifest.index:
        stem, a, b = CLIP_NAME.match(name).groups()
        g = rec.get(stem + ".ogg")
        assert g is not None, f"no annotated recording for {name}"
        g = g[(g.end_time > float(a)) & (g.start_time < float(b))]
        out[name] = {}
        for code, gg in g.groupby("ebird_code"):
            if code in HSN_SPECIES_LIST:
                out[name][HSN_SPECIES_LIST.index(code)] = np.stack(
                    [gg.start_time - float(a), gg.end_time - float(a), gg.low_freq, gg.high_freq], axis=1).astype(float)
    return out


def match_table(pooled: dict, extracted: dict, boxes: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    names = [Path(p).name for p in pooled["audio_path"]]
    labels = [set(HSN_SPECIES_LIST.index(c) for c in manifest.loc[n, "hsn_classes"]) for n in names]
    sites = site_of(pd.Series(names)).to_numpy()
    best = {n: pooled["proto_max"][i].max(axis=1) for i, n in enumerate(names)}   # [21] best activation anywhere in the clip
    rows, empty = [], np.zeros((0, 4))
    for s, k, idx in top_matches(pooled["proto_max"], pooled["targets"]):
        p = s * 5 + k
        cells = []
        for i in idx:
            e = extracted[names[i]]
            row, col = divmod(int(e["cell"][p]), trust.GRID_TIME)
            here = labels[i] | set(boxes[names[i]])
            sims = e["at_peak"][p].astype(float).reshape(21, 5).max(axis=1)      # best prototype of each species here
            own = float(e["at_peak"][p][p])
            rivals = sims.copy()
            rivals[s] = -np.inf
            j = int(rivals.argmax())
            window = trust.cell_window(row, col)
            present = s in here
            hit = trust.overlaps(window, boxes[names[i]].get(s, empty))
            cells.append((window, present))
            present_species = np.zeros(21, dtype=bool)
            present_species[list(here)] = True
            rows.append({"species": HSN_SPECIES_LIST[s], "prototype": k, "clip": names[i], "site": sites[i], "own": own,
                         "margin": float(rivals[j] - own), "rival_species": HSN_SPECIES_LIST[j], "rival_index": j,
                         "rival_present": j in here, "present_species": present_species,
                         "rival_anywhere": float(np.delete(best[names[i]], s).max()),
                         "label": trust.label_match(present, hit), "in_own_box": hit})
        # chance of a hit: this prototype's match cells (of clips with the species) shuffled among those clips
        with_species = [(i, w) for i, (w, present) in zip(idx, cells) if present]
        windows = [w for _, w in with_species]
        chance = {names[i]: float(np.mean([trust.overlaps(w, boxes[names[i]].get(s, empty)) for w in windows]))
                  for i, _ in with_species}
        for r in rows[-len(idx):]:
            r["chance_in_box"] = chance.get(r["clip"], np.nan)
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["agreement", "extract", "competition", "controls"])
    p.add_argument("--dataset", default=results.DEFAULT_DATASET)
    p.add_argument("--splits-dir", type=Path, default=Path("data/HSN/splits"))
    p.add_argument("--events", type=Path, default=Path("data/HSN/raw/metadata/HSN_metadata_test.parquet"))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--max-clips", type=int, default=None, help="debug: run only this many clips and save nothing")
    args = p.parse_args()
    store = results.Store(args.dataset)
    if args.step == "extract":
        extract(args, store)
        return
    pooled = pooled_responses(store)
    if args.step == "agreement":
        out = trust.agreement(pooled["proto_max"], pooled["targets"], pooled["weights"], HSN_SPECIES_LIST)
        headline = {k: out[k] for k in ("mean_estimated_r", "mean_true_r")}
    else:
        manifest = non_test_manifest(args.splits_dir)
        extracted = torch.load(store.features("pretrained-xcl") / "trust_matches.pt", weights_only=False)
        table = match_table(pooled, extracted, boxes_by_clip(pd.read_parquet(args.events), manifest), manifest)
        if args.step == "controls":
            out = trust.location_controls(table, n_boot=args.n_boot)
            headline = {k: out["auc"][k] for k in ("margin_same_place", "margin_anywhere", "own_low")}
        else:
            out = trust.evaluate_competition(table, n_boot=args.n_boot)
            headline = {"contested_false": out["point"]["c1"], "difference": out["point"]["c2"], "margin_auc": out["point"]["c3"]}
    out = {**out, **results.provenance()}
    entry = results.save_summary(store, f"trust_{args.step}", out, headline=headline,
                                 entry={"note": "pretrained model, non-test soundscape clips (docs/TRUST_DIAGNOSTIC.md)"})
    if args.step == "competition":
        logger.info(f"passed {out['passed']} -> {out['decision']}")
    logger.info(f"saved {entry['paths']['summary']}")


if __name__ == "__main__":
    main()
