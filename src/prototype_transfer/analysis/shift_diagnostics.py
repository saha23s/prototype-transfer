"""Stage 2 diagnostics for the PRETRAINED AudioProtoPNet (inference only, no training).

Tasks (run in this order; each writes a cache that scripts/analysis/analyze_diagnostics.py reads):
  extract    pretrained responses on soundscape-train (label-free: no targets are stored), focal-train,
             focal-test and soundscape-test (labeled, for evaluation)
             -> outputs/features/hsn/pretrained-xcl/diag_responses.pt
             and the label-free background manifest (lowest --bg-quantile of max species probability on
             soundscape-train) -> data/HSN/splits/soundscape_background.parquet
  synthetic  corrupt focal-test clips with real soundscape background, an overlaid second species, or both,
             at several levels                                  -> outputs/features/hsn/pretrained-xcl/diag_synthetic.pt
  renorm     soundscape-test responses with the input normalisation re-estimated on soundscape-train
                                                                -> outputs/features/hsn/pretrained-xcl/diag_renorm.pt
Soundscape labels are never used to choose anything here; they are stored only for test-split evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from prototype_transfer.data.datasets import target_matrix, wav_loader
from prototype_transfer.data.preprocessing import AudioPreprocessor
from prototype_transfer.method.mixing import mix_at_snr, peak_guard
from prototype_transfer.models.classifier import HSNProtoPNetClassifier

logger = logging.getLogger(__name__)

SPLITS = Path("data/HSN/splits")
# --eval-split: the labeled (focal, soundscape) pair that is extracted / corrupted / scored, and the file-name suffix.
# test is the committed default; val (soundscape_val, focal_val) lets the diagnostics run without touching test.
EVAL_SETS = {"test": ("focal_test", "soundscape_test", ""), "val": ("focal_val", "soundscape_val", "_val")}
BG_SNR_DB = (20, 10, 5, 0, -5)      # background level: SNR of the clip over the added background
OVERLAP_REL_DB = (-10, -5, 0)       # level of the overlaid second species relative to the clip
BOTH = ((10, -5), (0, -5))          # (background SNR, overlap level) combinations


def to_device(pre: AudioPreprocessor, device: str) -> AudioPreprocessor:
    """Run the spectrogram transforms on the GPU (same operations as AudioPreprocessor.transform_to_spectrogram)."""
    for m in (pre.spec_transform, pre.mel_scale, pre.db_scale):
        m.to(device)
    return pre


def spec_of(pre: AudioPreprocessor, wav: torch.Tensor, mean: float | None = None, std: float | None = None) -> torch.Tensor:
    """[B, 1, T] waveforms -> standardised log-mel [B, 1, 256, 626].

    Kept 4-D on purpose: torchaudio's AmplitudeToDB applies top_db relative to the max over the last three
    dims, so a 3-D [B, mel, time] input would clamp against the whole batch instead of each clip."""
    db = pre.db_scale(pre.mel_scale(pre.spec_transform(wav)))  # [B, 1, 256, 626]
    return (db - (pre.mean if mean is None else mean)) / (pre.std if std is None else std)


@torch.no_grad()
def respond(model: HSNProtoPNetClassifier, pre: AudioPreprocessor, wav: torch.Tensor) -> dict[str, np.ndarray]:
    """probs (B,21), proto_max (B,21,5), peak_freq (B,21,5) uint8 for a batch of waveforms (B,1,160000)."""
    spec = spec_of(pre, wav.to(model.device))  # [B, 1, 256, 626]
    out = model(spec, return_heatmaps=True)
    heat = out["prototype_activations"]  # [B, 21, 5, 16, 38]
    W = heat.shape[-1]
    return {"probs": out["probs"].cpu().numpy(), "proto_max": out["prototype_max_activations"].cpu().numpy(),
            "peak_freq": (heat.flatten(-2).argmax(-1) // W).cpu().numpy().astype(np.uint8)}


def extract_set(model, pre, df: pd.DataFrame, is_sc: bool, labeled: bool, bs: int, workers: int, desc: str) -> dict[str, Any]:
    acc: dict[str, list] = {"probs": [], "proto_max": [], "peak_freq": []}
    for wav in tqdm(wav_loader(df, is_sc, bs, workers), desc=desc):
        for k, v in respond(model, pre, wav).items():
            acc[k].append(v)
    out: dict[str, Any] = {k: np.concatenate(v) for k, v in acc.items()}
    out["audio_path"] = df["audio_path"].tolist()
    if labeled:
        out["targets"] = target_matrix(df, is_sc)
    return out


def read_split(name: str, max_n: int | None) -> pd.DataFrame:
    df = pd.read_parquet(SPLITS / f"{name}.parquet").reset_index(drop=True)
    return df.sample(n=min(max_n, len(df)), random_state=0).reset_index(drop=True) if max_n else df


# ---------------------------------------------------------------------------
def task_extract(args, model, pre) -> None:
    out: dict[str, Any] = {"species": model.species_list, "seed": args.seed,
                           "last_layer_weights": model.hsn_last_layer_weights().detach().cpu().numpy(),  # [21, 5]
                           "bias": model.head.last_layer.bias[model.target_class_tensor].detach().cpu().numpy()}  # [21]
    f_eval, s_eval, sfx = EVAL_SETS[args.eval_split]
    for name, is_sc, labeled in (("soundscape_train", True, False), ("focal_train", False, True),
                                 (f_eval, False, True), (s_eval, True, True)):
        out[name] = extract_set(model, pre, read_split(name, args.max_n), is_sc, labeled, args.batch_size, args.num_workers, name)
    # label-free background manifest: soundscape-train clips the pretrained model finds empty
    sc = read_split("soundscape_train", args.max_n)
    max_prob = out["soundscape_train"]["probs"].max(axis=1)
    keep = max_prob <= np.quantile(max_prob, args.bg_quantile)
    bg = sc.loc[keep, ["audio_path"]].assign(max_prob=max_prob[keep]).reset_index(drop=True)
    if not args.max_n:
        bg.to_parquet(SPLITS / "soundscape_background.parquet", index=False)
    # DIAGNOSTIC ONLY (not used to choose anything): how many selected clips have no annotated species
    empty = sc["hsn_classes"].map(len).eq(0).to_numpy()
    out["background_check"] = {"n_selected": int(keep.sum()), "threshold": float(np.quantile(max_prob, args.bg_quantile)),
                               "frac_selected_empty": float(empty[keep].mean()), "frac_all_empty": float(empty.mean())}
    logger.info(f"background selection: {out['background_check']}")
    torch.save(out, args.out_dir / f"diag_responses{sfx}.pt")


def mix_snr_atten(x: torch.Tensor, b: torch.Tensor, snr_db: float) -> torch.Tensor:
    """g*x + b with g set so that power(g*x) / power(b) = snr_db: the background keeps its natural level and the
    bird is made quieter (a distant bird over a fixed noise floor). x, b: [B, 1, T]."""
    px, pb = x.pow(2).mean((-2, -1), keepdim=True).clamp_min(1e-12), b.pow(2).mean((-2, -1), keepdim=True)
    return x * torch.sqrt(pb * 10 ** (snr_db / 10) / px) + b


def task_synthetic(args, model, pre) -> None:
    rng = np.random.default_rng(args.seed)
    f_eval, _, sfx = EVAL_SETS[args.eval_split]
    focal = read_split(f_eval, args.max_n)
    bg_df = pd.read_parquet(SPLITS / "soundscape_background.parquet") if not args.max_n else \
        read_split("soundscape_train", args.max_n)
    ov_all = read_split("focal_train", args.max_n)
    bg_df = bg_df.sample(n=min(args.pool, len(bg_df)), random_state=args.seed).reset_index(drop=True)
    ov_df = ov_all.sample(n=min(args.pool, len(ov_all)), random_state=args.seed).reset_index(drop=True)
    load = lambda df, sc, d, win="peak_energy": torch.cat(list(tqdm(wav_loader(df, sc, 32, args.num_workers, win, args.seed), desc=d)))  # [N, 1, T]
    x, bgw, ovw = load(focal, False, f"{f_eval} wav", args.focal_window), load(bg_df, True, "background wav"), load(ov_df, False, "overlay wav")
    x_peak = x if args.focal_window == "peak_energy" else load(focal, False, f"{f_eval} wav (peak_energy)")
    sp_x, sp_ov = focal["target_class"].to_numpy(), ov_df["target_class"].to_numpy()
    bg_idx = rng.integers(0, len(bgw), len(x))
    ov_idx = np.array([rng.choice(np.nonzero(sp_ov != s)[0]) for s in sp_x])  # overlay a DIFFERENT species

    conds: list[tuple[str, float | None, float | None]] = [("clean", None, None)]
    if args.focal_window != "peak_energy":
        conds.append(("clean_peak", None, None))  # the same clips with the loudest-window selection, for direct comparison
    conds += [(f"bg_snr{s:g}", s, None) for s in args.bg_snrs] + [(f"ov_rel{r:g}", None, r) for r in args.overlap_rels]
    conds += [(f"both_snr{s:g}_rel{r:g}", s, r) for s, r in zip(args.both_snrs, args.both_rels, strict=True)]
    res: dict[str, list] = {"probs": [], "proto_max": [], "peak_freq": []}
    guard_frac, measured = [], []
    for name, snr, rel in conds:
        parts: dict[str, list] = {k: [] for k in res}
        n_guard, snr_meas = 0, []
        for i in tqdm(range(0, len(x), args.batch_size), desc=name):
            sl = slice(i, i + args.batch_size)
            xb = (x_peak if name == "clean_peak" else x)[sl].to(model.device)
            m = xb
            if snr is not None:
                bb = bgw[bg_idx[sl]].to(model.device)
                m = mix_at_snr(m, bb, snr) if args.mix_mode == "add" else mix_snr_atten(m, bb, snr)
            if rel is not None:  # second species placed `rel` dB relative to the clip
                ob = ovw[ov_idx[sl]].to(model.device)
                m = m + ob * torch.sqrt(xb.pow(2).mean((-2, -1), keepdim=True) / ob.pow(2).mean((-2, -1), keepdim=True).clamp_min(1e-12)) * 10 ** (rel / 20)
            if snr is not None and rel is None:  # achieved SNR of the background mix, before any peak guard
                sig, noi = (xb, m - xb) if args.mix_mode == "add" else (m - bb, bb)
                snr_meas.append(float((10 * torch.log10(sig.pow(2).mean((-2, -1)) / noi.pow(2).mean((-2, -1)).clamp_min(1e-12))).mean()))
            n_guard += int((m.abs().amax((-2, -1)) > 1).sum())
            m = peak_guard(m)
            for k, v in respond(model, pre, m).items():
                parts[k].append(v)
        for k in res:
            res[k].append(np.concatenate(parts[k]))
        guard_frac.append(n_guard / len(x)); measured.append(float(np.mean(snr_meas)) if snr_meas else None)
        logger.info(f"{name}: peak-guard on {guard_frac[-1]:.1%} of clips; measured SNR {measured[-1]}")
    torch.save({"conditions": [c[0] for c in conds], "snr_db": [c[1] for c in conds], "overlap_rel_db": [c[2] for c in conds],
                **{k: np.stack(v) for k, v in res.items()},  # [C, N, ...]
                "targets": target_matrix(focal, False), "audio_path": focal["audio_path"].tolist(),
                "bg_audio": bg_df["audio_path"].tolist(), "peak_guard_frac": guard_frac, "measured_snr_db": measured,
                "seed": args.seed, "focal_window": args.focal_window, "mix_mode": args.mix_mode}, args.out_dir / f"diag_synthetic{sfx}{args.tag}.pt")


@torch.no_grad()
def task_renorm(args, model, pre) -> None:
    sc_train = read_split("soundscape_train", args.max_n)
    n, s1, s2 = 0, 0.0, 0.0
    for wav in tqdm(wav_loader(sc_train, True, args.batch_size, args.num_workers), desc="soundscape-train dB stats"):
        db = pre.db_scale(pre.mel_scale(pre.spec_transform(wav.to(model.device)))).double()  # [B, 1, 256, 626] in dB
        n += db.numel(); s1 += float(db.sum()); s2 += float(db.pow(2).sum())
    mu = s1 / n; sd = (s2 / n - mu ** 2) ** 0.5
    logger.info(f"soundscape-train dB mean {mu:.3f} std {sd:.3f} (pretrained fixed: {pre.mean} / {pre.std})")
    pre_sc = to_device(AudioPreprocessor(mean=mu, std=sd), model.device)  # respond() reads pre.mean / pre.std
    _, s_eval, sfx = EVAL_SETS[args.eval_split]
    test = read_split(s_eval, args.max_n)
    out = extract_set(model, pre_sc, test, True, True, args.batch_size, args.num_workers, f"{s_eval} (renorm)")
    torch.save({"mean": mu, "std": sd, "default_mean": pre.mean, "default_std": pre.std, s_eval: out},
               args.out_dir / f"diag_renorm{sfx}.pt")


TASKS = {"extract": task_extract, "synthetic": task_synthetic, "renorm": task_renorm}
