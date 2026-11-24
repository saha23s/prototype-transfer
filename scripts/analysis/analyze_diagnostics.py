#!/usr/bin/env python3
"""Stage 2 analysis (CPU only) of the caches written by scripts/analysis/diagnose_shift.py.

2.0  consistency: the new extraction reproduces the committed pretrained numbers (model_comparison.json)
2.1  does a synthetic shift on FOCAL test clips reproduce the real per-prototype delta_k?
     delta_k = AUC(soundscape test) - AUC(focal test) of the prototype's max activation as a detector of its species
2.2  which label-free statistics of UNLABELED soundscape-train audio predict the real delta_k?
2.3  inference-only baselines: input-normalisation re-estimated on soundscape-train; zeroing drifting prototypes

Gate (plan, Stage 2): proceed to augmentation training only if Spearman(delta_k synthetic, delta_k real) > 0.5
for at least one corruption. Saves summaries/diagnostics_stage2[_val].json in the results store.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


from prototype_transfer.analysis.prototype_responses import grid_row_centers_khz
from prototype_transfer.data.splits import site_of
from prototype_transfer.evaluation import results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("analyze_diagnostics")
K = 5
FAIL, MIN_POS, PRIMARY_MIN_POS, GATE_RHO, N_BOOT = -0.10, 5, 30, 0.5, 1000


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def ap(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if y.sum() > 0 else float("nan")


def proto_auc(proto_max: np.ndarray, targets: np.ndarray, cols: list[int]) -> np.ndarray:
    """[len(cols) * 5] AUC of each prototype's max activation as a detector of its own species."""
    return np.array([auc(targets[:, c], proto_max[:, c, k]) for c in cols for k in range(K)])


def spear(a: np.ndarray, b: np.ndarray) -> float:
    m = ~(np.isnan(a) | np.isnan(b))
    return float(spearmanr(a[m], b[m])[0]) if m.sum() > 2 else float("nan")


def cluster_boot_rho(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> list[float]:
    """95% CI of Spearman(a, b), resampling species (clusters of 5 prototypes)."""
    n = len(a) // K
    rhos = []
    for _ in range(N_BOOT):
        idx = np.concatenate([np.arange(c * K, c * K + K) for c in rng.integers(0, n, n)])
        rhos.append(spear(a[idx], b[idx]))
    return [float(x) for x in np.nanpercentile(rhos, [2.5, 97.5])]


def logits_probs(proto_max: np.ndarray, w: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """probs = sigmoid(sum_k relu(w_ck) * a_ck * mask_ck + b_c); proto_max [N,21,5], w [21,5], b [21]."""
    m = np.ones_like(w) if mask is None else mask
    return 1 / (1 + np.exp(-((proto_max * np.maximum(w, 0) * m).sum(-1) + b)))


def macro(metric, y: np.ndarray, p: np.ndarray, cols: list[int]) -> float:
    return float(np.nanmean([metric(y[:, c], p[:, c]) for c in cols]))


def paired_boot(y, p0, p1, cols, sites, rng) -> dict:
    """Site-cluster bootstrap of macro AP / AUC difference (p1 - p0)."""
    uniq = np.unique(sites)
    by_site = {s: np.nonzero(sites == s)[0] for s in uniq}
    out = {"ap": [], "auc": []}
    for _ in range(N_BOOT):
        idx = np.concatenate([by_site[s] for s in rng.choice(uniq, len(uniq))])
        out["ap"].append(macro(ap, y[idx], p1[idx], cols) - macro(ap, y[idx], p0[idx], cols))
        out["auc"].append(macro(auc, y[idx], p1[idx], cols) - macro(auc, y[idx], p0[idx], cols))
    return {k: {"mean": float(np.nanmean(v)), "ci95": [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]} for k, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=results.DEFAULT_DATASET)
    p.add_argument("--features-dir", type=Path, default=None, help="default: outputs/features/<dataset>/pretrained-xcl/ (where diagnose_shift.py writes)")
    p.add_argument("--comparison", type=Path, default=None, help="default: the model_comparison summary")
    p.add_argument("--eval-split", choices=["test", "val"], default="test",
                   help="test: the committed Stage 2 analysis. val: the same analysis on soundscape_val + focal_val "
                        "(run diagnose_shift.py with --eval-split val first); default file names get a _val suffix")
    p.add_argument("--synthetic", default=None, help="synthetic cache in --features-dir (default diag_synthetic[_val].pt)")
    p.add_argument("--summary-name", default=None, help="default diagnostics_stage2[_val]")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    f_key, s_key, sfx = {"test": ("focal_test", "soundscape_test", ""), "val": ("focal_val", "soundscape_val", "_val")}[args.eval_split]
    args.synthetic = args.synthetic or f"diag_synthetic{sfx}.pt"
    store = results.Store(args.dataset)
    args.features_dir = args.features_dir or store.features("pretrained-xcl")
    args.comparison = args.comparison or store.summary("model_comparison")
    summary_name = args.summary_name or f"diagnostics_stage2{sfx}"
    ld = lambda n: torch.load(args.features_dir / n, map_location="cpu", weights_only=False)
    R, S, N = ld(f"diag_responses{sfx}.pt"), ld(args.synthetic), ld(f"diag_renorm{sfx}.pt")
    cmp = json.loads(args.comparison.read_text())
    sp = R["species"]
    ft, st, sr = R[f_key], R[s_key], R["soundscape_train"]
    assert (S["targets"] == ft["targets"]).all(), "synthetic and extract use different focal-test clips"
    npos = {"focal": ft["targets"].sum(0), "soundscape": st["targets"].sum(0)}
    cols = [c for c in range(len(sp)) if npos["focal"][c] >= MIN_POS and npos["soundscape"][c] >= MIN_POS]
    primary = [c for c in cols if npos["soundscape"][c] >= PRIMARY_MIN_POS]
    logger.info(f"{len(cols)} analysed species {[sp[c] for c in cols]}; {len(primary)} primary")
    res: dict = {"eval_split": args.eval_split, "species": [sp[c] for c in cols], "primary": [sp[c] for c in primary], "seed": args.seed}

    # ---- 2.0 consistency with the committed pretrained numbers
    a_f, a_s = proto_auc(ft["proto_max"], ft["targets"], cols), proto_auc(st["proto_max"], st["targets"], cols)
    m = cmp["models"]["pretrained-xcl"]
    same_species = args.eval_split == "test" and cmp["species"] == [sp[c] for c in cols]   # committed numbers are test-split only
    chk = {"same_species_as_committed": same_species}
    if same_species:
        chk |= {"max_abs_diff_proto_auc_focal": float(np.nanmax(np.abs(a_f - np.array(m["focal"]["proto_auc"])))),
                "max_abs_diff_proto_auc_soundscape": float(np.nanmax(np.abs(a_s - np.array(m["soundscape"]["proto_auc"])))),
                "max_abs_diff_class_ap_soundscape": float(np.nanmax(np.abs(
                    np.array([ap(st["targets"][:, c], st["probs"][:, c]) for c in cols]) - np.array(m["soundscape"]["per_species_ap"]))))}
    w, b = R["last_layer_weights"], R["bias"]
    chk["max_abs_diff_reconstructed_probs"] = float(np.abs(logits_probs(st["proto_max"], w, b) - st["probs"]).max())
    res["consistency"] = chk
    logger.info(f"2.0 consistency: {chk}")

    # ---- real delta_k
    d_real = a_s - a_f
    res["real"] = {"n_prototypes": int(len(d_real)), "median_delta": float(np.nanmedian(d_real)),
                   "n_fail": int((d_real < FAIL).sum()), "pooled_pos_act_median": {}}
    pos_med = lambda d: float(np.median(np.concatenate([d["proto_max"][d["targets"][:, c], c, :].ravel() for c in cols])))
    res["real"]["pooled_pos_act_median"] = {"focal": pos_med(ft), "soundscape": pos_med(st)}
    res["real"]["shrinkage_ratio"] = res["real"]["pooled_pos_act_median"]["soundscape"] / res["real"]["pooled_pos_act_median"]["focal"]

    # ---- 2.1 synthetic shift
    clean = S["conditions"].index("clean")
    syn = []
    d_alt = None  # real delta recomputed against the focal AUC under the SAME windowing as the corrupted clips
    if "clean_peak" in S["conditions"]:
        d_alt = a_s - proto_auc(S["proto_max"][clean], ft["targets"], cols)
        pm0 = lambda arr: float(np.median(np.concatenate([arr[ft["targets"][:, c], c, :].ravel() for c in cols])))
        res["window_effect"] = {"focal_window": S.get("focal_window"), "spearman_real_delta_peak_vs_window": spear(d_real, d_alt),
                                "n_fail_with_window_baseline": int((d_alt < FAIL).sum()), "median_delta_with_window_baseline": float(np.nanmedian(d_alt)),
                                "focal_class_macro_auc": {"peak_energy": macro(auc, ft["targets"], S["probs"][S["conditions"].index("clean_peak")], cols),
                                                          S.get("focal_window"): macro(auc, ft["targets"], S["probs"][clean], cols)},
                                "pos_activation_ratio_window_vs_peak": pm0(S["proto_max"][clean]) / pm0(S["proto_max"][S["conditions"].index("clean_peak")])}
    for i, name in enumerate(S["conditions"]):
        if name in ("clean", "clean_peak"):
            continue
        a_c = proto_auc(S["proto_max"][i], ft["targets"], cols)
        d_syn = a_c - proto_auc(S["proto_max"][clean], ft["targets"], cols)
        pm = lambda arr: float(np.median(np.concatenate([arr[ft["targets"][:, c], c, :].ravel() for c in cols])))
        syn.append({"condition": name, "spearman_vs_real": spear(d_syn, d_real),
                    "spearman_ci95": cluster_boot_rho(d_syn, d_real, rng), "mean_delta": float(np.nanmean(d_syn)),
                    "n_fail": int((d_syn < FAIL).sum()), "shrinkage_ratio": pm(S["proto_max"][i]) / pm(S["proto_max"][clean]),
                    "class_macro_auc": macro(auc, ft["targets"], S["probs"][i], cols),
                    "peak_guard_frac": S["peak_guard_frac"][i],
                    **({"spearman_vs_real_window_baseline": spear(d_syn, d_alt)} if d_alt is not None else {})})
    res["synthetic"] = syn
    res["real"]["class_macro_auc"] = {"focal": macro(auc, ft["targets"], ft["probs"], cols), "soundscape": macro(auc, st["targets"], st["probs"], cols)}
    best = max(syn, key=lambda r: np.nan_to_num(r["spearman_vs_real"], nan=-1))
    res["gate"] = {"threshold": GATE_RHO, "best_condition": best["condition"], "best_spearman": best["spearman_vs_real"],
                   "passed": bool(best["spearman_vs_real"] > GATE_RHO)}

    # ---- 2.2 label-free predictors from unlabeled soundscape-train audio
    row_khz = grid_row_centers_khz()
    pm_sr = sr["proto_max"][:, cols, :]  # [N, S, 5]
    pf_sr = sr["peak_freq"][:, cols, :]
    feats: dict[str, np.ndarray] = {k: np.full(len(cols) * K, np.nan) for k in
                                    ("fire_rate_gt0.1", "act_q50", "act_q90", "act_q99", "peak_khz_top5pct", "peak_khz_shift_vs_focal")}
    drift = np.zeros(len(cols) * K, dtype=bool)
    for si, c in enumerate(cols):
        for k in range(K):
            j = si * K + k
            a = pm_sr[:, si, k]
            top = a >= np.quantile(a, 0.95)
            sc_khz = float(np.median(row_khz[pf_sr[top, si, k]]))
            pos = R["focal_train"]["targets"][:, c]  # source labels are allowed
            f_khz = float(np.median(row_khz[R["focal_train"]["peak_freq"][pos, c, k]])) if pos.any() else np.nan
            feats["fire_rate_gt0.1"][j], feats["act_q50"][j] = (a > 0.1).mean(), np.quantile(a, 0.5)
            feats["act_q90"][j], feats["act_q99"][j] = np.quantile(a, 0.9), np.quantile(a, 0.99)
            feats["peak_khz_top5pct"][j], feats["peak_khz_shift_vs_focal"][j] = sc_khz, sc_khz - f_khz
            drift[j] = bool(f_khz > 1.0 and sc_khz < 0.5)  # a-priori rule: focal peak above 1 kHz, soundscape peak in the lowest band
    res["predictors"] = {k: {"spearman_vs_real_delta": spear(v, d_real), "ci95": cluster_boot_rho(v, d_real, rng)} for k, v in feats.items()}
    res["drift_rule"] = {"n_flagged": int(drift.sum()), "flagged": [f"{sp[cols[j // K]]} P{j % K + 1}" for j in np.nonzero(drift)[0]],
                         "mean_real_delta_flagged": float(np.nanmean(d_real[drift])) if drift.any() else None,
                         "mean_real_delta_rest": float(np.nanmean(d_real[~drift]))}

    # ---- 2.3 inference-only baselines on soundscape test (site-cluster bootstrap vs pretrained)
    sites = site_of(pd.Series([Path(q).name for q in st["audio_path"]])).to_numpy()
    y, p0 = st["targets"], st["probs"]
    mask = np.ones((len(sp), K))
    for j in np.nonzero(drift)[0]:
        mask[cols[j // K], j % K] = 0
    variants = {"renorm_input_stats": N[s_key]["probs"], "zero_drifting_prototypes": logits_probs(st["proto_max"], w, b, mask)}
    base = {"all_12": {"ap": macro(ap, y, p0, cols), "auc": macro(auc, y, p0, cols)},
            "primary": {"ap": macro(ap, y, p0, primary), "auc": macro(auc, y, p0, primary)}}
    res["baselines"] = {"pretrained": base, "renorm_stats": {"mean": N["mean"], "std": N["std"], "default_mean": N["default_mean"], "default_std": N["default_std"]}}
    for name, p1 in variants.items():
        res["baselines"][name] = {"all_12": {"ap": macro(ap, y, p1, cols), "auc": macro(auc, y, p1, cols)},
                                  "primary": {"ap": macro(ap, y, p1, primary), "auc": macro(auc, y, p1, primary)},
                                  "primary_vs_pretrained_boot": paired_boot(y, p0, p1, primary, sites, rng)}
    pa_ren = proto_auc(N[s_key]["proto_max"], st["targets"], cols)
    res["baselines"]["renorm_input_stats"]["n_fail"] = int(((pa_ren - a_f) < FAIL).sum())
    entry = results.save_summary(store, summary_name, res, entry={"note": "domain-shift diagnostics of the pretrained model"})

    # ---- printed summary
    print(f"\n2.1 synthetic shift on focal test (real: median delta {res['real']['median_delta']:+.3f}, {res['real']['n_fail']}/{len(d_real)} fail, "
          f"activation ratio {res['real']['shrinkage_ratio']:.2f}, class AUC focal {res['real']['class_macro_auc']['focal']:.3f} -> soundscape {res['real']['class_macro_auc']['soundscape']:.3f})")
    print(f"{'condition':22} {'rho':>6} {'95% CI':>16} {'mean d':>7} {'#fail':>5} {'act ratio':>9} {'class AUC':>9}")
    for r in syn:
        print(f"{r['condition']:22} {r['spearman_vs_real']:6.2f} [{r['spearman_ci95'][0]:5.2f},{r['spearman_ci95'][1]:5.2f}] {r['mean_delta']:+7.3f} {r['n_fail']:5d} {r['shrinkage_ratio']:9.2f} {r['class_macro_auc']:9.3f}")
    if "window_effect" in res:
        we = res["window_effect"]
        print(f"window effect ({we['focal_window']} vs peak_energy): focal class AUC {we['focal_class_macro_auc']}, activation ratio {we['pos_activation_ratio_window_vs_peak']:.2f}, "
              f"real delta vs window baseline: median {we['median_delta_with_window_baseline']:+.3f}, {we['n_fail_with_window_baseline']} fail, "
              f"rank agreement with the peak-energy delta {we['spearman_real_delta_peak_vs_window']:.2f}")
        print("rho vs real delta under the window baseline: " + "  ".join(f"{r['condition']}:{r['spearman_vs_real_window_baseline']:+.2f}" for r in syn))
    print(f"GATE (rho > {GATE_RHO}): {'PASS' if res['gate']['passed'] else 'FAIL'}  best {best['condition']} rho {best['spearman_vs_real']:.2f}")
    print("\n2.2 label-free predictors of real delta_k (Spearman, 95% species-bootstrap CI)")
    for k, v in res["predictors"].items():
        print(f"  {k:26} {v['spearman_vs_real_delta']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]")
    print(f"  drift rule flags {res['drift_rule']['n_flagged']}: mean real delta {res['drift_rule']['mean_real_delta_flagged']} vs rest {res['drift_rule']['mean_real_delta_rest']:+.3f}")
    print("\n2.3 baselines on soundscape test (macro AP / AUC, primary species; bootstrap diff vs pretrained)")
    print(f"  pretrained                 AP {base['primary']['ap']:.3f}  AUC {base['primary']['auc']:.3f}")
    for name in variants:
        v = res["baselines"][name]; d = v["primary_vs_pretrained_boot"]
        print(f"  {name:26} AP {v['primary']['ap']:.3f}  AUC {v['primary']['auc']:.3f}  dAP {d['ap']['mean']:+.3f} [{d['ap']['ci95'][0]:+.3f},{d['ap']['ci95'][1]:+.3f}]")
    print(f"consistency with committed pretrained results: {chk}\nSaved {entry['paths']['summary']}")


if __name__ == "__main__":
    main()
