#!/usr/bin/env python3
"""Build dashboard/plan_data.js for dashboard/plan.html (problem, decisions, plan, analysis of current work).

Reads  outputs/results/model_comparison.json, outputs/results/splits.json, dashboard/plan_content.json
Writes dashboard/plan_data.js  (window.PLAN_DATA; a .js file so the page works from file://)

Only the analysis block is computed here; decisions / stages / criteria text live in plan_content.json.
Needs no GPU, features or audio.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_plan_data")

DOMAINS = ("focal", "soundscape")
PRETRAINED, FOCAL_ONLY, SC_ONLY = "pretrained-xcl", "focal-only", "soundscape-only"
LABELS = {PRETRAINED: "Pretrained XCL", FOCAL_ONLY: "Focal-only", SC_ONLY: "Soundscape-only (oracle)"}
PRIMARY_MIN_POS = 30   # primary evaluation species: >= this many soundscape test positives
FAIL, IMPROVE = -0.10, 0.02   # Δ_k thresholds used by the existing dashboard findings
GAP_FRACTION = 0.5     # success criterion: close this fraction of the pretrained -> oracle gap
FOCAL_TOL = 0.01       # success criterion: focal macro AUC within this of pretrained


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def r3(x: float) -> float:
    return round(float(x), 3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comparison", type=Path, default=Path("outputs/results/model_comparison.json"))
    p.add_argument("--splits", type=Path, default=Path("outputs/results/splits.json"))
    p.add_argument("--content", type=Path, default=Path("dashboard/plan_content.json"))
    p.add_argument("--out", type=Path, default=Path("dashboard/plan_data.js"))
    args = p.parse_args()

    cmp = json.loads(args.comparison.read_text())
    spl = json.loads(args.splits.read_text())
    content = json.loads(args.content.read_text())
    sp, M = cmp["species"], cmp["models"]
    K = 5  # prototypes per species; model_comparison stores proto_auc species-major
    names = [PRETRAINED, FOCAL_ONLY, SC_ONLY]

    # ---- species positives and primary set
    npos = {d: [cmp["n_pos"][d][s] for s in sp] for d in DOMAINS}
    primary = [i for i, s in enumerate(sp) if npos["soundscape"][i] >= PRIMARY_MIN_POS]
    species = [{"code": s, "n_focal": npos["focal"][i], "n_soundscape": npos["soundscape"][i],
                "primary": i in primary} for i, s in enumerate(sp)]

    # ---- class-level scores on the primary set (mean of per-species AUC, as compare_models does for all 12)
    def macro(name: str, dom: str, idx: list[int], key: str = "per_species_auc") -> float:
        return float(np.mean([M[name][dom][key][i] for i in idx]))

    allidx = list(range(len(sp)))
    class_level = [{"name": n, "label": LABELS[n],
                    **{f"{d}_auc_primary": r3(macro(n, d, primary)) for d in DOMAINS},
                    **{f"{d}_auc_all": r3(macro(n, d, allidx)) for d in DOMAINS},
                    **{f"{d}_ap_primary": r3(macro(n, d, primary, "per_species_ap")) for d in DOMAINS}}
                   for n in names]

    # ---- per-prototype delta
    auc = {n: {d: np.array(M[n][d]["proto_auc"]) for d in DOMAINS} for n in names}
    delta = {n: auc[n]["soundscape"] - auc[n]["focal"] for n in names}
    protos = [{"species": sp[j // K], "k": j % K + 1,
               "delta": {n: r3(delta[n][j]) for n in names},
               "sc_auc": {n: r3(auc[n]["soundscape"][j]) for n in names},
               "focal_auc": {n: r3(auc[n]["focal"][j]) for n in names}} for j in range(len(sp) * K)]
    counts = [{"name": n, "label": LABELS[n], "fail": int((delta[n] < FAIL).sum()),
               "improve": int((delta[n] > IMPROVE).sum()), "total": len(delta[n]),
               "median_delta": r3(np.median(delta[n]))} for n in names]
    rho = {"pretrained_vs_focal_only": r3(spearman(delta[PRETRAINED], delta[FOCAL_ONLY])),
           "pretrained_vs_soundscape_only": r3(spearman(delta[PRETRAINED], delta[SC_ONLY]))}
    failing = delta[PRETRAINED] < FAIL
    groups = [{"label": f"{int(m.sum())} prototypes that failed under pretrained (Δ < {FAIL})" if lab == "fail"
               else f"{int(m.sum())} other prototypes",
               "n": int(m.sum()),
               "sc_auc": {n: r3(auc[n]["soundscape"][m].mean()) for n in names}}
              for lab, m in (("fail", failing), ("rest", ~failing))]

    # ---- prevalence (shows the cmAP gap is not a base-rate artefact)
    n_clips = {d: spl[d]["test"]["n_clips"] for d in DOMAINS}
    prev = {d: r3(np.mean([x / n_clips[d] for x in npos[d]])) for d in DOMAINS}
    low = [s["code"] for s in species if s["n_soundscape"] < PRIMARY_MIN_POS]

    # ---- splits table
    splits = []
    for d in DOMAINS:
        for part in ("train", "val", "test"):
            e = spl[d][part]
            groups_n = e.get("recordists") if d == "focal" else len(spl["soundscape"]["sites"][part])
            acc = content["splits_access"][f"{d}/{part}"]
            splits.append({"domain": d, "split": part, "n_clips": e["n_clips"],
                           "group_unit": "recordists" if d == "focal" else "sites", "n_groups": groups_n,
                           "background": e.get("background_clips"), "audio": acc["audio"], "labels": acc["labels"],
                           "role": content["splits_roles"][f"{d}/{part}"]})

    # ---- per-species positives for every split (all 21 species), primary species flagged
    all_sp = list(spl["focal"]["train"]["positives"])
    sp_split = [{"code": c, "primary": c in [sp[i] for i in primary],
                 "focal": [spl["focal"][k]["positives"][c] for k in ("train", "val", "test")],
                 "soundscape": [spl["soundscape"][k]["positives"][c] for k in ("train", "val", "test")]} for c in all_sp]
    sp_split.sort(key=lambda r: (-int(r["primary"]), -r["soundscape"][2], -r["focal"][2]))
    sc_val0 = [r["code"] for r in sp_split if r["soundscape"][1] == 0]
    sc_test0 = [r["code"] for r in sp_split if r["soundscape"][2] == 0]
    tot_tr = sum(r["soundscape"][0] for r in sp_split)
    top4 = sorted(sp_split, key=lambda r: -r["soundscape"][0])[:4]
    split_notes = [
        f"Soundscape val has no positives for {len(sc_val0)} of {len(all_sp)} species, so it cannot be used for selection.",
        f"{len(sc_test0)} species have no soundscape test positives ({', '.join(sc_test0)}); they can only be measured on focal clips.",
        f"Four species make up {sum(r['soundscape'][0] for r in top4) / tot_tr:.0%} of soundscape-train positives ({', '.join(r['code'] for r in top4)}), "
        f"so unlabeled soundscape train audio is dominated by them, which matters for pseudo-labeling.",
    ]

    # ---- success criteria with the numbers they imply
    pre, orc = class_level[0], class_level[2]
    ap0, ap1 = pre["soundscape_ap_primary"], orc["soundscape_ap_primary"]
    auc0, auc1 = pre["soundscape_auc_primary"], orc["soundscape_auc_primary"]
    criteria = [
        {"text": f"Soundscape macro AP on the {len(primary)} primary species closes at least {int(GAP_FRACTION * 100)}% of the gap "
                 f"between pretrained and the oracle, with a site-bootstrap CI that excludes zero improvement. (AP is the headline: "
                 f"AUC is nearly saturated on these species.)",
         "baseline": f"pretrained {ap0:.3f} \u2192 target \u2265 {ap0 + GAP_FRACTION * (ap1 - ap0):.3f} (oracle {ap1:.3f})"},
        {"text": "Soundscape macro AUC on the primary species does not drop below pretrained (secondary).",
         "baseline": f"pretrained {auc0:.3f} (oracle {auc1:.3f}; gap only {auc1 - auc0:.3f})"},
        {"text": f"Prototypes with \u0394_k < {FAIL} drop to at most half of the pretrained count.",
         "baseline": f"pretrained {counts[0]['fail']}/{counts[0]['total']} \u2192 target \u2264 {counts[0]['fail'] // 2}"},
        {"text": f"Focal macro AUC (primary species) stays within {FOCAL_TOL} of pretrained.",
         "baseline": f"pretrained {pre['focal_auc_primary']:.3f} \u2192 target \u2265 {pre['focal_auc_primary'] - FOCAL_TOL:.3f}"},
    ]

    g_fail, g_rest = groups
    findings = [
        {"title": "Failures are inherited from pretraining",
         "text": f"{counts[0]['fail']} of {counts[0]['total']} prototypes lose more than {abs(FAIL)} AUC on soundscapes under the pretrained model. "
                 f"Focal fine-tuning keeps the same ones failing (Spearman of Δ_k {rho['pretrained_vs_focal_only']}), while soundscape-only "
                 f"changes them ({rho['pretrained_vs_soundscape_only']})."},
        {"title": "Focal-only fine-tuning also damages prototypes that were fine",
         "text": f"The {g_rest['n']} prototypes that did not fail drop from {g_rest['sc_auc'][PRETRAINED]} to {g_rest['sc_auc'][FOCAL_ONLY]} soundscape AUC "
                 f"after focal fine-tuning, and its failing count rises to {counts[1]['fail']}. Any method must stay anchored to the pretrained weights."},
        {"title": "Soundscape labels repair most, not all, of the failures",
         "text": f"The {g_fail['n']} failing prototypes recover from {g_fail['sc_auc'][PRETRAINED]} to {g_fail['sc_auc'][SC_ONLY]} soundscape AUC "
                 f"under soundscape-only, still below the other prototypes ({g_rest['sc_auc'][SC_ONLY]}). Expect a ceiling for a label-free method."},
        {"title": "The headline metric is noisy for rare species",
         "text": f"{len(low)} of {len(sp)} species have fewer than {PRIMARY_MIN_POS} soundscape test positives ({', '.join(low)}). "
                 f"The 12-species means have no error bars yet, so the primary evaluation uses the {len(primary)} species above the threshold."},
        {"title": "On the primary species AUC is nearly saturated; AP discriminates",
         "text": f"Soundscape macro AUC is {auc0} (pretrained) vs {auc1} (oracle), a gap of {auc1 - auc0:.3f}. Macro AP is {ap0} vs {ap1}, so AP is the headline metric for the success criteria."},
        {"title": "Prevalence is not the confound",
         "text": f"Mean test prevalence is {prev['focal']} on focal and {prev['soundscape']} on soundscapes, so the cmAP gap is not mainly a base-rate effect."},
    ]

    # ---- one-to-two line caption per figure: what it shows + the insight (numbers computed above)
    g_fail, g_rest = groups
    npri_low = [s for s in species if not s["primary"]]
    figure_notes = {
        "ch-counts": {
            "shows": "How many of the 60 prototypes lose or gain detection AUC on soundscapes, per model.",
            "insight": f"Failures grow from {counts[0]['fail']} (pretrained) to {counts[1]['fail']} after focal fine-tuning and shrink to {counts[2]['fail']} with soundscape labels, so focal-only fine-tuning makes transfer worse."},
        "ch-groups": {
            "shows": f"Mean soundscape prototype AUC of the {g_fail['n']} prototypes that failed under pretrained, versus the other {g_rest['n']}, for each model.",
            "insight": f"Soundscape labels lift the failing group from {g_fail['sc_auc'][PRETRAINED]} to {g_fail['sc_auc'][SC_ONLY]}, still below the others ({g_rest['sc_auc'][SC_ONLY]}); focal-only also drags the others down from {g_rest['sc_auc'][PRETRAINED]} to {g_rest['sc_auc'][FOCAL_ONLY]}."},
        "ch-scatter": {
            "shows": "Each prototype's soundscape-minus-focal AUC (\u0394) under pretrained against its \u0394 under a fine-tuned model.",
            "insight": f"Focal-only keeps the same prototypes failing (\u03c1 = {rho['pretrained_vs_focal_only']}); soundscape-only reshuffles them (\u03c1 = {rho['pretrained_vs_soundscape_only']}) and moves most above the \u22120.1 line."},
        "ch-species": {
            "shows": "Number of test clips containing each species, soundscape (bars) and focal (dots), on a log scale.",
            "insight": f"Only {len(primary)} species reach {PRIMARY_MIN_POS} soundscape test positives; the other {len(npri_low)} have {min(s['n_soundscape'] for s in npri_low)}\u2013{max(s['n_soundscape'] for s in npri_low)}, so their scores are too noisy to headline."},
    }

    out = {**{k: content[k] for k in ("title", "subtitle", "problem", "shift_factors", "decisions", "stages", "blockers", "recipe")},
           "analysis": {"models": [{"name": n, "label": LABELS[n]} for n in names], "species": species,
                        "primary_min_pos": PRIMARY_MIN_POS, "primary_species": [sp[i] for i in primary],
                        "class_level": class_level, "proto_counts": counts, "proto": protos, "rho": rho,
                        "groups": groups, "prevalence": prev, "splits": splits, "species_split": sp_split, "split_notes": split_notes, "figure_notes": figure_notes, "criteria": criteria,
                        "thresholds": {"fail": FAIL, "improve": IMPROVE}, "findings": findings}}
    args.out.write_text("window.PLAN_DATA = " + json.dumps(out) + ";\n")
    logger.info(f"Wrote {args.out}: {len(protos)} prototypes, {len(primary)} primary species {[sp[i] for i in primary]}")
    for f in findings:
        logger.info(f"  - {f['title']}: {f['text']}")
    for c in criteria:
        logger.info(f"  * {c['baseline']}")


if __name__ == "__main__":
    main()
