"""Dashboard data bundle pieces (prototype_transfer.reporting.dashboard_data). Run: python tests/test_dashboard_data.py"""
import numpy as np

from prototype_transfer.reporting import dashboard_data as dd


def _domain(n, rng, species_col, other_col):
    t = np.zeros((n, 21), dtype=bool)
    t[: n // 4, species_col] = True        # the first quarter contains the species
    t[n // 2 : n // 2 + 20, other_col] = True
    act = rng.random((n, 21, 5)).astype(np.float32) * 0.1
    act[: n // 4, species_col, :] += 0.4   # strong on positives
    act[n // 2 : n // 2 + 20, species_col, 0] = 0.9   # prototype 1 also fires on clips of the other species
    return {"targets": t, "proto_max": act}


def test_cdf_is_monotone_and_reaches_one():
    c = dd.cdf(np.random.default_rng(0).random(500) * 0.5)
    assert all(b >= a for a, b in zip(c, c[1:])) and c[-1] == 1.0 and c[0] >= 0


def test_confusion_names_the_species_a_prototype_is_confused_with():
    rng = np.random.default_rng(0)
    sp = ["s%02d" % i for i in range(21)]
    data = {"soundscape": _domain(200, rng, 0, 1), "focal": _domain(100, rng, 0, 1)}
    conf = dd.confusion_stats(data, 0, sp, top_n=20)
    assert len(conf) == 5
    assert conf[0]["top_species"] == "s01" and conf[0]["top_count"] == 20 and conf[0]["frac_other"] == 1.0
    assert conf[1]["frac_other"] < conf[0]["frac_other"]


def test_species_activation_cdfs_has_every_view_for_both_domains():
    rng = np.random.default_rng(0)
    data = {"soundscape": _domain(200, rng, 0, 1), "focal": _domain(100, rng, 0, 1)}
    out = dd.species_activation_cdfs(data, 0)
    assert set(out["series"]) == {"all", "max", "1", "2", "3", "4", "5"}
    assert set(out["series"]["all"]) == {"focal_pos", "focal_neg", "soundscape_pos", "soundscape_neg"}
    assert out["n"]["soundscape_pos"] == 50 and out["medians"]["max"]["soundscape_pos"] > out["medians"]["max"]["soundscape_neg"]


def test_silent_clips_do_not_count_towards_the_peak_frequency():
    """An all-zero activation map has argmax 0, the lowest frequency row: it must not be reported as a peak there."""
    from prototype_transfer.analysis.prototype_responses import grid_row_centers_khz, prototype_transfer_stats
    n = 10
    targets = np.zeros((n, 1), dtype=bool)
    targets[:6, 0] = True
    act = np.zeros((n, 1, 5), dtype=np.float32)
    act[:2, 0, 0] = 0.5                       # prototype 1 fires on 2 of the 6 positives, at row 10; prototype 2 never fires
    rows = np.zeros((n, 1, 5), dtype=np.uint8)
    rows[:2, 0, 0] = 10
    d = {"targets": targets, "proto_max": act, "peak_freq": rows, "probs": np.random.default_rng(0).random((n, 1))}
    p1, p2 = prototype_transfer_stats(d, d, np.ones((1, 5)), ["s"], min_soundscape_pos=1)["species"]["s"]["domains"]["soundscape"]["prototypes"][:2]
    assert np.isclose(p1["peak_khz"], grid_row_centers_khz()[10]) and np.isclose(p1["silent_pos"], 4 / 6)
    assert p2["peak_khz"] is None and p2["silent_pos"] == 1.0


def test_model_blocks_read_the_results_store_and_label_models():
    from prototype_transfer.evaluation import results
    store = results.Store("hsn")
    block = dd.model_comparison_block(store.summary("model_comparison"))
    assert [m["name"] for m in block["models"]] == ["pretrained-xcl", "focal-only", "soundscape-only"]
    assert block["models"][0]["label"] == "Pretrained (XCL)"
    assert dd.model_comparison_block(store.summary("does-not-exist")) is None


def test_improving_prototypes_use_the_evaluation_threshold():
    import json
    import tempfile
    from pathlib import Path

    from prototype_transfer.evaluation import canonical, results

    deltas = [-0.2, 0.0, canonical.IMPROVE / 2, canonical.IMPROVE + 0.01, 0.1]   # two of these improve
    side = lambda auc: {"n_pos": 20, "class_auc": 0.9, "prototypes": [
        {"auc": a, "act_pos": 0.5, "act_neg": 0.1, "logit_share": 0.2, "peak_khz": 4.0} for a in auc]}
    stats = {"min_soundscape_pos": 10, "species": {"s00": {"analysed": True, "last_layer_weights": [1.0] * 5, "domains": {
        "focal": side([0.5] * 5), "soundscape": side([0.5 + d for d in deltas])}}}}
    with tempfile.TemporaryDirectory() as tmp:
        store = results.Store("toy", Path(tmp))
        store.run_dir("m").mkdir(parents=True)
        (store.run_dir("m") / "prototype_transfer.json").write_text(json.dumps(stats))
        summary = dd.model_prototypes_block(["m"], store)["models"][0]["summary"]
    assert (summary["n_improve"], summary["n_lose"], summary["n_protos"]) == (2, 1, 5), summary


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"{len(tests)} tests OK")
