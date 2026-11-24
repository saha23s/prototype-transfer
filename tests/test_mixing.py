"""Plain-assert tests for src/prototype_transfer/method/mixing.py and the mixing options of src/prototype_transfer/method/train_dataset.py.  Run: python tests/test_mixing.py"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from prototype_transfer.method.mixing import coloured_noise, mix_at_snr, mix_background, mix_overlap, peak_guard, power, select_background
from prototype_transfer.method.train_dataset import HSNPseudoDataset, HSNTrainDataset

SR, T = 32_000, 160_000


def _db(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(10 * torch.log10(power(a) / power(b)))


def test_requested_snr_equals_measured_snr():
    rng = np.random.default_rng(0)
    x, b = torch.randn(1, T) * 0.1, torch.randn(1, T) * 0.7
    for snr in (-5.0, 0.0, 7.5, 20.0):
        m = mix_at_snr(x, b, snr)
        assert abs(_db(x, m - x) - snr) < 0.5, (snr, _db(x, m - x))
    for _ in range(20):                                   # drawn SNRs stay in range and are the ones realised
        m, snr = mix_background(x, b, -5.0, 20.0, rng)
        assert -5.0 <= snr <= 20.0 and abs(_db(x, m - x) - snr) < 0.5


def test_coloured_noise_is_unit_power_with_the_right_slope():
    rng = np.random.default_rng(1)
    for kind, beta in (("white", 0), ("pink", 1), ("brown", 2)):
        n = coloured_noise(T, kind, rng)
        assert n.shape == (1, T) and abs(float(power(n)) - 1) < 1e-3
        ps = np.abs(np.fft.rfft(n[0].numpy())) ** 2
        f = np.arange(len(ps))
        band = (f >= 50) & (f <= 20000)
        slope = np.polyfit(np.log(f[band]), np.log(ps[band]), 1)[0]      # log-log slope of the power spectrum = -beta
        assert abs(slope + beta) < 0.15, (kind, slope)


def test_overlap_target_is_the_union_and_level_is_relative():
    rng = np.random.default_rng(2)
    x, y = torch.randn(1, T) * 0.2, torch.randn(1, T) * 0.5
    t_x, t_y = torch.zeros(21), torch.zeros(21)
    t_x[3], t_y[8] = 1, 1
    for _ in range(20):
        m, t, rel = mix_overlap(x, y, t_x, t_y, -10.0, 0.0, rng)
        assert -10.0 <= rel <= 0.0 and abs(_db(m - x, x) - rel) < 0.5      # second source sits `rel` dB re the clip
        assert t.sum() == 2 and t[3] == 1 and t[8] == 1                      # union of the two labels, two sources at most
    same = mix_overlap(x, y, t_x, t_x, -10.0, 0.0, rng)[1]
    assert torch.equal(same, t_x)                                            # same label twice stays a single label


def test_peak_guard_only_scales_down():
    quiet, loud = torch.full((1, 10), 0.5), torch.tensor([[0.0, 2.0, -4.0]])
    assert torch.equal(peak_guard(quiet), quiet) and float(peak_guard(loud).abs().max()) == 1.0


def test_background_selection_is_label_free_and_picks_the_quietest():
    max_prob = np.random.default_rng(3).random(1000)
    keep = select_background(max_prob, 0.3)
    assert abs(keep.mean() - 0.3) < 0.01 and max_prob[keep].max() <= max_prob[~keep].min()
    import inspect
    assert list(inspect.signature(select_background).parameters) == ["max_prob", "quantile"]   # no labels can be passed


def _make_dataset_files(root: Path):
    rng = np.random.default_rng(4)
    rows = []
    for i in range(6):
        p = root / f"focal_{i}.wav"
        sf.write(p, rng.standard_normal(6 * SR).astype("float32") * 0.1, SR)
        rows.append({"audio_path": str(p), "target_class": i % 3})
    manifest = root / "focal.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    bg = root / "bg.parquet"
    pd.DataFrame({"audio_path": [str(root / "focal_0.wav")]}).to_parquet(bg, index=False)
    return manifest, bg


def _old_getitem(ds: HSNTrainDataset, i: int) -> torch.Tensor:
    """The pre-mixing pipeline of HSNTrainDataset.__getitem__ (focal): crop, gain, spectrogram, masks."""
    wav = ds._load_crop(ds.df.iloc[i]["audio_path"])
    wav = wav * float(10 ** (np.random.uniform(-6, 6) / 20))
    spec = ds.pre.transform_to_spectrogram(wav)
    fill = float(spec.mean())
    for _ in range(ds.tm):
        w = np.random.randint(0, ds.mt + 1); t0 = np.random.randint(0, spec.shape[-1] - w + 1)
        spec[..., t0:t0 + w] = fill
    for _ in range(ds.fm):
        h = np.random.randint(0, ds.mf + 1); f0 = np.random.randint(0, spec.shape[-2] - h + 1)
        spec[..., f0:f0 + h, :] = fill
    return spec


def test_all_flags_off_equals_the_previous_dataset():
    with tempfile.TemporaryDirectory() as d:
        manifest, _ = _make_dataset_files(Path(d))
        ds = HSNTrainDataset(manifest, is_soundscape=False)
        for i in range(3):
            np.random.seed(10 + i); got = ds[i]["spectrogram"]
            np.random.seed(10 + i); want = _old_getitem(ds, i)
            assert torch.equal(got, want)
        assert ds.mixing_config()["bg_prob"] == 0.0 and ds.mixing_config()["overlap_prob"] == 0.0


def test_mixing_options_run_and_change_the_sample():
    with tempfile.TemporaryDirectory() as d:
        manifest, bg = _make_dataset_files(Path(d))
        base = HSNTrainDataset(manifest, is_soundscape=False)
        for kw in ({"bg_kind": "noise", "bg_prob": 1.0}, {"bg_kind": "real", "bg_manifest": bg, "bg_prob": 1.0},
                   {"overlap_prob": 1.0}, {"bg_kind": "noise", "bg_prob": 1.0, "overlap_prob": 1.0}):
            ds = HSNTrainDataset(manifest, is_soundscape=False, **kw)
            torch.manual_seed(0); np.random.seed(0); item = ds[1]
            assert item["spectrogram"].shape == (1, 256, 626) and torch.isfinite(item["spectrogram"]).all(), kw
            if "overlap_prob" in kw:
                assert item["target"].sum() == 2, kw                      # class 1 plus a different species
            else:
                assert item["target"].sum() == 1, kw                      # labels do not change with background mixing
            torch.manual_seed(0); np.random.seed(0)
            assert not torch.equal(item["spectrogram"], base[1]["spectrogram"]), kw
        try:
            HSNTrainDataset(manifest, is_soundscape=True, bg_kind="noise")
            raise AssertionError("mixing on soundscape training should be refused")
        except AssertionError as e:
            assert "focal training" in str(e)


def test_pseudo_dataset_serves_teacher_probabilities_and_no_annotations():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        sf.write(root / "s.wav", np.random.default_rng(5).standard_normal(5 * SR).astype("float32") * 0.05, SR)
        probs = np.linspace(0, 1, 21, dtype=np.float32)
        m = root / "pseudo.parquet"
        pd.DataFrame({"audio_path": [str(root / "s.wav")], "teacher_probs": [probs]}).to_parquet(m, index=False)
        item = HSNPseudoDataset(m)[0]
        assert torch.allclose(item["target"], torch.from_numpy(probs)) and item["spectrogram"].shape == (1, 256, 626)
        bad = root / "bad.parquet"
        pd.DataFrame({"audio_path": [str(root / "s.wav")], "teacher_probs": [probs], "hsn_classes": [["x"]]}).to_parquet(bad, index=False)
        try:
            HSNPseudoDataset(bad)
            raise AssertionError("a pseudo manifest with annotations should be refused")
        except AssertionError as e:
            assert "annotations" in str(e)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
