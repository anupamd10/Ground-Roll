import numpy as np
import pytest
import torch

from grsep import Band, GatherParams, GroundRollSeparator, SeparationLoss, leakage_ratio, make_gather, separate, split
from grsep.data import PatchDataset, patch_count, positions, prepare
from grsep.synth import Geometry, events_to_gather, ground_roll_mode

BAND = Band(25.0, 35.0)


@pytest.fixture(scope="module")
def gather():
    return make_gather(GatherParams().with_(nt=400, nx=100), np.random.default_rng(1))


def test_band_split_is_exact_and_zero_phase(gather):
    m, dt = gather["m"], gather["dt"]
    low, high = split(m, dt, BAND)
    assert np.allclose(low + high, m, atol=1e-5)
    spike = np.zeros((400, 1), np.float32)
    spike[200] = 1
    lo, _ = split(spike, dt, BAND)
    assert np.argmax(lo[:, 0]) == 200
    assert np.allclose(lo[200 - 30 : 200, 0], lo[201 : 231, 0][::-1], atol=1e-6)


def test_ground_roll_in_low_band(gather):
    assert leakage_ratio(gather["g"], gather["dt"], BAND) < 5e-3


def test_ground_roll_symmetric_about_source():
    g = ground_roll_mode(Geometry(nt=500, nx=101), 300.0, 5.0, 15.0, 0.15, 0.5e-3, 0.99)
    assert np.allclose(g, g[:, ::-1], atol=1e-6)
    assert np.abs(g[:, 50]).max() > 0 and np.abs(g[:, 30]).max() > 0


def test_ground_roll_sweep_is_relative_to_onset():
    geom = Geometry(nt=4000, nx=41, dt=0.0005)
    g = ground_roll_mode(geom, 200.0, 5.0, 15.0, 0.4, 0.0, 1.0)
    ref = g[:800, 20]
    for tr in (10, 15, 25, 30):
        lag = abs(tr - 20) * 100
        assert np.allclose(g[lag : lag + 800, tr], ref, atol=1e-9)


def test_loss_bounded_below():
    crit = SeparationLoss(lam=0.5)
    r, g = torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8)
    vals = [crit(r * s + 3 * s, g * s - 3 * s, r, g, r + g)[0].item() for s in (1, 10, 100, 1000)]
    assert min(vals) > -1e-6 and vals[-1] > vals[0]
    with pytest.raises(ValueError):
        SeparationLoss(lam=1.0)


def test_paper_loss_is_unbounded():
    r, g = torch.randn(64), torch.randn(64)
    def paper(rh, gh):
        return ((rh - r) ** 2).sum() + ((gh - g) ** 2).sum() + ((rh + gh - r - g) ** 2).sum() - 0.001 * (rh - gh).abs().sum()
    assert paper(r, g) > paper(r + 1e-4 * torch.sign(r - g), g - 1e-4 * torch.sign(r - g))


def test_patch_positions_cover_edges():
    assert positions(1000, 64, 10)[-1] == 936
    assert patch_count((1000, 200), 64, 10) == 95 * 15


def test_patch_dataset_labels_consistent(gather):
    ds = PatchDataset([gather], BAND, 64, 32)
    x, y = ds[3]
    assert torch.allclose(y[0] + y[1], x[0], atol=1e-4)


def test_stitching_identity(gather):
    class Oracle(torch.nn.Module):
        def forward(self, x):
            return x[:, :1] * 0.25, x[:, :1] * 0.75

    est = separate(Oracle(), gather["m"], gather["dt"], BAND, stride=24)
    low, high = split(gather["m"], gather["dt"], BAND)
    assert np.allclose(est["r_low"], 0.25 * low, atol=1e-4)
    assert np.allclose(est["g"], 0.75 * low, atol=1e-4)
    assert np.allclose(est["r"] + est["g"], gather["m"], atol=1e-4)


def test_small_gather_padding():
    m = np.random.randn(50, 30).astype(np.float32)
    est = separate(GroundRollSeparator(width=8), m, 0.002, BAND)
    assert est["r"].shape == m.shape


def test_model_shapes():
    net = GroundRollSeparator(width=16)
    r, g = net(torch.randn(2, 2, 64, 64))
    assert r.shape == g.shape == (2, 1, 64, 64)


def test_quiet_patches_stay_bounded():
    geom = Geometry(nt=400, nx=64)
    r = events_to_gather(geom, [np.full(64, 0.3)], [1.0], 40.0).astype(np.float32)
    ds = PatchDataset([{"m": r, "r": r, "dt": 0.002}], BAND, 64, 8)
    worst = max(max(ds[i][0].abs().max().item(), ds[i][1].abs().max().item()) for i in range(len(ds)))
    assert worst < 20


def test_noise_goes_to_reflection_label():
    s = make_gather(GatherParams().with_(nt=300, nx=80, noise_ratio=0.1), np.random.default_rng(2))
    _, y = prepare(s, BAND)
    g_low, _ = split(s["g"], s["dt"], BAND)
    assert np.allclose(y[1], g_low, atol=1e-5)
