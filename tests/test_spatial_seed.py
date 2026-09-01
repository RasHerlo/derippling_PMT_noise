"""Tests for spatial line-scan hints and fx-column contrast."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "reference" / "gpt"))

from batch_defringe.spatial_seed import (  # noqa: E402
    col_contrast,
    fft_axis_scores,
    fft_log_amp,
    fx_family_from_q,
    geometry_fx_mask,
    geometry_fy_mask,
    notch_fx,
    notch_fy,
    period_to_q,
    periodic_baseline,
    reconstruct_from_mask,
    sine_fit,
    spatial_periods,
    spectral_peak_mask,
    synthesize_fringe,
)
from pmt_fringe_raw_adaptive import contiguous_ranges  # noqa: E402

H, W = 128, 128
Q_Y = 8
Q_X = 16


def _xvalid(width: int) -> np.ndarray:
    cx = width // 2
    fx = np.arange(width) - cx
    return (np.abs(fx) > 5) & (np.abs(fx) < cx - 10)


def _yvalid(height: int) -> np.ndarray:
    cy = height // 2
    fy = np.arange(height) - cy
    return (np.abs(fy) > 5) & (np.abs(fy) < cy - 10)


def _vgrating(q: float = Q_X, amp: float = 50.0) -> np.ndarray:
    """Vertical stripes: period along x, energy on the fx axis."""
    x = np.arange(W)[None, :]
    return np.broadcast_to(amp * np.sin(2.0 * np.pi * q * x / W), (H, W)).copy()


def _hgrating(q: float = Q_Y, n_fx: int = 7, amp: float = 50.0) -> np.ndarray:
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    img = np.zeros((H, W), dtype=np.float64)
    half = n_fx // 2
    used = 0
    for df in range(-half, half + 1):
        fxc = 16 + df
        if abs(fxc) < 6:
            continue
        img += np.sin(2.0 * np.pi * q * y / H + 2.0 * np.pi * fxc * x / W)
        used += 1
    return img * (amp / max(used, 1))


def _fy_family(q: float, fx: float) -> dict:
    xw = np.zeros(W, dtype=float)
    cx = W // 2
    for fxc in (fx, -fx):
        for i in range(W):
            xw[i] += np.exp(-0.5 * ((i - cx - fxc) / 4.0) ** 2)
    xw /= max(float(xw.max()), 1e-12)
    return {
        "q": float(q),
        "hi": float(H // 2 - q),
        "paired": True,
        "x_weight": xw,
        "fx_ranges": contiguous_ranges((np.arange(W) - cx)[xw > 0.20]),
    }


def test_period_to_q():
    assert period_to_q(None, 128) is None
    q = period_to_q(128 / 8, 128)
    assert q is not None and abs(q - 8) < 0.2


def test_col_contrast_prefers_vertical_grating():
    v = 80.0 + _vgrating()
    h = 80.0 + _hgrating()
    scores = fft_axis_scores(fft_log_amp(v))
    assert scores["col_peak"] > scores["row_peak"]
    assert abs(scores["col_peak_q"] - Q_X) <= 3
    scores_h = fft_axis_scores(fft_log_amp(h))
    assert scores_h["row_peak"] > scores_h["col_peak"]


def test_spatial_hint_vertical_is_qx():
    img = 80.0 + _vgrating()
    sp = spatial_periods(img)
    assert sp["qx_hint"] is not None
    assert abs(sp["qx_hint"] - Q_X) <= 3


def test_spatial_hint_horizontal_is_qy():
    img = 80.0 + _hgrating()
    sp = spatial_periods(img)
    assert sp["qy_hint"] is not None
    assert abs(sp["qy_hint"] - Q_Y) <= 3


def test_fx_notch_removes_vertical_grating():
    img = 80.0 + _vgrating()
    logamp = fft_log_amp(img)
    fam = fx_family_from_q(logamp, Q_X, y_z=1.5)
    out = notch_fx(img, fam, Q_X, search=4)
    assert out["gate"] > 0, out
    assert out["removed_rms"] > 5.0
    assert abs(out["q"] - Q_X) <= 3


def test_fy_notch_still_removes_horizontal():
    img = 80.0 + _hgrating()
    fam = _fy_family(Q_Y, 16.0)
    out = notch_fy(img, fam, Q_Y, search=2)
    assert out["gate"] > 0, out
    assert out["removed_rms"] > 5.0


def test_sine_fit_matches_known_period():
    n = 128
    period = 16.0
    k = np.arange(n)
    hp = np.sin(2.0 * np.pi * k / period)
    fit = sine_fit(hp, period)
    assert fit is not None
    corr = float(np.corrcoef(hp, fit)[0, 1])
    assert corr > 0.99, corr


def test_geometry_masks_are_thin_bands():
    fy = geometry_fy_mask(H, W, Q_Y)
    fx = geometry_fx_mask(H, W, Q_X)
    assert fy.mean() < 0.25
    assert fx.mean() < 0.25
    cy, cx = H // 2, W // 2
    assert fy[cy + Q_Y, :].mean() > 0.9
    assert fx[:, cx + Q_X].mean() > 0.9
    assert fx[:, cx].mean() == 0.0


def test_period_prefers_fringe_band_over_grain():
    n = 512
    t = np.arange(n, dtype=np.float64)
    fringe = 10.0 * np.sin(2.0 * np.pi * t / 46.0)
    grain = 20.0 * np.sin(2.0 * np.pi * t / 13.0)
    sig = 80.0 + fringe + grain
    _base, period, _alpha = periodic_baseline(sig, length=n)
    q = period_to_q(period, n)
    assert q is not None and q <= 20.0, (period, q)
    assert period is not None and abs(period - 13.0) > 5.0


def test_periodic_baseline_is_the_sine_not_the_cells():
    n = 256
    t = np.arange(n, dtype=np.float64)
    sine = 10.0 * np.sin(2.0 * np.pi * t / 20.0)
    sig = 80.0 + sine
    sig[40:48] += 80.0
    sig[120:125] += 100.0
    base, period, _alpha = periodic_baseline(sig)
    assert period is not None and abs(period - 20.0) <= 2.0, period
    osc = base - float(np.median(base))
    corr = float(np.corrcoef(sine, osc)[0, 1])
    assert corr > 0.95, corr
    leftover = sig - base
    assert float(leftover[43]) > 40.0
    assert float(leftover[122]) > 40.0


def test_reconstruct_peak_mask_looks_like_vertical_grating():
    img = 80.0 + _vgrating()
    mask = spectral_peak_mask(H, W, qx=float(Q_X))
    rec = reconstruct_from_mask(img, mask)
    g = img - float(np.median(img))
    corr = float(np.corrcoef(g.ravel(), rec.ravel())[0, 1])
    assert corr > 0.9, corr
    # whole-column geometry would also correlate, but the peak mask is sparse
    assert float(mask.mean()) < 0.02


def test_synthesize_fringe_is_a_grating():
    synth = synthesize_fringe(H, W, qx=float(Q_X))
    x = np.arange(W)[None, :]
    g = np.broadcast_to(np.cos(2.0 * np.pi * Q_X * x / W), (H, W))
    corr = abs(float(np.corrcoef(g.ravel(), synth.ravel())[0, 1]))
    assert corr > 0.99, corr
    assert float(np.std(synth[:, 0])) < 1e-6


def test_spatial_traces_store_backbone_and_fit():
    img = 80.0 + _hgrating()
    sp = spatial_periods(img)
    tr = sp["traces"]["horizontal"]
    assert "smooth" in tr and "sine" in tr
    assert tr["smooth"].shape == tr["raw"].shape
    assert tr["axis"] == "fx"
    np.testing.assert_allclose(tr["fit"], tr["smooth"])
    np.testing.assert_allclose(tr["hp"], tr["raw"] - tr["fit"])


if __name__ == "__main__":
    test_period_to_q()
    test_col_contrast_prefers_vertical_grating()
    test_spatial_hint_vertical_is_qx()
    test_spatial_hint_horizontal_is_qy()
    test_fx_notch_removes_vertical_grating()
    test_fy_notch_still_removes_horizontal()
    test_sine_fit_matches_known_period()
    test_geometry_masks_are_thin_bands()
    test_period_prefers_fringe_band_over_grain()
    test_periodic_baseline_is_the_sine_not_the_cells()
    test_reconstruct_peak_mask_looks_like_vertical_grating()
    test_synthesize_fringe_is_a_grating()
    test_spatial_traces_store_backbone_and_fit()
    print("ok")
