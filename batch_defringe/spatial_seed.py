"""Probe: spatial line-scan periods + fx-column contrast as seed hints.

Score-and-print only. Does not overwrite production TIFFs.

``python -m batch_defringe.spatial_seed`` defaults to Haj Grant ChanA frame 160.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

_REPO = Path(__file__).resolve().parents[1]
_GPT = _REPO / "reference" / "gpt"
if str(_GPT) not in sys.path:
    sys.path.insert(0, str(_GPT))

from scipy.ndimage import gaussian_filter1d

from pmt_fringe_raw_adaptive import fft_log_amp, row_contrast, search_q  # noqa: E402
from pmt_fringe_raw_adaptive_v21 import _attenuate_family_on_amp, _effective_max_alpha  # noqa: E402

from .image_check import (
    DIAGONALS,
    diagonal_sample,
    score_removed,
    seed_mask_at_q,
)
from .process import PACK_D
from .readout import _jsonable, _percentile_limits, _signed_limit
from .seed import hydrate_families

OUTPUT_SUBDIR = "spatial_seed"
MIN_Q = 5
MAX_HINT_Q = 32
MASK_RADIUS = 2
TRACE_AXIS = {
    "vertical": "fy",
    "horizontal": "fx",
    "main": "diag",
    "anti": "diag",
}
GATE_LOW = float(PACK_D["gate_low"])
GATE_HIGH = float(PACK_D["gate_high"])


def _yvalid(height: int) -> np.ndarray:
    cy = height // 2
    fy = np.arange(height) - cy
    return (np.abs(fy) > 5) & (np.abs(fy) < cy - 10)


def _y_axis_band(height: int, half: int = 8) -> np.ndarray:
    """fy bins around DC, including fy=0. Vertical stripes sit on that row at fx≠0."""
    cy = height // 2
    m = np.zeros(height, dtype=bool)
    m[max(0, cy - half) : min(height, cy + half + 1)] = True
    return m


def _xvalid(width: int) -> np.ndarray:
    cx = width // 2
    fx = np.arange(width) - cx
    return (np.abs(fx) > 5) & (np.abs(fx) < cx - 10)


def _col_contrast_on(logamp: np.ndarray, dx: int, ymask: np.ndarray, inner: int = 5, outer: int = 9) -> float:
    w = logamp.shape[1]
    cx = w // 2
    vals = []
    for sgn in (-1, +1):
        x = cx + sgn * int(dx)
        if not (0 <= x < w):
            continue
        rp = np.percentile(logamp[ymask, x], 95)
        bgcols = []
        for off in list(range(-outer, -inner + 1)) + list(range(inner, outer + 1)):
            xx = x + off
            if 0 <= xx < w:
                bgcols.append(np.percentile(logamp[ymask, xx], 95))
        if bgcols:
            vals.append(rp - np.median(bgcols))
    return float(np.median(vals)) if vals else float("-inf")


def col_contrast(logamp: np.ndarray, dx: int, yvalid: np.ndarray | None = None, inner: int = 5, outer: int = 9) -> float:
    """fx-column analog of ``row_contrast``.

    Takes the stronger of: energy on the fx-axis (vertical stripes, fy≈0) and
    energy in the off-axis fy band (tilted / textured column ridges).
    """
    h = logamp.shape[0]
    axis = _col_contrast_on(logamp, dx, _y_axis_band(h), inner=inner, outer=outer)
    off = _col_contrast_on(
        logamp, dx, yvalid if yvalid is not None else _yvalid(h), inner=inner, outer=outer
    )
    return float(max(axis, off))


def family_score_fx(logamp: np.ndarray, q: int, paired: bool, yvalid: np.ndarray) -> float:
    w = logamp.shape[1]
    cx = w // 2
    scores = [col_contrast(logamp, q, yvalid)]
    if paired:
        scores.append(col_contrast(logamp, cx - q, yvalid))
    return float(np.median(scores))


def search_fx(logamp: np.ndarray, q0: float, paired: bool, yvalid: np.ndarray, radius: int) -> tuple[float, float]:
    cx = logamp.shape[1] // 2
    lo = max(MIN_Q, int(round(q0)) - radius)
    hi = min(cx - MIN_Q, int(round(q0)) + radius)
    qs = np.arange(lo, hi + 1)
    if len(qs) == 0:
        return float(q0), float("-inf")
    scores = np.array([family_score_fx(logamp, int(q), paired, yvalid) for q in qs])
    j = int(np.argmax(scores))
    return float(qs[j]), float(scores[j])


def period_to_q(period_px: float | None, n: int) -> float | None:
    if period_px is None or period_px < 4.0:
        return None
    q = float(n) / float(period_px)
    if q < MIN_Q or q > (n // 2) - MIN_Q:
        return None
    return q


def sine_fit(hp: np.ndarray, period_px: float | None) -> np.ndarray | None:
    """Least-squares sinusoid at a fixed period."""
    if period_px is None or period_px < 4.0:
        return None
    y = np.asarray(hp, dtype=np.float64)
    n = int(y.size)
    k = np.arange(n, dtype=np.float64)
    w = 2.0 * np.pi / float(period_px)
    a = np.column_stack([np.sin(w * k), np.cos(w * k)])
    coef, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
    return a @ coef


def _fundamental_acf_period(y: np.ndarray, *, length: int | None = None) -> float | None:
    """ACF lag of the fringe, not soma spacing and not pixel grain.

    Bright cells can put a tall peak at the soma-to-soma distance. Grain can put
    an earlier peak at a few pixels. Keep lags whose implied q = length/P sits in
    the fringe band, then take the strongest of those.
    """
    y = np.asarray(y, dtype=np.float64)
    s = 1.4826 * float(np.median(np.abs(y - np.median(y)))) + 1e-12
    z = np.tanh(y / (3.0 * s))
    z = z - float(z.mean())
    if z.size < 24 or float(np.std(z)) < 1e-12:
        return None
    ac = np.correlate(z, z, mode="full")
    mid = len(z) - 1
    ac = ac[mid:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    min_lag = 4
    max_lag = max(min_lag + 2, z.size // 3)
    thresh = 0.12
    nq = int(length) if length is not None else int(z.size)
    best_k = None
    best_val = thresh
    for k in range(min_lag + 1, min(max_lag, len(ac) - 1)):
        val = float(ac[k])
        if not (val >= thresh and val >= float(ac[k - 1]) and val >= float(ac[k + 1])):
            continue
        q = period_to_q(float(k), nq)
        if q is None or q > MAX_HINT_Q:
            continue
        if val >= best_val:
            best_val = val
            best_k = k
    return None if best_k is None else float(best_k)


def _chirp_design(n: int, period: float, alpha: float) -> np.ndarray:
    """sin/cos columns whose instantaneous period is P·(1 + α·(t/T − ½))."""
    t = np.arange(n, dtype=np.float64)
    u = t / max(n - 1, 1) - 0.5
    inst_p = np.clip(float(period) * (1.0 + alpha * u), 4.0, max(4.0, n / 4.0))
    ph = 2.0 * np.pi * np.cumsum(1.0 / inst_p)
    ph -= ph[0]
    return np.column_stack([np.sin(ph), np.cos(ph)])


def _fit_irls_positive_outliers(y: np.ndarray, A: np.ndarray, niter: int = 8) -> np.ndarray:
    """Fit A @ coef to y, downweighting bright (positive) outliers — cells on the fringe."""
    n = int(y.size)
    w = np.ones(n, dtype=np.float64)
    fit = np.zeros(n, dtype=np.float64)
    for _ in range(niter):
        coef, _, _, _ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        fit = A @ coef
        r = y - fit
        s = 1.4826 * float(np.median(np.abs(r))) + 1e-12
        pos = np.maximum(r / (2.5 * s), 0.0)
        w = 1.0 / (1.0 + pos ** 2)
    return fit


def periodic_baseline(
    sig: np.ndarray, *, length: int | None = None
) -> tuple[np.ndarray, float | None, float]:
    """The periodic pattern *is* the baseline. Biology sits on top as bright outliers.

    Returns ``(baseline, period_px, chirp_alpha)``. Residual ``sig − baseline`` is leftover
    (cells / unstructured). Shutter traces should be almost pure baseline.
    ``length`` is the axis used for q = length/P (image height/width for V/H traces).
    """
    y0 = np.asarray(sig, dtype=np.float64)
    dc = float(np.median(y0))
    y = y0 - dc
    nq = int(length) if length is not None else int(y.size)
    period = _fundamental_acf_period(y, length=nq)
    if period is None:
        return np.full_like(y0, dc), None, 0.0
    n = int(y.size)
    best_fit = None
    best_err = np.inf
    best_alpha = 0.0
    for alpha in np.linspace(-0.5, 0.5, 11):
        fit = _fit_irls_positive_outliers(y, _chirp_design(n, period, float(alpha)))
        r = y - fit
        keep = r <= np.percentile(r, 80.0)
        err = float(np.median(np.abs(r[keep]))) if np.any(keep) else float(np.median(np.abs(r)))
        if err < best_err:
            best_err = err
            best_fit = fit
            best_alpha = float(alpha)
    return dc + best_fit, float(period), best_alpha


def reconstruct_from_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """IFFT of the FFT coefficients inside ``mask`` — the spatial pattern that q would take."""
    x = np.asarray(frame, dtype=np.float32)
    w = np.asarray(mask, dtype=np.float32)
    if w.shape != x.shape:
        return np.zeros_like(x, dtype=np.float32)
    offset = float(np.median(x))
    F = np.fft.fftshift(np.fft.fft2(x - offset))
    rec = np.real(np.fft.ifft2(np.fft.ifftshift(F * w)))
    return rec.astype(np.float32)


def spectral_peak_mask(
    h: int,
    w: int,
    *,
    qy: float | None = None,
    qx: float | None = None,
    radius: int = MASK_RADIUS,
) -> np.ndarray:
    """Compact conjugate blobs at ±q on the fy / fx axes — not whole rows or columns.

    A y-periodic fringe lives at ``(fy=±qy, fx=0)``. An x-periodic fringe lives at
    ``(fy=0, fx=±qx)``. Those pixels are not DC. Inverting this mask is a grating,
    not a bandpass of the whole FOV.
    """
    mask = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2
    sigma = max(0.6, radius / 1.5)

    def stamp(fy: int, fx: int) -> None:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                y = cy + fy + dy
                x = cx + fx + dx
                if not (0 <= y < h and 0 <= x < w):
                    continue
                if y == cy and x == cx:
                    continue
                wt = float(np.exp(-0.5 * (dy * dy + dx * dx) / (sigma * sigma)))
                if wt > mask[y, x]:
                    mask[y, x] = wt

    if qy is not None:
        q = int(round(qy))
        stamp(q, 0)
        stamp(-q, 0)
    if qx is not None:
        q = int(round(qx))
        stamp(0, q)
        stamp(0, -q)
    return mask


def synthesize_fringe(
    h: int,
    w: int,
    *,
    qy: float | None = None,
    qx: float | None = None,
) -> np.ndarray:
    """Unit-amplitude cosine field implied by q — the intended fringe, no cell energy."""
    yy = np.arange(h, dtype=np.float64)[:, None]
    xx = np.arange(w, dtype=np.float64)[None, :]
    img = np.zeros((h, w), dtype=np.float32)
    if qy is not None:
        img += np.cos(2.0 * np.pi * float(qy) * yy / float(h)).astype(np.float32)
    if qx is not None:
        img += np.cos(2.0 * np.pi * float(qx) * xx / float(w)).astype(np.float32)
    return img


def geometry_fy_mask(h: int, w: int, q: float) -> np.ndarray:
    """Thin fy bands at ±q and ±(cy−q) — the q-guess before fx support is learned."""
    mask = np.zeros((h, w), dtype=np.float32)
    cy = h // 2
    for d in (float(q), float(cy) - float(q)):
        for sgn in (-1.0, 1.0):
            yc = cy + int(round(sgn * d))
            for off in range(-MASK_RADIUS, MASK_RADIUS + 1):
                y = yc + off
                if 0 <= y < h:
                    mask[y, :] = 1.0
    return mask


def geometry_fx_mask(h: int, w: int, q: float) -> np.ndarray:
    """Thin fx columns at ±q and ±(cx−q). Never touches |fx|<5 (true DC neighborhood)."""
    mask = np.zeros((h, w), dtype=np.float32)
    cx = w // 2
    for d in (float(q), float(cx) - float(q)):
        for sgn in (-1.0, 1.0):
            xc = cx + int(round(sgn * d))
            for off in range(-MASK_RADIUS, MASK_RADIUS + 1):
                x = xc + off
                if 0 <= x < w and abs(x - cx) >= 5:
                    mask[:, x] = 1.0
    return mask


def fx_seed_image(shape_hw: tuple[int, int], family: dict) -> np.ndarray:
    """Data-adapted fx mask: family y_weight painted on the ±q columns."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.float32)
    yw = family.get("y_weight")
    if yw is None:
        return mask
    yw = np.asarray(yw, dtype=np.float32)
    if yw.shape[0] != h:
        return mask
    cx = w // 2
    qs = [float(family["q"])]
    if family.get("hi") is not None:
        qs.append(float(family["hi"]))
    for d in qs:
        for sgn in (-1, +1):
            xc = cx + sgn * int(round(d))
            for off in range(-MASK_RADIUS, MASK_RADIUS + 1):
                x = xc + off
                if 0 <= x < w and abs(x - cx) >= 5:
                    mask[:, x] = np.maximum(mask[:, x], yw)
    return mask


def _strongest_trace(img: np.ndarray, axis: int) -> np.ndarray:
    """axis=0: pick the x-column with max std (vertical scan). axis=1: y-row (horizontal)."""
    arr = np.asarray(img, dtype=np.float64)
    std = np.std(arr, axis=axis)
    idx = int(np.argmax(std))
    return arr[:, idx] if axis == 0 else arr[idx, :]


def spatial_periods(img: np.ndarray) -> dict:
    """Line scans → periodic baseline (the fringe) → ACF period → implied FFT bins."""
    arr = np.asarray(img, dtype=np.float64)
    h, w = arr.shape
    traces = {
        "vertical": _strongest_trace(arr, 0),
        "horizontal": _strongest_trace(arr, 1),
    }
    for which in DIAGONALS:
        _, traces[which] = diagonal_sample(arr, which)

    out: dict = {"traces": {}, "period_px": {}, "q_hint": {}, "q_length": {}}
    for name, sig in traces.items():
        n = int(sig.size)
        length = h if name == "vertical" else w if name == "horizontal" else n
        baseline, period, alpha = periodic_baseline(sig, length=length)
        leftover = np.asarray(sig, dtype=np.float64) - baseline
        q = period_to_q(period, length)
        osc = baseline - float(np.median(baseline))
        out["traces"][name] = {
            "t": np.linspace(0.0, 1.0, n),
            "raw": sig,
            "smooth": baseline,
            "hp": leftover,
            "sine": osc,
            "fit": baseline,
            "chirp": alpha,
            "n": n,
            "length": length,
            "axis": TRACE_AXIS[name],
        }
        out["period_px"][name] = period
        out["q_hint"][name] = q
        out["q_length"][name] = length

    # L/C/R along x for the resonant chirp (period in y, as now).
    from .image_check import _period_by_x_third

    thirds = _period_by_x_third(arr)
    out["period_px"]["y_left"] = thirds["left"]
    out["period_px"]["y_center"] = thirds["center"]
    out["period_px"]["y_right"] = thirds["right"]
    out["q_hint"]["y_left"] = period_to_q(thirds["left"], h)
    out["q_hint"]["y_center"] = period_to_q(thirds["center"], h)
    out["q_hint"]["y_right"] = period_to_q(thirds["right"], h)

    fy_cands = [out["q_hint"][k] for k in ("vertical", "y_left", "y_center", "y_right")]
    out["qy_hint"] = _median_hint(fy_cands)
    out["qx_hint"] = out["q_hint"].get("horizontal")
    return out


def _median_hint(vals: list[float | None]) -> float | None:
    good = [float(v) for v in vals if v is not None]
    return float(np.median(good)) if good else None


def fft_axis_scores(logamp: np.ndarray, max_q: int = 80) -> dict:
    h, w = logamp.shape
    cy, cx = h // 2, w // 2
    xvalid = _xvalid(w)
    yvalid = _yvalid(h)
    qs = np.arange(MIN_Q, max(MIN_Q + 1, min(max_q, cy - MIN_Q, cx - MIN_Q)))
    if qs.size == 0:
        qs = np.array([MIN_Q], dtype=int)
    row = np.array([row_contrast(logamp, int(q), xvalid) for q in qs], dtype=float)
    col = np.array([col_contrast(logamp, int(q), yvalid) for q in qs], dtype=float)
    combo = np.maximum(row, col)
    return {
        "qs": qs,
        "row": row,
        "col": col,
        "combo": combo,
        "row_peak_q": float(qs[int(np.argmax(row))]),
        "row_peak": float(np.max(row)),
        "col_peak_q": float(qs[int(np.argmax(col))]),
        "col_peak": float(np.max(col)),
        "combo_peak_q": float(qs[int(np.argmax(combo))]),
        "combo_peak": float(np.max(combo)),
    }


def _weight_from_z(z: np.ndarray, z_thresh: float = 2.5) -> np.ndarray:
    from scipy.ndimage import binary_dilation, gaussian_filter1d as gf

    support = z > z_thresh
    if not np.any(support):
        w = np.ones(z.size, dtype=float)
        return w / max(float(w.max()), 1e-12)
    support = binary_dilation(support, iterations=1)
    weight = gf(support.astype(float), sigma=1.0)
    if weight.max() > 0:
        weight = weight / weight.max()
    return weight


def fy_family_from_q(logamp: np.ndarray, q: float, *, x_z: float = 2.5) -> dict:
    from pmt_fringe_raw_adaptive import ridge_z_at_row

    h, w = logamp.shape
    fam = {"q": float(q), "hi": float(h // 2 - q), "paired": True, "row_score": 0.0}
    return hydrate_families([fam], logamp, x_z_thresh=x_z)[0]


def fx_family_from_q(logamp: np.ndarray, q: float, *, y_z: float = 2.5) -> dict:
    """fx-column family: y_weight is which fy rows may be notched on those columns."""
    h, w = logamp.shape
    ymask = _yvalid(h) | _y_axis_band(h)
    zy = []
    for sgn in (-1, +1):
        x = w // 2 + sgn * int(round(q))
        if 0 <= x < w:
            zy.append(logamp[:, x])
    profile = np.max(np.stack(zy), axis=0) if zy else logamp.mean(axis=1)
    med = float(np.median(profile[ymask]))
    mad = float(np.median(np.abs(profile[ymask] - med))) + 1e-12
    z = (profile - med) / (1.4826 * mad)
    z[~ymask] = 0.0
    y_weight = _weight_from_z(z, y_z)
    y_weight[~ymask] = 0.0
    return {
        "q": float(q),
        "hi": float(w // 2 - q),
        "paired": True,
        "axis": "fx",
        "y_weight": y_weight,
        "row_score": 0.0,
    }


def _attenuate_fx_family_on_amp(
    src_amp: np.ndarray,
    dst_amp: np.ndarray,
    family: dict,
    q: float,
    gate: float,
    *,
    max_alpha: float,
    ratio_start: float,
    ratio_full: float,
    x_sigma: float = 1.0,
    x_radius: int = 2,
) -> None:
    """Column-wise analog of ``_attenuate_family_on_amp``."""
    h, w = src_amp.shape
    cx = w // 2
    dxs = [q]
    if family.get("paired", True):
        dxs.append(cx - q)
    xoffs = np.arange(-x_radius, x_radius + 1)
    xweights = np.exp(-0.5 * (xoffs / max(1e-6, x_sigma)) ** 2)
    yweight = np.asarray(family["y_weight"], dtype=float)

    for d in dxs:
        for sgn in (-1, +1):
            xc = cx + sgn * int(round(d))
            for off, wx in zip(xoffs, xweights):
                x = xc + int(off)
                if not (0 <= x < w):
                    continue
                if abs(x - cx) < 5:
                    continue
                bgcols = []
                for boff in list(range(-9, -4)) + list(range(5, 10)):
                    xx = x + boff
                    if 0 <= xx < w:
                        bgcols.append(src_amp[:, xx])
                if not bgcols:
                    continue
                bg = np.median(np.stack(bgcols), axis=0)
                ratio = src_amp[:, x] / (bg + 1e-12)
                local_conf = np.clip(
                    (ratio - ratio_start) / max(1e-9, ratio_full - ratio_start),
                    0.0,
                    1.0,
                )
                alpha = max_alpha * gate * float(wx) * yweight * local_conf
                excess = np.maximum(src_amp[:, x] - bg, 0.0)
                candidate = src_amp[:, x] - alpha * excess
                dst_amp[:, x] = np.minimum(dst_amp[:, x], np.maximum(candidate, bg))


def notch_fy(frame: np.ndarray, family: dict, q_pred: float, *, search: int = 10) -> dict:
    return _notch_axis(frame, family, q_pred, axis="fy", search=search)


def notch_fx(frame: np.ndarray, family: dict, q_pred: float, *, search: int = 10) -> dict:
    return _notch_axis(frame, family, q_pred, axis="fx", search=search)


def _notch_axis(
    frame: np.ndarray,
    family: dict,
    q_pred: float,
    *,
    axis: str,
    search: int,
) -> dict:
    params = dict(PACK_D)
    params.update(y_sigma=1.0, y_radius=2)
    orig_dtype = frame.dtype
    x = np.asarray(frame, dtype=np.float32)
    offset = float(np.median(x))
    x0 = x - offset
    F = np.fft.fftshift(np.fft.fft2(x0))
    amp = np.abs(F)
    phase = np.exp(1j * np.angle(F))
    logamp = np.log1p(amp)
    h, w = x.shape
    newamp = amp.copy()

    if axis == "fy":
        xvalid = _xvalid(w)
        q, strength = search_q(logamp, q_pred, True, xvalid, int(search))
    else:
        yvalid = _yvalid(h)
        q, strength = search_fx(logamp, q_pred, True, yvalid, int(search))

    gate = float(np.clip((strength - GATE_LOW) / max(1e-9, GATE_HIGH - GATE_LOW), 0.0, 1.0))
    eff_alpha = _effective_max_alpha(
        gate,
        strength,
        True,
        max_alpha=params["max_alpha"],
        max_alpha_high=params["max_alpha_high"],
        high_gate=params["high_gate"],
        high_strength=params["high_strength"],
        strength_span=params["strength_span"],
    )
    if gate > 0:
        if axis == "fy":
            _attenuate_family_on_amp(
                amp,
                newamp,
                family,
                q,
                gate,
                max_alpha=eff_alpha,
                ratio_start=params["ratio_start"],
                ratio_full=params["ratio_full"],
                y_sigma=params["y_sigma"],
                y_radius=int(params["y_radius"]),
            )
        else:
            _attenuate_fx_family_on_amp(
                amp,
                newamp,
                family,
                q,
                gate,
                max_alpha=eff_alpha,
                ratio_start=params["ratio_start"],
                ratio_full=params["ratio_full"],
            )

    applied = np.clip((amp - newamp) / (amp + 1e-12), 0.0, 1.0).astype(np.float32)
    cleaned = np.real(np.fft.ifft2(np.fft.ifftshift(newamp * phase))) + offset
    removed = x - cleaned
    if np.issubdtype(orig_dtype, np.integer):
        lim = np.iinfo(orig_dtype)
        cleaned_w = np.clip(np.rint(cleaned), lim.min, lim.max).astype(orig_dtype)
    else:
        cleaned_w = cleaned.astype(orig_dtype)
    score = score_removed(removed) if gate > 0 else None
    return {
        "axis": axis,
        "q_pred": float(q_pred),
        "q": float(q),
        "strength": float(strength),
        "gate": gate,
        "cleaned": cleaned_w,
        "removed": removed.astype(np.float32),
        "applied": applied,
        "score": score,
        "passed": bool(score["passed"]) if score else False,
        "removed_rms": float(np.sqrt(np.mean(np.asarray(removed, dtype=np.float64) ** 2))),
    }


def combo_gate_fy(frame: np.ndarray, family: dict, q_pred: float, *, search: int = 10) -> dict:
    """Existing fy notch, but gate uses max(row_contrast, col_contrast at same bin).

    This is the cheap add-on: no extra FFT, one extra column-score per candidate q.
    """
    x = np.asarray(frame, dtype=np.float32)
    logamp = fft_log_amp(x)
    h, w = logamp.shape
    xvalid = _xvalid(w)
    yvalid = _yvalid(h)
    lo = max(MIN_Q, int(round(q_pred)) - search)
    hi = min(h // 2 - MIN_Q, int(round(q_pred)) + search)
    best_q, best_s = float(q_pred), float("-inf")
    for q in range(lo, hi + 1):
        s = max(
            float(row_contrast(logamp, q, xvalid)),
            float(col_contrast(logamp, q, yvalid)),
        )
        if s > best_s:
            best_s = s
            best_q = float(q)
    # Notch still fy-rows at best_q; gate from the combined score.
    result = notch_fy(frame, family, best_q, search=0)
    # Override gate/strength to the combo (search=0 already used row-only).
    # Re-notch with forced gate from combo if row-only missed.
    gate = float(np.clip((best_s - GATE_LOW) / max(1e-9, GATE_HIGH - GATE_LOW), 0.0, 1.0))
    if result["gate"] <= 0 and gate > 0:
        result = _notch_fy_forced(frame, family, best_q, gate, best_s)
    result["combo_strength"] = best_s
    result["combo_q"] = best_q
    result["axis"] = "fy_combo"
    return result


def _notch_fy_forced(frame, family, q, gate, strength) -> dict:
    params = dict(PACK_D)
    params.update(y_sigma=1.0, y_radius=2)
    orig_dtype = frame.dtype
    x = np.asarray(frame, dtype=np.float32)
    offset = float(np.median(x))
    x0 = x - offset
    F = np.fft.fftshift(np.fft.fft2(x0))
    amp = np.abs(F)
    phase = np.exp(1j * np.angle(F))
    newamp = amp.copy()
    eff_alpha = _effective_max_alpha(
        gate,
        strength,
        True,
        max_alpha=params["max_alpha"],
        max_alpha_high=params["max_alpha_high"],
        high_gate=params["high_gate"],
        high_strength=params["high_strength"],
        strength_span=params["strength_span"],
    )
    _attenuate_family_on_amp(
        amp,
        newamp,
        family,
        q,
        gate,
        max_alpha=eff_alpha,
        ratio_start=params["ratio_start"],
        ratio_full=params["ratio_full"],
        y_sigma=params["y_sigma"],
        y_radius=int(params["y_radius"]),
    )
    applied = np.clip((amp - newamp) / (amp + 1e-12), 0.0, 1.0).astype(np.float32)
    cleaned = np.real(np.fft.ifft2(np.fft.ifftshift(newamp * phase))) + offset
    removed = x - cleaned
    if np.issubdtype(orig_dtype, np.integer):
        lim = np.iinfo(orig_dtype)
        cleaned_w = np.clip(np.rint(cleaned), lim.min, lim.max).astype(orig_dtype)
    else:
        cleaned_w = cleaned.astype(orig_dtype)
    score = score_removed(removed)
    return {
        "axis": "fy_combo",
        "q_pred": float(q),
        "q": float(q),
        "strength": float(strength),
        "gate": float(gate),
        "cleaned": cleaned_w,
        "removed": removed.astype(np.float32),
        "applied": applied,
        "score": score,
        "passed": bool(score["passed"]),
        "removed_rms": float(np.sqrt(np.mean(np.asarray(removed, dtype=np.float64) ** 2))),
    }


def probe_frame(frame: np.ndarray, *, fy_seed_q: float | None = 10.0) -> dict:
    logamp = fft_log_amp(frame)
    spatial = spatial_periods(frame)
    axes = fft_axis_scores(logamp)
    trials = []

    h, w = frame.shape
    qy_spatial = spatial["qy_hint"]
    qx_spatial = spatial["qx_hint"]
    qy = qy_spatial if qy_spatial is not None else fy_seed_q
    qx = qx_spatial if qx_spatial is not None else axes["col_peak_q"]

    def _open_fy(q: float) -> dict:
        xw = _xvalid(w).astype(float)
        return {
            "q": float(q),
            "hi": float(h // 2 - q),
            "paired": True,
            "x_weight": xw,
            "row_score": 0.0,
        }

    if qy is not None:
        fam_y = fy_family_from_q(logamp, qy)
        trials.append(("spatial_fy", notch_fy(frame, fam_y, qy, search=10)))
        trials.append(("spatial_fy_open", notch_fy(frame, _open_fy(qy), qy, search=10)))
        trials.append(("combo_fy", combo_gate_fy(frame, fam_y, qy, search=10)))
    if fy_seed_q is not None:
        fam_seed = fy_family_from_q(logamp, fy_seed_q)
        trials.append(("shutter_fy", notch_fy(frame, fam_seed, fy_seed_q, search=10)))

    if qx is not None:
        fam_x = fx_family_from_q(logamp, qx)
        trials.append(("spatial_fx", notch_fx(frame, fam_x, qx, search=10)))

    fam_col = fx_family_from_q(logamp, axes["col_peak_q"])
    trials.append(("fft_col_peak", notch_fx(frame, fam_col, axes["col_peak_q"], search=4)))

    fy_geom = (
        geometry_fy_mask(h, w, qy_spatial) if qy_spatial is not None else np.zeros((h, w), dtype=np.float32)
    )
    fx_geom = (
        geometry_fx_mask(h, w, qx_spatial) if qx_spatial is not None else np.zeros((h, w), dtype=np.float32)
    )
    fy_adapt = (
        seed_mask_at_q((h, w), fy_family_from_q(logamp, qy_spatial), qy_spatial)
        if qy_spatial is not None
        else np.zeros((h, w), dtype=np.float32)
    )
    fx_adapt = (
        fx_seed_image((h, w), fx_family_from_q(logamp, qx_spatial))
        if qx_spatial is not None
        else np.zeros((h, w), dtype=np.float32)
    )

    peak = spectral_peak_mask(h, w, qy=qy_spatial, qx=qx_spatial)
    synth = synthesize_fringe(h, w, qy=qy_spatial, qx=qx_spatial)
    rec_peak = reconstruct_from_mask(frame, peak)
    rec_adapt = reconstruct_from_mask(frame, np.maximum(fy_adapt, fx_adapt))
    orig = np.asarray(frame, dtype=np.float64)

    return {
        "spatial": {k: spatial[k] for k in ("period_px", "q_hint", "qy_hint", "qx_hint", "q_length")},
        "traces": spatial["traces"],
        "fft_axes": {k: axes[k] for k in axes if k != "qs"},
        "fft_curves": axes,
        "logamp": logamp,
        "trials": trials,
        "preview": {
            "fy_geom": fy_geom,
            "fx_geom": fx_geom,
            "fy_adapt": fy_adapt,
            "fx_adapt": fx_adapt,
            "peak": peak,
            "qy": qy_spatial,
            "qx": qx_spatial,
            "synth": synth,
            "rec_peak": rec_peak,
            "rec_adapt": rec_adapt,
            "leftover_peak": (orig - rec_peak).astype(np.float32),
            "leftover_adapt": (orig - rec_adapt).astype(np.float32),
        },
    }


def _fmt(p: float | None) -> str:
    return "-" if p is None else f"{p:.1f}"


def _plot_trace_fit(ax_raw, ax_hp, name: str, tr: dict, sp: dict, titles: dict) -> None:
    t = tr["t"]
    ax_raw.plot(t, tr["raw"], color="0.45", lw=0.8, label="trace")
    ax_raw.plot(t, tr["fit"], color="C1", lw=1.4, label="periodic baseline")
    ax_raw.set_xlim(0, 1)
    ax_raw.tick_params(labelsize=6)
    ax_raw.legend(fontsize=5.5, loc="upper right", frameon=False)
    per = sp["period_px"].get(name)
    qh = sp["q_hint"].get(name)
    axis = tr.get("axis", "?")
    nlen = tr.get("length", tr.get("n"))
    chirp = tr.get("chirp", 0.0)
    chirp_s = f"  chirp α={chirp:+.2f}" if abs(float(chirp)) >= 0.05 else ""
    ax_raw.set_title(
        f"{titles[name]}  P={_fmt(per)} px   q={nlen}/{_fmt(per)}={_fmt(qh)}  → {axis}{chirp_s}",
        fontsize=8,
    )
    ax_hp.plot(t, tr["hp"], color="0.25", lw=0.8, label="leftover (cells / other)")
    ax_hp.axhline(0.0, color="0.7", lw=0.6)
    ax_hp.set_xlim(0, 1)
    ax_hp.set_xlabel("position along trace", fontsize=7)
    ax_hp.tick_params(labelsize=6)
    ax_hp.legend(fontsize=5.5, loc="upper right", frameon=False)


def _overlay_mask(ax, mask: np.ndarray, color, *, alpha: float = 0.45) -> None:
    hit = np.ma.masked_where(mask < 0.05, mask)
    cmap = color
    ax.imshow(hit, cmap=cmap, alpha=alpha, interpolation="nearest", vmin=0.0, vmax=1.0)


def write_probe_pdf(path: Path, frame: np.ndarray, result: dict, *, frame_idx: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = _percentile_limits(frame)
    logamp = result["logamp"]
    slo, shi = _percentile_limits(logamp, (3.0, 99.7))
    curves = result["fft_curves"]
    qs = curves["qs"]

    fig = plt.figure(figsize=(11.69, 8.27))
    with PdfPages(path) as pdf:
        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.97, f"Spatial + fx seed probe  ·  frame {frame_idx}", fontsize=13, fontweight="bold", va="top")
        sp = result["spatial"]
        lines = [
            f"spatial periods (px):  V {_fmt(sp['period_px'].get('vertical'))}  "
            f"H {_fmt(sp['period_px'].get('horizontal'))}  "
            f"TLBR {_fmt(sp['period_px'].get('main'))}  "
            f"TRBL {_fmt(sp['period_px'].get('anti'))}",
            f"y L/C/R px:  {_fmt(sp['period_px'].get('y_left'))} / "
            f"{_fmt(sp['period_px'].get('y_center'))} / {_fmt(sp['period_px'].get('y_right'))}",
            f"hints:  qy={_fmt(sp['qy_hint'])}  qx={_fmt(sp['qx_hint'])}",
            f"FFT peaks:  fy q={curves['row_peak_q']:.0f} sc={curves['row_peak']:.3f}   "
            f"fx q={curves['col_peak_q']:.0f} sc={curves['col_peak']:.3f}   "
            f"max(fy,fx) q={curves['combo_peak_q']:.0f} sc={curves['combo_peak']:.3f}",
            f"gate_low={GATE_LOW:.2f}  (fy-only on this frame was below gate)",
            "",
            "trial           axis       q_pred  q_used  strength  gate   rms     image-test",
            "--------------  ---------  ------  ------  --------  -----  ------  ----------",
        ]
        for name, tr in result["trials"]:
            img_t = "n/a" if tr["gate"] <= 0 else ("PASS" if tr["passed"] else "FAIL")
            lines.append(
                f"{name:<14}  {tr['axis']:<9}  {tr['q_pred']:6.1f}  {tr['q']:6.1f}  "
                f"{tr['strength']:8.3f}  {tr['gate']:5.2f}  {tr['removed_rms']:6.3g}  {img_t}"
            )
        fig.text(0.06, 0.91, "\n".join(lines), fontsize=8, va="top", family="monospace", transform=fig.transFigure)
        pdf.savefig(fig, dpi=140)

        # Traces: backbone + period fit
        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(
            0.06,
            0.97,
            f"Periodic baseline is the fringe  ·  frame {frame_idx}",
            fontsize=12,
            fontweight="bold",
            va="top",
        )
        fig.text(
            0.06,
            0.935,
            "Orange = chirped sinusoid fitted to the trace itself (the ripples). Cells are bright outliers "
            "above that floor — leftover below. Shutter should be almost pure orange; live traces diverge where biology sits. "
            "q = length / P. Horizontal → fx. Vertical → fy. Diagonals mixed (shown, not mixed into qy/qx).",
            fontsize=8,
            va="top",
            color="0.3",
        )
        outer = fig.add_gridspec(2, 2, left=0.06, right=0.98, top=0.90, bottom=0.06, wspace=0.22, hspace=0.32)
        order = ("horizontal", "vertical", "main", "anti")
        titles = {
            "horizontal": "horizontal (along x)",
            "vertical": "vertical (along y)",
            "main": "TL-BR diagonal",
            "anti": "TR-BL diagonal",
        }
        for i, name in enumerate(order):
            inner = outer[i // 2, i % 2].subgridspec(2, 1, hspace=0.08, height_ratios=[1.05, 1.0])
            ax_raw = fig.add_subplot(inner[0])
            ax_hp = fig.add_subplot(inner[1], sharex=ax_raw)
            _plot_trace_fit(ax_raw, ax_hp, name, result["traces"][name], sp, titles)
        pdf.savefig(fig, dpi=140)

        # P → q → mask
        fig.clear()
        fig.patch.set_facecolor("white")
        prev = result["preview"]
        qy, qx = prev["qy"], prev["qx"]
        fig.text(
            0.06,
            0.97,
            f"Period to FFT mask  ·  frame {frame_idx}  ·  qy={_fmt(qy)}  qx={_fmt(qx)}",
            fontsize=12,
            fontweight="bold",
            va="top",
        )
        fig.text(
            0.06,
            0.935,
            "q = N/P. Blobs = conjugate peaks at ±q on the fy/fx axes (the grating). Full rows/columns below are only "
            "where that q sits — inverting them would bandpass the whole FOV, including cells. Right = thin ridge support "
            "(production-like: a few fy × some fx, or a few fx × some fy).",
            fontsize=8,
            va="top",
            color="0.3",
        )
        gs = fig.add_gridspec(2, 3, left=0.04, right=0.99, top=0.89, bottom=0.08, wspace=0.16, hspace=0.28)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(logamp, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest")
        ax.set_title("log |FFT|  (fy up, fx right)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(logamp, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest")
        _overlay_mask(ax, prev["peak"], "Reds")
        ax.set_title("q as conjugate peaks  (not whole rows)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(logamp, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest")
        if qy is not None:
            _overlay_mask(ax, prev["fy_adapt"], "Blues")
        if qx is not None:
            _overlay_mask(ax, prev["fx_adapt"], "Oranges")
        ax.set_title("adapted ridge support at those q", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(prev["fy_geom"], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"fy q location only  qy={_fmt(qy)}  (≠ notch)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[1, 1])
        ax.imshow(prev["fx_geom"], cmap="Oranges", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"fx q location only  qx={_fmt(qx)}  (≠ notch)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[1, 2])
        ax.plot(qs, curves["row"], color="C0", lw=1.0, label="fy row contrast")
        ax.plot(qs, curves["col"], color="C1", lw=1.0, label="fx col contrast")
        ax.axhline(GATE_LOW, color="0.5", ls="--", lw=0.8, label=f"gate_low={GATE_LOW:.2f}")
        if qy is not None:
            ax.axvline(qy, color="C0", ls=":", lw=1.1, label=f"qy hint {_fmt(qy)}")
        if qx is not None:
            ax.axvline(qx, color="C1", ls=":", lw=1.1, label=f"qx hint {_fmt(qx)}")
        ax.set_xlim(5, min(80, int(qs[-1])))
        ax.set_xlabel("q (FFT bins from DC)", fontsize=8)
        ax.set_ylabel("contrast", fontsize=8)
        ax.legend(fontsize=6, loc="upper right", frameon=False)
        ax.set_title("search starts at the hint, then walks nearby", fontsize=8)
        pdf.savefig(fig, dpi=140)

        # Spatial reconstruction of the q-mask (what that guess would remove)
        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(
            0.06,
            0.97,
            f"What this q-guess would remove  ·  frame {frame_idx}",
            fontsize=12,
            fontweight="bold",
            va="top",
        )
        fig.text(
            0.06,
            0.935,
            "Not a full-row/column IFFT (that is a FOV bandpass and looks 'cleaned'). Left: pure cosine at q. "
            "Middle: this frame's energy in those conjugate peaks — stripes, not cells. Right: energy in the "
            "adapted ridge (production-like support). Bottom = original minus that pattern.",
            fontsize=8,
            va="top",
            color="0.3",
        )
        rec_peak = prev["rec_peak"]
        rec_adapt = prev["rec_adapt"]
        leftover_peak = prev["leftover_peak"]
        leftover_adapt = prev["leftover_adapt"]
        synth = prev["synth"]
        rec_lim = _signed_limit(np.concatenate([rec_peak.ravel(), rec_adapt.ravel()]))
        syn_lim = _signed_limit(synth)
        gs = fig.add_gridspec(2, 3, left=0.04, right=0.99, top=0.89, bottom=0.08, wspace=0.16, hspace=0.28)
        rec_panels = (
            (frame, "gray", (vmin, vmax), "original"),
            (synth, "RdBu_r", (-syn_lim, syn_lim), f"intended grating  qy={_fmt(qy)} qx={_fmt(qx)}"),
            (rec_peak, "RdBu_r", (-rec_lim, rec_lim), "ifft of conjugate peaks (this frame)"),
            (rec_adapt, "RdBu_r", (-rec_lim, rec_lim), "ifft of adapted ridge (this frame)"),
            (leftover_peak, "gray", (vmin, vmax), "original − peak ifft"),
            (leftover_adapt, "gray", (vmin, vmax), "original − adapted-ridge ifft"),
        )
        for j, (im, cmap, lim, title) in enumerate(rec_panels):
            ax = fig.add_subplot(gs[j // 3, j % 3])
            ax.imshow(im, cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        pdf.savefig(fig, dpi=140)

        # Each trial
        for name, tr in result["trials"]:
            fig.clear()
            fig.patch.set_facecolor("white")
            color = "#1b7f3a" if tr["passed"] else "#b42318" if tr["gate"] > 0 else "0.35"
            fig.text(
                0.06,
                0.97,
                f"{name}  ·  {tr['axis']}  q {tr['q_pred']:.0f}→{tr['q']:.0f}  "
                f"gate={tr['gate']:.2f}  strength={tr['strength']:.3f}",
                fontsize=12,
                fontweight="bold",
                va="top",
                color=color,
            )
            gs = fig.add_gridspec(2, 4, left=0.04, right=0.99, top=0.90, bottom=0.08, wspace=0.16, hspace=0.28)
            rlim = _signed_limit(tr["removed"])
            panels = (
                (frame, "gray", (vmin, vmax), "original"),
                (tr["cleaned"], "gray", (vmin, vmax), "after"),
                (tr["removed"], "RdBu_r", (-rlim, rlim), f"removed rms={tr['removed_rms']:.3g}"),
                (tr["applied"], "hot", (0.0, max(0.15, float(np.percentile(tr['applied'][tr['applied'] > 0.02], 99)) if np.any(tr['applied'] > 0.02) else 1.0)), "applied FFT"),
            )
            for j, (im, cmap, lim, title) in enumerate(panels):
                ax = fig.add_subplot(gs[0, j])
                ax.imshow(im, cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest")
                ax.set_title(title, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            ax = fig.add_subplot(gs[1, 0])
            ax.imshow(logamp, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest")
            ax.set_title("log |FFT| before", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax = fig.add_subplot(gs[1, 1])
            ax.imshow(fft_log_amp(tr["cleaned"]), cmap="gray", vmin=slo, vmax=shi, interpolation="nearest")
            ax.set_title("log |FFT| after", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax = fig.add_subplot(gs[1, 2:])
            ax.axis("off")
            if tr["score"] is None:
                txt = "gate=0 — nothing written"
            else:
                rows = [f"{'PASS' if tr['passed'] else 'FAIL'}  rms={tr['removed_rms']:.3g}"]
                for t in tr["score"]["traits"]:
                    rows.append(f"  {t['name']:<9} {'Y' if t['passed'] else 'N'}  {t['detail']}")
                txt = "\n".join(rows)
            ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=8, family="monospace", color=color, transform=ax.transAxes)
            pdf.savefig(fig, dpi=140)

    plt.close(fig)
    return path


def result_light(result: dict) -> dict:
    trials = []
    for name, tr in result["trials"]:
        trials.append(
            {
                "name": name,
                "axis": tr["axis"],
                "q_pred": tr["q_pred"],
                "q": tr["q"],
                "strength": tr["strength"],
                "gate": tr["gate"],
                "removed_rms": tr["removed_rms"],
                "passed": tr["passed"],
                "traits": None if tr["score"] is None else tr["score"].get("traits"),
            }
        )
    return {
        "spatial": result["spatial"],
        "fft_axes": result["fft_axes"],
        "trials": trials,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tif", type=Path, default=None)
    ap.add_argument("--frame", type=int, default=160)
    ap.add_argument("--fy-seed", type=float, default=10.0, help="Shutter fy hint to compare against")
    args = ap.parse_args(argv)
    tif = args.tif
    if tif is None:
        tif = Path(r"F:\bPACNewData2026\Haj Grant Example\DATA\ChanA\ChanA_stk.tif")
    if not tif.is_file():
        print(f"MISSING {tif}")
        return 1
    with tifffile.TiffFile(tif) as tf:
        frame = np.asarray(tf.pages[int(args.frame)].asarray())
    print(f"probe frame {args.frame}  {tif}  shape={frame.shape}", flush=True)
    result = probe_frame(frame, fy_seed_q=args.fy_seed)
    sp = result["spatial"]
    print(f"  periods px  V={sp['period_px'].get('vertical')} H={sp['period_px'].get('horizontal')} "
          f"diag={sp['period_px'].get('main')}/{sp['period_px'].get('anti')}", flush=True)
    print(f"  hints qy={sp['qy_hint']} qx={sp['qx_hint']}", flush=True)
    ax = result["fft_axes"]
    print(f"  FFT fy peak q={ax['row_peak_q']:.0f} sc={ax['row_peak']:.3f}  "
          f"fx peak q={ax['col_peak_q']:.0f} sc={ax['col_peak']:.3f}", flush=True)
    for name, tr in result["trials"]:
        print(
            f"  {name}: {tr['axis']} q {tr['q_pred']:.0f}->{tr['q']:.0f} "
            f"str={tr['strength']:.3f} gate={tr['gate']:.2f} rms={tr['removed_rms']:.3g} "
            f"{'PASS' if tr['passed'] else ('FAIL' if tr['gate']>0 else 'off')}",
            flush=True,
        )
    out_dir = tif.parent / "defringe_v22" / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = write_probe_pdf(out_dir / f"frame_{args.frame}.pdf", frame, result, frame_idx=args.frame)
    (out_dir / f"frame_{args.frame}.json").write_text(
        json.dumps(_jsonable(result_light(result)), indent=2), encoding="utf-8"
    )
    print(f"  wrote {pdf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
