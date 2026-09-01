"""Recursive image-domain check on what an adaptive spectral notch removes.

Score-and-print only: does not overwrite production TIFFs and does not write
the catalog. Propose one FFT-family candidate, notch it, judge ``removed`` by
spatial traits (not a period template), then look at leftover or stop.

``python -m batch_defringe.image_check`` writes a dedicated PDF under
``<channel>/defringe_v22/image_check/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import find_objects, gaussian_filter1d, label, map_coordinates

_REPO = Path(__file__).resolve().parents[1]
_GPT = _REPO / "reference" / "gpt"
if str(_GPT) not in sys.path:
    sys.path.insert(0, str(_GPT))

from pmt_fringe_raw_adaptive import detect_families, family_score, fft_log_amp, row_contrast, search_q  # noqa: E402
from pmt_fringe_raw_adaptive_v21 import (  # noqa: E402
    _attenuate_family_on_amp,
    _effective_max_alpha,
    ridge_excess_score,
)

from .library import catalog_status, format_catalog_line, lookup_prior
from .process import PACK_D
from .readout import _jsonable, _percentile_limits, _signed_limit, fft_mask_image
from .seed import EVAL_ANCHOR_FRAMES, TRACK_SEARCH, hydrate_families, library_family_supported
from .shutter_seed_test import (
    SHUTTER_DEFAULT,
    _is_nyquist_self_pair,
    learn_shutter_families,
    low_std_runs,
    scan_frame_stats,
)

LIVE_DEFAULT = EVAL_ANCHOR_FRAMES + (700,)
MAX_ROUNDS = 4
MAX_CONSECUTIVE_REJECTS = 2
Q_CLUSTER_TOL = 3.0
MAX_LEFTOVER_FX_BINS = 80
OUTPUT_SUBDIR = "image_check"

# Trait cuts — printed on the PDF so a human can see why a frame passed/failed.
THRESHOLDS = {
    "coverage_min": 0.40,  # fraction of tiles with structure
    "even_min": 0.35,  # min(left, right) / max(left, right) tile coverage
    "half_min": 0.18,  # each half must itself have some active tiles
    "ridge_frac_min": 0.35,  # strips with a clear y-autocorr peak
    "lag_consistency_min": 0.45,  # 1 - MAD(lag)/median(lag)
    "min_ridge_strips": 3,
    "blob_frac_max": 0.35,  # compact-component area / all-mask area
    "n_compact_max": 4,
    "blob_frac_grain": 0.05,  # tiny compact area = grain on a field, not somata
    "tile_rel": 0.25,  # tile active if RMS > this * max tile RMS
    "n_tiles": 8,
    "n_strips": 16,
    "acf_peak_min": 0.12,
}

DETECT_LEFTOVER = dict(
    row_z_thresh=3.5,
    pair_z_min=2.0,
    x_z_thresh=2.5,
    max_families=4,
    allow_standalone=True,
)


def _xvalid(width: int) -> np.ndarray:
    cx = width // 2
    fx = np.arange(width) - cx
    return (np.abs(fx) > 5) & (np.abs(fx) < cx - 10)


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    med = float(np.median(x))
    return float(np.median(np.abs(x - med))) + 1e-12


HP_FRAC = 0.08  # high-pass sigma as a fraction of the diagonal length
DIAGONALS = ("main", "anti")


def diagonal_sample(img: np.ndarray, which: str = "main") -> tuple[np.ndarray, np.ndarray]:
    """Sample a diagonal. ``main`` = top-left to bottom-right; ``anti`` = top-right to bottom-left.

    ``t=0`` is the top end of that diagonal.
    """
    arr = np.asarray(img, dtype=np.float64)
    h, w = arr.shape
    n = int(max(h, w))
    ys = np.linspace(0.0, h - 1.0, n)
    if which == "anti":
        xs = np.linspace(w - 1.0, 0.0, n)
    else:
        xs = np.linspace(0.0, w - 1.0, n)
    vals = map_coordinates(arr, [ys, xs], order=1, mode="nearest")
    t = np.linspace(0.0, 1.0, n)
    return t, vals


def _highpass_1d(sig: np.ndarray, n: int) -> np.ndarray:
    sigma = max(8.0, HP_FRAC * n)
    return sig - gaussian_filter1d(sig, sigma=sigma)


def _acf_period_px(sig: np.ndarray) -> float | None:
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - float(sig.mean())
    if sig.size < 24 or float(np.std(sig)) < 1e-12:
        return None
    ac = np.correlate(sig, sig, mode="full")
    mid = len(sig) - 1
    ac = ac[mid:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    min_lag = 4
    max_lag = max(min_lag + 2, sig.size // 3)
    best_k = None
    best_val = float(THRESHOLDS["acf_peak_min"])
    for k in range(min_lag + 1, min(max_lag, len(ac) - 1)):
        val = float(ac[k])
        if val >= best_val and val >= float(ac[k - 1]) and val >= float(ac[k + 1]):
            best_val = val
            best_k = k
    return None if best_k is None else float(best_k)


def _period_by_x_third(img: np.ndarray) -> dict[str, float | None]:
    """ACF period (px) along y in left / center / right x-bands of the image."""
    arr = np.asarray(img, dtype=np.float64)
    h, w = arr.shape
    bands = {
        "left": (0, max(1, int(0.25 * w))),
        "center": (int(0.375 * w), max(int(0.375 * w) + 1, int(0.625 * w))),
        "right": (int(0.75 * w), w),
    }
    out: dict[str, float | None] = {}
    for name, (x0, x1) in bands.items():
        strip = arr[:, x0:x1]
        col_std = np.std(strip, axis=0)
        if float(np.max(col_std)) < 1e-12:
            out[name] = None
            continue
        sig = strip[:, int(np.argmax(col_std))]
        hp = _highpass_1d(sig, int(sig.size))
        out[name] = _acf_period_px(hp)
    return out


def _one_diagonal_profiles(
    raw: np.ndarray, cleaned: np.ndarray, removed: np.ndarray, which: str
) -> dict:
    t, raw_p = diagonal_sample(raw, which)
    _, cln_p = diagonal_sample(cleaned, which)
    _, rem_p = diagonal_sample(removed, which)
    n = int(t.size)
    raw_hp = _highpass_1d(raw_p, n)
    cln_hp = _highpass_1d(cln_p, n)
    rms_raw_hp = float(np.sqrt(np.mean(raw_hp * raw_hp)))
    rms_cln_hp = float(np.sqrt(np.mean(cln_hp * cln_hp)))
    rms_rem = float(np.sqrt(np.mean(rem_p * rem_p)))
    written = np.asarray(raw, dtype=np.float64) - np.asarray(cleaned, dtype=np.float64)
    written_rms = float(np.sqrt(np.mean(written * written)))
    return {
        "t": t,
        "raw": raw_p,
        "cleaned": cln_p,
        "removed": rem_p,
        "raw_hp": raw_hp,
        "cleaned_hp": cln_hp,
        "rms_raw_hp": rms_raw_hp,
        "rms_cleaned_hp": rms_cln_hp,
        "rms_removed": rms_rem,
        "written_rms": written_rms,
        "frac_left": float(rms_cln_hp / max(rms_raw_hp, 1e-12)),
        "period_in": _period_by_x_third(raw),
        "period_after": _period_by_x_third(cleaned),
    }


def line_scan_profiles(
    raw: np.ndarray, cleaned: np.ndarray, removed: np.ndarray
) -> dict:
    """Both diagonals. Arrays for plotting; scalars also nested under main/anti."""
    out = {which: _one_diagonal_profiles(raw, cleaned, removed, which) for which in DIAGONALS}
    fracs = [out[w]["frac_left"] for w in DIAGONALS]
    out["frac_left"] = float(np.mean(fracs))
    out["written_rms"] = float(
        np.sqrt(np.mean((np.asarray(raw, dtype=np.float64) - np.asarray(cleaned, dtype=np.float64)) ** 2))
    )
    out["float_removed_rms"] = float(
        np.sqrt(np.mean(np.asarray(removed, dtype=np.float64) ** 2))
    )
    return out


def line_scan_summary(raw: np.ndarray, cleaned: np.ndarray, removed: np.ndarray) -> dict:
    p = line_scan_profiles(raw, cleaned, removed)
    keys = ("rms_raw_hp", "rms_cleaned_hp", "rms_removed", "frac_left", "period_in", "period_after")
    summary = {which: {k: p[which][k] for k in keys} for which in DIAGONALS}
    summary["frac_left"] = p["frac_left"]
    summary["written_rms"] = p["written_rms"]
    summary["float_removed_rms"] = p["float_removed_rms"]
    return summary


def _tile_rms(img: np.ndarray, n: int) -> np.ndarray:
    h, w = img.shape
    th, tw = h // n, w // n
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        y0, y1 = i * th, h if i == n - 1 else (i + 1) * th
        for j in range(n):
            x0, x1 = j * tw, w if j == n - 1 else (j + 1) * tw
            patch = img[y0:y1, x0:x1]
            out[i, j] = float(np.sqrt(np.mean(patch * patch)))
    return out


def _coverage_and_even(removed: np.ndarray) -> tuple[dict, dict]:
    n = int(THRESHOLDS["n_tiles"])
    abs_r = np.abs(np.asarray(removed, dtype=np.float64))
    tiles = _tile_rms(abs_r, n)
    peak = float(np.max(tiles))
    if peak <= 1e-12:
        cov = {
            "name": "coverage",
            "value": 0.0,
            "threshold": THRESHOLDS["coverage_min"],
            "passed": False,
            "detail": "removed is empty",
        }
        even = {
            "name": "even",
            "value": 0.0,
            "threshold": THRESHOLDS["even_min"],
            "passed": False,
            "detail": "removed is empty",
        }
        return cov, even
    active = tiles > (THRESHOLDS["tile_rel"] * peak)
    coverage = float(np.mean(active))
    mid = n // 2
    left = float(np.mean(active[:, :mid])) if mid else 0.0
    right = float(np.mean(active[:, mid:])) if mid else 0.0
    even_ratio = float(min(left, right) / max(left, right, 1e-12))
    cov = {
        "name": "coverage",
        "value": coverage,
        "threshold": THRESHOLDS["coverage_min"],
        "passed": coverage >= THRESHOLDS["coverage_min"],
        "detail": (
            f"{coverage:.2f} of {n}×{n} tiles active "
            f"(RMS > {THRESHOLDS['tile_rel']:.2f}× peak)"
        ),
    }
    even = {
        "name": "even",
        "value": even_ratio,
        "threshold": THRESHOLDS["even_min"],
        "passed": (
            even_ratio >= THRESHOLDS["even_min"]
            and left >= THRESHOLDS["half_min"]
            and right >= THRESHOLDS["half_min"]
        ),
        "detail": (
            f"L={left:.2f} R={right:.2f} ratio={even_ratio:.2f} "
            f"(need both halves ≥{THRESHOLDS['half_min']:.2f}, ratio ≥{THRESHOLDS['even_min']:.2f})"
        ),
    }
    return cov, even


def _ridge_traits(removed: np.ndarray) -> dict:
    n_strips = int(THRESHOLDS["n_strips"])
    arr = np.asarray(removed, dtype=np.float64)
    h, w = arr.shape
    sw = max(4, w // n_strips)
    min_lag = 3
    max_lag = max(min_lag + 1, h // 3)
    lags: list[int] = []
    for i in range(n_strips):
        x0 = i * sw
        x1 = w if i == n_strips - 1 else min(w, x0 + sw)
        strip = arr[:, x0:x1]
        col_std = np.std(strip, axis=0)
        if float(np.max(col_std)) < 1e-12:
            continue
        sig = strip[:, int(np.argmax(col_std))]
        sig = sig - float(sig.mean())
        if float(np.std(sig)) < 1e-12:
            continue
        ac = np.correlate(sig, sig, mode="full")
        mid = len(sig) - 1
        ac = ac[mid:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        hi = min(max_lag, len(ac) - 1)
        if hi <= min_lag + 2:
            continue
        best_k = None
        best_val = THRESHOLDS["acf_peak_min"]
        for k in range(min_lag + 1, hi - 1):
            val = float(ac[k])
            if val >= best_val and val >= float(ac[k - 1]) and val >= float(ac[k + 1]):
                best_val = val
                best_k = k
        if best_k is not None:
            lags.append(best_k)
    ridge_frac = float(len(lags) / max(n_strips, 1))
    if lags:
        lag_arr = np.asarray(lags, dtype=np.float64)
        med = float(np.median(lag_arr))
        rel = float(np.median(np.abs(lag_arr - med)) / max(med, 1e-9))
        lag_consistency = float(max(0.0, 1.0 - rel))
        lag_detail = f"median lag {med:.1f} px, consistency {lag_consistency:.2f}"
    else:
        lag_consistency = 0.0
        lag_detail = "no ACF peaks"
    passed = (
        ridge_frac >= THRESHOLDS["ridge_frac_min"]
        and lag_consistency >= THRESHOLDS["lag_consistency_min"]
        and len(lags) >= int(THRESHOLDS["min_ridge_strips"])
    )
    return {
        "name": "ridges",
        "value": ridge_frac,
        "threshold": THRESHOLDS["ridge_frac_min"],
        "passed": passed,
        "detail": (
            f"{len(lags)}/{n_strips} strips peaked, {lag_detail} "
            f"(need ≥{THRESHOLDS['ridge_frac_min']:.2f} strips, "
            f"consistency ≥{THRESHOLDS['lag_consistency_min']:.2f})"
        ),
        "lag_consistency": lag_consistency,
        "n_strips_peaked": len(lags),
    }


def _is_compact(slc: tuple, mask: np.ndarray) -> bool:
    ys, xs = slc
    patch = mask[ys, xs]
    area = int(np.sum(patch))
    if area < 20 or area > 2000:
        return False
    bh = int(ys.stop - ys.start)
    bw = int(xs.stop - xs.start)
    if min(bh, bw) <= 0:
        return False
    aspect = max(bh, bw) / min(bh, bw)
    extent = area / float(bh * bw)
    return aspect < 2.8 and extent > 0.45


def _blob_analysis(removed: np.ndarray) -> tuple[dict, np.ndarray, list[tuple[int, int, int, int]]]:
    """Blob-rule traits plus the compact components to draw on the report."""
    arr = np.abs(np.asarray(removed, dtype=np.float64))
    floor = 2.5 * _mad(arr)
    mask = arr > floor
    h, w = arr.shape
    compact_mask = np.zeros((h, w), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    empty = {
        "name": "blob",
        "value": 0.0,
        "threshold": THRESHOLDS["blob_frac_max"],
        "passed": True,
        "detail": "no thresholded components (not blob-like)",
        "n_compact": 0,
    }
    n_pix = int(mask.size)
    labeled, n_lab = label(mask)
    if n_lab == 0 or int(np.sum(mask)) == 0:
        return empty, compact_mask, boxes
    slices = find_objects(labeled)
    compact_area = 0
    n_compact = 0
    aspects: list[float] = []
    for i, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        ys, xs = slc
        area = int(np.sum(labeled[slc] == i))
        if area < 20:
            continue
        bh = int(ys.stop - ys.start)
        bw = int(xs.stop - xs.start)
        if min(bh, bw) <= 0:
            continue
        aspect = max(bh, bw) / min(bh, bw)
        aspects.append(aspect)
        if _is_compact(slc, labeled == i):
            n_compact += 1
            compact_area += area
            compact_mask |= labeled == i
            boxes.append((int(ys.start), int(ys.stop), int(xs.start), int(xs.stop)))
    mask_area = int(np.sum(mask))
    blob_frac = float(compact_area / max(mask_area, 1))
    median_aspect = float(np.median(aspects)) if aspects else 0.0
    elongated = median_aspect >= 3.0
    grain_only = blob_frac <= float(THRESHOLDS["blob_frac_grain"])
    passed = elongated or grain_only or (
        blob_frac <= THRESHOLDS["blob_frac_max"] and n_compact <= int(THRESHOLDS["n_compact_max"])
    )
    traits = {
        "name": "blob",
        "value": blob_frac,
        "threshold": THRESHOLDS["blob_frac_max"],
        "passed": passed,
        "detail": (
            f"{n_compact} compact blobs, {blob_frac:.2f} of mask, "
            f"median aspect {median_aspect:.1f}"
            + ("; grain on field (tiny compact area) — pass" if grain_only else "")
            + ("" if passed else " (FAIL: compact/round, not just grain)")
        ),
        "n_compact": n_compact,
        "n_components": int(n_lab),
        "mask_frac": float(mask_area / max(n_pix, 1)),
        "median_aspect": median_aspect,
    }
    return traits, compact_mask, boxes


def _blob_traits(removed: np.ndarray) -> dict:
    traits, _, _ = _blob_analysis(removed)
    return traits


def compact_blob_mask(removed: np.ndarray) -> np.ndarray:
    _, mask, _ = _blob_analysis(removed)
    return mask


def score_removed(removed: np.ndarray) -> dict:
    """Image-domain traits of ``removed``. Does not use the proposed q."""
    rem = np.asarray(removed, dtype=np.float64)
    rms = float(np.sqrt(np.mean(rem * rem)))
    cov, even = _coverage_and_even(rem)
    ridges = _ridge_traits(rem)
    blob = dict(_blob_traits(rem))
    # Compact spots on a passing ridge field are ridge-edge grain, not somata.
    if ridges["passed"]:
        blob["passed"] = True
        if int(blob.get("n_compact") or 0) > 0:
            note = "; ridge-edge compact spots — not a reject"
            if note not in str(blob.get("detail", "")):
                blob["detail"] = str(blob.get("detail", "")) + note
    traits = [cov, even, ridges, blob]
    return {
        "rms": rms,
        "traits": traits,
        "passed": all(bool(t["passed"]) for t in traits),
        "by_name": {t["name"]: t for t in traits},
    }


def family_public_light(fam: dict) -> dict:
    xw = fam.get("x_weight")
    n_fx = int(np.sum(np.asarray(xw) > 0.20)) if xw is not None else 0
    return {
        "q": float(fam["q"]),
        "hi": None if fam.get("hi") is None else float(fam["hi"]),
        "paired": bool(fam.get("paired", True)),
        "row_score": float(fam.get("row_score", 0.0)),
        "fx_ranges": fam.get("fx_ranges"),
        "n_fx_bins": n_fx,
    }


@dataclass
class Candidate:
    source: str
    family: dict
    note: str = ""

    @property
    def q(self) -> float:
        return float(self.family["q"])


@dataclass
class FrameEval:
    frame: int
    role: str
    gate: float
    q: float
    removed_rms: float
    active: bool
    score: dict
    passed: bool
    raw: np.ndarray
    cleaned: np.ndarray
    removed: np.ndarray
    skip_reason: str | None = None
    line_scan: dict = field(default_factory=dict)
    applied: np.ndarray | None = None
    seed_mask: np.ndarray | None = None
    compact_mask: np.ndarray | None = None
    compact_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    q_proposed: float | None = None


@dataclass
class RoundResult:
    round_index: int
    candidate: Candidate
    verdict: str
    reason: str
    frames: list[FrameEval] = field(default_factory=list)

    @property
    def q(self) -> float:
        return self.candidate.q


@dataclass
class ImageCheckResult:
    rounds: list[RoundResult]
    accepted: list[dict]
    stop_reason: str
    eval_index: list[int]
    roles: dict[int, str]
    shutter_idx: list[int]
    catalog: dict | None = None
    low_std_runs: list[list[int]] | None = None
    source_tif: str | None = None
    computer: str | None = None
    channel: str | None = None
    source_frames: dict[int, np.ndarray] = field(default_factory=dict)
    mark_qs: list[float] = field(default_factory=list)


def seed_mask_at_q(shape_hw: tuple[int, int], family: dict, q: float) -> np.ndarray:
    """Seed fx-support at this frame's q (where the notch *may* fire)."""
    fam = dict(family)
    fam["q"] = float(q)
    h = int(shape_hw[0])
    if fam.get("paired", True):
        fam["hi"] = float(h // 2) - float(q)
    else:
        fam["hi"] = None
    return fft_mask_image(shape_hw, [fam])


def notch_one(
    frame: np.ndarray,
    family: dict,
    *,
    q_pred: float | None = None,
    search: int = 2,
    pack: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray, np.ndarray]:
    """Notch one family and return cleaned, removed, tracking, applied, seed.

    ``applied`` is (amp − newamp) / amp in the shifted FFT — a heatmap over fy×fx,
    not a binary full-row mask. ``seed`` is the family's x_weight at this frame's q.
    """
    params = dict(PACK_D)
    params.update(y_sigma=1.0, y_radius=2)
    if pack:
        params.update(pack)
    params["frame_search"] = int(search)
    q0 = float(family["q"] if q_pred is None else q_pred)

    orig_dtype = frame.dtype
    x = np.asarray(frame, dtype=np.float32)
    offset = float(np.median(x))
    x0 = x - offset
    F = np.fft.fftshift(np.fft.fft2(x0))
    amp = np.abs(F)
    phase = np.exp(1j * np.angle(F))
    logamp = np.log1p(amp)
    h, w = x.shape
    fx = np.arange(w) - (w // 2)
    xvalid = (np.abs(fx) > 5) & (np.abs(fx) < (w // 2) - 10)
    newamp = amp.copy()
    tracking: list[dict] = []

    q, strength = search_q(
        logamp,
        q0,
        bool(family.get("paired", True)),
        xvalid,
        int(params["frame_search"]),
    )
    gate = float(
        np.clip(
            (strength - params["gate_low"]) / max(1e-9, params["gate_high"] - params["gate_low"]),
            0.0,
            1.0,
        )
    )
    eff_alpha = _effective_max_alpha(
        gate,
        strength,
        bool(family.get("paired", True)),
        max_alpha=params["max_alpha"],
        max_alpha_high=params["max_alpha_high"],
        high_gate=params["high_gate"],
        high_strength=params["high_strength"],
        strength_span=params["strength_span"],
    )
    entry = {
        "q": q,
        "strength": strength,
        "gate": gate,
        "eff_max_alpha": eff_alpha,
        "residual_pass": 0,
        "residual_strength": 0.0,
    }
    tracking.append(entry)
    if gate > 0:
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
        if params.get("residual_pass", True) and gate >= params["high_gate"] and family.get("paired", True):
            log_after = np.log1p(newamp)
            r_strength = ridge_excess_score(log_after, family, q, xvalid)
            entry["residual_strength"] = float(r_strength)
            if r_strength >= params["residual_strength_min"]:
                residual_gate = float(
                    np.clip(
                        (r_strength - params["residual_strength_min"])
                        / max(1e-9, params["residual_strength_min"]),
                        0.0,
                        1.0,
                    )
                )
                after_src = newamp.copy()
                _attenuate_family_on_amp(
                    after_src,
                    newamp,
                    family,
                    q,
                    residual_gate,
                    max_alpha=params["residual_alpha"],
                    ratio_start=params["ratio_start"],
                    ratio_full=max(params["ratio_full"], params["ratio_start"] + 0.5),
                    y_sigma=params["y_sigma"],
                    y_radius=int(params["y_radius"]),
                )
                entry["residual_pass"] = 1

    applied = np.clip((amp - newamp) / (amp + 1e-12), 0.0, 1.0).astype(np.float32)
    seed = seed_mask_at_q((h, w), family, float(q))
    Fnew = newamp * phase
    cleaned = np.real(np.fft.ifft2(np.fft.ifftshift(Fnew))) + offset
    removed = x - cleaned
    if np.issubdtype(orig_dtype, np.integer):
        lim = np.iinfo(orig_dtype)
        cleaned_write = np.clip(np.rint(cleaned), lim.min, lim.max).astype(orig_dtype)
    else:
        cleaned_write = cleaned.astype(orig_dtype)
    return cleaned_write, removed.astype(np.float32), tracking, applied, seed


def _q_used(q: float, used: list[float]) -> bool:
    return any(abs(q - u) < Q_CLUSTER_TOL for u in used)


def _unique_candidates(cands: list[Candidate], used: list[float]) -> list[Candidate]:
    out: list[Candidate] = []
    seen: list[float] = list(used)
    for c in cands:
        if _q_used(c.q, seen):
            continue
        out.append(c)
        seen.append(c.q)
    return out


def _detect_on_frames(frames: list[np.ndarray]) -> list[dict]:
    if not frames:
        return []
    medspec = np.median(np.stack([fft_log_amp(f) for f in frames]), axis=0)
    height = int(medspec.shape[0])
    families, _, _ = detect_families(medspec, **DETECT_LEFTOVER)
    if not families:
        return []
    hydrated = hydrate_families(families, medspec, x_z_thresh=float(DETECT_LEFTOVER["x_z_thresh"]))
    out = []
    for fam in hydrated:
        if _is_nyquist_self_pair(fam, height):
            continue
        xw = fam.get("x_weight")
        n_fx = int(np.sum(np.asarray(xw) > 0.20)) if xw is not None else 0
        if n_fx > MAX_LEFTOVER_FX_BINS:
            continue
        out.append(fam)
    return out


def _catalog_candidates(
    catalog_families: list[dict],
    medspec: np.ndarray | None,
) -> list[Candidate]:
    out: list[Candidate] = []
    if medspec is None:
        return out
    for src in catalog_families:
        if not library_family_supported(src, medspec):
            continue
        fam = hydrate_families([src], medspec)[0]
        out.append(Candidate(source="catalog", family=fam, note="library hint, fx support on this stack"))
    return out


def propose_candidates(
    leftover_frames: list[np.ndarray],
    *,
    used_q: list[float],
    catalog_families: list[dict] | None = None,
    shutter_families: list[dict] | None = None,
    original_medspec: np.ndarray | None = None,
    extra: list[Candidate] | None = None,
    detect: bool = True,
) -> list[Candidate]:
    cands: list[Candidate] = []
    if extra:
        cands.extend(extra)
    cands.extend(_catalog_candidates(catalog_families or [], original_medspec))
    for fam in shutter_families or []:
        cands.append(
            Candidate(
                source="shutter_hint",
                family=fam,
                note="quiet shutter window (incomplete: families those frames show)",
            )
        )
    if detect:
        for fam in _detect_on_frames(leftover_frames):
            cands.append(
                Candidate(
                    source="leftover_fft",
                    family=fam,
                    note="paired/standalone ridge on leftover eval frames",
                )
            )
    return _unique_candidates(cands, used_q)


def _aggregate_verdict(frames: list[FrameEval]) -> tuple[str, str]:
    active = [f for f in frames if f.active]
    if not active:
        return "inactive", "notch did not gate on any eval frame"
    shutter = [f for f in active if f.role == "shutter"]
    live = [f for f in active if f.role != "shutter"]

    def _majority(group: list[FrameEval]) -> bool:
        return sum(1 for f in group if f.passed) >= (len(group) + 1) // 2

    if shutter and not _majority(shutter):
        n_ok = sum(1 for f in shutter if f.passed)
        return "reject", f"shutter window failed image traits ({n_ok}/{len(shutter)} pass)"
    if live and not _majority(live):
        n_ok = sum(1 for f in live if f.passed)
        return "reject", f"live removed failed image traits ({n_ok}/{len(live)} pass)"
    if not any(f.passed for f in active):
        return "reject", "no active eval frame passed all traits"
    n_ok = sum(1 for f in active if f.passed)
    if not live:
        return (
            "accept",
            f"{n_ok}/{len(active)} active frames passed all traits (no live frame gated - shutter-only)",
        )
    return "accept", f"{n_ok}/{len(active)} active frames passed all traits"


def run_image_check(
    frames: dict[int, np.ndarray],
    *,
    roles: dict[int, str],
    extra_candidates: list[Candidate] | None = None,
    catalog_families: list[dict] | None = None,
    shutter_families: list[dict] | None = None,
    original_medspec: np.ndarray | None = None,
    detect_on_leftover: bool = True,
    max_rounds: int = MAX_ROUNDS,
    pack: dict | None = None,
) -> ImageCheckResult:
    """Propose → notch one family → judge removed → leftover or stop."""
    order = sorted(frames)
    working = {i: np.asarray(frames[i]) for i in order}
    used_q: list[float] = []
    accepted: list[dict] = []
    rounds: list[RoundResult] = []
    consecutive_rejects = 0
    stop_reason = "no candidates"

    for rnd in range(max_rounds):
        leftover_list = [working[i] for i in order]
        cands = propose_candidates(
            leftover_list,
            used_q=used_q,
            catalog_families=catalog_families,
            shutter_families=shutter_families,
            original_medspec=original_medspec,
            extra=extra_candidates,
            detect=detect_on_leftover,
        )
        if not cands:
            stop_reason = "no remaining candidates"
            break
        cand = cands[0]
        used_q.append(cand.q)
        evals: list[FrameEval] = []
        for idx in order:
            role = roles.get(idx, "live")
            search = 2 if role == "shutter" else TRACK_SEARCH
            cleaned, removed, tracking, applied, seed = notch_one(
                working[idx], cand.family, search=search, pack=pack
            )
            gate = float(tracking[0].get("gate", 0.0)) if tracking else 0.0
            q_used = float(tracking[0].get("q", cand.q)) if tracking else cand.q
            rms = float(np.sqrt(np.mean(np.asarray(removed, dtype=np.float64) ** 2)))
            active = gate > 0.0
            score = score_removed(removed)
            _, compact_mask, compact_boxes = _blob_analysis(removed)
            if active:
                passed = bool(score["passed"])
                skip = None
            else:
                passed = False
                skip = "gate=0 (notch inactive)"
            evals.append(
                FrameEval(
                    frame=int(idx),
                    role=role,
                    gate=gate,
                    q=q_used,
                    removed_rms=rms,
                    active=active,
                    score=score,
                    passed=passed,
                    raw=np.asarray(working[idx]),
                    cleaned=np.asarray(cleaned),
                    removed=np.asarray(removed),
                    skip_reason=skip,
                    line_scan=line_scan_summary(working[idx], cleaned, removed),
                    applied=applied,
                    seed_mask=seed,
                    compact_mask=compact_mask,
                    compact_boxes=compact_boxes,
                    q_proposed=float(cand.q),
                )
            )
        verdict, reason = _aggregate_verdict(evals)
        rounds.append(
            RoundResult(round_index=rnd + 1, candidate=cand, verdict=verdict, reason=reason, frames=evals)
        )
        if verdict == "accept":
            accepted.append(family_public_light(cand.family))
            for ev in evals:
                working[ev.frame] = ev.cleaned
            consecutive_rejects = 0
            continue
        if verdict == "inactive":
            continue
        consecutive_rejects += 1
        if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
            stop_reason = "next candidate also failed the image test — stop"
            break
        stop_reason = "rejected a candidate; leftover still open to another proposal"

    if rounds and rounds[-1].verdict == "accept":
        leftover_list = [working[i] for i in order]
        more = propose_candidates(
            leftover_list,
            used_q=used_q,
            catalog_families=catalog_families,
            shutter_families=shutter_families,
            original_medspec=original_medspec,
            detect=detect_on_leftover,
        )
        stop_reason = "accepted; leftover has no further candidate" if not more else stop_reason
        if more and len(rounds) >= max_rounds:
            stop_reason = f"hit max_rounds={max_rounds}"

    mark_qs: list[float] = []
    for rnd in rounds:
        q = float(rnd.q)
        if not any(abs(q - e) < 0.5 for e in mark_qs):
            mark_qs.append(q)
    for fam in shutter_families or []:
        q = float(fam["q"])
        if not any(abs(q - e) < 0.5 for e in mark_qs):
            mark_qs.append(q)

    return ImageCheckResult(
        rounds=rounds,
        accepted=accepted,
        stop_reason=stop_reason,
        eval_index=order,
        roles=dict(roles),
        shutter_idx=[i for i in order if roles.get(i) == "shutter"],
        source_frames={i: np.asarray(frames[i]) for i in order},
        mark_qs=mark_qs,
    )


def result_as_dict(result: ImageCheckResult) -> dict:
    rounds = []
    for rnd in result.rounds:
        frames = []
        for ev in rnd.frames:
            frames.append(
                {
                    "frame": ev.frame,
                    "role": ev.role,
                    "gate": ev.gate,
                    "q": ev.q,
                    "q_proposed": ev.q_proposed,
                    "removed_rms": ev.removed_rms,
                    "active": ev.active,
                    "passed": ev.passed,
                    "skip_reason": ev.skip_reason,
                    "traits": ev.score.get("traits", []),
                    "line_scan": ev.line_scan,
                    "applied_max": float(np.max(ev.applied)) if ev.applied is not None else 0.0,
                    "applied_n_pix": int(np.sum(ev.applied > 0.02)) if ev.applied is not None else 0,
                    "n_compact": int(ev.score.get("by_name", {}).get("blob", {}).get("n_compact", len(ev.compact_boxes))),
                }
            )
        rounds.append(
            {
                "round": rnd.round_index,
                "q": rnd.q,
                "source": rnd.candidate.source,
                "note": rnd.candidate.note,
                "family": family_public_light(rnd.candidate.family),
                "verdict": rnd.verdict,
                "reason": rnd.reason,
                "frames": frames,
            }
        )
    return {
        "source_tif": result.source_tif,
        "computer": result.computer,
        "channel": result.channel,
        "catalog": result.catalog,
        "catalog_line": format_catalog_line(result.catalog),
        "eval_frames": result.eval_index,
        "roles": {str(k): v for k, v in result.roles.items()},
        "shutter_frames": result.shutter_idx,
        "low_std_runs": result.low_std_runs,
        "thresholds": THRESHOLDS,
        "stop_reason": result.stop_reason,
        "accepted": result.accepted,
        "rounds": rounds,
    }


def _trait_color(passed: bool, inactive: bool) -> str:
    if inactive:
        return "0.45"
    return "#1b7f3a" if passed else "#b42318"


def _draw_cover(fig, result: ImageCheckResult) -> None:
    fig.clear()
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.97, "Recursive image-domain check", fontsize=14, fontweight="bold", va="top")
    sub = (
        f"{result.source_tif or ''}  ·  {result.computer or ''} / {result.channel or ''}  ·  "
        "stack not overwritten"
    )
    fig.text(0.06, 0.935, sub, fontsize=8, va="top", color="0.3")
    fig.text(0.06, 0.905, format_catalog_line(result.catalog), fontsize=8, va="top", color="0.2")

    accepted_q = ", ".join(f"{a['q']:.0f}" for a in result.accepted) or "(none)"
    lines = [
        f"Stop: {result.stop_reason}",
        f"Accepted q: {accepted_q}",
        f"Eval frames: {result.eval_index}",
        f"Shutter (quiet window, incomplete inventory): {result.shutter_idx or '(none)'}",
        "",
        "Each round: propose one FFT ridge → adaptive spectral notch → score removed",
        "in image domain (coverage, left–right even, parallel ridges).",
        "q is not an accept criterion. Rejected leftover can try one more candidate;",
        "a second image-test failure stops. Human-in-the-loop can come later.",
        "Cyan = TL-BR diagonal, orange = TR-BL. Shared y-scale on each.",
        "",
        "Round  src            q    verdict    why",
        "-----  -------------  ---  ---------  --------------------------------",
    ]
    for rnd in result.rounds:
        lines.append(
            f"{rnd.round_index:<5}  {rnd.candidate.source:<13}  {rnd.q:>3.0f}  "
            f"{rnd.verdict:<9}  {rnd.reason}"
        )
    if not result.rounds:
        lines.append("(no rounds)")
    fig.text(0.06, 0.87, "\n".join(lines), fontsize=8, va="top", family="monospace", transform=fig.transFigure)

    cuts = [
        "Trait cuts (PASS needs all four on an active frame):",
        f"  coverage  ≥ {THRESHOLDS['coverage_min']:.2f} of {int(THRESHOLDS['n_tiles'])}×{int(THRESHOLDS['n_tiles'])} tiles",
        f"  even      both halves ≥ {THRESHOLDS['half_min']:.2f} tiles, ratio ≥ {THRESHOLDS['even_min']:.2f}",
        f"  ridges    ≥ {THRESHOLDS['ridge_frac_min']:.2f} of strips with similar y-period (not matched to q)",
        f"  blob      compact spots on a ridge field are ignored (ridge-edge grain);",
        "            else compact fraction / count still veto soma-like removal",
        "  inactive  gate=0 — does not vote accept or reject",
        "  family    majority of active shutter AND of active live must pass",
    ]
    fig.text(0.06, 0.22, "\n".join(cuts), fontsize=8, va="bottom", family="monospace", transform=fig.transFigure)


def _overlay_diagonals(ax, shape_hw: tuple[int, int]) -> None:
    h, w = shape_hw
    ax.plot([0, w - 1], [0, h - 1], color="C0", lw=0.9, alpha=0.9)
    ax.plot([w - 1, 0], [0, h - 1], color="C1", lw=0.9, alpha=0.9)


def _overlay_compact_blobs(ax, ev: FrameEval) -> None:
    """Boxes around compact spots the blob-rule counted — shown on pass and fail."""
    from matplotlib.patches import Rectangle

    boxes = ev.compact_boxes or []
    if not boxes:
        return
    blob_failed = False
    if ev.active:
        blob = (ev.score.get("by_name") or {}).get("blob") or {}
        blob_failed = not bool(blob.get("passed", True))
    color = "#ff00aa" if blob_failed else "#00a8e8"
    lw = 1.4 if blob_failed else 0.9
    for y0, y1, x0, x1 in boxes:
        ax.add_patch(
            Rectangle(
                (x0 - 0.5, y0 - 0.5),
                max(x1 - x0, 1),
                max(y1 - y0, 1),
                fill=False,
                edgecolor=color,
                lw=lw,
            )
        )
    mask = ev.compact_mask
    if mask is not None and np.any(mask):
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=0.5, alpha=0.7)
    label = (
        f"{len(boxes)} compact spots (blob fail)"
        if blob_failed
        else f"{len(boxes)} ridge-edge spots (not a reject)"
    )
    ax.plot([], [], color=color, lw=1.5, label=label)
    ax.legend(fontsize=6, loc="upper right", frameon=True, fancybox=False, edgecolor=color)


def _applied_vmax(applied: np.ndarray | None) -> float:
    if applied is None:
        return 1.0
    hit = applied[applied > 0.02]
    if hit.size == 0:
        return 1.0
    return float(max(0.15, np.percentile(hit, 99.0)))


def _imshow_applied(ax, applied: np.ndarray | None, *, title: str) -> None:
    if applied is None:
        ax.set_facecolor("0.92")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    vmax = _applied_vmax(applied)
    ax.imshow(applied, cmap="hot", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _plus_q_zoom_ylim(h: int, q: float, pad: int = 8) -> tuple[int, int]:
    cy = h // 2
    y0 = max(0, cy + int(round(q)) - pad)
    y1 = min(h, cy + int(round(q)) + pad + 1)
    return y0, y1


def _draw_zoom_box(ax, y0: int, y1: int, width: int) -> None:
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (-0.5, y0 - 0.5),
            width,
            max(y1 - y0, 1),
            fill=False,
            linestyle=(0, (2, 2)),
            edgecolor="#7ec8ff",
            lw=1.3,
        )
    )


def _fmt_period(p: float | None) -> str:
    return "-" if p is None else f"{p:.0f}px"


def _plot_one_diagonal(ax_dc, ax_hp, p: dict, *, title: str, xlabel: str) -> None:
    t = p["t"]
    ax_dc.plot(t, p["raw"], color="0.35", lw=0.9, label="leftover in")
    ax_dc.plot(t, p["cleaned"], color="C0", lw=0.9, label="after notch")
    ax_dc.set_ylabel("ADU", fontsize=7)
    ax_dc.set_title(title, fontsize=8)
    ax_dc.legend(fontsize=5.5, loc="upper right", frameon=False)
    ax_dc.tick_params(labelsize=6)
    lo = float(min(np.min(p["raw"]), np.min(p["cleaned"])))
    hi = float(max(np.max(p["raw"]), np.max(p["cleaned"])))
    pad = 0.05 * (hi - lo + 1e-9)
    ax_dc.set_ylim(lo - pad, hi + pad)
    ax_dc.set_xlim(0, 1)

    ax_hp.plot(t, p["raw_hp"], color="0.35", lw=0.8, label=f"in {p['rms_raw_hp']:.3g}")
    ax_hp.plot(t, p["cleaned_hp"], color="C0", lw=0.8, label=f"after {p['rms_cleaned_hp']:.3g}")
    ax_hp.plot(t, p["removed"], color="C3", lw=0.8, alpha=0.85, label=f"removed {p['rms_removed']:.3g}")
    ax_hp.set_ylabel("ADU (high-pass)", fontsize=7)
    ax_hp.set_xlabel(xlabel, fontsize=7)
    ax_hp.set_title(
        f"leftover frac {p['frac_left']:.2f}  period in  "
        f"L {_fmt_period((p.get('period_in') or {}).get('left'))}  "
        f"C {_fmt_period((p.get('period_in') or {}).get('center'))}  "
        f"R {_fmt_period((p.get('period_in') or {}).get('right'))}",
        fontsize=7,
    )
    ax_hp.legend(fontsize=5.5, loc="upper right", frameon=False)
    ax_hp.tick_params(labelsize=6)
    hp_stack = np.concatenate([p["raw_hp"], p["cleaned_hp"], p["removed"]])
    lim = float(np.percentile(np.abs(hp_stack), 99.5))
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    ax_hp.set_ylim(-lim, lim)
    ax_hp.set_xlim(0, 1)
    ax_hp.axvline(0.5, color="0.75", lw=0.5)
    ax_hp.axvline(0.25, color="0.85", lw=0.4, ls=":")
    ax_hp.axvline(0.75, color="0.85", lw=0.4, ls=":")


def _draw_line_scans(ax_dc_main, ax_hp_main, ax_dc_anti, ax_hp_anti, ev: FrameEval) -> None:
    p = line_scan_profiles(ev.raw, ev.cleaned, ev.removed)
    _plot_one_diagonal(
        ax_dc_main,
        ax_hp_main,
        p["main"],
        title="Cyan TL-BR (shared scale)",
        xlabel="TL left  --  center  --  BR right",
    )
    _plot_one_diagonal(
        ax_dc_anti,
        ax_hp_anti,
        p["anti"],
        title="Orange TR-BL (shared scale)",
        xlabel="TR right  --  center  --  BL left",
    )


def _draw_frame_page(fig, rnd: RoundResult, ev: FrameEval) -> None:
    fig.clear()
    fig.patch.set_facecolor("white")
    color = "#1b7f3a" if rnd.verdict == "accept" else "#b42318" if rnd.verdict == "reject" else "0.35"
    fig.text(
        0.06,
        0.97,
        f"Round {rnd.round_index}  ·  q≈{rnd.q:.0f}  ·  {rnd.candidate.source}  ·  "
        f"{rnd.verdict.upper()}  ·  frame {ev.frame} ({ev.role})",
        fontsize=12,
        fontweight="bold",
        va="top",
        color=color,
    )
    fig.text(
        0.06,
        0.935,
        f"{rnd.reason}  ·  {rnd.candidate.note}",
        fontsize=8,
        va="top",
        color="0.3",
    )
    gs = fig.add_gridspec(
        3,
        4,
        left=0.04,
        right=0.99,
        top=0.89,
        bottom=0.06,
        hspace=0.42,
        wspace=0.18,
        height_ratios=[1.05, 0.72, 0.78],
    )
    vmin, vmax = _percentile_limits(ev.raw)
    rlim = _signed_limit(ev.removed)
    applied_max = float(np.max(ev.applied)) if ev.applied is not None else 0.0
    n_app = int(np.sum(ev.applied > 0.02)) if ev.applied is not None else 0
    images = (ev.raw, ev.cleaned, ev.removed)
    cmaps = ("gray", "gray", "RdBu_r")
    limits = ((vmin, vmax), (vmin, vmax), (-rlim, rlim))
    labels = (
        "leftover in  (cyan TL-BR, orange TR-BL)",
        "after this notch",
        f"removed  rms={ev.removed_rms:.3g}  gate={ev.gate:.2f}",
    )
    for j in range(3):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(images[j], cmap=cmaps[j], vmin=limits[j][0], vmax=limits[j][1], interpolation="nearest")
        _overlay_diagonals(ax, ev.raw.shape)
        if j == 2:
            _overlay_compact_blobs(ax, ev)
        ax.set_title(labels[j], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_app = fig.add_subplot(gs[0, 3])
    app_title = (
        f"FFT applied  max={applied_max:.2f}  n={n_app}"
        if applied_max > 0
        else "FFT applied  (gate off — empty)"
    )
    _imshow_applied(ax_app, ev.applied, title=app_title)

    ax_dc_main = fig.add_subplot(gs[1, 0])
    ax_hp_main = fig.add_subplot(gs[2, 0])
    ax_dc_anti = fig.add_subplot(gs[1, 1])
    ax_hp_anti = fig.add_subplot(gs[2, 1])
    _draw_line_scans(ax_dc_main, ax_hp_main, ax_dc_anti, ax_hp_anti, ev)

    ax = fig.add_subplot(gs[1:, 2:])
    ax.axis("off")
    inactive = not ev.active
    head = "INACTIVE" if inactive else ("PASS" if ev.passed else "FAIL")
    head_c = _trait_color(ev.passed, inactive)
    ls = ev.line_scan or {}
    main = ls.get("main") or {}
    anti = ls.get("anti") or {}
    failed = [t for t in ev.score.get("traits", []) if ev.active and not t["passed"]]
    why = ""
    if inactive:
        why = "NOTCH OFF (gate=0) — nothing written"
    elif failed:
        why = "FRAME FAIL: " + ", ".join(t["name"] for t in failed)
    elif ev.passed:
        why = "FRAME PASS"
    rows = [
        f"frame {ev.frame}  {ev.role}  q={ev.q:.0f}  {head}",
        why,
        f"written RMS (in-after) {ls.get('written_rms', float('nan')):.3g}",
        f"float notch RMS        {ls.get('float_removed_rms', ev.removed_rms):.3g}",
        f"leftover frac  mean {ls.get('frac_left', float('nan')):.2f}",
        f"  TL-BR {main.get('frac_left', float('nan')):.2f}   "
        f"TR-BL {anti.get('frac_left', float('nan')):.2f}",
        "period in (L / C / R px):",
        "  TL-BR  "
        f"{_fmt_period((main.get('period_in') or {}).get('left'))} / "
        f"{_fmt_period((main.get('period_in') or {}).get('center'))} / "
        f"{_fmt_period((main.get('period_in') or {}).get('right'))}",
        "  TR-BL  "
        f"{_fmt_period((anti.get('period_in') or {}).get('left'))} / "
        f"{_fmt_period((anti.get('period_in') or {}).get('center'))} / "
        f"{_fmt_period((anti.get('period_in') or {}).get('right'))}",
        f"applied FFT  max {applied_max:.3f}  pixels>0.02 {n_app}",
        f"ridge-edge spots boxed (not a reject): {len(ev.compact_boxes)}",
    ]
    if ev.skip_reason:
        rows.append(ev.skip_reason)
    for t in ev.score.get("traits", []):
        mark = "PASS" if t["passed"] else "FAIL"
        if inactive:
            mark = "n/a"
        rows.append(f"  {t['name']:<9} {mark:<4}  {t['detail']}")
    ax.text(0.0, 1.0, "\n".join(rows), va="top", ha="left", fontsize=7, family="monospace", color=head_c, transform=ax.transAxes)


def _draw_fft_page(
    fig,
    *,
    frame_idx: int,
    role: str,
    image: np.ndarray,
    mark_qs: list[float],
    ev: FrameEval | None = None,
) -> None:
    """Before/after spatial + FFT, plus this frame's applied heatmap."""
    fig.clear()
    fig.patch.set_facecolor("white")
    log_before = fft_log_amp(image)
    h, w = log_before.shape
    cy = h // 2
    xvalid = _xvalid(w)
    gate_low = float(PACK_D["gate_low"])
    qs_axis = np.arange(5, cy - 5)
    scores = np.array([row_contrast(log_before, int(q), xvalid) for q in qs_axis], dtype=float)
    applied = None if ev is None else ev.applied
    seed = None if ev is None else ev.seed_mask
    q_used = None if ev is None else float(ev.q)
    q_prop = None if ev is None or ev.q_proposed is None else float(ev.q_proposed)
    gate = 0.0 if ev is None else float(ev.gate)
    applied_max = float(np.max(applied)) if applied is not None else 0.0
    after_img = image if ev is None else np.asarray(ev.cleaned)
    log_after = fft_log_amp(after_img)
    qz = q_used if q_used is not None else (mark_qs[0] if mark_qs else 10.0)
    zy0, zy1 = _plus_q_zoom_ylim(h, qz)

    fig.text(
        0.06,
        0.97,
        f"FFT  ·  frame {frame_idx} ({role})"
        + (
            f"  ·  proposed q={q_prop:.0f} → used q={q_used:.0f}  gate={gate:.2f}"
            if ev is not None and q_prop is not None
            else (f"  ·  used q={q_used:.0f}  gate={gate:.2f}" if ev is not None else "")
        ),
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.06,
        0.935,
        "Applied heatmap is this frame. Dotted box = zoom region at +q. "
        "Dashed line = proposed q; solid = q after per-frame search.",
        fontsize=8,
        va="top",
        color="0.3",
    )
    gs = fig.add_gridspec(
        2, 4, left=0.04, right=0.99, top=0.89, bottom=0.07, wspace=0.16, hspace=0.28
    )

    vmin, vmax = _percentile_limits(image)
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_raw.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax_raw.set_title("original", fontsize=8)
    ax_raw.set_xticks([])
    ax_raw.set_yticks([])

    ax_cln = fig.add_subplot(gs[0, 1])
    ax_cln.imshow(after_img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax_cln.set_title("after attenuation", fontsize=8)
    ax_cln.set_xticks([])
    ax_cln.set_yticks([])

    slo, shi = _percentile_limits(log_before, (3.0, 99.7))
    ax_fft0 = fig.add_subplot(gs[0, 2])
    ax_fft0.imshow(log_before, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest")
    ax_fft0.set_title("log |FFT|  before", fontsize=8)
    ax_fft0.set_xticks([])
    ax_fft0.set_yticks([])

    ax_fft1 = fig.add_subplot(gs[0, 3])
    ax_fft1.imshow(log_after, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest")
    ax_fft1.set_title("log |FFT|  after", fontsize=8)
    ax_fft1.set_xticks([])
    ax_fft1.set_yticks([])

    ax_seed = fig.add_subplot(gs[1, 0])
    if seed is None:
        ax_seed.set_facecolor("0.92")
        ax_seed.set_title("seed support (no map)", fontsize=8)
    else:
        ax_seed.imshow(seed, cmap="Blues", vmin=0.0, vmax=max(0.2, float(np.max(seed))), interpolation="nearest")
        ax_seed.set_title(f"seed at used q={q_used:.0f}", fontsize=8)
    ax_seed.set_xticks([])
    ax_seed.set_yticks([])

    ax_app = fig.add_subplot(gs[1, 1])
    app_title = (
        f"applied this frame  max={applied_max:.2f}"
        if applied_max > 0.02
        else "applied this frame  (gate off)"
    )
    _imshow_applied(ax_app, applied, title=app_title)
    if applied is not None:
        _draw_zoom_box(ax_app, zy0, zy1, w)

    ax_z = fig.add_subplot(gs[1, 2])
    zoom_src = applied if applied is not None else seed
    if zoom_src is None:
        ax_z.set_facecolor("0.92")
        ax_z.set_title("zoom of dotted box", fontsize=8)
        ax_z.set_xticks([])
        ax_z.set_yticks([])
    else:
        crop = zoom_src[zy0:zy1, :]
        if applied is not None and applied_max > 0.02:
            ax_z.imshow(
                crop,
                cmap="hot",
                vmin=0.0,
                vmax=_applied_vmax(applied),
                interpolation="nearest",
                aspect="auto",
            )
            ax_z.set_title(f"zoom of box  (+q={qz:.0f})", fontsize=8)
        else:
            ax_z.imshow(
                crop,
                cmap="Blues",
                vmin=0.0,
                vmax=max(0.2, float(np.max(crop))),
                interpolation="nearest",
                aspect="auto",
            )
            ax_z.set_title(f"seed zoom  (+q={qz:.0f}, gate off)", fontsize=8)
        ax_z.set_xlabel("fx", fontsize=7)
        ax_z.set_ylabel("fy in box", fontsize=7)
        ax_z.tick_params(labelsize=6)

    q_colors = ["C0", "C1", "C2", "C3"]
    ax2 = fig.add_subplot(gs[1, 3])
    ax2.plot(qs_axis, scores, color="0.2", lw=0.8)
    ax2.axhline(gate_low, color="0.5", ls="--", lw=0.8, label=f"gate_low={gate_low:.2f}")
    for i, q in enumerate(mark_qs):
        col = q_colors[i % len(q_colors)]
        ax2.axvline(q, color=col, lw=0.9, ls="--")
        sc = float(family_score(log_before, int(round(q)), True, xvalid))
        ax2.scatter([q], [sc], color=col, s=28, zorder=3)
        ax2.annotate(
            f"proposed {q:.0f}\n{sc:.3f}",
            (q, sc),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=6.5,
            color=col,
        )
    if q_used is not None:
        scu = float(family_score(log_before, int(round(q_used)), True, xvalid))
        ax2.axvline(q_used, color="C3", lw=1.3, label=f"used q={q_used:.0f}")
        ax2.scatter([q_used], [scu], color="C3", s=36, zorder=4)
        if q_prop is None or abs(q_used - q_prop) >= 0.5:
            ax2.annotate(
                f"used {q_used:.0f}\n{scu:.3f}",
                (q_used, scu),
                textcoords="offset points",
                xytext=(6, -18),
                fontsize=6.5,
                color="C3",
            )
    ax2.set_xlim(5, min(80, cy - 6))
    ax2.set_xlabel("q (FFT bins from DC)", fontsize=8)
    ax2.set_ylabel("row contrast", fontsize=8)
    ax2.set_title("Proposed vs used q", fontsize=8)
    ax2.legend(fontsize=6, loc="upper right", frameon=False)
    ax2.tick_params(labelsize=7)


def _first_round_eval(result: ImageCheckResult, frame_idx: int) -> FrameEval | None:
    if not result.rounds:
        return None
    for ev in result.rounds[0].frames:
        if ev.frame == frame_idx:
            return ev
    return None


def write_image_check_pdf(path: Path, result: ImageCheckResult) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.69, 8.27))
    png_path = path.with_suffix(".png")
    with PdfPages(path) as pdf:
        _draw_cover(fig, result)
        fig.savefig(png_path, dpi=130)
        pdf.savefig(fig, dpi=140)
        mark_qs = list(result.mark_qs or [])
        src = result.source_frames or {}
        fft_order = []
        for i in result.eval_index:
            if result.roles.get(i) != "shutter":
                fft_order.append(i)
        for i in result.shutter_idx[:1]:
            fft_order.append(i)
        for idx in fft_order:
            if idx not in src:
                continue
            _draw_fft_page(
                fig,
                frame_idx=idx,
                role=result.roles.get(idx, "live"),
                image=src[idx],
                mark_qs=mark_qs,
                ev=_first_round_eval(result, idx),
            )
            pdf.savefig(fig, dpi=140)
        for rnd in result.rounds:
            for ev in rnd.frames:
                _draw_frame_page(fig, rnd, ev)
                pdf.savefig(fig, dpi=140)
    plt.close(fig)
    return path


def _pick_eval_frames(
    n: int,
    *,
    shutter: tuple[int, ...] | None,
    live: tuple[int, ...],
    stats: list[dict] | None,
) -> tuple[list[int], dict[int, str], list[int], list[list[int]]]:
    runs = low_std_runs(stats) if stats else []
    shutter_idx: list[int] = []
    if shutter is None:
        if runs:
            shutter_idx = [i for i in runs[0] if 0 <= i < n]
    else:
        shutter_idx = [i for i in shutter if 0 <= i < n]
    live_idx = [i for i in live if 0 <= i < n and i not in shutter_idx]
    roles = {i: "shutter" for i in shutter_idx}
    roles.update({i: "live" for i in live_idx})
    eval_idx = sorted(set(shutter_idx + live_idx))
    return eval_idx, roles, shutter_idx, [[int(x) for x in r] for r in runs]


def run_on_stack(
    tif_path: Path,
    *,
    shutter: tuple[int, ...] | None = SHUTTER_DEFAULT,
    live: tuple[int, ...] = LIVE_DEFAULT,
    out_dir: Path | None = None,
) -> dict:
    from .discover import job_for_stack

    tif_path = Path(tif_path)
    job = job_for_stack(tif_path)
    out_dir = Path(out_dir) if out_dir is not None else tif_path.parent / "defringe_v22" / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    lib_hit = None
    if job.fingerprint:
        lib_hit = lookup_prior(
            computer=job.computer,
            channel=job.channel,
            fingerprint=job.fingerprint,
            recording_date=job.date_utc,
            batch_root=job.trial_dir,
        )
    catalog = catalog_status(lib_hit, used=False)
    catalog_families = (lib_hit or {}).get("families") or []

    with tifffile.TiffFile(tif_path) as tf:
        n = int(tf.series[0].shape[0])
        print(f"  scanning {n} frame stats...", flush=True)
        stats = scan_frame_stats(tf)
        eval_idx, roles, shutter_idx, runs = _pick_eval_frames(
            n, shutter=shutter, live=live, stats=stats
        )
        if not eval_idx:
            raise SystemExit("no eval frames in range")
        loaded = {i: np.asarray(tf.pages[i].asarray()) for i in eval_idx}
        original_medspec = np.median(np.stack([fft_log_amp(loaded[i]) for i in eval_idx]), axis=0)

    shutter_frames = [loaded[i] for i in shutter_idx]
    shutter_families: list[dict] = []
    if shutter_frames:
        shutter_families, _ = learn_shutter_families(shutter_frames)
        print(
            f"  shutter frames {shutter_idx}  families q={[float(f['q']) for f in shutter_families]}",
            flush=True,
        )
    else:
        print("  no shutter quiet window in eval set", flush=True)

    supported = []
    rejected = []
    if catalog_families:
        for fam in catalog_families:
            q = float(fam["q"])
            if library_family_supported(fam, original_medspec):
                supported.append(q)
            else:
                rejected.append(q)
        catalog = catalog_status(
            lib_hit, used=False, supported_qs=supported, rejected_qs=rejected
        )

    result = run_image_check(
        loaded,
        roles=roles,
        catalog_families=catalog_families,
        shutter_families=shutter_families,
        original_medspec=original_medspec,
        detect_on_leftover=True,
    )
    result.source_tif = str(tif_path)
    result.computer = job.computer
    result.channel = job.channel
    result.catalog = catalog
    result.low_std_runs = runs
    result.shutter_idx = shutter_idx

    payload = result_as_dict(result)
    json_path = out_dir / "image_check.json"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    pdf_path = write_image_check_pdf(out_dir / "overview.pdf", result)
    payload["overview_pdf"] = str(pdf_path)
    payload["json_path"] = str(json_path)

    print(f"  stop: {result.stop_reason}", flush=True)
    print(f"  accepted q={[a['q'] for a in result.accepted]}", flush=True)
    for rnd in result.rounds:
        print(
            f"  round {rnd.round_index}: q={rnd.q:.0f} {rnd.candidate.source} -> {rnd.verdict} ({rnd.reason})",
            flush=True,
        )
        for ev in rnd.frames:
            marks = " ".join(
                f"{t['name']}={'Y' if t['passed'] else 'N'}" for t in ev.score.get("traits", [])
            )
            state = "inactive" if not ev.active else ("PASS" if ev.passed else "FAIL")
            n_app = int(np.sum(ev.applied > 0.02)) if ev.applied is not None else 0
            app_max = float(np.max(ev.applied)) if ev.applied is not None else 0.0
            print(
                f"    fr {ev.frame:<5} {ev.role:<7} gate={ev.gate:.2f} q={ev.q:.0f} {state}  "
                f"applied_max={app_max:.2f} n={n_app}  compact={len(ev.compact_boxes)}  "
            f"left={ev.line_scan.get('frac_left', float('nan')):.2f}"
            f"(TLBR={ev.line_scan.get('main', {}).get('frac_left', float('nan')):.2f}"
            f" TRBL={ev.line_scan.get('anti', {}).get('frac_left', float('nan')):.2f})  {marks}",
                flush=True,
            )
    print(f"  wrote {pdf_path}", flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tif",
        type=Path,
        help="Raw stack TIFF. Default: Haj Grant Example ChanA.",
    )
    ap.add_argument(
        "--shutter",
        type=str,
        default="756-760",
        help="Inclusive shutter frame range, e.g. 756-760, or 'auto' for low-std run, or 'none'.",
    )
    ap.add_argument(
        "--live",
        type=str,
        default="160,700,1061",
        help="Comma-separated live frame indices to score.",
    )
    args = ap.parse_args(argv)

    tif = args.tif
    if tif is None:
        tif = Path(r"F:\bPACNewData2026\Haj Grant Example\DATA\ChanA\ChanA_stk.tif")
    if not tif.is_file():
        print(f"MISSING {tif}")
        return 1

    live = tuple(int(x.strip()) for x in args.live.split(",") if x.strip())
    shutter: tuple[int, ...] | None
    key = args.shutter.strip().lower()
    if key == "none":
        shutter = ()
    elif key == "auto":
        shutter = None
    elif "-" in args.shutter:
        a, b = args.shutter.split("-", 1)
        shutter = tuple(range(int(a), int(b) + 1))
    else:
        shutter = tuple(int(x.strip()) for x in args.shutter.split(",") if x.strip())

    print(f"\n{tif}", flush=True)
    run_on_stack(tif, shutter=shutter, live=live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
