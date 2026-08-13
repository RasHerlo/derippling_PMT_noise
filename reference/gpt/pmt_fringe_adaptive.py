#!/usr/bin/env python3
"""Adaptive Fourier-domain removal of narrow PMT/electronic fringe artifacts.

Designed for 2-D TIFF images or 3-D TIFF stacks (frames, y, x).

Key idea
--------
1. Detect recurring, unusually narrow Fourier peaks from a sampled set of frames.
   Detection can use the RAW stack itself (recommended) or an optional processed
   reference stack such as SUPPORT if the fringe is much easier to see there.
2. For every RAW frame, search a few Fourier pixels around each nominal peak,
   estimate the local spectral background, and attenuate only the excess peak.
   This automatically adapts to frame-to-frame changes in fringe phase/position,
   amplitude, and small frequency drift.
3. Save the cleaned TIFF and optional diagnostics/temporal tracking CSV.

The filter does NOT subtract a spatial template and does NOT copy amplitudes from
an optional reference stack into the raw data.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import tifffile
from scipy.ndimage import maximum_filter, median_filter


def _as_stack(a: np.ndarray) -> Tuple[np.ndarray, bool]:
    if a.ndim == 2:
        return a[None, ...], True
    if a.ndim == 3:
        return a, False
    raise ValueError(f"Expected a 2-D image or 3-D stack (frames,y,x); got shape {a.shape}")


def _sample_indices(n: int, n_sample: int) -> np.ndarray:
    n_sample = max(1, min(int(n_sample), n))
    if n_sample == n:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, n_sample).round().astype(int))


def detect_recurring_peaks(
    stack: np.ndarray,
    sample_frames: int = 64,
    local_size: int = 11,
    peak_separation: int = 7,
    dc_radius: float = 20.0,
    detect_z: float = 8.0,
    max_peaks: int = 16,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    """Detect recurring narrow Fourier peaks.

    Returns
    -------
    peaks : list of (dy, dx) offsets from the shifted FFT center
    score : aggregate local log-spectral excess image
    zmap  : robust z-score version of score
    used_indices : sampled frame indices
    """
    s, _ = _as_stack(np.asarray(stack))
    n, h, w = s.shape
    used = _sample_indices(n, sample_frames)

    wy = np.hanning(h).astype(np.float64)[:, None]
    wx = np.hanning(w).astype(np.float64)[None, :]
    window = wy * wx

    residuals = []
    for i in used:
        im = s[i].astype(np.float64, copy=False)
        x = (im - np.median(im)) * window
        F = np.fft.fftshift(np.fft.fft2(x))
        lm = np.log1p(np.abs(F))
        bg = median_filter(lm, size=int(local_size), mode="reflect")
        residuals.append(lm - bg)

    # Median emphasizes peaks recurring at stable frequency while suppressing
    # frame-specific biological structure.
    score = np.median(np.stack(residuals, axis=0), axis=0)

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    valid = rr >= float(dc_radius)

    vals = score[valid]
    med = float(np.median(vals))
    mad = float(1.4826 * np.median(np.abs(vals - med)))
    if mad <= 1e-12:
        mad = float(np.std(vals) + 1e-12)
    zmap = (score - med) / mad

    locmax = score == maximum_filter(score, size=int(peak_separation), mode="reflect")
    coords = np.argwhere(valid & locmax & (zmap >= float(detect_z)))
    ranked = sorted(
        [(float(zmap[y, x]), int(y - cy), int(x - cx)) for y, x in coords],
        reverse=True,
    )
    ranked = ranked[: int(max_peaks)]
    peaks = [(dy, dx) for _, dy, dx in ranked]

    # Keep a stable, readable order by dy then dx after ranking/capping.
    peaks = sorted(peaks)
    return peaks, score, zmap, used


def _annulus_median(mag: np.ndarray, y0: int, x0: int, r_in: float, r_out: float) -> float:
    h, w = mag.shape
    r = int(np.ceil(r_out))
    y1, y2 = max(0, y0 - r), min(h, y0 + r + 1)
    x1, x2 = max(0, x0 - r), min(w, x0 + r + 1)
    patch = mag[y1:y2, x1:x2]
    yy, xx = np.ogrid[y1:y2, x1:x2]
    rr2 = (yy - y0) ** 2 + (xx - x0) ** 2
    mask = (rr2 >= r_in * r_in) & (rr2 <= r_out * r_out)
    v = patch[mask]
    if v.size == 0:
        return float(np.median(patch))
    return float(np.median(v))


def adaptive_filter_frame(
    image: np.ndarray,
    peaks: Sequence[Tuple[int, int]],
    search_radius: int = 2,
    sigma: float = 1.6,
    min_ratio: float = 1.8,
    annulus_inner: float = 3.5,
    annulus_outer: float = 7.0,
    max_reduction: float = 0.98,
) -> Tuple[np.ndarray, List[dict]]:
    """Adaptively attenuate detected Fourier peaks in one raw frame."""
    im = image.astype(np.float64, copy=False)
    h, w = im.shape
    cy, cx = h // 2, w // 2

    F = np.fft.fftshift(np.fft.fft2(im))
    mag = np.abs(F)
    attenuation = np.ones_like(mag, dtype=np.float64)
    tracking: List[dict] = []

    sr = int(search_radius)
    sigma = float(sigma)
    support_r = max(2, int(np.ceil(3.0 * sigma)))

    for dy, dx in peaks:
        yn = cy + int(dy)
        xn = cx + int(dx)
        if not (0 <= yn < h and 0 <= xn < w):
            continue

        # Small local search lets the nominal fringe frequency drift slightly.
        y1, y2 = max(0, yn - sr), min(h, yn + sr + 1)
        x1, x2 = max(0, xn - sr), min(w, xn + sr + 1)
        p = mag[y1:y2, x1:x2]
        iy, ix = np.unravel_index(np.argmax(p), p.shape)
        yp, xp = y1 + int(iy), x1 + int(ix)

        baseline = _annulus_median(mag, yp, xp, annulus_inner, annulus_outer)
        peak_mag = float(mag[yp, xp])
        ratio = peak_mag / max(baseline, 1e-12)

        applied = ratio >= float(min_ratio)
        if applied:
            yy1, yy2 = max(0, yp - support_r), min(h, yp + support_r + 1)
            xx1, xx2 = max(0, xp - support_r), min(w, xp + support_r + 1)
            yy, xx = np.ogrid[yy1:yy2, xx1:xx2]
            d2 = (yy - yp) ** 2 + (xx - xp) ** 2
            weight = np.exp(-0.5 * d2 / (sigma * sigma))

            local_mag = mag[yy1:yy2, xx1:xx2]
            # Only reduce coefficients that rise above the estimated local
            # background. Preserve their phase and smoothly blend to untouched
            # neighboring coefficients.
            target_factor = np.minimum(1.0, baseline / np.maximum(local_mag, 1e-12))
            reduction = weight * (1.0 - target_factor)
            reduction = np.minimum(reduction, float(max_reduction))
            local_att = 1.0 - reduction
            attenuation[yy1:yy2, xx1:xx2] = np.minimum(
                attenuation[yy1:yy2, xx1:xx2], local_att
            )

        tracking.append(
            {
                "nominal_dy": int(dy),
                "nominal_dx": int(dx),
                "actual_dy": int(yp - cy),
                "actual_dx": int(xp - cx),
                "peak_to_background": float(ratio),
                "applied": int(applied),
            }
        )

    Ff = F * attenuation
    out = np.fft.ifft2(np.fft.ifftshift(Ff)).real
    return out, tracking


def _cast_like(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(x), info.min, info.max).astype(dtype)
    return x.astype(dtype, copy=False)


def save_detection_png(path: Path, zmap: np.ndarray, peaks: Sequence[Tuple[int, int]]) -> None:
    import matplotlib.pyplot as plt

    h, w = zmap.shape
    cy, cx = h // 2, w // 2
    fig, ax = plt.subplots(figsize=(7, 7))
    vmax = max(3.0, float(np.percentile(zmap, 99.95)))
    im = ax.imshow(zmap, origin="upper", vmin=-1, vmax=vmax)
    if peaks:
        xs = [cx + dx for dy, dx in peaks]
        ys = [cy + dy for dy, dx in peaks]
        ax.scatter(xs, ys, s=60, facecolors="none", edgecolors="white", linewidths=1.2)
    ax.set_title("Recurring narrow spectral peaks (robust z-score)")
    ax.set_xlabel("Fourier x pixel")
    ax.set_ylabel("Fourier y pixel")
    fig.colorbar(im, ax=ax, shrink=0.8, label="local spectral excess (robust z)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input", help="Raw TIFF image or stack to clean")
    ap.add_argument("-o", "--output", required=True, help="Output cleaned TIFF")
    ap.add_argument("--reference", default=None,
                    help="Optional TIFF used ONLY to detect fringe frequencies (e.g. SUPPORT). Raw input is filtered.")
    ap.add_argument("--sample-frames", type=int, default=64,
                    help="Number of evenly spaced frames used for frequency detection")
    ap.add_argument("--detect-z", type=float, default=8.0,
                    help="Robust z threshold for recurring spectral peak detection")
    ap.add_argument("--max-peaks", type=int, default=16,
                    help="Maximum number of individual Fourier peaks (including conjugates)")
    ap.add_argument("--dc-radius", type=float, default=20.0,
                    help="Do not detect peaks this close to Fourier DC")
    ap.add_argument("--search-radius", type=int, default=2,
                    help="Per-frame local frequency drift allowed around each nominal peak")
    ap.add_argument("--sigma", type=float, default=1.6,
                    help="Gaussian width (Fourier pixels) of adaptive attenuation")
    ap.add_argument("--min-ratio", type=float, default=1.8,
                    help="Per-frame peak/local-background ratio required before filtering")
    ap.add_argument("--diagnostics-dir", default=None,
                    help="Optional directory for detected-peaks PNG, temporal CSV, and summary text")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.output)
    raw0 = tifffile.imread(input_path)
    raw, was_2d = _as_stack(raw0)

    if args.reference:
        ref0 = tifffile.imread(args.reference)
        ref, _ = _as_stack(ref0)
        if ref.shape[-2:] != raw.shape[-2:]:
            raise ValueError(f"Reference XY shape {ref.shape[-2:]} != raw XY shape {raw.shape[-2:]}")
    else:
        ref = raw

    peaks, score, zmap, used = detect_recurring_peaks(
        ref,
        sample_frames=args.sample_frames,
        dc_radius=args.dc_radius,
        detect_z=args.detect_z,
        max_peaks=args.max_peaks,
    )

    if not peaks:
        raise RuntimeError(
            "No recurring narrow Fourier peaks passed the detection threshold. "
            "Inspect diagnostics or try a lower --detect-z (for example 6)."
        )

    print("Detected nominal Fourier peaks (dy, dx):")
    for p in peaks:
        print(f"  {p}")

    cleaned = np.empty_like(raw)
    all_tracking: List[dict] = []
    for i in range(raw.shape[0]):
        out, tr = adaptive_filter_frame(
            raw[i],
            peaks,
            search_radius=args.search_radius,
            sigma=args.sigma,
            min_ratio=args.min_ratio,
        )
        cleaned[i] = _cast_like(out, raw.dtype)
        for row in tr:
            row = dict(row)
            row["frame"] = i
            all_tracking.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_write = cleaned[0] if was_2d else cleaned
    tifffile.imwrite(out_path, to_write, photometric="minisblack")

    if args.diagnostics_dir:
        d = Path(args.diagnostics_dir)
        d.mkdir(parents=True, exist_ok=True)
        save_detection_png(d / "detected_peaks.png", zmap, peaks)
        with open(d / "temporal_tracking.csv", "w", newline="") as f:
            fieldnames = [
                "frame", "nominal_dy", "nominal_dx", "actual_dy", "actual_dx",
                "peak_to_background", "applied"
            ]
            wri = csv.DictWriter(f, fieldnames=fieldnames)
            wri.writeheader()
            for row in all_tracking:
                wri.writerow(row)
        with open(d / "summary.txt", "w") as f:
            f.write(f"input={input_path}\n")
            f.write(f"reference={args.reference or input_path}\n")
            f.write(f"shape={raw.shape}\n")
            f.write(f"sampled_frames={used.tolist()}\n")
            f.write(f"detected_peaks={peaks}\n")
            if all_tracking:
                ratios = np.array([r["peak_to_background"] for r in all_tracking], dtype=float)
                applied = np.array([r["applied"] for r in all_tracking], dtype=int)
                f.write(f"median_peak_to_background={np.median(ratios):.4g}\n")
                f.write(f"filter_application_fraction={applied.mean():.4f}\n")

    print(f"Saved cleaned TIFF: {out_path}")
    if args.diagnostics_dir:
        print(f"Saved diagnostics: {args.diagnostics_dir}")


if __name__ == "__main__":
    main()
