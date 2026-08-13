#!/usr/bin/env python3
"""
deripple.py — remove periodic PMT / scan-electronics fringe noise from .tif stacks.

Core method adapted from Claude's `pmt_fringe_denoise.py` (full-width harmonic
row-band notches + magnitude replacement with phase preserved).

Recommended SUPPORT + raw pipeline
----------------------------------
1. Detect harmonic rows on a SUPPORT / enhanced stack (`--reference`) where
   fringes are clearer.
2. Apply the same notch mask to the RAW stack (`input`).
3. Denoise the defringed raw afterward (avoids SUPPORT inventing block artifacts
   on still-fringed data).

    python deripple.py Raw_frame1t10.tif ^
        --reference Frame1t10.tif ^
        -o Raw_frame1t10_derippled.tif ^
        --diagnostics diagnostics.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


def detect_noise_harmonics(
    reference_stack,
    dc_col_protect=6,
    bg_median_size=9,
    min_peak_distance=15,
    height_sigma=2.5,
):
    """
    Detect Fourier-row offsets (dy from center) where persistent fringe energy
    lives, by averaging magnitude spectra and finding outlier rows.
    """
    stack = np.asarray(reference_stack)
    if stack.ndim == 2:
        stack = stack[None, ...]
    h, w = stack.shape[1:]
    cy, cx = h // 2, w // 2

    mags = [np.abs(np.fft.fftshift(np.fft.fft2(f.astype(np.float64)))) for f in stack]
    mean_mag = np.mean(mags, axis=0)
    logmag = np.log(mean_mag + 1)
    bg = median_filter(logmag, size=bg_median_size)
    resid = logmag - bg

    resid_masked = resid.copy()
    resid_masked[:, max(0, cx - dc_col_protect) : cx + dc_col_protect + 1] = 0
    row_profile = np.max(resid_masked, axis=1)

    med = np.median(row_profile)
    mad = np.median(np.abs(row_profile - med)) + 1e-9
    height_thresh = med + height_sigma * mad

    peaks, _ = find_peaks(row_profile, height=height_thresh, distance=min_peak_distance)
    dy_harmonics = sorted({int(p - cy) for p in peaks})
    return dy_harmonics


def build_rowband_mask(
    shape,
    dy_harmonics,
    sigma_y=2.2,
    dc_col_protect=6,
    dc_radius=6,
):
    """
    Smooth [0..1] attenuation mask: full-width horizontal bands at each
    harmonic row. 1 = fully replace with local magnitude background.
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]

    mask = np.zeros((h, w), dtype=np.float64)
    for dy0 in dy_harmonics:
        g = np.exp(-((yy - cy - dy0) ** 2) / (2 * sigma_y**2))
        mask = np.maximum(mask, g)

    # spare central column (large-scale horizontal structure / gradients)
    mask[:, max(0, cx - dc_col_protect) : cx + dc_col_protect + 1] *= 0.05

    # always protect true DC / very low frequencies
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask[dist < dc_radius] = 0

    return np.clip(mask, 0, 1)


def clean_frame(img, mask, bg_median_size=9):
    """Notch one frame: blend magnitude toward local background; keep phase."""
    img64 = img.astype(np.float64, copy=False)
    F = np.fft.fftshift(np.fft.fft2(img64))
    mag = np.abs(F)
    phase = np.angle(F)
    bg = median_filter(mag, size=bg_median_size)
    newmag = mag * (1 - mask) + bg * mask
    Fc = newmag * np.exp(1j * phase)
    out = np.fft.ifft2(np.fft.ifftshift(Fc)).real
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        return np.clip(np.rint(out), info.min, info.max).astype(img.dtype)
    return out.astype(img.dtype, copy=False)


def clean_stack(
    stack,
    dy_harmonics=None,
    reference_stack=None,
    sigma_y=2.2,
    dc_col_protect=6,
    bg_median_size=9,
    height_sigma=2.5,
):
    """
    Remove periodic fringe noise from a YX / TYX stack.

    Prefer passing ``reference_stack`` (e.g. SUPPORT) for detection and ``stack``
    as the RAW data to clean.
    """
    arr = np.asarray(stack)
    single_frame = arr.ndim == 2
    if single_frame:
        arr = arr[None, ...]

    if dy_harmonics is None:
        ref = reference_stack if reference_stack is not None else arr
        dy_harmonics = detect_noise_harmonics(
            ref,
            dc_col_protect=dc_col_protect,
            bg_median_size=bg_median_size,
            height_sigma=height_sigma,
        )

    mask = build_rowband_mask(
        arr.shape[1:],
        dy_harmonics,
        sigma_y=sigma_y,
        dc_col_protect=dc_col_protect,
    )

    cleaned = np.empty_like(arr)
    for i in range(arr.shape[0]):
        cleaned[i] = clean_frame(arr[i], mask, bg_median_size=bg_median_size)

    if single_frame:
        cleaned = cleaned[0]
    return cleaned, dy_harmonics


def save_diagnostics(path, stack, cleaned, dy_harmonics, mask):
    """Before/after + mask figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame0 = stack[0] if stack.ndim == 3 else stack
    frame0_c = cleaned[0] if cleaned.ndim == 3 else cleaned
    vmin, vmax = np.percentile(frame0, 1), np.percentile(frame0, 99)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(frame0, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Input (frame 0)")
    axes[1].imshow(frame0_c, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("Cleaned (frame 0)")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(f"Notch mask\nharmonics dy={dy_harmonics}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, help="Target .tif to clean (often RAW)")
    p.add_argument("-o", "--output", type=Path, required=True, help="Cleaned .tif path")
    p.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional stack for harmonic detection only (e.g. SUPPORT-enhanced)",
    )
    p.add_argument("--sigma-y", type=float, default=2.2, help="Notch band width (FFT row px)")
    p.add_argument("--dc-col-protect", type=int, default=6, help="Protected DC-column half-width")
    p.add_argument(
        "--height-sigma",
        type=float,
        default=2.5,
        help="Harmonic detection sensitivity (lower = more rows)",
    )
    p.add_argument(
        "--diagnostics",
        type=Path,
        nargs="?",
        const=True,
        default=True,
        help="Diagnostic PNG path. Default: <output_stem>_diagnostics.png next to -o. "
        "Pass a path to override. Use --no-diagnostics to skip.",
    )
    p.add_argument("--no-diagnostics", action="store_true", help="Do not write a diagnostic PNG")
    args = p.parse_args()

    stack = tifffile.imread(args.input)
    ref = tifffile.imread(args.reference) if args.reference else stack
    print(f"target: {args.input}  shape={stack.shape} dtype={stack.dtype}")
    if args.reference:
        print(f"reference: {args.reference}  shape={ref.shape} dtype={ref.dtype}")

    dy_harmonics = detect_noise_harmonics(
        ref, dc_col_protect=args.dc_col_protect, height_sigma=args.height_sigma
    )
    print(f"Detected {len(dy_harmonics)} harmonic rows: {dy_harmonics}")

    cleaned, dy_harmonics = clean_stack(
        stack,
        dy_harmonics=dy_harmonics,
        sigma_y=args.sigma_y,
        dc_col_protect=args.dc_col_protect,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(args.output, cleaned)
    print(f"Wrote {args.output}")

    if not args.no_diagnostics:
        if args.diagnostics is True or args.diagnostics is None:
            diag_path = args.output.with_name(args.output.stem + "_diagnostics.png")
        else:
            diag_path = Path(args.diagnostics)
        shape = stack.shape[1:] if stack.ndim == 3 else stack.shape
        mask = build_rowband_mask(
            shape, dy_harmonics, sigma_y=args.sigma_y, dc_col_protect=args.dc_col_protect
        )
        save_diagnostics(diag_path, stack, cleaned, dy_harmonics, mask)
        print(f"Wrote {diag_path}")


if __name__ == "__main__":
    main()
