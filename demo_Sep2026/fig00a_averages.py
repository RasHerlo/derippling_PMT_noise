"""Figure 0a: mean real-space frame, circular ROI z-profile, mean Fourier frame.

``python -m demo_Sep2026.fig00a_averages``
"""

from __future__ import annotations

import argparse

import numpy as np

from .common import _percentile_limits, boxed_text, fft_log_amp, iter_frames, load_frame, new_figure, save_figure
from .paths import CHANA_TIF

ROI_X = (96, 116)
ROI_Y = (305, 325)
ROI_DIAMETER = 20
SHUTTER = (756, 760)


def roi_center(x_range=ROI_X, y_range=ROI_Y) -> tuple[float, float]:
    """Centre of the half-open pixel box ``x[x0:x1]``, ``y[y0:y1]``."""
    return 0.5 * (x_range[0] + x_range[1] - 1), 0.5 * (y_range[0] + y_range[1] - 1)


def circular_roi_mask(shape_hw: tuple[int, int], *, x_range=ROI_X, y_range=ROI_Y, diameter: float = ROI_DIAMETER) -> np.ndarray:
    h, w = shape_hw
    cx, cy = roi_center(x_range, y_range)
    radius = float(diameter) / 2.0
    yy, xx = np.indices((h, w))
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius + 1e-9


def accumulate(max_frames: int | None = None) -> dict:
    first, n_frames = load_frame(0)
    h, w = first.shape
    n_use = n_frames if max_frames is None else min(int(max_frames), n_frames)
    mask = circular_roi_mask((h, w))
    n_roi = int(mask.sum())
    if n_roi == 0:
        raise RuntimeError("circular ROI is empty")

    sum_real = np.zeros((h, w), dtype=np.float64)
    sum_fft = np.zeros((h, w), dtype=np.float64)
    z_profile = np.zeros(n_use, dtype=np.float64)

    for i, raw, _n in iter_frames():
        if i >= n_use:
            break
        x = np.asarray(raw, dtype=np.float64)
        sum_real += x
        sum_fft += fft_log_amp(raw)
        z_profile[i] = float(x[mask].mean())
        if i == 0 or (i + 1) % 200 == 0 or i + 1 == n_use:
            print(f"  average frames {i + 1}/{n_use}", flush=True)

    inv = 1.0 / float(n_use)
    return {
        "n_frames": n_frames,
        "n_used": n_use,
        "shape_hw": [h, w],
        "mean_real": sum_real * inv,
        "mean_fft": sum_fft * inv,
        "z_profile": z_profile,
        "roi_n_pixels": n_roi,
        "roi_center_xy": list(roi_center()),
        "mask": mask,
    }


def draw(*, max_frames: int | None = None) -> dict:
    from matplotlib.patches import Circle

    acc = accumulate(max_frames=max_frames)
    mean_real = acc["mean_real"]
    mean_fft = acc["mean_fft"]
    z = acc["z_profile"]
    h, w = acc["shape_hw"]
    cy_fft, cx_fft = h // 2, w // 2
    cx, cy = roi_center()
    radius = ROI_DIAMETER / 2.0

    fig = new_figure()
    fig.suptitle("Figure 0a  ·  Stack averages  ·  ChanA", fontsize=16, fontweight="bold", y=0.97)
    gs = fig.add_gridspec(2, 2, left=0.05, right=0.98, top=0.88, bottom=0.11, wspace=0.22, hspace=0.32, height_ratios=[1.35, 0.85])

    ax = fig.add_subplot(gs[0, 0])
    lo, hi = _percentile_limits(mean_real)
    ax.imshow(mean_real, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.add_patch(Circle((cx, cy), radius, fill=False, edgecolor=(1.0, 0.85, 0.1), linewidth=1.8))
    ax.set_title("Mean real space", fontsize=12)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    ax = fig.add_subplot(gs[0, 1])
    fft_show = mean_fft.copy()
    fft_show[cy_fft - 2 : cy_fft + 3, cx_fft - 2 : cx_fft + 3] = np.median(fft_show)
    slo, shi = np.percentile(fft_show, (8.0, 99.9))
    extent = (-cx_fft - 0.5, w - cx_fft - 0.5, h - cy_fft - 0.5, -cy_fft - 0.5)
    ax.imshow(mean_fft, cmap="gray", vmin=float(slo), vmax=float(shi), interpolation="nearest", extent=extent)
    ax.set_title("Mean Fourier space", fontsize=12)
    ax.set_xlabel("fx (frequency steps)")
    ax.set_ylabel("fy (frequency steps)")

    ax = fig.add_subplot(gs[1, :])
    frames = np.arange(z.size)
    ax.plot(frames, z, color="0.15", lw=0.7)
    ax.axvspan(SHUTTER[0] - 0.5, SHUTTER[1] + 0.5, color=(0.85, 0.12, 0.12), alpha=0.28, zorder=0)
    ax.plot(np.arange(SHUTTER[0], SHUTTER[1] + 1), z[SHUTTER[0] : SHUTTER[1] + 1], color=(0.85, 0.12, 0.12), lw=1.4)
    ax.annotate(
        f"shutter {SHUTTER[0]}–{SHUTTER[1]}",
        xy=(SHUTTER[0] + 2, float(z[SHUTTER[0] : SHUTTER[1] + 1].mean())),
        xytext=(SHUTTER[0] + 90, float(z.min()) + 40),
        fontsize=9,
        color=(0.75, 0.12, 0.10),
        arrowprops=dict(arrowstyle="->", color=(0.75, 0.12, 0.10), lw=0.8),
    )
    ax.set_title(rf"Mean $z$-profile in the {ROI_DIAMETER}-pixel circular ROI", fontsize=12)
    ax.set_xlabel("frame")
    ax.set_ylabel("mean ADU")
    ax.set_xlim(0, max(z.size - 1, 1))

    boxed_text(
        fig,
        [0.045, 0.015, 0.91, 0.08],
        (
            f"ROI centre x,y = ({cx:.1f}, {cy:.1f}), diameter {ROI_DIAMETER} px, "
            f"box x[{ROI_X[0]}:{ROI_X[1]}] y[{ROI_Y[0]}:{ROI_Y[1]}].  "
            f"Red band: shutter frames {SHUTTER[0]}–{SHUTTER[1]}.  "
            "Fourier panel is the mean of per-frame log(1+|FFT|), not the FFT of the mean image."
        ),
        fontsize=8.5,
        color="0.3",
    )

    numbers = {
        "stack": str(CHANA_TIF),
        "n_frames": acc["n_frames"],
        "n_used": acc["n_used"],
        "shape_hw": acc["shape_hw"],
        "roi_x": list(ROI_X),
        "roi_y": list(ROI_Y),
        "roi_diameter_px": ROI_DIAMETER,
        "roi_center_xy": acc["roi_center_xy"],
        "roi_n_pixels": acc["roi_n_pixels"],
        "z_profile_min": float(z.min()),
        "z_profile_max": float(z.max()),
        "z_profile_mean": float(z.mean()),
        "z_shutter_mean": float(z[SHUTTER[0] : SHUTTER[1] + 1].mean()) if z.size > SHUTTER[1] else None,
        "real_display_limits": [lo, hi],
        "fft_display_limits": [float(slo), float(shi)],
        "fft_note": "mean of per-frame log(1+|FFT|) after each frame's median subtraction",
    }
    written = save_figure(fig, "fig00a_stack_averages", numbers)
    numbers["written"] = [str(p) for p in written]
    return numbers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-frames", type=int, default=None, help="Use only the first N frames (debug)")
    args = ap.parse_args(argv)
    numbers = draw(max_frames=args.max_frames)
    for p in numbers["written"]:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
