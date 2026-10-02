"""Figure 1, Frame-inspection: one raw frame in real space (left) and Fourier space (right).

``python -m demo_Sep2026.fig01_frame_inspection --frame 1288``
"""

from __future__ import annotations

import argparse

import numpy as np

from .common import _percentile_limits, fft_log_amp, load_frame, new_figure, save_figure


def draw(frame_idx: int) -> dict:
    raw, n_frames = load_frame(frame_idx)
    h, w = raw.shape
    logamp = fft_log_amp(raw)
    cy, cx = h // 2, w // 2

    fig = new_figure()
    fig.suptitle(f"Frame-inspection  ·  ChanA  ·  frame {frame_idx}", fontsize=16, fontweight="bold", y=0.97)
    gs = fig.add_gridspec(1, 2, left=0.04, right=0.97, top=0.88, bottom=0.08, wspace=0.18)

    ax = fig.add_subplot(gs[0])
    lo, hi = _percentile_limits(raw)
    ax.imshow(raw, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title("Real space (image)", fontsize=13)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    ax = fig.add_subplot(gs[1])
    slo, shi = _percentile_limits(logamp, (3.0, 99.7))
    extent = (-cx - 0.5, w - cx - 0.5, h - cy - 0.5, -cy - 0.5)
    ax.imshow(logamp, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest", extent=extent)
    ax.set_title("Fourier space (log amplitude, centre = 0)", fontsize=13)
    ax.set_xlabel("fx (frequency steps)")
    ax.set_ylabel("fy (frequency steps)")

    numbers = {
        "frame": frame_idx,
        "n_frames": n_frames,
        "shape_hw": [h, w],
        "dtype": str(raw.dtype),
        "raw_min": float(raw.min()),
        "raw_median": float(np.median(raw)),
        "raw_max": float(raw.max()),
        "real_display_limits": [lo, hi],
        "fft_display_limits": [slo, shi],
        "fft_note": "log(1 + |FFT|) of frame minus its median, centred (same as v4 fft_log_amp)",
    }
    written = save_figure(fig, f"fig01_frame_inspection_frame{frame_idx}", numbers)
    numbers["written"] = [str(p) for p in written]
    return numbers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", type=int, default=1288)
    args = ap.parse_args(argv)
    numbers = draw(args.frame)
    for p in numbers["written"]:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
