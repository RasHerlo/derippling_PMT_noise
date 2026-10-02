"""Figure 04a, Shutter-frames: detected quiet frames and the figure-0a z-profile.

``python -m demo_Sep2026.fig04a_shutter_frames``
"""

from __future__ import annotations

import argparse

import numpy as np
import tifffile

from batch_defringe.shutter_detect import detect_shutter_windows, format_shutter_span

from .common import boxed_text, iter_frames, load_frame, new_figure, save_figure
from .fig00a_averages import ROI_DIAMETER, circular_roi_mask, roi_center
from .paths import CHANA_TIF


def _shutter_limits(frames: list[np.ndarray]) -> tuple[float, float]:
    """Shared median-centred stretch so the row is comparable."""
    mags: list[np.ndarray] = []
    for fr in frames:
        x = np.asarray(fr, dtype=np.float64)
        mags.append(np.abs(x - float(np.median(x))))
    lim = float(np.percentile(np.concatenate([m.ravel() for m in mags]), 99.2)) or 1.0
    return -lim, lim


def _scan_stack() -> dict:
    first, n_frames = load_frame(0)
    mask = circular_roi_mask(first.shape)
    z_profile = np.zeros(n_frames, dtype=np.float64)
    stats: list[dict] = []
    for i, raw, _n in iter_frames():
        x = np.asarray(raw, dtype=np.float64)
        z_profile[i] = float(x[mask].mean())
        stats.append({"frame": int(i), "mean": float(x.mean()), "std": float(x.std())})
        if i == 0 or (i + 1) % 200 == 0 or i + 1 == n_frames:
            print(f"  shutter scan {i + 1}/{n_frames}", flush=True)
    det = detect_shutter_windows(stats)
    return {"n_frames": n_frames, "shape_hw": list(first.shape), "z_profile": z_profile, "det": det}


def draw() -> dict:
    scan = _scan_stack()
    det = scan["det"]
    idxs = [int(i) for i in (det.get("frames") or [])]
    if not idxs:
        raise RuntimeError("no shutter window detected on the demo ChanA stack")

    with tifffile.TiffFile(CHANA_TIF) as tf:
        frames = [np.asarray(tf.pages[i].asarray()) for i in idxs]
    vlo, vhi = _shutter_limits(frames)
    z = scan["z_profile"]
    lo_i, hi_i = idxs[0], idxs[-1]
    cx, cy = roi_center()

    fig = new_figure()
    fig.suptitle("Figure 04a  ·  Shutter-frames  ·  ChanA", fontsize=16, fontweight="bold", y=0.97)
    n = len(frames)
    gs = fig.add_gridspec(
        2,
        n,
        left=0.06,
        right=0.96,
        top=0.86,
        bottom=0.24,
        wspace=0.10,
        hspace=0.42,
        height_ratios=[1.15, 0.90],
    )

    for col, (idx, fr) in enumerate(zip(idxs, frames)):
        ax = fig.add_subplot(gs[0, col])
        x = np.asarray(fr, dtype=np.float64)
        ax.imshow(x - float(np.median(x)), cmap="gray", vmin=vlo, vmax=vhi, interpolation="nearest")
        ax.set_title(f"frame {idx}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("High-contrast shutter", fontsize=10)

    ax = fig.add_subplot(gs[1, :])
    frames_x = np.arange(z.size)
    ax.plot(frames_x, z, color="0.15", lw=0.7)
    ax.axvspan(lo_i - 0.5, hi_i + 0.5, color=(0.85, 0.12, 0.12), alpha=0.28, zorder=0)
    ax.plot(np.arange(lo_i, hi_i + 1), z[lo_i : hi_i + 1], color=(0.85, 0.12, 0.12), lw=1.4)
    ax.annotate(
        f"shutter {lo_i}–{hi_i}",
        xy=(lo_i + 2, float(z[lo_i : hi_i + 1].mean())),
        xytext=(lo_i + 90, float(z.min()) + 40),
        fontsize=9,
        color=(0.75, 0.12, 0.10),
        arrowprops=dict(arrowstyle="->", color=(0.75, 0.12, 0.10), lw=0.8),
    )
    ax.set_title(rf"Mean $z$-profile in the {ROI_DIAMETER}-pixel circular ROI (same as figure 0a)", fontsize=12)
    ax.set_xlabel("frame")
    ax.set_ylabel("mean ADU")
    ax.set_xlim(0, max(z.size - 1, 1))

    boxed_text(
        fig,
        [0.06, 0.02, 0.88, 0.16],
        (
            f"Quiet window from v4 shutter detect: frames {lo_i}–{hi_i} "
            f"({len(idxs)} frames). Each panel is median-centred grayscale, "
            "shared stretch to the 99.2nd percentile of |pixel|.\n"
            f"Trace: figure-0a ROI mean (centre x,y = ({cx:.1f}, {cy:.1f}), "
            f"diameter {ROI_DIAMETER} px). v4 cuts on FOV std, not this ROI mean; "
            "both mark the same cliff on this stack."
        ),
        fontsize=8.5,
        color="0.3",
    )

    numbers = {
        "stack": str(CHANA_TIF),
        "n_frames": scan["n_frames"],
        "shape_hw": scan["shape_hw"],
        "shutter_frames": idxs,
        "shutter_span": format_shutter_span(det),
        "shutter_display_limits": [vlo, vhi],
        "shutter_contrast": "median-centred grayscale, shared ±99.2 percentile of |pixel|",
        "z_profile_min": float(z.min()),
        "z_profile_max": float(z.max()),
        "z_shutter_mean": float(z[lo_i : hi_i + 1].mean()),
        "roi_center_xy": [cx, cy],
        "roi_diameter_px": ROI_DIAMETER,
        "detect_live_std": det.get("live_std"),
        "detect_std_thresh": det.get("std_thresh"),
    }
    try:
        written = save_figure(fig, "fig04a_shutter_frames", numbers)
    except OSError:
        written = save_figure(fig, "fig04a_shutter_frames_new", numbers)
    numbers["written"] = [str(p) for p in written]
    return numbers


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    numbers = draw()
    for p in numbers["written"]:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
