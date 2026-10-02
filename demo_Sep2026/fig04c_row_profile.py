"""Figure 04c: collapse S_med to a 1-D row profile, Z-score, and first-pass ridge rows.

``python -m demo_Sep2026.fig04c_row_profile``
"""

from __future__ import annotations

import argparse

import numpy as np
import tifffile

from .common import _percentile_limits, boxed_text, fft_log_amp, new_figure, save_figure
from .fig04b_median_fourier import _shutter_indices
from .paths import CHANA_TIF
from pmt_fringe_raw_adaptive import robust_local_z  # noqa: E402

ROW_Z_THRESH = 3.0  # first pass in learn_shutter_families
DC_FX = 5
EDGE_MARGIN = 10


def _median_spectrum(idxs: list[int]) -> np.ndarray:
    with tifffile.TiffFile(CHANA_TIF) as tf:
        specs = [fft_log_amp(tf.pages[int(i)].asarray()) for i in idxs]
    return np.median(np.stack(specs, axis=0), axis=0)


def _row_collapse(medspec: np.ndarray) -> dict:
    h, w = medspec.shape
    cy, cx = h // 2, w // 2
    fx = np.arange(w) - cx
    xvalid = (np.abs(fx) > DC_FX) & (np.abs(fx) < cx - EDGE_MARGIN)
    row_profile = np.percentile(medspec[:, xvalid], 95, axis=1)
    row_z, row_diff = robust_local_z(row_profile, radius=8, exclude=1)
    candidates: list[dict] = []
    for dy in range(5, cy - 5):
        y = cy + dy
        if row_z[y] < ROW_Z_THRESH:
            continue
        lo, hi = max(cy + 5, y - 2), min(h - 5, y + 3)
        if row_z[y] >= np.max(row_z[lo:hi]):
            candidates.append(
                {
                    "dy": int(dy),
                    "y": int(y),
                    "fy": float(dy),
                    "row_z": float(row_z[y]),
                    "row_diff": float(row_diff[y]),
                    "row_profile": float(row_profile[y]),
                }
            )
    candidates.sort(key=lambda c: c["row_z"], reverse=True)
    return {
        "row_profile": np.asarray(row_profile, dtype=np.float64),
        "row_z": np.asarray(row_z, dtype=np.float64),
        "xvalid": xvalid,
        "candidates": candidates,
        "cy": cy,
        "cx": cx,
    }


def draw() -> dict:
    idxs = _shutter_indices()
    medspec = _median_spectrum(idxs)
    h, w = medspec.shape
    coll = _row_collapse(medspec)
    cy, cx = coll["cy"], coll["cx"]
    fy = np.arange(h) - cy
    cands = coll["candidates"]
    extent = (-cx - 0.5, w - cx - 0.5, h - cy - 0.5, -cy - 0.5)
    slo, shi = _percentile_limits(medspec, (3.0, 99.7))
    edge = cx - EDGE_MARGIN

    fig = new_figure()
    fig.suptitle("Figure 04c  ·  Row profile and ridge proposals  ·  ChanA", fontsize=16, fontweight="bold", y=0.97)
    gs = fig.add_gridspec(1, 3, left=0.055, right=0.94, top=0.88, bottom=0.38, wspace=0.30)

    ax = fig.add_subplot(gs[0])
    ax.imshow(medspec, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest", extent=extent)
    ax.axvspan(-DC_FX, DC_FX, color=(0.85, 0.15, 0.12), alpha=0.22, lw=0)
    ax.axvspan(-cx, -edge, color=(0.20, 0.35, 0.80), alpha=0.18, lw=0)
    ax.axvspan(edge, cx, color=(0.20, 0.35, 0.80), alpha=0.18, lw=0)
    for c in cands:
        ax.axhline(c["fy"], color=(1.0, 0.85, 0.15), lw=0.9, alpha=0.95)
    ax.set_xlim(-256, 255)
    ax.set_ylim(255, -256)
    ax.set_xlabel(r"$f_x$")
    ax.set_ylabel(r"$f_y$")
    ax.set_title(r"$S_{\mathrm{med}}$  (excluded $f_x$ shaded)", fontsize=11)

    axp = fig.add_subplot(gs[1])
    axp.plot(coll["row_profile"], fy, color="0.15", lw=0.9)
    for c in cands:
        axp.plot(c["row_profile"], c["fy"], "o", color=(0.85, 0.15, 0.12), ms=4.5, zorder=3)
    axp.set_ylim(255, -256)
    axp.set_xlabel("95th percentile on $\\mathcal{V}$")
    axp.set_ylabel(r"$f_y$")
    axp.set_title(r"Row profile", fontsize=11)

    axz = fig.add_subplot(gs[2])
    axz.plot(coll["row_z"], fy, color="0.15", lw=0.9)
    axz.axvline(ROW_Z_THRESH, color=(0.85, 0.12, 0.12), ls="--", lw=1.1, label=rf"$Z={ROW_Z_THRESH:.1f}$")
    axz.axhspan(-5, 5, color="0.85", alpha=0.45, lw=0, zorder=0)
    label_fy = {10, 30, 128, 246}
    for c in cands:
        axz.plot(c["row_z"], c["fy"], "o", color=(0.85, 0.15, 0.12), ms=4.5, zorder=3)
        if int(c["dy"]) in label_fy:
            axz.annotate(
                rf"$f_y=+{c['dy']}$",
                (c["row_z"], c["fy"]),
                textcoords="offset points",
                xytext=(-52, 3),
                fontsize=8,
                color=(0.55, 0.08, 0.08),
            )
    zmax = max(float(coll["row_z"].max()), ROW_Z_THRESH) * 1.08
    axz.set_xlim(-2.0, zmax)
    axz.set_ylim(255, -256)
    axz.set_xlabel(r"row $Z$ (robust, radius 8)")
    axz.set_ylabel(r"$f_y$")
    axz.set_title("Ridge-row proposals", fontsize=11)
    axz.legend(fontsize=8, loc="lower right", frameon=False)

    boxed_text(
        fig,
        [0.05, 0.02, 0.90, 0.26],
        (
            "Red band on S_med: DC neighbourhood |fx| <= 5. Blue bands: outer edge |fx| >= cx-10. "
            "The 95th percentile of each row is taken only on the remaining columns V."
            "\n"
            "row Z is a robust local score of that profile (median/MAD, window +/-8, exclude the bin itself). "
            f"Yellow lines / red dots: first-pass proposals = local maxima of row Z with Z >= {ROW_Z_THRESH:.1f} "
            "and dy in 5 ... cy-5 (positive fy only). Labels mark fy = +10, +30, +128, +246. "
            "Pairing is figure 04d; ranking and harmonics are figure 04e."
        ),
        fontsize=8.5,
        color="0.3",
    )

    numbers = {
        "stack": str(CHANA_TIF),
        "shutter_frames": idxs,
        "shape_hw": [h, w],
        "dc_fx": DC_FX,
        "edge_margin": EDGE_MARGIN,
        "row_z_thresh": ROW_Z_THRESH,
        "n_proposals": len(cands),
        "proposals_fy": [c["fy"] for c in cands],
        "proposals_row_z": [c["row_z"] for c in cands],
        "note": "first-pass detect_families candidates on S_med; no pairing yet",
    }
    try:
        written = save_figure(fig, "fig04c_row_profile", numbers)
    except OSError:
        written = save_figure(fig, "fig04c_row_profile_new", numbers)
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
