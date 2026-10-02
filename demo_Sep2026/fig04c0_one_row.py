"""Figure 04c0: how one ridge row is proposed (fy = +10).

Goes between 04b (median spectrum) and 04c (all proposals).

``python -m demo_Sep2026.fig04c0_one_row``
"""

from __future__ import annotations

import argparse

import numpy as np

from .common import _percentile_limits, boxed_text, new_figure, save_figure
from .fig04b_median_fourier import _shutter_indices
from .fig04c_row_profile import DC_FX, EDGE_MARGIN, ROW_Z_THRESH, _median_spectrum, _row_collapse
from .paths import CHANA_TIF

EXAMPLE_FY = 10
WIN = 8


def draw() -> dict:
    idxs = _shutter_indices()
    medspec = _median_spectrum(idxs)
    h, w = medspec.shape
    coll = _row_collapse(medspec)
    cy, cx = coll["cy"], coll["cx"]
    y = cy + EXAMPLE_FY
    fx = np.arange(w) - cx
    fy = np.arange(h) - cy
    xvalid = coll["xvalid"]
    row = np.asarray(medspec[y], dtype=np.float64)
    p95 = float(coll["row_profile"][y])
    z_here = float(coll["row_z"][y])
    prof = coll["row_profile"]
    neigh = [j for j in range(max(0, y - WIN), min(h, y + WIN + 1)) if abs(j - y) > 1]
    neigh_vals = prof[neigh]
    med = float(np.median(neigh_vals))
    mad = float(np.median(np.abs(neigh_vals - med))) + 1e-9
    lo_loc, hi_loc = max(cy + 5, y - 2), min(h - 5, y + 3)
    loc_max = z_here >= float(np.max(coll["row_z"][lo_loc:hi_loc]))
    proposed = z_here >= ROW_Z_THRESH and loc_max
    extent = (-cx - 0.5, w - cx - 0.5, h - cy - 0.5, -cy - 0.5)
    slo, shi = _percentile_limits(medspec, (3.0, 99.7))
    edge = cx - EDGE_MARGIN

    fig = new_figure()
    fig.suptitle(
        rf"Figure 04c0  ·  Proposing one ridge row  ·  $f_y=+{EXAMPLE_FY}$",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    gs = fig.add_gridspec(1, 3, left=0.055, right=0.97, top=0.88, bottom=0.40, wspace=0.28)

    ax = fig.add_subplot(gs[0])
    ax.imshow(medspec, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest", extent=extent)
    ax.axvspan(-DC_FX, DC_FX, color=(0.85, 0.15, 0.12), alpha=0.22, lw=0)
    ax.axvspan(-cx, -edge, color=(0.20, 0.35, 0.80), alpha=0.18, lw=0)
    ax.axvspan(edge, cx, color=(0.20, 0.35, 0.80), alpha=0.18, lw=0)
    ax.axhline(EXAMPLE_FY, color=(1.0, 0.85, 0.15), lw=1.4)
    ax.set_xlim(-256, 255)
    ax.set_ylim(255, -256)
    ax.set_xlabel(r"$f_x$")
    ax.set_ylabel(r"$f_y$")
    ax.set_title(r"$S_{\mathrm{med}}$, one row", fontsize=11)

    axr = fig.add_subplot(gs[1])
    axr.plot(fx, row, color="0.2", lw=0.7)
    axr.axvspan(-DC_FX, DC_FX, color=(0.85, 0.15, 0.12), alpha=0.22, lw=0)
    axr.axvspan(-cx, -edge, color=(0.20, 0.35, 0.80), alpha=0.18, lw=0)
    axr.axvspan(edge, cx, color=(0.20, 0.35, 0.80), alpha=0.18, lw=0)
    axr.axhline(p95, color=(0.85, 0.12, 0.12), ls="--", lw=1.2, label=f"95th pct on V = {p95:.2f}")
    axr.plot(fx[xvalid], row[xvalid], color="0.05", lw=0.9)
    axr.set_xlim(-256, 255)
    axr.set_xlabel(r"$f_x$")
    axr.set_ylabel(r"$S_{\mathrm{med}}$")
    axr.set_title(rf"That row, $f_y=+{EXAMPLE_FY}$", fontsize=11)
    axr.legend(fontsize=8, loc="upper right", frameon=False)

    axz = fig.add_subplot(gs[2])
    axz.plot(fy, coll["row_z"], color="0.2", lw=0.8)
    axz.axvspan(EXAMPLE_FY - WIN, EXAMPLE_FY + WIN, color=(1.0, 0.85, 0.15), alpha=0.25, lw=0, zorder=0)
    axz.axhline(ROW_Z_THRESH, color=(0.85, 0.12, 0.12), ls="--", lw=1.1, label=f"threshold {ROW_Z_THRESH:.1f}")
    axz.plot([EXAMPLE_FY], [z_here], "o", color=(0.85, 0.12, 0.12), ms=7, zorder=4)
    axz.annotate(
        f"Z = {z_here:.2f}",
        (EXAMPLE_FY, z_here),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
        color=(0.55, 0.08, 0.08),
    )
    axz.set_xlim(-5, 45)
    axz.set_xlabel(r"$f_y$")
    axz.set_ylabel("row Z")
    axz.set_title("Local Z of the row profile", fontsize=11)
    axz.legend(fontsize=8, loc="upper right", frameon=False)

    boxed_text(
        fig,
        [0.05, 0.02, 0.90, 0.28],
        (
            f"DC neighbourhood: columns |fx| <= {DC_FX} around zero frequency (the image mean). "
            f"They are always bright, so they are left out of V. Blue: outer edge |fx| >= {edge}."
            "\n"
            f"This row is collapsed to one number: the 95th percentile of S_med on V (= {p95:.2f}). "
            f"row Z compares that number to the neighbouring rows' profile values "
            f"(window +/-{WIN}, exclude this row): median {med:.2f}, MAD {mad:.3f}. "
            f"Z = (p95 - median) / (1.4826 MAD) = {z_here:.2f}. "
            "The 3.0 cut is unitless: three robust local standard deviations."
            "\n"
            f"First-pass proposal if Z >= {ROW_Z_THRESH:.1f} and this row is a local max of Z "
            f"in a 5-bin window. Here both hold (local max = {loc_max}), so fy = +{EXAMPLE_FY} is proposed. "
            "First-pass = the stricter detect (Z >= 3.0) before a softer fallback (Z >= 2.2) if nothing is found."
        ),
        fontsize=8.5,
        color="0.3",
    )

    numbers = {
        "stack": str(CHANA_TIF),
        "shutter_frames": idxs,
        "example_fy": EXAMPLE_FY,
        "p95": p95,
        "neighbor_median": med,
        "neighbor_mad": mad,
        "row_z": z_here,
        "row_z_thresh": ROW_Z_THRESH,
        "local_max": loc_max,
        "proposed": proposed,
        "z_formula": "(p95 - median_neighbors) / (1.4826 * MAD)",
    }
    try:
        written = save_figure(fig, "fig04c0_one_row", numbers)
    except OSError:
        written = save_figure(fig, "fig04c0_one_row_new", numbers)
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
