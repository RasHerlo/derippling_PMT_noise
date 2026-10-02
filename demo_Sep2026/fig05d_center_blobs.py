"""Figure 05d: linescan-centre blobs (seed_peak_mask).

``python -m demo_Sep2026.fig05d_center_blobs``
"""

from __future__ import annotations

import argparse

import numpy as np

from batch_defringe.congruence import MASK_RADIUS, seed_peak_mask
from batch_defringe.process import PACK_D

from .common import _percentile_limits, boxed_text, fft_log_amp, new_figure
from .fig05_common import LIVE_FRAME, _save, seed_for

MAX_ALPHA = float(PACK_D["max_alpha"])


def draw() -> dict:
    raw, seed, _n = seed_for(LIVE_FRAME)
    h, w = raw.shape
    cy, cx = h // 2, w // 2
    logamp = fft_log_amp(raw)
    mask = seed_peak_mask(h, w, seed)
    slo, shi = _percentile_limits(logamp, (3.0, 99.7))
    extent = (-cx - 0.5, w - cx - 0.5, h - cy - 0.5, -cy - 0.5)
    qy, qx = seed.get("qy"), seed.get("qx")
    win = seed.get("winner")

    fig = new_figure()
    fig.suptitle(
        f"Figure 05d  ·  Linescan-centre blobs  ·  ChanA frame {LIVE_FRAME}",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    gs = fig.add_gridspec(1, 3, left=0.05, right=0.96, top=0.88, bottom=0.40, wspace=0.28)

    ax = fig.add_subplot(gs[0])
    ax.imshow(logamp, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest", extent=extent)
    if win == "fx" and qx is not None:
        q = int(round(float(qx)))
        ax.plot([q, -q], [0, 0], "o", color=(0.85, 0.12, 0.12), ms=6)
    elif win == "fy" and qy is not None:
        q = int(round(float(qy)))
        ax.plot([0, 0], [q, -q], "o", color=(0.85, 0.12, 0.12), ms=6)
    elif win == "tilted" and qy is not None and qx is not None:
        iy, ix = int(round(float(qy))), int(round(float(qx)))
        for fy in (iy, -iy):
            for fx in (ix, -ix):
                ax.plot([fx], [fy], "o", color=(0.85, 0.12, 0.12), ms=6)
    ax.set_xlim(-256, 255)
    ax.set_ylim(255, -256)
    ax.set_xlabel(r"$f_x$")
    ax.set_ylabel(r"$f_y$")
    ax.set_title(r"$\log(1+|F|)$  +  blob centres", fontsize=11)

    axm = fig.add_subplot(gs[1])
    axm.imshow(mask, cmap="inferno", vmin=0, vmax=1, interpolation="nearest", extent=extent)
    axm.set_xlim(-256, 255)
    axm.set_ylim(255, -256)
    axm.set_xlabel(r"$f_x$")
    axm.set_ylabel(r"$f_y$")
    axm.set_title(r"seed_peak_mask (full plane)", fontsize=11)

    axz = fig.add_subplot(gs[2])
    hits = np.argwhere(mask > 1e-3)
    if hits.size:
        fys = hits[:, 0] - cy
        fxs = hits[:, 1] - cx
        fy0, fy1 = int(fys.min()) - 6, int(fys.max()) + 6
        fx0, fx1 = int(fxs.min()) - 6, int(fxs.max()) + 6
    else:
        fy0, fy1, fx0, fx1 = -12, 12, -25, 25
    ya, yb = max(0, cy + fy0), min(h, cy + fy1 + 1)
    xa, xb = max(0, cx + fx0), min(w, cx + fx1 + 1)
    patch = mask[ya:yb, xa:xb]
    zext = (xa - cx - 0.5, xb - cx - 0.5, yb - cy - 0.5, ya - cy - 0.5)
    im = axz.imshow(patch, cmap="inferno", vmin=0, vmax=1, interpolation="nearest", extent=zext, aspect="auto")
    axz.set_xlabel(r"$f_x$")
    axz.set_ylabel(r"$f_y$")
    axz.set_title(rf"Zoom  ·  radius {MASK_RADIUS}", fontsize=11)
    fig.colorbar(im, ax=axz, fraction=0.046, pad=0.04, label="weight")

    boxed_text(
        fig,
        [0.05, 0.02, 0.90, 0.28],
        (
            "If congruence is not none, the centre contribution is thin conjugate blobs, "
            f"not a ridge row. Radius {MASK_RADIUS}, Gaussian, DC skipped. "
            "fx: (fy=0, fx=+/-qx). fy: (+/-qy, fx=0). tilted: all four (+/-qy, +/-qx)."
            "\n"
            f"This frame: winner = {win}, qy = {qy}, qx = {qx}. "
            f"Apply is {MAX_ALPHA:g} x this field, then max with any ridge attenuation. "
            "No search_q, no leftover Z, no local_conf, no gate. "
            "Dropped only if catalog or shutter already claimed the same (axis, q)."
        ),
        fontsize=8.5,
        color="0.3",
    )
    return _save(
        fig,
        "fig05d_center_blobs",
        {
            "frame": LIVE_FRAME,
            "winner": win,
            "qy": qy,
            "qx": qx,
            "radius": MASK_RADIUS,
            "n_bins": int(np.sum(mask > 1e-3)),
            "max_alpha": MAX_ALPHA,
        },
    )


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    numbers = draw()
    for p in numbers["written"]:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
