"""Figure 04h: stored shutter mask weight, in the same zoom as figure 03-right.

What is stored and passed on is x_weight on the family rows, with the later
+/-2 fade already drawn. local_conf is not stored; that is measured later.

``python -m demo_Sep2026.fig04h_stored_weight``
"""

from __future__ import annotations

import argparse

import numpy as np

from batch_defringe.process_v4 import Y_RADIUS

from .common import boxed_text, new_figure, save_figure
from .fig04g_shutter_mask import _learn
from .paths import CHANA_TIF

Y_SIGMA = 1.0


def stored_weight(fam: dict, h: int, w: int) -> np.ndarray:
    """2-D template passed forward: w_y(d) * x_weight, no local_conf and no gate."""
    out = np.zeros((h, w), dtype=np.float64)
    cy = h // 2
    xw = np.asarray(fam["x_weight"], dtype=np.float64)
    yoffs = np.arange(-Y_RADIUS, Y_RADIUS + 1)
    yweights = np.exp(-0.5 * (yoffs / Y_SIGMA) ** 2)
    components = [float(fam["q"])]
    if fam.get("hi") is not None:
        components.append(float(fam["hi"]))
    for d in components:
        for sgn in (-1, +1):
            yc = cy + sgn * int(round(d))
            for off, wy in zip(yoffs, yweights):
                y = yc + int(off)
                if 0 <= y < h:
                    out[y, :] = np.maximum(out[y, :], wy * xw)
    return out


def _zoom_box(weight: np.ndarray, q: int, cx: int, cy: int) -> tuple[int, int, int, int]:
    h, w = weight.shape
    y_band = cy + q
    hits = np.where(weight[y_band] > 1e-3)[0]
    if hits.size:
        fx_m0 = int((hits - cx).min()) - 6
        fx_m1 = int((hits - cx).max()) + 6
    else:
        fx_m0, fx_m1 = -25, 25
    fy_m0 = -q - Y_RADIUS - 3
    fy_m1 = q + Y_RADIUS + 3
    ya, yb = max(0, cy + fy_m0), min(h, cy + fy_m1 + 1)
    xa, xb = max(0, cx + fx_m0), min(w, cx + fx_m1 + 1)
    return ya, yb, xa, xb


def _draw_mask_zoom(ax, weight: np.ndarray, q: int, cx: int, cy: int, *, title: str):
    ya, yb, xa, xb = _zoom_box(weight, q, cx, cy)
    patch = weight[ya:yb, xa:xb]
    extent = (xa - cx - 0.5, xb - cx - 0.5, yb - cy - 0.5, ya - cy - 0.5)
    im = ax.imshow(
        patch,
        cmap="inferno",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=extent,
        aspect="auto",
    )
    ax.set_xticks(np.arange(int(np.ceil((xa - cx) / 5.0)) * 5, xb - cx, 5))
    fy_span = (yb - cy) - (ya - cy)
    fy_step = 1 if fy_span <= 30 else 5
    ax.set_yticks(np.arange(int(np.ceil((ya - cy) / fy_step) * fy_step), yb - cy, fy_step))
    ax.tick_params(labelsize=6)
    ax.grid(True, color="0.35", lw=0.3, alpha=0.5)
    ax.axhline(q, color=(0.85, 0.12, 0.12), lw=0.7, alpha=0.85)
    ax.axhline(-q, color=(0.85, 0.12, 0.12), lw=0.7, ls="--", alpha=0.85)
    ax.set_xlabel("fx")
    ax.set_ylabel("fy")
    ax.set_title(title, fontsize=11)
    return im


def draw() -> dict:
    families, idxs = _learn()
    h = w = 512
    cy = cx = h // 2
    by_q = {int(round(float(f["q"]))): f for f in families}

    fig = new_figure()
    fig.suptitle(
        "Figure 04h  ·  Stored shutter mask weight  ·  ChanA",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    axes = [
        fig.add_axes([0.07, 0.40, 0.38, 0.48]),
        fig.add_axes([0.50, 0.40, 0.38, 0.48]),
    ]
    im = None
    shown: list[dict] = []
    for ax, q in zip(axes, (10, 30)):
        fam = by_q[q]
        weight = stored_weight(fam, h, w)
        im = _draw_mask_zoom(
            ax,
            weight,
            q,
            cx,
            cy,
            title=rf"$q={q}$  stored weight  ·  fade $\pm {Y_RADIUS}$",
        )
        shown.append(
            {
                "q": q,
                "hi": None if fam.get("hi") is None else float(fam["hi"]),
                "n_fx": int(np.sum(np.asarray(fam["x_weight"]) > 0.20)),
                "n_bins_template": int(np.sum(weight > 1e-3)),
            }
        )

    cax = fig.add_axes([0.90, 0.40, 0.015, 0.48])
    fig.colorbar(im, cax=cax, label="weight")

    boxed_text(
        fig,
        [0.05, 0.02, 0.90, 0.28],
        (
            "Same kind of panel as figure 03-right: mask weight on a zoom of +/-q, "
            f"with the +/-{Y_RADIUS} row fade. This is the stored shutter contribution, "
            "not a frame's final applied mask."
            "\n"
            "What is stored and passed on is x_weight (the peak fx intervals from 04f) "
            "together with q and hi. The picture here is that 1-D weight copied onto "
            "+/-q and +/-hi, then faded with w_y = exp(-d^2 / 2). "
            "local_conf and gate are not stored; each later frame measures those "
            "(figure 04g) and multiplies this template. Partner rows hi = 246 and 224 "
            "carry the same x_weight; they sit outside this +/-q zoom."
        ),
        fontsize=8.5,
        color="0.3",
    )

    numbers = {
        "stack": str(CHANA_TIF),
        "shutter_frames": idxs,
        "y_radius": Y_RADIUS,
        "y_sigma": Y_SIGMA,
        "families": shown,
        "note": "stored x_weight * w_y; no local_conf, no gate",
    }
    try:
        written = save_figure(fig, "fig04h_stored_weight", numbers)
    except OSError:
        written = save_figure(fig, "fig04h_stored_weight_new", numbers)
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
