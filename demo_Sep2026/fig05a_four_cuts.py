"""Figure 05a: the four real-space cuts used by linescan centre.

``python -m demo_Sep2026.fig05a_four_cuts``
"""

from __future__ import annotations

import argparse

import numpy as np

from .common import _percentile_limits, boxed_text, new_figure
from .fig05_common import CUT_COLOR, LIVE_FRAME, _save, plot_cut, seed_for


def draw() -> dict:
    raw, seed, _n = seed_for(LIVE_FRAME)
    h, w = raw.shape
    cuts = seed.get("cuts") or {}
    row = cuts.get("row")
    col = cuts.get("col")
    lo, hi = _percentile_limits(raw)

    fig = new_figure()
    fig.suptitle(
        f"Figure 05a  ·  Four linescan cuts  ·  ChanA frame {LIVE_FRAME}",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    gs = fig.add_gridspec(
        2,
        3,
        left=0.05,
        right=0.98,
        top=0.88,
        bottom=0.36,
        wspace=0.28,
        hspace=0.38,
        width_ratios=[1.15, 1.0, 1.0],
    )

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(raw, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    if row is not None:
        ax.axhline(int(row), color=CUT_COLOR["horizontal"], lw=1.2, label=f"H row {int(row)}")
    if col is not None:
        ax.axvline(int(col), color=CUT_COLOR["vertical"], lw=1.2, label=f"V col {int(col)}")
    n = int(max(h, w))
    ys = np.linspace(0.0, h - 1.0, n)
    ax.plot(np.linspace(0.0, w - 1.0, n), ys, color=CUT_COLOR["main"], lw=1.15, label="main TL-BR")
    ax.plot(np.linspace(w - 1.0, 0.0, n), ys, color=CUT_COLOR["anti"], lw=1.15, label="anti TR-BL")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Real space", fontsize=11)
    ax.legend(fontsize=7, loc="lower left", frameon=False)

    plot_cut(fig.add_subplot(gs[0, 1]), raw, seed, "horizontal", "H")
    plot_cut(fig.add_subplot(gs[0, 2]), raw, seed, "vertical", "V")
    plot_cut(fig.add_subplot(gs[1, 1]), raw, seed, "main", "main")
    plot_cut(fig.add_subplot(gs[1, 2]), raw, seed, "anti", "anti")

    boxed_text(
        fig,
        [0.05, 0.02, 0.90, 0.26],
        (
            "After the probe search (figure 05a0). Linescan centre starts on this frame, not on the movie. "
            "The H/V traces shown are the ones attached to the winning hypothesis "
            f"(or the first finite-q cut if congruence is none). "
            f"Here winner = {seed.get('winner')}  "
            f"qy = {seed.get('qy')}  qx = {seed.get('qx')}."
            "\n"
            "Grey: raw 1-D cut. Dots: k=4 lowest marks. Coloured: rloess smooth. "
            "P and q are the median period and median q of that smooth. "
            "The two diagonals supply the veto periods for figure 05c."
        ),
        fontsize=8.5,
        color="0.3",
    )
    return _save(
        fig,
        "fig05a_four_cuts",
        {
            "frame": LIVE_FRAME,
            "winner": seed.get("winner"),
            "qy": seed.get("qy"),
            "qx": seed.get("qx"),
            "score": seed.get("score"),
            "row": row,
            "col": col,
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
