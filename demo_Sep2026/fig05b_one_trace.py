"""Figure 05b: one H cut becomes median qx.

``python -m demo_Sep2026.fig05b_one_trace``
"""

from __future__ import annotations

import argparse

import numpy as np

from batch_defringe.baseline_smooth import SEED_K, _pack

from .common import boxed_text, new_figure
from .fig05_common import LIVE_FRAME, _save, seed_for


def draw() -> dict:
    raw, seed, _n = seed_for(LIVE_FRAME)
    row = (seed.get("cuts") or {}).get("row")
    if row is None:
        raise RuntimeError("no H row on this frame")
    sig = np.asarray(raw[int(row)], dtype=np.float64)
    rec = _pack(sig, SEED_K, int(sig.size))
    xs, ps, qs = rec["xs"], rec["ps"], rec["qs"]
    x = np.arange(sig.size)
    marked = np.asarray(rec["marked"], dtype=bool)

    fig = new_figure()
    fig.suptitle(
        f"Figure 05b  ·  One H cut to $q_x$  ·  ChanA frame {LIVE_FRAME}, row {int(row)}",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.98, top=0.86, bottom=0.38, wspace=0.28)

    ax = fig.add_subplot(gs[0])
    ax.plot(x, sig, color="0.55", lw=0.6, label="raw")
    ax.plot(x[marked], sig[marked], ".", color="0.05", ms=3.5, label=f"k={SEED_K} lows")
    ax.plot(x, rec["smooth"], color=(0.80, 0.12, 0.12), lw=1.15, label="rloess")
    ax.set_xlim(0, sig.size - 1)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("intensity")
    ax.set_title("Mark lows, then smooth", fontsize=11)
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax.tick_params(labelsize=7)

    axp = fig.add_subplot(gs[1])
    axp.plot(xs, ps, color="0.15", lw=0.9)
    if rec["p_median"] is not None:
        axp.axhline(rec["p_median"], color=(0.80, 0.12, 0.12), ls="--", lw=1.0, label=f"median P = {rec['p_median']:.2f}")
    axp.set_xlim(0, sig.size - 1)
    axp.set_xlabel("x (pixels)")
    axp.set_ylabel("P (pixels)")
    axp.set_title("Sliding period", fontsize=11)
    axp.legend(fontsize=7, loc="upper right", frameon=False)
    axp.tick_params(labelsize=7)

    axq = fig.add_subplot(gs[2])
    axq.plot(xs, qs, color="0.15", lw=0.9)
    if rec["q_median"] is not None:
        axq.axhline(rec["q_median"], color=(0.80, 0.12, 0.12), ls="--", lw=1.0, label=f"median qx = {rec['q_median']:.2f}")
    axq.set_xlim(0, sig.size - 1)
    axq.set_xlabel("x (pixels)")
    axq.set_ylabel("q = length / P")
    axq.set_title("Sliding q, then median", fontsize=11)
    axq.legend(fontsize=7, loc="upper right", frameon=False)
    axq.tick_params(labelsize=7)

    boxed_text(
        fig,
        [0.05, 0.02, 0.90, 0.26],
        (
            f"_pack on the H row attached to the seed (row {int(row)}). "
            f"Mark the {SEED_K} lowest samples in a rolling window, smooth those marks with rloess, "
            "then measure local period P(x) from the rFFT of a sliding window. "
            "q(x) = frame width / P(x). The cut's q is the median of the finite q(x) values."
            "\n"
            f"Here median P = {rec['p_median']:.2f} px and median qx = {rec['q_median']:.2f}. "
            f"Frame winner = {seed.get('winner')}, qx = {seed.get('qx')}. "
            "This q is only a hypothesis until the diagonals accept it (figure 05c)."
        ),
        fontsize=8.5,
        color="0.3",
    )
    return _save(
        fig,
        "fig05b_one_trace",
        {
            "frame": LIVE_FRAME,
            "row": int(row),
            "k": SEED_K,
            "p_median": rec["p_median"],
            "q_median": rec["q_median"],
            "winner": seed.get("winner"),
            "qx": seed.get("qx"),
            "n_marked": rec["n_marked"],
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
