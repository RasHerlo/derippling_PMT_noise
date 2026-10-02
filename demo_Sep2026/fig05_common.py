"""Shared helpers for the 05 linescan-centre figures."""

from __future__ import annotations

from batch_defringe.congruence import congruent_seed
from batch_defringe.image_check import diagonal_sample

from .common import load_frame
from .paths import CHANA_TIF

LIVE_FRAME = 1288
SHUTTER_FRAME = 758

CUT_COLOR = {
    "horizontal": (0.80, 0.12, 0.12),
    "vertical": (0.15, 0.35, 0.80),
    "main": (0.85, 0.50, 0.08),
    "anti": (0.48, 0.22, 0.65),
}


def seed_for(frame_idx: int) -> tuple:
    raw, n = load_frame(frame_idx)
    return raw, congruent_seed(raw), n


def cut_trace(raw, seed: dict, name: str):
    import numpy as np

    arr = np.asarray(raw, dtype=np.float64)
    cuts = seed.get("cuts") or {}
    tr = (cuts.get("traces") or {}).get(name) or {}
    if name == "horizontal":
        row = cuts.get("row")
        sig = arr[int(row)] if row is not None else None
    elif name == "vertical":
        col = cuts.get("col")
        sig = arr[:, int(col)] if col is not None else None
    else:
        _, sig = diagonal_sample(arr, name)
    return tr, None if sig is None else np.asarray(sig, dtype=np.float64)


def plot_cut(ax, raw, seed: dict, name: str, title: str) -> None:
    import numpy as np

    tr, sig = cut_trace(raw, seed, name)
    smooth = tr.get("smooth")
    if sig is None or smooth is None:
        ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes, color="0.4")
        ax.set_axis_off()
        return
    y = np.asarray(smooth, dtype=float)
    n = int(min(y.size, sig.size))
    x = np.arange(n)
    ax.plot(x, sig[:n], color="0.65", lw=0.5)
    marked = tr.get("marked")
    if marked is not None:
        marked = np.asarray(marked)[:n]
        if marked.dtype == bool:
            ax.plot(x[marked], sig[:n][marked], ".", color="0.1", ms=2.0)
    ax.plot(x, y[:n], color=CUT_COLOR[name], lw=1.05)
    ax.set_xlim(0, n - 1)
    p, q = tr.get("period"), tr.get("q")
    ptxt = "-" if p is None else f"{float(p):.1f}"
    qtxt = "-" if q is None else f"{float(q):.1f}"
    ax.set_title(f"{title}   P={ptxt}   q={qtxt}", fontsize=9)
    ax.tick_params(labelsize=7)


def _save(fig, name: str, numbers: dict):
    from .common import save_figure

    try:
        written = save_figure(fig, name, numbers)
    except OSError:
        written = save_figure(fig, f"{name}_new", numbers)
    numbers["written"] = [str(p) for p in written]
    numbers["stack"] = str(CHANA_TIF)
    return numbers
