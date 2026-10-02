"""Figure 04g: peak fx weights become the shutter-hint mask on a frame.

``python -m demo_Sep2026.fig04g_shutter_mask``
"""

from __future__ import annotations

import argparse

import numpy as np
import tifffile

from batch_defringe.process import PACK_D
from batch_defringe.process_v4 import Line, Y_RADIUS, apply_lines
from batch_defringe.shutter_seed_test import learn_shutter_families
from batch_defringe.spatial_seed import GATE_HIGH, GATE_LOW

from .common import _percentile_limits, boxed_text, fft_log_amp, load_frame, new_figure, save_figure, search_q
from .fig04b_median_fourier import _shutter_indices
from .fig04c_row_profile import DC_FX, EDGE_MARGIN
from .fig04d_pairing import FAMILY_COLOR
from .paths import CHANA_TIF

EXAMPLE_Q = 10
SHUTTER_SEARCH = 2
MAX_ALPHA = float(PACK_D["max_alpha"])
RATIO_START = float(PACK_D["ratio_start"])
RATIO_FULL = float(PACK_D["ratio_full"])
BG_OFFS = list(range(-9, -4)) + list(range(5, 10))


def _learn() -> tuple[list[dict], list[int]]:
    idxs = _shutter_indices()
    with tifffile.TiffFile(CHANA_TIF) as tf:
        frames = [tf.pages[int(i)].asarray() for i in idxs]
    families, _ = learn_shutter_families(frames)
    return families, idxs


def _fft_amp(frame: np.ndarray) -> np.ndarray:
    x = np.asarray(frame, dtype=np.float32)
    x = x - float(np.median(x))
    return np.abs(np.fft.fftshift(np.fft.fft2(x)))


def _row_background(amp: np.ndarray, y: int) -> np.ndarray:
    h = amp.shape[0]
    rows = [amp[y + off] for off in BG_OFFS if 0 <= y + off < h]
    if not rows:
        return np.zeros(amp.shape[1], dtype=np.float64)
    return np.median(np.stack(rows, axis=0), axis=0)


def _xvalid(width: int) -> np.ndarray:
    cx = width // 2
    fx = np.arange(width) - cx
    return (np.abs(fx) > DC_FX) & (np.abs(fx) < cx - EDGE_MARGIN)


def draw() -> dict:
    families, idxs = _learn()
    frame_idx = int(idxs[len(idxs) // 2])
    raw, _n = load_frame(frame_idx)
    h, w = raw.shape
    cy, cx = h // 2, w // 2
    fx = np.arange(w) - cx
    logamp = fft_log_amp(raw)
    amp = _fft_amp(raw)
    xvalid = _xvalid(w)

    lines = [
        Line("fy", float(f["q"]), dict(f), "shutter_hint", kind="ridge", note="in-stack shutter learn")
        for f in families
    ]
    trial = apply_lines(raw, lines, max_alpha=MAX_ALPHA, search=SHUTTER_SEARCH)
    applied = np.asarray(trial["applied"], dtype=np.float64)

    prim = next(f for f in families if int(round(float(f["q"]))) == EXAMPLE_Q)
    q_used, strength = search_q(logamp, float(prim["q"]), True, xvalid, SHUTTER_SEARCH)
    gate = float(np.clip((strength - GATE_LOW) / max(1e-9, GATE_HIGH - GATE_LOW), 0.0, 1.0))
    y = cy + int(round(q_used))
    xw = np.asarray(prim["x_weight"], dtype=np.float64)
    bg = _row_background(amp, y)
    ratio = amp[y] / (bg + 1e-12)
    local_conf = np.clip((ratio - RATIO_START) / max(1e-9, RATIO_FULL - RATIO_START), 0.0, 1.0)
    product = xw * local_conf
    color = FAMILY_COLOR.get(EXAMPLE_Q, (0.10, 0.52, 0.38))
    extent = (-cx - 0.5, w - cx - 0.5, h - cy - 0.5, -cy - 0.5)
    slo, shi = _percentile_limits(logamp, (3.0, 99.7))
    edge = cx - EDGE_MARGIN

    fig = new_figure()
    fig.suptitle(
        f"Figure 04g  ·  Peak $f_x$ to shutter mask  ·  ChanA frame {frame_idx}",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    gs = fig.add_gridspec(
        1,
        3,
        left=0.05,
        right=0.96,
        top=0.88,
        bottom=0.40,
        wspace=0.26,
        width_ratios=[1.05, 1.15, 1.05],
    )

    ax = fig.add_subplot(gs[0])
    ax.imshow(logamp, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest", extent=extent)
    ax.axvspan(-DC_FX, DC_FX, color=(0.85, 0.15, 0.12), alpha=0.22, lw=0)
    ax.axvspan(-cx, -edge, color=(0.20, 0.35, 0.80), alpha=0.18, lw=0)
    ax.axvspan(edge, cx, color=(0.20, 0.35, 0.80), alpha=0.18, lw=0)
    for fam in families:
        q = int(round(float(fam["q"])))
        col = FAMILY_COLOR.get(q, (0.7, 0.7, 0.2))
        ax.axhline(q, color=col, lw=1.1)
        if fam.get("hi") is not None:
            ax.axhline(float(fam["hi"]), color=col, lw=1.1)
        for lo, hi in fam.get("fx_ranges") or []:
            ax.axvspan(float(lo), float(hi), color=col, alpha=0.22, lw=0)
    ax.set_xlim(-256, 255)
    ax.set_ylim(255, -256)
    ax.set_xlabel(r"$f_x$")
    ax.set_ylabel(r"$f_y$")
    ax.set_title(r"Frame log-amp + stored peak $f_x$", fontsize=11)

    axw = fig.add_subplot(gs[1])
    axw.plot(fx, xw, color=color, lw=1.2, label="stored peak fx weight")
    axw.plot(fx, local_conf, color=(0.35, 0.35, 0.35), lw=0.9, label="local_conf on this frame")
    axw.plot(fx, product, color=(0.75, 0.12, 0.12), lw=1.15, label="weight x local_conf")
    axw.set_xlim(-80, 80)
    axw.set_ylim(-0.05, 1.08)
    axw.set_xlabel(r"$f_x$")
    axw.set_ylabel("weight")
    axw.set_title(rf"$q={int(round(q_used))}$  on this frame", fontsize=11)
    axw.legend(fontsize=7, loc="upper left", frameon=False)

    axm = fig.add_subplot(gs[2])
    fy0, fy1 = -EXAMPLE_Q - Y_RADIUS - 6, EXAMPLE_Q + Y_RADIUS + 6
    fx0, fx1 = -80, 80
    ya, yb = max(0, cy + fy0), min(h, cy + fy1 + 1)
    xa, xb = max(0, cx + fx0), min(w, cx + fx1 + 1)
    patch = applied[ya:yb, xa:xb]
    mext = (xa - cx - 0.5, xb - cx - 0.5, yb - cy - 0.5, ya - cy - 0.5)
    im = axm.imshow(
        patch,
        cmap="inferno",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=mext,
        aspect="auto",
    )
    axm.axhline(int(round(q_used)), color=color, lw=0.8, alpha=0.9)
    axm.axhline(-int(round(q_used)), color=color, lw=0.8, ls="--", alpha=0.9)
    axm.set_xlabel(r"$f_x$")
    axm.set_ylabel(r"$f_y$")
    axm.set_title(rf"Applied mask  (fade $\pm {Y_RADIUS}$)", fontsize=11)
    fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04, ticks=[0, 0.5, 1])

    n_mask = int(np.sum(applied[y] > 1e-3))
    boxed_text(
        fig,
        [0.05, 0.02, 0.90, 0.28],
        (
            "Peak fx from 04f is stored as x_weight on the family. v4 copies that weight "
            "(it does not re-hydrate fx on the target frame). On a shutter frame, q snaps "
            f"by +/-{SHUTTER_SEARCH} (live frames snap by +/-10). "
            f"Here q stayed {int(round(q_used))} on frame {frame_idx}."
            "\n"
            f"The mask at each bin is not x_weight alone. "
            f"alpha = {MAX_ALPHA:g} x gate x w_y x x_weight x local_conf. "
            f"w_y = exp(-d^2 / 2) for d in -{Y_RADIUS}..+{Y_RADIUS}. "
            f"local_conf is 0 at {RATIO_START:g} x the median of rows 5-9 away, and 1 at "
            f"{RATIO_FULL:g} x that median, measured on this frame's |F|. "
            f"gate from search strength was {gate:.2f}. "
            f"On fy=+{int(round(q_used))}, {n_mask} bins have applied > 0. "
            "Partner rows +/- hi get the same x_weight. This panel isolates shutter_hint; "
            "catalog and linescan are not included."
        ),
        fontsize=8.5,
        color="0.3",
    )

    numbers = {
        "stack": str(CHANA_TIF),
        "shutter_frames": idxs,
        "apply_frame": frame_idx,
        "search": SHUTTER_SEARCH,
        "max_alpha": MAX_ALPHA,
        "ratio_start": RATIO_START,
        "ratio_full": RATIO_FULL,
        "q_stored": float(prim["q"]),
        "q_used": float(q_used),
        "strength": float(strength),
        "gate": gate,
        "n_mask_on_q_row": n_mask,
        "families_q": [float(f["q"]) for f in families],
        "note": "shutter_hint x_weight from S_med; local_conf from this frame",
    }
    try:
        written = save_figure(fig, "fig04g_shutter_mask", numbers)
    except OSError:
        written = save_figure(fig, "fig04g_shutter_mask_new", numbers)
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
