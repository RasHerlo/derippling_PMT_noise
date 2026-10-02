"""Figure 3, Local-background: how v4 turns a catalog row into mask spots.

For the catalog band row (fy = q), each fx column is compared to the median of
the same column in the 10 rows 5–9 steps above and below. Weight is 0 at 1.4×
that median and 1 at 3.5× (pack_D ratio_start / ratio_full).

``python -m demo_Sep2026.fig03_local_background --index 3 --frame 667``
"""

from __future__ import annotations

import argparse

import numpy as np

from batch_defringe.library import load_catalog
from batch_defringe.process import PACK_D
from batch_defringe.process_v4 import TRACK_SEARCH, Y_RADIUS, _catalog_line, _xvalid, apply_lines

from .common import fft_log_amp, load_frame, new_figure, save_figure, search_q
from .fig02_catalog_entry import MAX_ALPHA, source_stack

BG_OFFS = list(range(-9, -4)) + list(range(5, 10))
RATIO_START = float(PACK_D["ratio_start"])
RATIO_FULL = float(PACK_D["ratio_full"])


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


def draw(index: int, frame_idx: int) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    records = load_catalog()["records"]
    if not 0 <= index < len(records):
        raise IndexError(f"catalog index {index} outside 0..{len(records) - 1}")
    entry = records[index]
    fam = (entry.get("families") or [None])[0]
    if fam is None:
        raise ValueError("entry has no families")

    tif = source_stack(entry)
    raw, n_frames = load_frame(frame_idx, tif)
    h, w = raw.shape
    cy, cx = h // 2, w // 2
    logamp = fft_log_amp(raw)
    amp = _fft_amp(raw)

    line = _catalog_line(fam, logamp)
    if line is None:
        raise RuntimeError("catalog entry has no support on this frame")
    q_used, _ = search_q(logamp, float(line.q), True, _xvalid(w), TRACK_SEARCH)
    trial = apply_lines(raw, [line], max_alpha=MAX_ALPHA, search=TRACK_SEARCH)
    applied = np.asarray(trial["applied"], dtype=np.float64)
    q = int(round(float(q_used)))
    y_band = cy + q
    bg = _row_background(amp, y_band)
    row = amp[y_band]
    fx = np.arange(w) - cx
    fy = np.arange(h) - cy
    bg_fy = [q + off for off in BG_OFFS]

    x_weight = np.asarray((line.family or {}).get("x_weight"), dtype=float)
    if x_weight.size != w:
        x_weight = np.zeros(w, dtype=float)
    allowed = x_weight > 0.20
    fx_lo, fx_hi = float(fx[0]), float(fx[-1])

    fig = new_figure()
    fig.suptitle(
        f"Local-background  ·  {entry.get('channel')}  ·  fy = {q}  ·  frame {frame_idx}",
        fontsize=14,
        fontweight="bold",
        y=0.975,
    )

    # Fixed boxes so the 3-D axes cannot spill out of the slide.
    ax = fig.add_axes([0.02, 0.22, 0.33, 0.66], projection="3d")
    ax2 = fig.add_axes([0.40, 0.30, 0.32, 0.54])
    ax3 = fig.add_axes([0.76, 0.30, 0.16, 0.54])

    from scipy.ndimage import gaussian_filter

    Z = gaussian_filter(np.asarray(logamp, dtype=np.float64), sigma=0.8)
    XX, YY = np.meshgrid(fx, fy)
    norm = Normalize(vmin=float(np.percentile(Z, 5)), vmax=float(np.percentile(Z, 99.6)))
    faces = cm.gray(norm(Z))

    def _paint_row(yi: int, rgba: tuple[float, float, float, float]) -> None:
        for dy in (-1, 0, 1):
            yy = yi + dy
            if 0 <= yy < h:
                faces[yy, :] = rgba

    for fval in bg_fy:
        _paint_row(cy + int(fval), (0.95, 0.82, 0.12, 1.0))
    _paint_row(y_band, (0.85, 0.10, 0.10, 1.0))
    ax.plot_surface(
        XX, YY, Z, facecolors=faces, linewidth=0, antialiased=False, shade=True,
        rstride=1, cstride=2,
    )
    ax.view_init(elev=38, azim=-60)
    ax.set_xlim(fx_lo, fx_hi)
    ax.set_ylim(fx_lo, fx_hi)
    ax.set_box_aspect((1.15, 1.15, 0.38))
    ax.set_xlabel("fx", fontsize=8, labelpad=2)
    ax.set_ylabel("fy", fontsize=8, labelpad=2)
    ax.set_zlabel("log |FFT|", fontsize=8, labelpad=2)
    ax.tick_params(labelsize=6)
    ax.set_title("1. Fourier space (full ±256)", fontsize=10, pad=8)
    ax.legend(
        handles=[
            Line2D([0], [0], color=(0.85, 0.10, 0.10), lw=3, label=f"fy = {q}"),
            Line2D([0], [0], color=(0.95, 0.82, 0.12), lw=3, label="background 6±[5–9]"),
        ],
        loc="upper left",
        fontsize=7,
        frameon=False,
    )

    # Grey bands: fx columns hydrate allows. Only those can enter the mask.
    ylim_lo = max(float(np.min(bg[bg > 0])) * 0.4, 1.0)
    ylim_hi = float(np.max(row)) * 1.3
    in_band = False
    x0 = fx[0]
    for i, ok in enumerate(allowed):
        if ok and not in_band:
            x0 = fx[i]
            in_band = True
        elif (not ok or i == w - 1) and in_band:
            ax2.axvspan(x0, fx[i if not ok else i], color="0.85", lw=0, zorder=0, label="_allowed")
            in_band = False
    ax2.plot(fx, row, color=(0.75, 0.10, 0.10), lw=0.9, label=f"|FFT| on fy = {q}")
    ax2.plot(fx, bg, color=(0.85, 0.70, 0.05), lw=1.4, label="median of the 10 rows")
    ax2.plot(fx, RATIO_START * bg, color="0.35", ls=":", lw=1.2, label=f"{RATIO_START:g} × median")
    ax2.plot(fx, RATIO_FULL * bg, color="0.10", ls=":", lw=1.2, label=f"{RATIO_FULL:g} × median")
    in_mask = applied[y_band] > 1e-3
    if np.any(in_mask):
        ax2.scatter(fx[in_mask], row[in_mask], s=16, c="k", zorder=5, label="kept in the mask")
    ax2.set_xlim(fx_lo, fx_hi)
    ax2.set_ylim(ylim_lo, ylim_hi)
    ax2.set_yscale("log")
    ax2.set_xlabel("fx (frequency steps)")
    ax2.set_ylabel("|FFT| amplitude (log)")
    ax2.set_title("2. Row 6 vs local background (full ±256)", fontsize=10)
    handles, labels = ax2.get_legend_handles_labels()
    handles.insert(0, plt.Rectangle((0, 0), 1, 1, color="0.85", label="fx allowed by ridge support"))
    labels.insert(0, "fx allowed by ridge support")
    ax2.legend(handles, labels, fontsize=6.5, loc="upper right", frameon=False, ncol=1)

    fy_m0, fy_m1 = -q - Y_RADIUS - 3, q + Y_RADIUS + 3
    mask_row = applied[y_band]
    if np.any(mask_row > 1e-3):
        xs_hit = np.where(mask_row > 1e-3)[0]
        fx_hit = xs_hit - cx
        fx_m0 = int(fx_hit.min()) - 6
        fx_m1 = int(fx_hit.max()) + 6
    else:
        fx_m0, fx_m1 = -25, 25
    y_a, y_b = max(0, cy + fy_m0), min(h, cy + fy_m1 + 1)
    x_a, x_b = max(0, cx + fx_m0), min(w, cx + fx_m1 + 1)
    patch = applied[y_a:y_b, x_a:x_b]
    extent = (x_a - cx - 0.5, x_b - cx - 0.5, y_b - cy - 0.5, y_a - cy - 0.5)
    im = ax3.imshow(patch, cmap="inferno", vmin=0, vmax=1, interpolation="nearest", extent=extent, aspect="equal")
    ax3.set_xticks(np.arange(int(np.ceil((x_a - cx) / 5.0)) * 5, x_b - cx, 5))
    ax3.set_yticks(np.arange(y_a - cy, y_b - cy))
    ax3.tick_params(labelsize=6)
    ax3.grid(True, color="0.35", lw=0.3, alpha=0.5)
    ax3.axhline(q, color=(0.85, 0.12, 0.12), lw=0.7, alpha=0.85)
    ax3.axhline(-q, color=(0.85, 0.12, 0.12), lw=0.7, alpha=0.85, ls="--")
    ax3.set_xlabel("fx")
    ax3.set_ylabel("fy")
    ax3.set_title(f"3. Mask (zoom)  ·  fade ±{Y_RADIUS} rows", fontsize=10)
    cax = fig.add_axes([0.93, 0.30, 0.012, 0.54])
    fig.colorbar(im, cax=cax, label="weight")

    fig.text(
        0.02,
        0.16,
        (
            f"Grey bands in panel 2 = fx where this frame still has ridge support (the left–right check). "
            f"A column must sit in a grey band AND rise above {RATIO_START:g}× the yellow median. "
            f"Being above {RATIO_FULL:g}× is not enough if the column is outside those bands.\n"
            f"Background rows fy = {bg_fy}.  Weight along a kept column: 0 at {RATIO_START:g}×, 1 at {RATIO_FULL:g}×.  "
            f"Then fade ±{Y_RADIUS} rows.  Panel 1 height = log |FFT|; panel 2 = linear |FFT| on a log axis."
        ),
        va="top",
        fontsize=8,
        color="0.25",
    )

    numbers = {
        "catalog_index": index,
        "frame": frame_idx,
        "n_frames": n_frames,
        "q_stored": float(fam["q"]),
        "q_used": float(q_used),
        "band_row_image_index": int(y_band),
        "background_fy": bg_fy,
        "ratio_start": RATIO_START,
        "ratio_full": RATIO_FULL,
        "n_fx_above_start": int(np.sum(row > RATIO_START * bg)),
        "n_fx_above_full": int(np.sum(row > RATIO_FULL * bg)),
        "n_fx_above_full_but_not_allowed": int(np.sum((row > RATIO_FULL * bg) & ~allowed)),
        "n_fx_allowed": int(np.sum(allowed)),
        "fx_ranges_hydrated": (line.family or {}).get("fx_ranges"),
        "n_fx_in_mask_on_band": int(np.sum(applied[y_band] > 1e-3)),
        "fx_in_mask_on_band": fx[applied[y_band] > 1e-3].tolist(),
        "mask_max_near_q": float(np.max(patch)),
        "y_radius": Y_RADIUS,
        "source_stack": str(tif),
    }
    written = save_figure(
        fig,
        f"fig03_local_background_{index}_{entry.get('channel')}_q{q}_frame{frame_idx}",
        numbers,
    )
    numbers["written"] = [str(p) for p in written]
    return numbers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, default=3)
    ap.add_argument("--frame", type=int, default=667)
    args = ap.parse_args(argv)
    numbers = draw(args.index, args.frame)
    for p in numbers["written"]:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
