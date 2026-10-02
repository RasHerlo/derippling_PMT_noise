"""Figure 04b: five shutter log-amplitude FFTs and their pixelwise median.

``python -m demo_Sep2026.fig04b_median_fourier``
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import tifffile

from .common import _percentile_limits, boxed_text, fft_log_amp, new_figure, save_figure
from .paths import CHANA_TIF, DEMO_DIR

FALLBACK_SHUTTER = list(range(756, 761))


def _shutter_indices() -> list[int]:
    meta = DEMO_DIR / "fig04a_shutter_frames.json"
    if meta.is_file():
        idxs = [int(i) for i in json.loads(meta.read_text(encoding="utf-8")).get("shutter_frames") or []]
        if idxs:
            return idxs
    return list(FALLBACK_SHUTTER)


def _pick_example_bin(medspec: np.ndarray) -> tuple[int, int, float, float]:
    """A bright off-DC bin on the typical shutter ridge fy = +10."""
    h, w = medspec.shape
    cy, cx = h // 2, w // 2
    y = cy + 10
    fx = np.arange(w) - cx
    xvalid = (np.abs(fx) > 5) & (np.abs(fx) < cx - 10)
    row = np.asarray(medspec[y], dtype=np.float64)
    x = int(np.flatnonzero(xvalid)[int(np.argmax(row[xvalid]))])
    return y, x, float(y - cy), float(x - cx)


def draw() -> dict:
    idxs = _shutter_indices()
    with tifffile.TiffFile(CHANA_TIF) as tf:
        raws = [np.asarray(tf.pages[int(i)].asarray()) for i in idxs]
    specs = [fft_log_amp(r) for r in raws]
    stack = np.stack(specs, axis=0)
    medspec = np.median(stack, axis=0)
    h, w = medspec.shape
    cy, cx = h // 2, w // 2
    extent = (-cx - 0.5, w - cx - 0.5, h - cy - 0.5, -cy - 0.5)
    slo, shi = _percentile_limits(stack, (3.0, 99.7))
    y_ex, x_ex, fy_ex, fx_ex = _pick_example_bin(medspec)
    bin_vals = stack[:, y_ex, x_ex].astype(np.float64)
    bin_med = float(np.median(bin_vals))

    n = len(idxs)
    fig = new_figure()
    fig.suptitle("Figure 04b  ·  Median Fourier picture  ·  ChanA", fontsize=16, fontweight="bold", y=0.97)
    gs = fig.add_gridspec(
        2,
        n,
        left=0.05,
        right=0.96,
        top=0.86,
        bottom=0.22,
        wspace=0.12,
        hspace=0.38,
        height_ratios=[0.92, 1.15],
    )

    for col, (idx, spec) in enumerate(zip(idxs, specs)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(spec, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest", extent=extent)
        ax.plot([fx_ex], [fy_ex], "+", color=(0.90, 0.15, 0.12), ms=7, mew=1.1)
        ax.set_title(rf"$S_{{{idx}}}$", fontsize=11)
        ax.set_xlim(-256, 255)
        ax.set_ylim(255, -256)
        if col == 0:
            ax.set_ylabel(r"$f_y$")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(r"$f_x$")
        ax.tick_params(labelsize=7)

    axm = fig.add_subplot(gs[1, : n - 2])
    axm.imshow(medspec, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest", extent=extent)
    axm.plot([fx_ex], [fy_ex], "+", color=(0.90, 0.15, 0.12), ms=9, mew=1.3)
    axm.set_title(r"$S_{\mathrm{med}}$  (pixelwise median of the five $S$)", fontsize=12)
    axm.set_xlabel(r"$f_x$")
    axm.set_ylabel(r"$f_y$")
    axm.set_xlim(-256, 255)
    axm.set_ylim(255, -256)

    axb = fig.add_subplot(gs[1, n - 2 :])
    xpos = np.arange(n)
    axb.bar(xpos, bin_vals, color="0.55", width=0.65, edgecolor="0.2", lw=0.6)
    axb.axhline(bin_med, color=(0.85, 0.12, 0.12), lw=1.4, ls="--", label=rf"median $={bin_med:.3f}$")
    pad = max(0.02, 0.15 * (float(bin_vals.max()) - float(bin_vals.min()) + 1e-6))
    axb.set_ylim(float(min(bin_vals.min(), bin_med)) - pad, float(max(bin_vals.max(), bin_med)) + pad)
    axb.set_xticks(xpos)
    axb.set_xticklabels([str(i) for i in idxs], fontsize=8)
    axb.set_xlabel("shutter frame")
    axb.set_ylabel(r"$\log(1+|F|)$")
    axb.set_title(rf"One bin: $f_y={fy_ex:.0f}$, $f_x={fx_ex:.0f}$", fontsize=11)
    axb.legend(fontsize=8, loc="best", frameon=False)

    boxed_text(
        fig,
        [0.05, 0.02, 0.90, 0.16],
        (
            "Top: log-amplitude Fourier pictures of the detected shutter frames, "
            r"each $S=\log(1+|F|)$ after subtracting that frame's median and fftshift.  "
            "Same grey scale on every panel.  Red cross: the example bin at right.\n"
            r"$S_{\mathrm{med}}$ is the median over the five $S$ arrays at each "
            r"$(f_y,f_x)$.  It is not the FFT of the median image, and not a mean.  "
            "The bar panel is that one bin's five values and their median."
        ),
        fontsize=8.5,
        color="0.3",
    )

    numbers = {
        "stack": str(CHANA_TIF),
        "shutter_frames": idxs,
        "shape_hw": [h, w],
        "example_bin_xy": [x_ex, y_ex],
        "example_bin_fx_fy": [fx_ex, fy_ex],
        "example_bin_values": [float(v) for v in bin_vals],
        "example_bin_median": bin_med,
        "fft_display_limits": [slo, shi],
        "note": "S_med = median of per-frame fft_log_amp; not FFT of the median image",
    }
    try:
        written = save_figure(fig, "fig04b_median_fourier", numbers)
    except OSError:
        written = save_figure(fig, "fig04b_median_fourier_new", numbers)
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
