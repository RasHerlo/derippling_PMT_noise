"""Diff-guided biology cut, then baseline fit on the leftover samples.

Shutter-frame |dI| is the expected ripple slope. A rise steeper than that,
followed later by a matching fall back to the pre-rise floor, is treated as
biology and held out of the sine fit.

Probe only. Does not write FFT masks.

``python -m batch_defringe.baseline_diff --frame 160 --shutter 756``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

from .image_check import _acf_period_px
from .spatial_seed import OUTPUT_SUBDIR

MIN_WIDTH = 6
SHUTTER_FRAMES = (756, 757, 758, 759, 760)


def pick_fringe_row(shutter: np.ndarray) -> int:
    """Strongest shutter row whose max |diff| is typical (skip hot-pixel rows)."""
    arr = np.asarray(shutter, dtype=np.float64)
    std = np.std(arr, axis=1)
    mx = np.array([float(np.max(np.abs(np.diff(arr[r])))) for r in range(arr.shape[0])])
    cap = float(np.percentile(mx, 90))
    valid = mx <= cap
    idx = np.flatnonzero(valid)
    return int(idx[int(np.argmax(std[idx]))])


def pick_fringe_col(shutter: np.ndarray) -> int:
    """Strongest shutter column whose max |diff| is typical (skip hot-pixel columns)."""
    arr = np.asarray(shutter, dtype=np.float64)
    std = np.std(arr, axis=0)
    mx = np.array([float(np.max(np.abs(np.diff(arr[:, c])))) for c in range(arr.shape[1])])
    cap = float(np.percentile(mx, 90))
    valid = mx <= cap
    idx = np.flatnonzero(valid)
    return int(idx[int(np.argmax(std[idx]))])


def shutter_diff_cutoff(shutter_frames: list[np.ndarray], row: int) -> float:
    """Largest |diff| this row ever shows on the shutter frames."""
    mx = 0.0
    for img in shutter_frames:
        d = np.diff(np.asarray(img, dtype=np.float64)[row])
        mx = max(mx, float(np.max(np.abs(d))))
    return mx


def mark_biology(
    sig: np.ndarray,
    thresh: float,
    *,
    min_width: int = MIN_WIDTH,
    floor_margin: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """True where a super-shutter rise is later closed by a super-shutter fall.

    ``sig[i]`` is the floor just before the rise ``d[i] = sig[i+1]-sig[i]``.
    The interval is closed when a fall brings the trace back to that floor
    (plus a margin of the shutter ripple amplitude).
    """
    y = np.asarray(sig, dtype=np.float64)
    d = np.diff(y)
    n = int(y.size)
    bio = np.zeros(n, dtype=bool)
    spans: list[tuple[int, int]] = []
    i = 0
    while i < d.size:
        if d[i] <= thresh:
            i += 1
            continue
        floor = float(y[i])
        j = i + 1
        closed = False
        while j < d.size:
            if (j - i) >= min_width and d[j] < -thresh and float(y[j + 1]) <= floor + floor_margin:
                closed = True
                break
            j += 1
        if closed:
            a, b = i, min(n - 1, j + 1)
            bio[a : b + 1] = True
            spans.append((a, b))
            i = j + 1
            continue
        i += 1
    return bio, spans


def keep_within_shutter_diff(sig: np.ndarray, thresh: float) -> np.ndarray:
    """Keep samples that are not an endpoint of a step with |diff| > thresh."""
    y = np.asarray(sig, dtype=np.float64)
    n = int(y.size)
    keep = np.ones(n, dtype=bool)
    if n < 2:
        return keep
    bad = np.abs(np.diff(y)) > float(thresh)
    keep[:-1] &= ~bad
    keep[1:] &= ~bad
    return keep


def sine_on_kept(sig: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Least-squares DC + sinusoid, using only ``keep`` samples."""
    y = np.asarray(sig, dtype=np.float64)
    n = int(y.size)
    t = np.arange(n, dtype=np.float64)
    if int(np.sum(keep)) < 16:
        dc = float(np.median(y[keep])) if np.any(keep) else float(np.median(y))
        return np.full(n, dc), None
    filled = y.copy()
    hole = ~keep
    if np.any(hole) and np.any(keep):
        filled[hole] = np.interp(t[hole], t[keep], y[keep])
    period = _acf_period_px(filled - float(np.median(y[keep])))
    dc = float(np.median(y[keep]))
    if period is None or period < 4.0:
        return np.full(n, dc), None
    w = 2.0 * np.pi / float(period)
    a = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones(n)])
    coef, _, _, _ = np.linalg.lstsq(a[keep], y[keep], rcond=None)
    return a @ coef, float(period)
    """Least-squares DC + sinusoid, using only ``keep`` samples."""
    y = np.asarray(sig, dtype=np.float64)
    n = int(y.size)
    t = np.arange(n, dtype=np.float64)
    if int(np.sum(keep)) < 16:
        dc = float(np.median(y[keep])) if np.any(keep) else float(np.median(y))
        return np.full(n, dc), None
    filled = y.copy()
    hole = ~keep
    if np.any(hole) and np.any(keep):
        filled[hole] = np.interp(t[hole], t[keep], y[keep])
    period = _acf_period_px(filled - float(np.median(y[keep])))
    dc = float(np.median(y[keep]))
    if period is None or period < 4.0:
        return np.full(n, dc), None
    w = 2.0 * np.pi / float(period)
    a = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones(n)])
    coef, _, _, _ = np.linalg.lstsq(a[keep], y[keep], rcond=None)
    return a @ coef, float(period)


def write_diff_pdf(
    path: Path,
    *,
    shutter: np.ndarray,
    live: np.ndarray,
    shutter_stack: list[np.ndarray],
    row: int,
    live_idx: int,
    shutter_idx: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    sh = np.asarray(shutter, dtype=np.float64)[row]
    lv = np.asarray(live, dtype=np.float64)[row]
    thresh = shutter_diff_cutoff(shutter_stack, row)
    floor_margin = 0.5 * float(np.percentile(sh, 90) - np.percentile(sh, 10))
    d_sh = np.diff(sh)
    d_lv = np.diff(lv)
    x_sh = np.arange(d_sh.size) + 0.5
    x_lv = np.arange(d_lv.size) + 0.5
    x = np.arange(lv.size)

    bio_sh, spans_sh = mark_biology(sh, thresh, floor_margin=floor_margin)
    bio_lv, spans_lv = mark_biology(lv, thresh, floor_margin=floor_margin)
    keep = ~bio_lv
    fit, period = sine_on_kept(lv, keep)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def banner(fig, title: str, sub: str) -> None:
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.97, title, fontsize=13, fontweight="bold", va="top")
        fig.text(0.06, 0.935, sub, fontsize=8, va="top", color="0.3")

    fig = plt.figure(figsize=(11.69, 8.27))
    with PdfPages(path) as pdf:
        fig.clear()
        banner(
            fig,
            f"Diff-guided biology cut  ·  shutter {shutter_idx} then live {live_idx}  ·  row {row}",
            "Cutoff = max |dI| this row shows on shutter frames 756–760. "
            f"thresh={thresh:.1f} ADU/px.  A cell = rise > +thresh, later fall < −thresh, back to the pre-rise floor.  "
            f"min width {MIN_WIDTH} px.  Same row on both frames.  Not used for FFT masks.",
        )
        gs = fig.add_gridspec(2, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.28)
        ax = fig.add_subplot(gs[0])
        ax.plot(np.arange(sh.size), sh, color="0.25", lw=0.8)
        ax.set_xlim(0, sh.size - 1)
        ax.set_ylabel("ADU")
        ax.set_title(f"shutter frame {shutter_idx}  ·  horizontal row {row}  ·  this is the expected ripple", fontsize=9)
        ax = fig.add_subplot(gs[1])
        ax.plot(x_sh, d_sh, color="0.2", lw=0.7)
        ax.axhline(thresh, color="C3", lw=1.1, label=f"+thresh = {thresh:.0f}")
        ax.axhline(-thresh, color="C3", lw=1.1)
        ax.axhline(0.0, color="0.7", lw=0.5)
        ax.set_xlim(0, sh.size - 1)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("diff (ADU / px)")
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_title(
            f"shutter differential  ·  detector marks {int(bio_sh.sum())} px here "
            f"({len(spans_sh)} spans) — should be ~none",
            fontsize=9,
        )
        pdf.savefig(fig, dpi=140)

        fig.clear()
        banner(
            fig,
            f"Live frame {live_idx}  ·  same row {row}  ·  differential vs shutter cutoff",
            "Red lines are the shutter-guided cutoff. Spikes that cross it are candidate cell edges.",
        )
        gs = fig.add_gridspec(2, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.28)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, lv, color="0.25", lw=0.8)
        ax.set_xlim(0, lv.size - 1)
        ax.set_ylabel("ADU")
        ax.set_title(f"frame {live_idx}  ·  row {row}", fontsize=9)
        ax = fig.add_subplot(gs[1])
        ax.plot(x_lv, d_lv, color="0.2", lw=0.7)
        ax.axhline(thresh, color="C3", lw=1.1, label=f"+/− thresh = {thresh:.0f}  (from shutter)")
        ax.axhline(-thresh, color="C3", lw=1.1)
        ax.axhline(0.0, color="0.7", lw=0.5)
        ax.set_xlim(0, lv.size - 1)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("diff (ADU / px)")
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_title("live differential", fontsize=9)
        pdf.savefig(fig, dpi=140)

        fig.clear()
        banner(
            fig,
            f"Marked biology  ·  frame {live_idx}  ·  {len(spans_lv)} spans, {100.0 * float(bio_lv.mean()):.0f}% of the row",
            "Orange = rise→later fall, |d| beyond shutter, width "
            f"≥ {MIN_WIDTH} px, return to pre-rise floor (±{floor_margin:.0f} ADU shutter ripple).",
        )
        ax = fig.add_axes([0.07, 0.12, 0.90, 0.76])
        for a, b in spans_lv:
            ax.axvspan(a, b, color="C1", alpha=0.25, lw=0)
        ax.plot(x, lv, color="0.15", lw=0.8)
        ax.set_xlim(0, lv.size - 1)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("ADU")
        pdf.savefig(fig, dpi=140)

        fig.clear()
        ptxt = "-" if period is None else f"{period:.1f}"
        banner(
            fig,
            f"Baseline fit on leftover samples  ·  P={ptxt} px  ·  {int(np.sum(keep))} / {lv.size} points kept",
            "Orange sine+DC is least squares on the non-orange samples only. "
            "Period from ACF after linearly filling the holes.  Judge this overlay before any FFT use.",
        )
        gs = fig.add_gridspec(2, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.28)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, lv, color="0.75", lw=0.7, label="raw (biology in)")
        if np.any(keep):
            ax.plot(x[keep], lv[keep], ".", color="0.15", ms=3, label="kept")
        ax.plot(x, fit, color="C1", lw=1.4, label="fit on kept")
        ax.set_xlim(0, lv.size - 1)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_ylabel("ADU")
        ax.set_title("full row", fontsize=9)
        ax = fig.add_subplot(gs[1])
        ax.plot(x, lv, color="0.75", lw=0.8)
        if np.any(keep):
            ax.plot(x[keep], lv[keep], ".", color="0.15", ms=4)
        ax.plot(x, fit, color="C1", lw=1.6)
        mid0, mid1 = int(0.35 * lv.size), int(0.65 * lv.size)
        ax.set_xlim(mid0, mid1)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("ADU")
        ax.set_title("middle third", fontsize=9)
        pdf.savefig(fig, dpi=140)

    plt.close(fig)
    return path


def write_inband_pdf(
    path: Path,
    *,
    shutter: np.ndarray,
    live: np.ndarray,
    shutter_stack: list[np.ndarray],
    row: int,
    live_idx: int,
    shutter_idx: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    sh = np.asarray(shutter, dtype=np.float64)[row]
    lv = np.asarray(live, dtype=np.float64)[row]
    thresh = shutter_diff_cutoff(shutter_stack, row)
    d_sh = np.diff(sh)
    d_lv = np.diff(lv)
    x_sh = np.arange(d_sh.size) + 0.5
    x_lv = np.arange(d_lv.size) + 0.5
    x = np.arange(lv.size)
    keep_sh = keep_within_shutter_diff(sh, thresh)
    keep = keep_within_shutter_diff(lv, thresh)
    drop = ~keep
    fit, period = sine_on_kept(lv, keep)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def banner(fig, title: str, sub: str) -> None:
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.97, title, fontsize=13, fontweight="bold", va="top")
        fig.text(0.06, 0.935, sub, fontsize=8, va="top", color="0.3")

    fig = plt.figure(figsize=(11.69, 8.27))
    with PdfPages(path) as pdf:
        fig.clear()
        banner(
            fig,
            f"Keep |diff| inside shutter cutoff  ·  shutter {shutter_idx} then live {live_idx}  ·  row {row}",
            "No rise–fall pairing. Drop a sample if either step that touches it has |dI| above the shutter max. "
            f"thresh={thresh:.1f} ADU/px from frames 756–760.  Not used for FFT masks.",
        )
        gs = fig.add_gridspec(2, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.28)
        ax = fig.add_subplot(gs[0])
        ax.plot(np.arange(sh.size), sh, color="0.25", lw=0.8)
        ax.set_xlim(0, sh.size - 1)
        ax.set_ylabel("ADU")
        ax.set_title(f"shutter frame {shutter_idx}  ·  row {row}", fontsize=9)
        ax = fig.add_subplot(gs[1])
        ax.plot(x_sh, d_sh, color="0.2", lw=0.7)
        ax.axhline(thresh, color="C3", lw=1.1, label=f"+/− thresh = {thresh:.0f}")
        ax.axhline(-thresh, color="C3", lw=1.1)
        ax.axhline(0.0, color="0.7", lw=0.5)
        ax.set_xlim(0, sh.size - 1)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("diff (ADU / px)")
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_title(
            f"shutter differential  ·  dropped {int((~keep_sh).sum())} / {sh.size} samples here",
            fontsize=9,
        )
        pdf.savefig(fig, dpi=140)

        fig.clear()
        banner(
            fig,
            f"Live frame {live_idx}  ·  same row {row}  ·  differential vs shutter cutoff",
            "Grey = |d| inside shutter band (kept). Red = |d| outside (those two samples are dropped).",
        )
        gs = fig.add_gridspec(2, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.28)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, lv, color="0.25", lw=0.8)
        ax.set_xlim(0, lv.size - 1)
        ax.set_ylabel("ADU")
        ax.set_title(f"frame {live_idx}  ·  row {row}", fontsize=9)
        ax = fig.add_subplot(gs[1])
        inb = np.abs(d_lv) <= thresh
        ax.plot(x_lv[inb], d_lv[inb], ".", color="0.35", ms=3, label="|d| inside shutter")
        ax.plot(x_lv[~inb], d_lv[~inb], ".", color="C3", ms=4, label="|d| outside shutter")
        ax.axhline(thresh, color="C3", lw=1.0)
        ax.axhline(-thresh, color="C3", lw=1.0)
        ax.axhline(0.0, color="0.7", lw=0.5)
        ax.set_xlim(0, lv.size - 1)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("diff (ADU / px)")
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        pdf.savefig(fig, dpi=140)

        fig.clear()
        banner(
            fig,
            f"Kept vs dropped  ·  frame {live_idx}  ·  kept {int(np.sum(keep))} / {lv.size}",
            "Black = |diff| within shutter cutoff. Red = at least one touching step is outside.",
        )
        ax = fig.add_axes([0.07, 0.12, 0.90, 0.76])
        ax.plot(x, lv, color="0.82", lw=0.7)
        if np.any(keep):
            ax.plot(x[keep], lv[keep], ".", color="0.1", ms=4, label="kept")
        if np.any(drop):
            ax.plot(x[drop], lv[drop], ".", color="C3", ms=4, label="dropped")
        ax.set_xlim(0, lv.size - 1)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("ADU")
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        pdf.savefig(fig, dpi=140)

        fig.clear()
        ptxt = "-" if period is None else f"{period:.1f}"
        banner(
            fig,
            f"Baseline fit on kept samples  ·  P={ptxt} px  ·  {int(np.sum(keep))} / {lv.size} points",
            "Orange = DC + sinusoid least squares on the black points only.",
        )
        gs = fig.add_gridspec(2, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.28)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, lv, color="0.75", lw=0.7, label="raw")
        if np.any(keep):
            ax.plot(x[keep], lv[keep], ".", color="0.15", ms=3, label="kept")
        ax.plot(x, fit, color="C1", lw=1.4, label="fit on kept")
        ax.set_xlim(0, lv.size - 1)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_ylabel("ADU")
        ax.set_title("full row", fontsize=9)
        ax = fig.add_subplot(gs[1])
        ax.plot(x, lv, color="0.75", lw=0.8)
        if np.any(keep):
            ax.plot(x[keep], lv[keep], ".", color="0.15", ms=4)
        ax.plot(x, fit, color="C1", lw=1.6)
        ax.set_xlim(int(0.35 * lv.size), int(0.65 * lv.size))
        ax.set_xlabel("x (px)")
        ax.set_ylabel("ADU")
        ax.set_title("middle third", fontsize=9)
        pdf.savefig(fig, dpi=140)

    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tif", type=Path, default=None)
    ap.add_argument("--frame", type=int, default=160)
    ap.add_argument("--shutter", type=int, default=756)
    ap.add_argument(
        "--inband",
        action="store_true",
        help="Keep samples whose |diff| stays inside the shutter cutoff (no rise-fall pairing).",
    )
    args = ap.parse_args(argv)
    tif = args.tif
    if tif is None:
        tif = Path(r"F:\bPACNewData2026\Haj Grant Example\DATA\ChanA\ChanA_stk.tif")
    if not tif.is_file():
        print(f"MISSING {tif}")
        return 1
    with tifffile.TiffFile(tif) as tf:
        n_pages = len(tf.pages)
        live = np.asarray(tf.pages[int(args.frame)].asarray())
        shutter = np.asarray(tf.pages[int(args.shutter)].asarray())
        stack = []
        for fi in SHUTTER_FRAMES:
            if 0 <= fi < n_pages:
                stack.append(np.asarray(tf.pages[fi].asarray()))
        if not stack:
            stack = [shutter]
        row = pick_fringe_row(shutter)
        thresh = shutter_diff_cutoff(stack, row)
        lv = np.asarray(live, dtype=np.float64)[row]
        if args.inband:
            keep = keep_within_shutter_diff(lv, thresh)
            print(
                f"diff-inband  live={args.frame} shutter={args.shutter} row={row} "
                f"thresh={thresh:.1f} kept={int(np.sum(keep))}/{lv.size}",
                flush=True,
            )
            out = tif.parent / "defringe_v22" / OUTPUT_SUBDIR / f"diffcut_inband_row{row}_frame_{args.frame}.pdf"
            write_inband_pdf(
                out,
                shutter=shutter,
                live=live,
                shutter_stack=stack,
                row=row,
                live_idx=int(args.frame),
                shutter_idx=int(args.shutter),
            )
        else:
            sh = np.asarray(shutter, dtype=np.float64)[row]
            margin = 0.5 * float(np.percentile(sh, 90) - np.percentile(sh, 10))
            bio, spans = mark_biology(lv, thresh, floor_margin=margin)
            print(
                f"diff-cut  live={args.frame} shutter={args.shutter} row={row} "
                f"thresh={thresh:.1f} spans={len(spans)} bio={100.0 * float(bio.mean()):.0f}%",
                flush=True,
            )
            out = tif.parent / "defringe_v22" / OUTPUT_SUBDIR / f"diffcut_row{row}_frame_{args.frame}.pdf"
            write_diff_pdf(
                out,
                shutter=shutter,
                live=live,
                shutter_stack=stack,
                row=row,
                live_idx=int(args.frame),
                shutter_idx=int(args.shutter),
            )
        print(f"  wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
