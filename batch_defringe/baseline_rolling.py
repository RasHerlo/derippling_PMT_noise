"""Rolling-window lowest-k samples, then sine fit on the marked set.

Window = 10 samples. In each window mark the k lowest (k = 3, 4, 5).
Fit is DC + sinusoid on the union of marked samples only.

Probe only. Does not write FFT masks.

``python -m batch_defringe.baseline_rolling --frame 160``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

from .baseline_diff import pick_fringe_row, sine_on_kept
from .spatial_seed import OUTPUT_SUBDIR

WINDOW = 10
K_LIST = (3, 4, 5)


def mark_lowest_rolling(sig: np.ndarray, *, window: int = WINDOW, n_lowest: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(sig, dtype=np.float64)
    n = int(y.size)
    k = int(n_lowest)
    if k < 1 or k >= window:
        raise ValueError("n_lowest must be in 1 .. window-1")
    marked = np.zeros(n, dtype=bool)
    votes = np.zeros(n, dtype=np.int32)
    for i in range(0, n - window + 1):
        sl = y[i : i + window]
        idx = np.argpartition(sl, k)[:k]
        marked[i + idx] = True
        votes[i + idx] += 1
    return marked, votes


def _one_fit(sig: np.ndarray, k: int) -> dict:
    marked, votes = mark_lowest_rolling(sig, n_lowest=k)
    fit, period = sine_on_kept(sig, marked)
    return {
        "k": k,
        "marked": marked,
        "votes": votes,
        "fit": fit,
        "period": period,
        "n_marked": int(np.sum(marked)),
    }


def write_rolling_pdf(
    path: Path,
    *,
    shutter: np.ndarray,
    live: np.ndarray,
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
    sh_fits = [_one_fit(sh, k) for k in K_LIST]
    lv_fits = [_one_fit(lv, k) for k in K_LIST]
    xs = np.arange(sh.size)
    xl = np.arange(lv.size)
    mid0, mid1 = int(0.35 * lv.size), int(0.65 * lv.size)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def banner(fig, title: str, sub: str) -> None:
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.97, title, fontsize=13, fontweight="bold", va="top")
        fig.text(0.06, 0.935, sub, fontsize=8, va="top", color="0.3")

    def panel_trace(ax, x, raw, rec, *, zoom=False, ylabel=True):
        ax.plot(x, raw, color="0.75", lw=0.7)
        ax.plot(x[rec["marked"]], raw[rec["marked"]], ".", color="0.1", ms=3)
        ax.plot(x, rec["fit"], color="C1", lw=1.3)
        p = "-" if rec["period"] is None else f"{rec['period']:.1f}"
        ax.set_title(
            f"k={rec['k']} lowest / {WINDOW}   marked {rec['n_marked']}/{raw.size}   P={p}",
            fontsize=8,
        )
        if zoom:
            ax.set_xlim(mid0, mid1)
        else:
            ax.set_xlim(0, raw.size - 1)
        if ylabel:
            ax.set_ylabel("ADU")

    fig = plt.figure(figsize=(11.69, 8.27))
    with PdfPages(path) as pdf:
        fig.clear()
        banner(
            fig,
            f"Rolling window {WINDOW}: lowest k, then sine on marked  ·  shutter {shutter_idx}  ·  row {row}",
            "Black dots = samples that were among the k lowest in at least one window of 10. "
            "Orange = DC + sinusoid least-squares on those dots only.  Pure fringe: dots should sit on the troughs.",
        )
        gs = fig.add_gridspec(3, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.32)
        for i, rec in enumerate(sh_fits):
            ax = fig.add_subplot(gs[i])
            panel_trace(ax, xs, sh, rec)
        ax.set_xlabel("x (px)")
        pdf.savefig(fig, dpi=140)

        fig.clear()
        banner(
            fig,
            f"Live frame {live_idx}  ·  same row {row}  ·  full trace",
            "Same rule. Biology is bright, so it should rarely be among the lowest k in a window of 10.",
        )
        gs = fig.add_gridspec(3, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.32)
        for i, rec in enumerate(lv_fits):
            ax = fig.add_subplot(gs[i])
            panel_trace(ax, xl, lv, rec)
        ax.set_xlabel("x (px)")
        pdf.savefig(fig, dpi=140)

        fig.clear()
        banner(
            fig,
            f"Live frame {live_idx}  ·  middle third",
            "Same three fits, zoomed. Judge whether orange sits on the ripples, not the cells.",
        )
        gs = fig.add_gridspec(3, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.32)
        for i, rec in enumerate(lv_fits):
            ax = fig.add_subplot(gs[i])
            panel_trace(ax, xl, lv, rec, zoom=True)
        ax.set_xlabel("x (px)")
        pdf.savefig(fig, dpi=140)

    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tif", type=Path, default=None)
    ap.add_argument("--frame", type=int, default=160)
    ap.add_argument("--shutter", type=int, default=756)
    args = ap.parse_args(argv)
    tif = args.tif
    if tif is None:
        tif = Path(r"F:\bPACNewData2026\Haj Grant Example\DATA\ChanA\ChanA_stk.tif")
    if not tif.is_file():
        print(f"MISSING {tif}")
        return 1
    with tifffile.TiffFile(tif) as tf:
        live = np.asarray(tf.pages[int(args.frame)].asarray())
        shutter = np.asarray(tf.pages[int(args.shutter)].asarray())
    row = pick_fringe_row(shutter)
    lv = np.asarray(live, dtype=np.float64)[row]
    for k in K_LIST:
        rec = _one_fit(lv, k)
        p = "-" if rec["period"] is None else f"{rec['period']:.1f}"
        print(f"  k={k}  marked {rec['n_marked']}/{lv.size}  P={p}", flush=True)
    out = tif.parent / "defringe_v22" / OUTPUT_SUBDIR / f"rolling_w{WINDOW}_row{row}_frame_{args.frame}.pdf"
    print(f"rolling-lowest  live={args.frame} shutter={args.shutter} row={row}", flush=True)
    write_rolling_pdf(
        out,
        shutter=shutter,
        live=live,
        row=row,
        live_idx=int(args.frame),
        shutter_idx=int(args.shutter),
    )
    print(f"  wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
