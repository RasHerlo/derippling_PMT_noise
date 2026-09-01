"""Demo: rolling lowest-k samples → robust local smooth → local P and q.

Does not seed FFT masks. k=5 and global sine are not used.

``python -m batch_defringe.baseline_smooth --frame 160``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

from .baseline_diff import pick_fringe_col, pick_fringe_row
from .baseline_rolling import mark_lowest_rolling
from .image_check import seed_mask_at_q
from .readout import _percentile_limits, _signed_limit
from .spatial_seed import (
    MAX_HINT_Q,
    MIN_Q,
    OUTPUT_SUBDIR,
    fft_log_amp,
    fx_family_from_q,
    fx_seed_image,
    fy_family_from_q,
    period_to_q,
    reconstruct_from_mask,
    spectral_peak_mask,
    synthesize_fringe,
)

SEED_K = 4

K_SHOW = (3, 4)
BANDWIDTH_PX = 12.0  # below one fringe period; 18 damped the wiggle too much
SLIDE_WIN = 96
SLIDE_STEP = 8
RLOESS_ITERS = 4


def rloess_marked(
    y: np.ndarray,
    marked: np.ndarray,
    *,
    bandwidth: float = BANDWIDTH_PX,
    n_iter: int = RLOESS_ITERS,
) -> np.ndarray:
    """Robust locally linear smooth through marked samples, evaluated on every x.

    Tricube kernel in x (pixels), then bisquare reweights on residual (Cleveland rloess).
    """
    y = np.asarray(y, dtype=np.float64)
    n = int(y.size)
    x = np.arange(n, dtype=np.float64)
    xm = x[marked]
    ym = y[marked]
    n_m = int(xm.size)
    if n_m < 5:
        return np.full(n, float(np.median(y)))

    def predict(xout: np.ndarray, rob: np.ndarray) -> np.ndarray:
        out = np.empty(xout.size, dtype=np.float64)
        for j, x0 in enumerate(xout):
            dist = np.abs(xm - x0)
            bw = float(bandwidth)
            w = np.zeros(n_m, dtype=np.float64)
            for _ in range(6):
                u = dist / max(bw, 1e-6)
                w = np.clip(1.0 - u ** 3, 0.0, 1.0) ** 3
                w *= rob
                if int(np.count_nonzero(w > 1e-9)) >= 4 and float(w.sum()) > 0:
                    break
                bw *= 1.6
            sw = float(w.sum())
            if sw < 1e-12:
                out[j] = float(np.median(ym))
                continue
            dx = xm - x0
            a11 = sw
            a12 = float(np.dot(w, dx))
            a22 = float(np.dot(w, dx * dx))
            b1 = float(np.dot(w, ym))
            b2 = float(np.dot(w, dx * ym))
            det = a11 * a22 - a12 * a12
            out[j] = (a22 * b1 - a12 * b2) / det if abs(det) > 1e-12 else b1 / a11
        return out

    rob = np.ones(n_m, dtype=np.float64)
    yhat_m = predict(xm, rob)
    for _ in range(int(n_iter)):
        r = ym - yhat_m
        s = 6.0 * (float(np.median(np.abs(r))) + 1e-12)
        u = r / s
        rob = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        if float(rob.sum()) < 1e-9:
            rob[:] = 1.0
        yhat_m = predict(xm, rob)
    return predict(x, rob)


def period_from_rfft(seg: np.ndarray, *, p_lo: float, p_hi: float) -> float | None:
    """Dominant period in ``[p_lo, p_hi]`` from a short 1-D FFT (no hole-filling ACF)."""
    y = np.asarray(seg, dtype=np.float64)
    y = y - float(np.median(y))
    n = int(y.size)
    if n < 16 or float(np.std(y)) < 1e-9:
        return None
    mag = np.abs(np.fft.rfft(y))
    mag[0] = 0.0
    k_lo = max(1, int(np.floor(n / float(p_hi))))
    k_hi = min(int(mag.size) - 1, int(np.ceil(n / float(p_lo))))
    if k_hi < k_lo:
        return None
    sl = mag[k_lo : k_hi + 1]
    if float(np.max(sl)) <= 0:
        return None
    j = k_lo + int(np.argmax(sl))
    if 1 <= j < mag.size - 1:
        a, b, c = float(mag[j - 1]), float(mag[j]), float(mag[j + 1])
        den = a - 2.0 * b + c
        if abs(den) > 1e-12:
            j = j + 0.5 * (a - c) / den
    p = n / float(j)
    if p < p_lo or p > p_hi:
        return None
    return float(p)


def _period_band(length: int) -> tuple[float, float]:
    p_lo = float(length) / float(MAX_HINT_Q)
    p_hi = float(length) / float(MIN_Q)
    return p_lo, p_hi


def sliding_period_q(
    curve: np.ndarray,
    *,
    length: int,
    win: int = SLIDE_WIN,
    step: int = SLIDE_STEP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local period and q = length/P along a smooth 1-D curve."""
    y = np.asarray(curve, dtype=np.float64)
    n = int(y.size)
    p_lo, p_hi = _period_band(length)
    xs: list[float] = []
    ps: list[float] = []
    qs: list[float] = []
    for i0 in range(0, max(1, n - win + 1), step):
        p = period_from_rfft(y[i0 : i0 + win], p_lo=p_lo, p_hi=p_hi)
        q = period_to_q(p, length)
        xs.append(i0 + 0.5 * min(win, n - i0))
        ps.append(np.nan if p is None else float(p))
        qs.append(np.nan if q is None else float(q))
    return np.asarray(xs), np.asarray(ps), np.asarray(qs)


def segment_periods(curve: np.ndarray, length: int, n_seg: int = 5) -> list[dict]:
    y = np.asarray(curve, dtype=np.float64)
    n = int(y.size)
    p_lo, p_hi = _period_band(length)
    out = []
    for i in range(n_seg):
        a = int(round(i * n / n_seg))
        b = int(round((i + 1) * n / n_seg))
        p = period_from_rfft(y[a:b], p_lo=p_lo, p_hi=p_hi)
        q = period_to_q(p, length)
        out.append({"i": i, "a": a, "b": b, "period": p, "q": q})
    return out


def _pack(y: np.ndarray, k: int, length: int) -> dict:
    marked, _votes = mark_lowest_rolling(y, n_lowest=k)
    smooth = rloess_marked(y, marked)
    xs, ps, qs = sliding_period_q(smooth, length=length)
    segs = segment_periods(smooth, length)
    good = qs[np.isfinite(qs)]
    return {
        "k": k,
        "marked": marked,
        "smooth": smooth,
        "xs": xs,
        "ps": ps,
        "qs": qs,
        "segs": segs,
        "q_median": float(np.median(good)) if good.size else None,
        "p_median": float(np.median(ps[np.isfinite(ps)])) if np.any(np.isfinite(ps)) else None,
        "n_marked": int(np.sum(marked)),
    }


def write_smooth_pdf(
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
    n = int(lv.size)
    x = np.arange(n)
    mid0, mid1 = int(0.35 * n), int(0.65 * n)
    sh_pack = {k: _pack(sh, k, n) for k in K_SHOW}
    lv_pack = {k: _pack(lv, k, n) for k in K_SHOW}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def banner(fig, title: str, sub: str) -> None:
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.97, title, fontsize=13, fontweight="bold", va="top")
        fig.text(0.06, 0.935, sub, fontsize=8, va="top", color="0.3")

    def _fmt(v: float | None) -> str:
        return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.1f}"

    fig = plt.figure(figsize=(11.69, 8.27))
    with PdfPages(path) as pdf:
        fig.clear()
        banner(
            fig,
            f"rloess on rolling lowest-k  ·  demo only  ·  row {row}",
            "Not a q-seed yet. No global sine. Bandwidth "
            f"{BANDWIDTH_PX:.0f} px (half a ~36 px fringe). Horizontal row → qx = W/P.",
        )
        lines = [
            "1. Rolling window of 10: mark the k lowest samples (k=3 and k=4; not 5).",
            "2. Robust local linear smooth (rloess) through those dots only, evaluated on every x.",
            "3. Sliding local FFT (96 px) on the smooth curve → P(x) → q(x)=W/P. Median q is the later seed.",
            "   Edge q is chirp, not a second family. Five equal segments are the piecewise version of the same.",
            "",
            "k=5 is omitted: linear-fill ACF locked onto 5 px grain. Global sine is omitted: one P cannot chirp.",
        ]
        fig.text(0.08, 0.82, "\n".join(lines), fontsize=10, va="top")
        pdf.savefig(fig, dpi=140)

        fig.clear()
        banner(
            fig,
            f"Shutter {shutter_idx}  ·  row {row}  ·  dots + rloess",
            "Pure fringe control. Orange should sit on the ripples for both k.",
        )
        gs = fig.add_gridspec(2, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.30)
        for i, k in enumerate(K_SHOW):
            ax = fig.add_subplot(gs[i])
            rec = sh_pack[k]
            ax.plot(x, sh, color="0.75", lw=0.7)
            ax.plot(x[rec["marked"]], sh[rec["marked"]], ".", color="0.1", ms=3)
            ax.plot(x, rec["smooth"], color="C1", lw=1.4)
            ax.set_xlim(0, n - 1)
            ax.set_ylabel("ADU")
            ax.set_title(
                f"k={k}  marked {rec['n_marked']}/{n}  median P={_fmt(rec['p_median'])}  "
                f"median qx={_fmt(rec['q_median'])}",
                fontsize=9,
            )
        ax.set_xlabel("x (px)")
        pdf.savefig(fig, dpi=140)

        fig.clear()
        banner(
            fig,
            f"Live {live_idx}  ·  same row  ·  dots + rloess",
            "Black = rolling lowest-k. Orange = rloess. This is the baseline curve, not a sinusoid.",
        )
        gs = fig.add_gridspec(2, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.30)
        for i, k in enumerate(K_SHOW):
            ax = fig.add_subplot(gs[i])
            rec = lv_pack[k]
            ax.plot(x, lv, color="0.75", lw=0.7)
            ax.plot(x[rec["marked"]], lv[rec["marked"]], ".", color="0.1", ms=3)
            ax.plot(x, rec["smooth"], color="C1", lw=1.4)
            ax.set_xlim(0, n - 1)
            ax.set_ylabel("ADU")
            ax.set_title(
                f"k={k}  marked {rec['n_marked']}/{n}  median P={_fmt(rec['p_median'])}  "
                f"median qx={_fmt(rec['q_median'])}",
                fontsize=9,
            )
        ax.set_xlabel("x (px)")
        pdf.savefig(fig, dpi=140)

        fig.clear()
        banner(
            fig,
            f"Live {live_idx}  ·  middle third",
            "Same overlays, zoomed. Judge whether orange follows the chirped ripples.",
        )
        gs = fig.add_gridspec(2, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.30)
        for i, k in enumerate(K_SHOW):
            ax = fig.add_subplot(gs[i])
            rec = lv_pack[k]
            ax.plot(x, lv, color="0.75", lw=0.8)
            ax.plot(x[rec["marked"]], lv[rec["marked"]], ".", color="0.1", ms=4)
            ax.plot(x, rec["smooth"], color="C1", lw=1.6)
            ax.set_xlim(mid0, mid1)
            ax.set_ylabel("ADU")
            ax.set_title(f"k={k}", fontsize=9)
        ax.set_xlabel("x (px)")
        pdf.savefig(fig, dpi=140)

        rec = lv_pack[4]
        fig.clear()
        banner(
            fig,
            f"Local P(x) and qx(x) from the k=4 smooth curve  ·  live {live_idx}",
            f"Local FFT window {SLIDE_WIN} px on the smooth curve. Median qx is the later seed. "
            "Edges moving away from that median are chirp, not a new family.",
        )
        gs = fig.add_gridspec(2, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.28)
        ax = fig.add_subplot(gs[0])
        ax.plot(rec["xs"], rec["ps"], "o-", color="C0", ms=3, lw=1.0)
        ax.axhline(rec["p_median"] or np.nan, color="C1", ls="--", lw=1.0, label=f"median P={_fmt(rec['p_median'])}")
        ax.set_xlim(0, n - 1)
        ax.set_ylabel("P (px)")
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_title("period along the row", fontsize=9)
        ax = fig.add_subplot(gs[1])
        ax.plot(rec["xs"], rec["qs"], "o-", color="C0", ms=3, lw=1.0)
        ax.axhline(rec["q_median"] or np.nan, color="C1", ls="--", lw=1.0, label=f"median qx={_fmt(rec['q_median'])}")
        ax.set_xlim(0, n - 1)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("qx = W/P")
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_title("q-guess along the row", fontsize=9)
        pdf.savefig(fig, dpi=140)

        fig.clear()
        banner(
            fig,
            f"Five segments and 1-D FFT of the k=4 smooth curve  ·  live {live_idx}",
            "Piecewise P→q is the simple version of the same chirp. FFT of the smooth 1-D curve should peak near median qx.",
        )
        gs = fig.add_gridspec(2, 1, left=0.07, right=0.97, top=0.86, bottom=0.08, hspace=0.32)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, rec["smooth"] - float(np.median(rec["smooth"])), color="C1", lw=1.0)
        for s in rec["segs"]:
            ax.axvline(s["a"], color="0.7", lw=0.6)
        ax.axvline(rec["segs"][-1]["b"] - 1, color="0.7", lw=0.6)
        ax.set_xlim(0, n - 1)
        ax.set_ylabel("smooth − median")
        seg_txt = "   ".join(
            f"[{s['a']}-{s['b']}) P={_fmt(s['period'])} qx={_fmt(s['q'])}" for s in rec["segs"]
        )
        ax.set_title(seg_txt, fontsize=7)
        ax = fig.add_subplot(gs[1])
        yc = rec["smooth"] - float(np.median(rec["smooth"]))
        mag = np.abs(np.fft.rfft(yc))
        mag[0] = 0.0
        kk = np.arange(mag.size)
        ax.plot(kk[:80], mag[:80], color="0.2", lw=1.0)
        if rec["q_median"] is not None:
            ax.axvline(rec["q_median"], color="C1", ls="--", lw=1.1, label=f"median qx={_fmt(rec['q_median'])}")
        ax.set_xlim(0, 80)
        ax.set_xlabel("FFT bin k  (= qx for this horizontal cut)")
        ax.set_ylabel("|rFFT|")
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_title("1-D FFT of rloess curve", fontsize=9)
        pdf.savefig(fig, dpi=140)

        fig.clear()
        banner(
            fig,
            "Later: scanning across the xy field  (not implemented)",
            "This 1-D recipe is a proposer, not a tile cleaner. Production remains a global ridge notch per family.",
        )
        notes = [
            "Horizontal strips (along x, this demo) → qx(x). Vertical strips (along y) → qy(y).",
            "Do not mix a diagonal cut into either seed.",
            "",
            "Walk overlapping x-windows (and later y-windows). In each window:",
            "  rolling lowest-k on one line or a median of a few lines → rloess → median q in that window.",
            "That q only votes into the existing small FFT search on the whole frame. No per-tile notch.",
            "",
            "Why this helps a walk:",
            "  • lowest-k ignores bright cells without a shutter |diff| cutoff (that cutoff failed when cells are dense);",
            "  • rloess absorbs scanner chirp inside a window so ACF is not forced to one P;",
            "  • q(x) / q(y) maps show whether one q + TRACK_SEARCH is enough, or the peak is smeared;",
            "  • edge vs center disagreement is chirp, not a second family.",
            "",
            "Cost: rloess is heavier than ACF. For a walk, run it on every Nth line or on a strip-mean of marked samples.",
            "Do not IFFT the 1-D smooth back into the image as the removal — that is still the 2-D ridge notch.",
        ]
        fig.text(0.08, 0.84, "\n".join(notes), fontsize=10, va="top")
        pdf.savefig(fig, dpi=140)

    plt.close(fig)
    return path


def _overlay_mask(ax, mask: np.ndarray, cmap, *, alpha: float = 0.45) -> None:
    hit = np.ma.masked_where(mask < 0.05, mask)
    ax.imshow(hit, cmap=cmap, alpha=alpha, interpolation="nearest", vmin=0.0, vmax=1.0)


def write_reconstruct_pdf(
    path: Path,
    frame: np.ndarray,
    *,
    shutter: np.ndarray,
    frame_idx: int,
    k: int = SEED_K,
) -> Path:
    """IFFT of the thin masks at rloess median qx / qy — the fringe those seeds would take."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    arr = np.asarray(frame, dtype=np.float64)
    h, w = arr.shape
    row = pick_fringe_row(shutter)
    col = pick_fringe_col(shutter)
    hx = _pack(arr[row], k, w)
    vy = _pack(arr[:, col], k, h)
    qx = hx["q_median"]
    qy = vy["q_median"]
    qx_edge = None
    segs = hx["segs"]
    if segs and segs[0]["q"] is not None and segs[-1]["q"] is not None:
        qx_edge = 0.5 * (float(segs[0]["q"]) + float(segs[-1]["q"]))

    logamp = fft_log_amp(arr)
    peak = spectral_peak_mask(h, w, qy=qy, qx=qx)
    peak_x = spectral_peak_mask(h, w, qx=qx)
    peak_y = spectral_peak_mask(h, w, qy=qy)
    fy_adapt = (
        seed_mask_at_q((h, w), fy_family_from_q(logamp, qy), qy)
        if qy is not None
        else np.zeros((h, w), dtype=np.float32)
    )
    fx_adapt = (
        fx_seed_image((h, w), fx_family_from_q(logamp, qx))
        if qx is not None
        else np.zeros((h, w), dtype=np.float32)
    )
    adapt = np.maximum(fy_adapt, fx_adapt)
    rec_peak = reconstruct_from_mask(arr, peak)
    rec_x = reconstruct_from_mask(arr, peak_x)
    rec_y = reconstruct_from_mask(arr, peak_y)
    rec_adapt = reconstruct_from_mask(arr, adapt)
    synth = synthesize_fringe(h, w, qy=qy, qx=qx)
    leftover_peak = arr - rec_peak
    leftover_adapt = arr - rec_adapt
    vmin, vmax = _percentile_limits(arr)
    slo, shi = _percentile_limits(logamp, (3.0, 99.7))
    rec_lim = _signed_limit(np.concatenate([rec_peak.ravel(), rec_adapt.ravel(), rec_x.ravel(), rec_y.ravel()]))
    syn_lim = _signed_limit(synth)

    def _fmt(v: float | None) -> str:
        return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.1f}"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.69, 8.27))
    with PdfPages(path) as pdf:
        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(
            0.06,
            0.97,
            f"rloess q-seeds → FFT mask IFFT  ·  frame {frame_idx}  ·  k={k}",
            fontsize=13,
            fontweight="bold",
            va="top",
        )
        fig.text(
            0.06,
            0.935,
            "Horizontal row → qx. Vertical col → qy. Median q is the seed. "
            "Masks are conjugate peaks / adapted ridges — not whole rows or columns.",
            fontsize=8,
            va="top",
            color="0.3",
        )
        lines = [
            f"row {row}  median P={_fmt(hx['p_median'])}  median qx={_fmt(qx)}  "
            f"edge qx={_fmt(qx_edge)}  (H segs: "
            + ", ".join(_fmt(s['q']) for s in hx['segs'])
            + ")",
            f"col {col}  median P={_fmt(vy['p_median'])}  median qy={_fmt(qy)}  "
            f"(V segs: " + ", ".join(_fmt(s['q']) for s in vy['segs']) + ")",
        ]
        fig.text(0.06, 0.88, "\n".join(lines), fontsize=9, va="top", family="monospace")
        gs = fig.add_gridspec(2, 1, left=0.07, right=0.97, top=0.78, bottom=0.08, hspace=0.32)
        ax = fig.add_subplot(gs[0])
        xr = np.arange(w)
        ax.plot(xr, arr[row], color="0.7", lw=0.6)
        ax.plot(xr[hx["marked"]], arr[row][hx["marked"]], ".", color="0.1", ms=2)
        ax.plot(xr, hx["smooth"], color="C1", lw=1.2)
        ax.set_xlim(0, w - 1)
        ax.set_title(f"horizontal row {row}  →  qx={_fmt(qx)}", fontsize=9)
        ax = fig.add_subplot(gs[1])
        yc = np.arange(h)
        ax.plot(yc, arr[:, col], color="0.7", lw=0.6)
        ax.plot(yc[vy["marked"]], arr[:, col][vy["marked"]], ".", color="0.1", ms=2)
        ax.plot(yc, vy["smooth"], color="C1", lw=1.2)
        ax.set_xlim(0, h - 1)
        ax.set_xlabel("position (px)")
        ax.set_title(f"vertical col {col}  →  qy={_fmt(qy)}", fontsize=9)
        pdf.savefig(fig, dpi=140)

        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(
            0.06,
            0.97,
            f"Masks at those q  ·  frame {frame_idx}  ·  qy={_fmt(qy)}  qx={_fmt(qx)}",
            fontsize=13,
            fontweight="bold",
            va="top",
        )
        fig.text(
            0.06,
            0.935,
            "Left: log |FFT|. Middle: conjugate peaks at ±q (the grating bins). "
            "Right: adapted ridge support at the same q (production-like, still not a full row/column).",
            fontsize=8,
            va="top",
            color="0.3",
        )
        gs = fig.add_gridspec(1, 3, left=0.04, right=0.99, top=0.88, bottom=0.10, wspace=0.14)
        ax = fig.add_subplot(gs[0])
        ax.imshow(logamp, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest")
        ax.set_title("log |FFT|", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[1])
        ax.imshow(logamp, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest")
        _overlay_mask(ax, peak, "Reds")
        ax.set_title("conjugate peaks", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[2])
        ax.imshow(logamp, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest")
        if qy is not None:
            _overlay_mask(ax, fy_adapt, "Blues")
        if qx is not None:
            _overlay_mask(ax, fx_adapt, "Oranges")
        ax.set_title("adapted ridge  (fy blue / fx orange)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        pdf.savefig(fig, dpi=140)

        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(
            0.06,
            0.97,
            f"IFFT of those masks  ·  frame {frame_idx}",
            fontsize=13,
            fontweight="bold",
            va="top",
        )
        fig.text(
            0.06,
            0.935,
            "Should look like stripes, not a smoothed FOV. "
            "Intended grating is a unit cosine at the seed q. Peak IFFT is this frame's energy in those bins.",
            fontsize=8,
            va="top",
            color="0.3",
        )
        gs = fig.add_gridspec(2, 3, left=0.04, right=0.99, top=0.88, bottom=0.08, wspace=0.16, hspace=0.28)
        panels = (
            (arr, "gray", (vmin, vmax), "original"),
            (synth, "RdBu_r", (-syn_lim, syn_lim), f"intended  qy={_fmt(qy)} qx={_fmt(qx)}"),
            (rec_peak, "RdBu_r", (-rec_lim, rec_lim), "ifft conjugate peaks"),
            (rec_adapt, "RdBu_r", (-rec_lim, rec_lim), "ifft adapted ridge"),
            (leftover_peak, "gray", (vmin, vmax), "original − peak ifft"),
            (leftover_adapt, "gray", (vmin, vmax), "original − adapted ifft"),
        )
        for j, (im, cmap, lim, title) in enumerate(panels):
            ax = fig.add_subplot(gs[j // 3, j % 3])
            ax.imshow(im, cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        pdf.savefig(fig, dpi=140)

        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(
            0.06,
            0.97,
            f"Split families  ·  frame {frame_idx}",
            fontsize=13,
            fontweight="bold",
            va="top",
        )
        fig.text(
            0.06,
            0.935,
            "qx-only should be vertical stripes. qy-only should be horizontal bands. "
            f"Edge qx={_fmt(qx_edge)} is chirp, shown only as a peak ifft for comparison.",
            fontsize=8,
            va="top",
            color="0.3",
        )
        rec_edge = (
            reconstruct_from_mask(arr, spectral_peak_mask(h, w, qx=qx_edge))
            if qx_edge is not None
            else np.zeros((h, w), dtype=np.float32)
        )
        e_lim = _signed_limit(rec_edge)
        gs = fig.add_gridspec(2, 3, left=0.04, right=0.99, top=0.88, bottom=0.08, wspace=0.16, hspace=0.28)
        split = (
            (rec_x, "RdBu_r", (-rec_lim, rec_lim), f"ifft qx={_fmt(qx)} peaks"),
            (rec_y, "RdBu_r", (-rec_lim, rec_lim), f"ifft qy={_fmt(qy)} peaks"),
            (rec_peak, "RdBu_r", (-rec_lim, rec_lim), "ifft both medians"),
            (arr - rec_x, "gray", (vmin, vmax), "original − qx ifft"),
            (arr - rec_y, "gray", (vmin, vmax), "original − qy ifft"),
            (rec_edge, "RdBu_r", (-e_lim, e_lim), f"ifft edge qx={_fmt(qx_edge)} (chirp)"),
        )
        for j, (im, cmap, lim, title) in enumerate(split):
            ax = fig.add_subplot(gs[j // 3, j % 3])
            ax.imshow(im, cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        pdf.savefig(fig, dpi=140)

    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tif", type=Path, default=None)
    ap.add_argument("--frame", type=int, default=160)
    ap.add_argument("--shutter", type=int, default=756)
    ap.add_argument(
        "--reconstruct",
        action="store_true",
        help="IFFT the thin FFT masks at rloess median qx/qy (no production write).",
    )
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
    if args.reconstruct:
        out = tif.parent / "defringe_v22" / OUTPUT_SUBDIR / f"rloess_recon_k{SEED_K}_frame_{args.frame}.pdf"
        print(f"rloess reconstruct  live={args.frame} shutter={args.shutter} k={SEED_K}", flush=True)
        rec_h = _pack(np.asarray(live, dtype=np.float64)[row], SEED_K, live.shape[1])
        col = pick_fringe_col(shutter)
        rec_v = _pack(np.asarray(live, dtype=np.float64)[:, col], SEED_K, live.shape[0])
        print(f"  row {row} median qx={rec_h['q_median']}  col {col} median qy={rec_v['q_median']}", flush=True)
        write_reconstruct_pdf(out, live, shutter=shutter, frame_idx=int(args.frame), k=SEED_K)
        print(f"  wrote {out}", flush=True)
        return 0
    n = live.shape[1]
    lv = np.asarray(live, dtype=np.float64)[row]
    print(f"rloess demo  live={args.frame} shutter={args.shutter} row={row}", flush=True)
    for k in K_SHOW:
        rec = _pack(lv, k, n)
        print(
            f"  k={k} marked {rec['n_marked']}/{n}  median P={rec['p_median']}  median qx={rec['q_median']}",
            flush=True,
        )
        for s in rec["segs"]:
            print(
                f"    seg {s['i']} x={s['a']}:{s['b']}  P={s['period']}  qx={s['q']}",
                flush=True,
            )
    out = tif.parent / "defringe_v22" / OUTPUT_SUBDIR / f"rloess_row{row}_frame_{args.frame}.pdf"
    write_smooth_pdf(
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
