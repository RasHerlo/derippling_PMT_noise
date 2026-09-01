"""Step-through of the current 1-D periodic-baseline fit, TL-BR diagonal only.

Does not change the fitter. Writes a PDF of what ``periodic_baseline`` actually
does to one trace.

``python -m batch_defringe.baseline_steps --frame 160``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

from .image_check import diagonal_sample
from .readout import _percentile_limits
from .spatial_seed import (
    MAX_HINT_Q,
    MIN_Q,
    OUTPUT_SUBDIR,
    _chirp_design,
    _fit_irls_positive_outliers,
    period_to_q,
    periodic_baseline,
)


def _acf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - float(z.mean())
    ac = np.correlate(z, z, mode="full")
    mid = len(z) - 1
    ac = ac[mid:]
    if ac[0] <= 0:
        return np.zeros(len(z), dtype=np.float64)
    return ac / ac[0]


def _local_maxima(ac: np.ndarray, *, min_lag: int = 5, max_lag: int | None = None, thresh: float = 0.08) -> list[tuple[int, float]]:
    n = int(ac.size)
    hi = min(n - 1, max_lag if max_lag is not None else max(min_lag + 2, n // 3))
    out: list[tuple[int, float]] = []
    for k in range(min_lag, hi):
        val = float(ac[k])
        if val >= thresh and val >= float(ac[k - 1]) and val >= float(ac[k + 1]):
            out.append((k, val))
    return out


def collect_steps(raw: np.ndarray, *, length: int | None = None) -> dict:
    """Replay ``periodic_baseline`` and keep the intermediates."""
    y0 = np.asarray(raw, dtype=np.float64)
    n = int(y0.size)
    nq = int(length) if length is not None else n
    dc = float(np.median(y0))
    y = y0 - dc
    mad = 1.4826 * float(np.median(np.abs(y - np.median(y)))) + 1e-12
    z = np.tanh(y / (3.0 * mad))
    zc = z - float(z.mean())
    ac = _acf(zc)
    max_lag = max(6, n // 3)
    peaks = []
    for k, val in _local_maxima(ac, min_lag=5, max_lag=max_lag, thresh=0.08):
        q = period_to_q(float(k), nq)
        in_band = q is not None and MIN_Q <= q <= MAX_HINT_Q
        peaks.append({"lag": k, "ac": val, "q": q, "in_band": in_band})

    in_band = [p for p in peaks if p["in_band"] and p["ac"] >= 0.12]
    chosen_peak = max(in_band, key=lambda p: p["ac"]) if in_band else None
    period = None if chosen_peak is None else float(chosen_peak["lag"])

    alphas = np.linspace(-0.5, 0.5, 11)
    chirp_grid = []
    best_fit = None
    best_err = np.inf
    best_alpha = 0.0
    if period is not None:
        for alpha in alphas:
            fit = _fit_irls_positive_outliers(y, _chirp_design(n, period, float(alpha)))
            r = y - fit
            keep = r <= np.percentile(r, 80.0)
            err = float(np.median(np.abs(r[keep]))) if np.any(keep) else float(np.median(np.abs(r)))
            chirp_grid.append({"alpha": float(alpha), "err": err, "fit": fit})
            if err < best_err:
                best_err = err
                best_fit = fit
                best_alpha = float(alpha)

    irls = []
    if period is not None:
        A = _chirp_design(n, period, best_alpha)
        w = np.ones(n, dtype=np.float64)
        fit = np.zeros(n, dtype=np.float64)
        for i in range(8):
            coef, _, _, _ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
            fit = A @ coef
            r = y - fit
            s = 1.4826 * float(np.median(np.abs(r))) + 1e-12
            pos = np.maximum(r / (2.5 * s), 0.0)
            w = 1.0 / (1.0 + pos ** 2)
            irls.append({"iter": i + 1, "fit": fit.copy(), "w": w.copy()})

    base, per_fn, alpha_fn = periodic_baseline(y0, length=nq)
    leftover = y0 - base
    osc = np.zeros_like(y) if best_fit is None else best_fit
    corr = 0.0
    if float(np.std(osc)) > 1e-12 and float(np.std(y)) > 1e-12:
        corr = float(np.corrcoef(y, osc)[0, 1])

    return {
        "raw": y0,
        "n": n,
        "nq": nq,
        "dc": dc,
        "y": y,
        "mad": mad,
        "z": z,
        "ac": ac,
        "max_lag": max_lag,
        "peaks": peaks,
        "period": period,
        "per_fn": per_fn,
        "alphas": alphas,
        "chirp_grid": chirp_grid,
        "best_alpha": best_alpha,
        "best_err": None if best_fit is None else best_err,
        "alpha_fn": alpha_fn,
        "irls": irls,
        "baseline": base,
        "leftover": leftover,
        "corr": corr,
    }


def write_tlbr_steps_pdf(path: Path, frame: np.ndarray, *, frame_idx: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    t, raw = diagonal_sample(frame, "main")
    st = collect_steps(raw, length=int(raw.size))
    n = st["n"]
    x = np.arange(n)
    vmin, vmax = _percentile_limits(frame)
    h, w = frame.shape

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _banner(fig, title: str, sub: str) -> None:
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.97, title, fontsize=13, fontweight="bold", va="top")
        fig.text(0.06, 0.935, sub, fontsize=8, va="top", color="0.3")

    def _zoom(ax, lo=0.35, hi=0.65) -> None:
        i0, i1 = int(lo * n), int(hi * n)
        ax.set_xlim(i0, i1)

    fig = plt.figure(figsize=(11.69, 8.27))
    with PdfPages(path) as pdf:
        # 0 — pipeline map + sampling
        fig.clear()
        _banner(
            fig,
            f"TL–BR baseline fit  ·  current process  ·  frame {frame_idx}",
            "This PDF only documents what the code does today. It does not change the fitter.",
        )
        ax = fig.add_axes([0.06, 0.38, 0.38, 0.50])
        ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.plot([0, w - 1], [0, h - 1], color="C1", lw=1.4)
        ax.set_title("step 0  sample main diagonal (TL→BR)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_axes([0.50, 0.38, 0.46, 0.50])
        ax.plot(x, st["raw"], color="0.35", lw=0.7)
        ax.set_xlim(0, n - 1)
        ax.set_title("sampled trace  (linear interpolation, n = max(H,W))", fontsize=9)
        ax.set_xlabel("sample along diagonal (px)")
        ax.set_ylabel("ADU")
        lines = [
            "Code path:  diagonal_sample(which='main')  →  periodic_baseline(sig, length=n)",
            "",
            "1. DC = median(trace)",
            "2. y = trace − DC",
            "3. Compress bright spikes: z = tanh(y / (3·MAD))",
            "4. Autocorrelation of z. Local-max lags with q = n/P in [5, 32] and ACF≥0.12.",
            "   Keep the strongest of those lags → P.",
            "5. If no such lag: return constant DC  (fit is a flat line).",
            "6. Else grid chirp α ∈ [−0.5, 0.5] (11 values). Instantaneous period",
            "   P·(1 + α·(t/T − ½)). sin/cos of that phase.",
            "7. For each α: 8× IRLS, downweight positive residuals (cells).",
            "8. Pick α with smallest median |residual| of the lower 80%.",
            "9. baseline = DC + that sinusoid. leftover = trace − baseline.",
        ]
        fig.text(0.06, 0.34, "\n".join(lines), fontsize=8, va="top", family="monospace")
        pdf.savefig(fig, dpi=140)

        # 1 — DC
        fig.clear()
        _banner(
            fig,
            f"Step 1–2  median DC  ·  frame {frame_idx}",
            f"DC = median = {st['dc']:.2f} ADU.  y = trace − DC.  This is the only 'slow' term; there is no rolling envelope.",
        )
        gs = fig.add_gridspec(2, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.28)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, st["raw"], color="0.35", lw=0.7, label="raw")
        ax.axhline(st["dc"], color="C1", lw=1.4, label=f"median DC = {st['dc']:.1f}")
        ax.set_xlim(0, n - 1)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_ylabel("ADU")
        ax.set_title("full trace", fontsize=9)
        ax = fig.add_subplot(gs[1])
        ax.plot(x, st["y"], color="0.25", lw=0.7)
        ax.axhline(0.0, color="0.7", lw=0.6)
        ax.set_xlim(0, n - 1)
        ax.set_xlabel("sample along diagonal (px)")
        ax.set_ylabel("ADU")
        ax.set_title("y = raw − DC", fontsize=9)
        pdf.savefig(fig, dpi=140)

        # 2 — tanh
        fig.clear()
        _banner(
            fig,
            f"Step 3  tanh compression  ·  frame {frame_idx}",
            f"MAD = {st['mad']:.2f}.  z = tanh(y / (3·MAD)).  Intended to flatten cell spikes so ACF sees the ripple, not soma spacing.",
        )
        gs = fig.add_gridspec(2, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.28)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, st["y"], color="0.45", lw=0.7, label="y")
        ax.plot(x, st["z"] * (3.0 * st["mad"]), color="C0", lw=0.9, label="tanh(y/(3 MAD))  (drawn back in ADU)")
        ax.set_xlim(0, n - 1)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_title("full", fontsize=9)
        ax = fig.add_subplot(gs[1])
        ax.plot(x, st["y"], color="0.45", lw=0.8, label="y")
        ax.plot(x, st["z"] * (3.0 * st["mad"]), color="C0", lw=1.0, label="compressed")
        _zoom(ax)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_xlabel("sample along diagonal (px)")
        ax.set_title("middle third  (ripples should still be visible in blue)", fontsize=9)
        pdf.savefig(fig, dpi=140)

        # 3 — ACF
        fig.clear()
        _banner(
            fig,
            f"Step 4  autocorrelation → P  ·  frame {frame_idx}",
            f"q = n/P with n={st['nq']}.  Keep local maxima with ACF≥0.12 and q in [{MIN_Q}, {MAX_HINT_Q}].  Strongest of those is P.",
        )
        ax = fig.add_axes([0.07, 0.42, 0.90, 0.46])
        lag = np.arange(st["ac"].size)
        ax.plot(lag[: st["max_lag"]], st["ac"][: st["max_lag"]], color="0.2", lw=0.9)
        ax.axhline(0.12, color="0.5", ls="--", lw=0.8, label="ACF gate 0.12")
        for p in st["peaks"]:
            color = "C1" if p["in_band"] else "0.6"
            ax.plot(p["lag"], p["ac"], "o", color=color, ms=4)
        if st["period"] is not None:
            ax.axvline(st["period"], color="C3", lw=1.2, label=f"chosen P={st['period']:.0f}")
        ax.set_xlim(0, st["max_lag"])
        ax.set_ylim(-0.2, 1.02)
        ax.set_xlabel("lag (px)")
        ax.set_ylabel("ACF")
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        rows = ["lag   ACF    q=n/P    in [5,32]?", "---  -----  --------  ----------"]
        for p in st["peaks"][:16]:
            qtxt = "-" if p["q"] is None else f"{p['q']:.2f}"
            mark = "yes" if p["in_band"] else "no"
            star = "  ← chosen" if st["period"] is not None and p["lag"] == int(st["period"]) else ""
            rows.append(f"{p['lag']:3d}  {p['ac']:.3f}  {qtxt:>8}  {mark}{star}")
        if st["period"] is None:
            rows.append("")
            rows.append("No lag in the q-band with ACF≥0.12 → baseline is the constant DC.")
        fig.text(0.07, 0.36, "\n".join(rows), fontsize=7.5, va="top", family="monospace")
        pdf.savefig(fig, dpi=140)

        # 4 — chirp grid (or skip explanation)
        fig.clear()
        if st["period"] is None:
            _banner(
                fig,
                f"Steps 5–8 skipped  ·  frame {frame_idx}",
                "No period. The function returns a constant equal to the median. That is the whole fit.",
            )
            ax = fig.add_axes([0.07, 0.18, 0.86, 0.68])
            ax.plot(x, st["raw"], color="0.35", lw=0.7, label="raw")
            ax.plot(x, st["baseline"], color="C1", lw=1.5, label="baseline = DC")
            ax.set_xlim(0, n - 1)
            ax.legend(fontsize=8, loc="upper right", frameon=False)
            ax.set_xlabel("sample along diagonal (px)")
            ax.set_ylabel("ADU")
            pdf.savefig(fig, dpi=140)
        else:
            _banner(
                fig,
                f"Step 5–6  chirp grid at P={st['period']:.0f}  ·  frame {frame_idx}",
                "α stretches period linearly along the trace. Score = median |y − fit| of points below the 80th residual percentile.",
            )
            gs = fig.add_gridspec(2, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.32)
            ax = fig.add_subplot(gs[0])
            avals = [c["alpha"] for c in st["chirp_grid"]]
            errs = [c["err"] for c in st["chirp_grid"]]
            ax.plot(avals, errs, "o-", color="C0", lw=1.2)
            ax.axvline(st["best_alpha"], color="C3", lw=1.1, label=f"chosen α={st['best_alpha']:+.2f}")
            ax.set_xlabel("chirp α")
            ax.set_ylabel("robust |resid|")
            ax.legend(fontsize=8, loc="upper right", frameon=False)
            ax.set_title("grid score", fontsize=9)
            ax = fig.add_subplot(gs[1])
            ax.plot(x, st["y"], color="0.7", lw=0.6, label="y")
            for c in st["chirp_grid"]:
                ax.plot(x, c["fit"], color="C0", lw=0.4, alpha=0.25)
            ax.plot(x, st["chirp_grid"][int(np.argmin(errs))]["fit"], color="C3", lw=1.3, label="best α")
            _zoom(ax)
            ax.legend(fontsize=8, loc="upper right", frameon=False)
            ax.set_title("middle third: all 11 chirps (faint) and the winner", fontsize=9)
            ax.set_xlabel("sample along diagonal (px)")
            pdf.savefig(fig, dpi=140)

            # IRLS
            fig.clear()
            _banner(
                fig,
                f"Step 7  IRLS at α={st['best_alpha']:+.2f}  ·  frame {frame_idx}",
                "Each iter: least squares with weights w. Residual r=y−fit. Positive r (above the sine) is downweighted: w=1/(1+(r₊/(2.5 MAD))²).",
            )
            gs = fig.add_gridspec(2, 1, left=0.07, right=0.97, top=0.88, bottom=0.08, hspace=0.32)
            ax = fig.add_subplot(gs[0])
            ax.plot(x, st["y"], color="0.7", lw=0.5)
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(st["irls"])))
            for rec, col in zip(st["irls"], colors):
                ax.plot(x, rec["fit"], color=col, lw=0.9, label=f"iter {rec['iter']}")
            _zoom(ax)
            ax.legend(fontsize=7, loc="upper right", frameon=False, ncol=4)
            ax.set_title("fit over IRLS iterations  (middle third)", fontsize=9)
            ax = fig.add_subplot(gs[1])
            last = st["irls"][-1]
            ax.plot(x, last["w"], color="C1", lw=0.8)
            ax.set_xlim(0, n - 1)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("weight")
            ax.set_xlabel("sample along diagonal (px)")
            ax.set_title("final weights  (near 0 = treated as a bright cell, ignored)", fontsize=9)
            pdf.savefig(fig, dpi=140)

        # result
        fig.clear()
        _banner(
            fig,
            f"Step 8–9  result  ·  frame {frame_idx}",
            f"P={st['period']}  α={st['best_alpha']:+.2f}  corr(y, sinusoid)={st['corr']:.3f}  "
            f"raw std={float(np.std(st['raw'])):.1f}  leftover std={float(np.std(st['leftover'])):.1f}",
        )
        gs = fig.add_gridspec(3, 1, left=0.07, right=0.97, top=0.88, bottom=0.07, hspace=0.32)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, st["raw"], color="0.4", lw=0.7, label="raw")
        ax.plot(x, st["baseline"], color="C1", lw=1.3, label="baseline returned to the PDF")
        ax.set_xlim(0, n - 1)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_title("full trace", fontsize=9)
        ax = fig.add_subplot(gs[1])
        ax.plot(x, st["raw"], color="0.4", lw=0.8, label="raw")
        ax.plot(x, st["baseline"], color="C1", lw=1.4, label="baseline")
        _zoom(ax)
        ax.legend(fontsize=8, loc="upper right", frameon=False)
        ax.set_title("middle third — this is the overlay that looked wrong", fontsize=9)
        ax = fig.add_subplot(gs[2])
        ax.plot(x, st["leftover"], color="0.2", lw=0.7)
        ax.axhline(0.0, color="0.7", lw=0.6)
        ax.set_xlim(0, n - 1)
        ax.set_xlabel("sample along diagonal (px)")
        ax.set_title("leftover = raw − baseline", fontsize=9)
        pdf.savefig(fig, dpi=140)

    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tif", type=Path, default=None)
    ap.add_argument("--frame", type=int, default=160)
    args = ap.parse_args(argv)
    tif = args.tif
    if tif is None:
        tif = Path(r"F:\bPACNewData2026\Haj Grant Example\DATA\ChanA\ChanA_stk.tif")
    if not tif.is_file():
        print(f"MISSING {tif}")
        return 1
    with tifffile.TiffFile(tif) as tf:
        frame = np.asarray(tf.pages[int(args.frame)].asarray())
    out = tif.parent / "defringe_v22" / OUTPUT_SUBDIR / f"tlbr_steps_frame_{args.frame}.pdf"
    print(f"TL-BR steps  frame {args.frame}  {tif}  shape={frame.shape}", flush=True)
    t, raw = diagonal_sample(frame, "main")
    st = collect_steps(raw, length=int(raw.size))
    print(
        f"  n={st['n']} DC={st['dc']:.2f} MAD={st['mad']:.2f} P={st['period']} "
        f"alpha={st['best_alpha']:+.2f} corr={st['corr']:.3f}",
        flush=True,
    )
    write_tlbr_steps_pdf(out, frame, frame_idx=int(args.frame))
    print(f"  wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
