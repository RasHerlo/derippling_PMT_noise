"""Congruent spatial seed from four line scans: H, V, and both diagonals.

Whenever a fringe is present, all four cuts are measured. They must describe
the *same* 2-D frequency (qy, qx). Independent qy from a vertical cut and qx
from a horizontal cut is not a seed.

Hypotheses
  fx-only  — vertical stripes: qy=0, qx from the horizontal scan
  fy-only  — horizontal bands: qx=0, qy from the vertical scan
  tilted   — both; energy at off-axis (±qy, ±qx)

Diagonals are the veto: predicted P must match the measured main/anti periods.

``python -m batch_defringe.congruence --frame 160``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from .baseline_smooth import SEED_K, _pack
from .image_check import DIAGONALS, diagonal_sample
from .readout import _jsonable, _percentile_limits, _signed_limit
from .spatial_seed import OUTPUT_SUBDIR, fft_log_amp, reconstruct_from_mask

REL_OK = 0.25
MASK_RADIUS = 2
N_PROBE_LINES = 9


def _strongest_index(img: np.ndarray, axis: int) -> int:
    std = np.std(np.asarray(img, dtype=np.float64), axis=axis)
    return int(np.argmax(std))


def _trace_period_q(sig: np.ndarray, length: int, *, k: int = SEED_K) -> dict:
    rec = _pack(np.asarray(sig, dtype=np.float64), k, length)
    return {
        "period": rec["p_median"],
        "q": rec["q_median"],
        "marked": rec["marked"],
        "smooth": rec["smooth"],
        "segs": rec["segs"],
    }


def predicted_diag_period(
    qy: float,
    qx: float,
    h: int,
    w: int,
    which: str,
) -> float | None:
    """Period in samples along a diagonal for a plane wave at (qy, qx) bins."""
    n = int(max(h, w))
    if n < 2:
        return None
    fy_step = float(qy) * (h - 1) / ((n - 1) * h)
    fx_step = float(qx) * (w - 1) / ((n - 1) * w)
    freq = fy_step + fx_step if which == "main" else fy_step - fx_step
    freq = abs(float(freq))
    if freq < 1e-6:
        return None
    return 1.0 / freq


def _rel_err(measured: float | None, predicted: float | None) -> float:
    if predicted is None and measured is None:
        return 0.0
    if predicted is None or measured is None:
        return 10.0
    return abs(float(measured) - float(predicted)) / max(abs(float(predicted)), 1.0)


def _hyp_score(p_main: float | None, p_anti: float | None, pred_main: float | None, pred_anti: float | None) -> float:
    return 0.5 * (_rel_err(p_main, pred_main) + _rel_err(p_anti, pred_anti))


def _probe_indices(n: int, n_lines: int = N_PROBE_LINES) -> np.ndarray:
    lo = max(1, n // 16)
    hi = max(lo + 1, n - n // 16)
    return np.unique(np.linspace(lo, hi - 1, n_lines, dtype=int))


def measure_four_cuts(frame: np.ndarray, *, k: int = SEED_K) -> dict:
    """Always: several H rows, several V cols, and both diagonals."""
    arr = np.asarray(frame, dtype=np.float64)
    h, w = arr.shape
    h_cands = []
    for r in np.unique(np.concatenate([_probe_indices(h), [_strongest_index(arr, 1)]])):
        rec = _trace_period_q(arr[int(r)], w, k=k)
        rec["index"] = int(r)
        h_cands.append(rec)
    v_cands = []
    for c in np.unique(np.concatenate([_probe_indices(w), [_strongest_index(arr, 0)]])):
        rec = _trace_period_q(arr[:, int(c)], h, k=k)
        rec["index"] = int(c)
        v_cands.append(rec)
    traces = {}
    for which in DIAGONALS:
        _, sig = diagonal_sample(arr, which)
        traces[which] = {"axis": "diag", "index": None, **_trace_period_q(sig, int(sig.size), k=k)}
    return {
        "h": h,
        "w": w,
        "h_cands": h_cands,
        "v_cands": v_cands,
        "traces": traces,
    }


def choose_hypothesis(
    h: int,
    w: int,
    *,
    qx_h: float | None,
    qy_v: float | None,
    p_main: float | None,
    p_anti: float | None,
) -> dict:
    """Score fx-only / fy-only / tilted against the two diagonal periods."""
    hyps: list[dict] = []
    if qx_h is not None:
        pred_m = predicted_diag_period(0.0, qx_h, h, w, "main")
        pred_a = predicted_diag_period(0.0, qx_h, h, w, "anti")
        hyps.append(
            {
                "name": "fx",
                "qy": None,
                "qx": float(qx_h),
                "pred_main": pred_m,
                "pred_anti": pred_a,
                "score": _hyp_score(p_main, p_anti, pred_m, pred_a),
            }
        )
    if qy_v is not None:
        pred_m = predicted_diag_period(qy_v, 0.0, h, w, "main")
        pred_a = predicted_diag_period(qy_v, 0.0, h, w, "anti")
        hyps.append(
            {
                "name": "fy",
                "qy": float(qy_v),
                "qx": None,
                "pred_main": pred_m,
                "pred_anti": pred_a,
                "score": _hyp_score(p_main, p_anti, pred_m, pred_a),
            }
        )
    if qx_h is not None and qy_v is not None:
        pred_m = predicted_diag_period(qy_v, qx_h, h, w, "main")
        pred_a = predicted_diag_period(qy_v, qx_h, h, w, "anti")
        hyps.append(
            {
                "name": "tilted",
                "qy": float(qy_v),
                "qx": float(qx_h),
                "pred_main": pred_m,
                "pred_anti": pred_a,
                "score": _hyp_score(p_main, p_anti, pred_m, pred_a),
            }
        )
    winner = min(hyps, key=lambda d: d["score"]) if hyps else None
    if winner is not None and winner["score"] > REL_OK:
        winner = None
    return {"winner": winner, "hypotheses": hyps}


def congruent_seed(frame: np.ndarray, *, k: int = SEED_K) -> dict:
    """Always run H, V, and both diagonals; keep one congruent (qy, qx) or none.

    Several rows and columns are tried. The diagonals veto: a cell-heavy line
    that does not predict the diagonal periods is not a seed.
    """
    meas = measure_four_cuts(frame, k=k)
    h, w = meas["h"], meas["w"]
    tr = meas["traces"]
    p_main = tr["main"]["period"]
    p_anti = tr["anti"]["period"]
    hyps: list[dict] = []
    for hc in meas["h_cands"]:
        if hc["q"] is None:
            continue
        picked = choose_hypothesis(h, w, qx_h=hc["q"], qy_v=None, p_main=p_main, p_anti=p_anti)
        for hyp in picked["hypotheses"]:
            hyp["row"] = hc["index"]
            hyp["col"] = None
            hyp["P_h"] = hc["period"]
            hyp["P_v"] = None
            hyps.append(hyp)
    for vc in meas["v_cands"]:
        if vc["q"] is None:
            continue
        picked = choose_hypothesis(h, w, qx_h=None, qy_v=vc["q"], p_main=p_main, p_anti=p_anti)
        for hyp in picked["hypotheses"]:
            hyp["row"] = None
            hyp["col"] = vc["index"]
            hyp["P_h"] = None
            hyp["P_v"] = vc["period"]
            hyps.append(hyp)
    for hc in meas["h_cands"]:
        if hc["q"] is None:
            continue
        for vc in meas["v_cands"]:
            if vc["q"] is None:
                continue
            picked = choose_hypothesis(
                h, w, qx_h=hc["q"], qy_v=vc["q"], p_main=p_main, p_anti=p_anti
            )
            for hyp in picked["hypotheses"]:
                if hyp["name"] != "tilted":
                    continue
                hyp["row"] = hc["index"]
                hyp["col"] = vc["index"]
                hyp["P_h"] = hc["period"]
                hyp["P_v"] = vc["period"]
                hyps.append(hyp)

    winner = min(hyps, key=lambda d: d["score"]) if hyps else None
    if winner is not None and winner["score"] > REL_OK:
        winner = None

    def _by_index(cands: list[dict], idx: int | None) -> dict | None:
        if idx is None:
            return None
        for c in cands:
            if c.get("index") == idx:
                return c
        return None

    h_show = _by_index(meas["h_cands"], None if winner is None else winner.get("row"))
    v_show = _by_index(meas["v_cands"], None if winner is None else winner.get("col"))
    if h_show is None:
        h_show = next((c for c in meas["h_cands"] if c.get("q") is not None), meas["h_cands"][0] if meas["h_cands"] else None)
    if v_show is None:
        v_show = next((c for c in meas["v_cands"] if c.get("q") is not None), meas["v_cands"][0] if meas["v_cands"] else None)
    meas["row"] = None if h_show is None else h_show["index"]
    meas["col"] = None if v_show is None else v_show["index"]
    meas["traces"]["horizontal"] = {"axis": "x", **(h_show or {})}
    meas["traces"]["vertical"] = {"axis": "y", **(v_show or {})}

    return {
        "ok": winner is not None,
        "winner": None if winner is None else winner["name"],
        "qy": None if winner is None else winner["qy"],
        "qx": None if winner is None else winner["qx"],
        "score": None if winner is None else winner["score"],
        "hypotheses": sorted(hyps, key=lambda d: d["score"])[:8],
        "measured": {
            "P_h": None if h_show is None else h_show.get("period"),
            "P_v": None if v_show is None else v_show.get("period"),
            "P_main": p_main,
            "P_anti": p_anti,
            "qx_from_H": None if h_show is None else h_show.get("q"),
            "qy_from_V": None if v_show is None else v_show.get("q"),
        },
        "cuts": meas,
        "k": k,
    }


def seed_peak_mask(h: int, w: int, result: dict, radius: int = MASK_RADIUS) -> np.ndarray:
    """Thin conjugate blobs for the congruent seed — not whole rows/columns.

    fx: (fy=0, fx=±qx). fy: (fy=±qy, fx=0). tilted: off-axis (±qy, ±qx).
    """
    mask = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2
    sigma = max(0.6, radius / 1.5)

    def stamp(fy: int, fx: int) -> None:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                y = cy + fy + dy
                x = cx + fx + dx
                if not (0 <= y < h and 0 <= x < w):
                    continue
                if y == cy and x == cx:
                    continue
                wt = float(np.exp(-0.5 * (dy * dy + dx * dx) / (sigma * sigma)))
                if wt > mask[y, x]:
                    mask[y, x] = wt

    qy, qx = result.get("qy"), result.get("qx")
    name = result.get("winner")
    if name == "fx" and qx is not None:
        q = int(round(qx))
        stamp(0, q)
        stamp(0, -q)
    elif name == "fy" and qy is not None:
        q = int(round(qy))
        stamp(q, 0)
        stamp(-q, 0)
    elif name == "tilted" and qy is not None and qx is not None:
        iy, ix = int(round(qy)), int(round(qx))
        for fy in (iy, -iy):
            for fx in (ix, -ix):
                stamp(fy, fx)
    return mask


def write_congruence_pdf(path: Path, frame: np.ndarray, result: dict, *, frame_idx: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    arr = np.asarray(frame, dtype=np.float64)
    h, w = arr.shape
    tr = result["cuts"]["traces"]
    vmin, vmax = _percentile_limits(arr)
    logamp = fft_log_amp(arr)
    slo, shi = _percentile_limits(logamp, (3.0, 99.7))
    mask = seed_peak_mask(h, w, result)
    rec = reconstruct_from_mask(arr, mask)
    leftover = arr - rec
    rec_lim = _signed_limit(rec)

    def _fmt(v: float | None) -> str:
        if v is None:
            return "-"
        try:
            if not np.isfinite(float(v)):
                return "-"
        except (TypeError, ValueError):
            return "-"
        return f"{float(v):.1f}"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.69, 8.27))
    with PdfPages(path) as pdf:
        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(
            0.06,
            0.97,
            f"Congruent seed  ·  four cuts  ·  frame {frame_idx}",
            fontsize=13,
            fontweight="bold",
            va="top",
        )
        win = result["winner"] or "none"
        fig.text(
            0.06,
            0.935,
            f"Winner: {win}   qy={_fmt(result['qy'])}  qx={_fmt(result['qx'])}  "
            f"diag rel-err={_fmt(result['score'])}  (accept if ≤ {REL_OK:.2f})",
            fontsize=9,
            va="top",
            color="0.3",
        )
        m = result["measured"]
        rows = [
            "cut          P_meas   q_meas    role",
            f"H row {str(result['cuts'].get('row')):<5} {_fmt(m['P_h']):>7}  {_fmt(m['qx_from_H']):>7}   → qx",
            f"V col {str(result['cuts'].get('col')):<5} {_fmt(m['P_v']):>7}  {_fmt(m['qy_from_V']):>7}   → qy (vetoed unless diags agree)",
            f"TL–BR        {_fmt(m['P_main']):>7}  {_fmt(tr['main']['q']):>7}   veto",
            f"TR–BL        {_fmt(m['P_anti']):>7}  {_fmt(tr['anti']['q']):>7}   veto",
            "",
            "hypothesis   qy     qx     pred P_main  pred P_anti  rel-err",
        ]
        for hyp in result["hypotheses"]:
            mark = "  ← winner" if hyp["name"] == result["winner"] else ""
            rows.append(
                f"{hyp['name']:<12} {_fmt(hyp['qy']):>6} {_fmt(hyp['qx']):>6}  "
                f"{_fmt(hyp['pred_main']):>11}  {_fmt(hyp['pred_anti']):>11}  "
                f"{hyp['score']:.3f}{mark}"
            )
        fig.text(0.06, 0.88, "\n".join(rows), fontsize=8.5, va="top", family="monospace")
        pdf.savefig(fig, dpi=140)

        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.97, f"The four scans  ·  frame {frame_idx}", fontsize=13, fontweight="bold", va="top")
        fig.text(
            0.06,
            0.935,
            "Black dots = rolling lowest-4. Orange = rloess. All four always run when we look for a seed.",
            fontsize=8,
            va="top",
            color="0.3",
        )
        gs = fig.add_gridspec(2, 2, left=0.07, right=0.98, top=0.90, bottom=0.07, hspace=0.32, wspace=0.22)
        order = (
            ("horizontal", f"horizontal  row {result['cuts']['row']}"),
            ("vertical", f"vertical  col {result['cuts']['col']}"),
            ("main", "TL–BR diagonal"),
            ("anti", "TR–BL diagonal"),
        )
        for i, (name, title) in enumerate(order):
            ax = fig.add_subplot(gs[i // 2, i % 2])
            t = tr[name]
            y = t["smooth"]
            n = int(np.asarray(y).size)
            raw = arr[result["cuts"]["row"]] if name == "horizontal" else (
                arr[:, result["cuts"]["col"]] if name == "vertical" else None
            )
            if raw is None:
                _, raw = diagonal_sample(arr, name)
            x = np.arange(n)
            ax.plot(x, raw, color="0.7", lw=0.6)
            ax.plot(x[t["marked"]], np.asarray(raw)[t["marked"]], ".", color="0.1", ms=2)
            ax.plot(x, t["smooth"], color="C1", lw=1.2)
            ax.set_xlim(0, n - 1)
            ax.set_title(
                f"{title}  P={_fmt(t['period'])}  q={_fmt(t['q'])}",
                fontsize=8,
            )
        pdf.savefig(fig, dpi=140)

        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(
            0.06,
            0.97,
            f"Congruent mask IFFT  ·  frame {frame_idx}  ·  {win}",
            fontsize=13,
            fontweight="bold",
            va="top",
        )
        fig.text(
            0.06,
            0.935,
            "Only the winning (qy, qx). Peaks, not whole rows/columns. "
            "IFFT should be stripes. Leftover looking cleaner is OK if the IFFT has no cells.",
            fontsize=8,
            va="top",
            color="0.3",
        )
        gs = fig.add_gridspec(2, 3, left=0.04, right=0.99, top=0.88, bottom=0.08, wspace=0.16, hspace=0.28)
        hit = np.ma.masked_where(mask < 0.05, mask)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title("original", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(logamp, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest")
        ax.imshow(hit, cmap="Reds", alpha=0.55, interpolation="nearest", vmin=0, vmax=1)
        ax.set_title("log |FFT| + congruent peaks", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(rec, cmap="RdBu_r", vmin=-rec_lim, vmax=rec_lim, interpolation="nearest")
        ax.set_title("ifft (should be the fringe)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(leftover, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title("original − ifft", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[1, 1:])
        ax.axis("off")
        note = (
            "fx: vertical stripes, peaks on the fx axis.\n"
            "fy: horizontal bands, peaks on the fy axis.\n"
            "tilted: off-axis (±qy, ±qx), not a fake fy family plus a fake fx family.\n"
            "A vertical cut with no real y-period is not qy — the diagonals reject it."
        )
        ax.text(0.0, 0.9, note, va="top", fontsize=9, transform=ax.transAxes)
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
    print(f"congruence  frame {args.frame}  {tif}  shape={frame.shape}", flush=True)
    result = congruent_seed(frame)
    print(
        f"  winner={result['winner']}  qy={result['qy']}  qx={result['qx']}  "
        f"score={result['score']}",
        flush=True,
    )
    m = result["measured"]
    print(
        f"  P  H={m['P_h']} V={m['P_v']} main={m['P_main']} anti={m['P_anti']}",
        flush=True,
    )
    for hyp in result["hypotheses"]:
        pm = hyp["pred_main"]
        pa = hyp["pred_anti"]
        print(
            f"  hyp {hyp['name']}: qy={hyp['qy']} qx={hyp['qx']} "
            f"pred_main={pm if pm is None else f'{pm:.1f}'} "
            f"pred_anti={pa if pa is None else f'{pa:.1f}'} "
            f"err={hyp['score']:.3f}",
            flush=True,
        )
    out_dir = tif.parent / "defringe_v22" / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = write_congruence_pdf(out_dir / f"congruence_frame_{args.frame}.pdf", frame, result, frame_idx=int(args.frame))
    light = {kk: result[kk] for kk in ("ok", "winner", "qy", "qx", "score", "measured", "k")}
    light["hypotheses"] = [
        {k: h[k] for k in ("name", "qy", "qx", "pred_main", "pred_anti", "score")} for h in result["hypotheses"]
    ]
    (out_dir / f"congruence_frame_{args.frame}.json").write_text(
        json.dumps(_jsonable(light), indent=2), encoding="utf-8"
    )
    print(f"  wrote {pdf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
