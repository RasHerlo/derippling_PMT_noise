"""v4 overview PDF: cover, shutter, traces, means, rung stories, undone, linescans."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .congruence import seed_peak_mask
from .readout import _percentile_limits, _signed_limit
from .shutter_detect import draw_shutter_page, format_shutter_span
from .spatial_seed import spectral_peak_mask
from .v3_report import _plot_linescan


def _lines_mask(h: int, w: int, lines: list[dict], families: list[dict] | None = None) -> np.ndarray:
    """Public line dicts: thin blobs for linescan peaks, ridges as short bands."""
    mask = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2
    rad = 2
    for ln in lines or []:
        q = float(ln.get("q") or 0)
        axis = ln.get("axis")
        kind = ln.get("kind") or "ridge"
        if kind == "peak":
            if axis == "fy":
                mask = np.maximum(mask, spectral_peak_mask(h, w, qy=q))
            else:
                mask = np.maximum(mask, spectral_peak_mask(h, w, qx=q))
            continue
        iq = int(round(q))
        if axis == "fy":
            for sgn in (-1, +1):
                y = cy + sgn * iq
                if 0 <= y < h:
                    mask[max(0, y - rad) : min(h, y + rad + 1), :] = np.maximum(
                        mask[max(0, y - rad) : min(h, y + rad + 1), :], 0.7
                    )
        else:
            for sgn in (-1, +1):
                x = cx + sgn * iq
                if 0 <= x < w and abs(x - cx) >= 5:
                    mask[:, max(0, x - rad) : min(w, x + rad + 1)] = np.maximum(
                        mask[:, max(0, x - rad) : min(w, x + rad + 1)], 0.7
                    )
    return mask


def write_v4_seed10_pdf(
    path: Path,
    *,
    tif_path: Path,
    channel: str,
    computer: str,
    shutter: dict,
    results: list[dict],
    mean_raw,
    mean_cleaned,
    mean_removed,
    mean_predicted,
    summary: dict,
) -> Path:
    return write_v4_report(
        path,
        tif_path=tif_path,
        channel=channel,
        computer=computer,
        shutter=shutter,
        rows=results,
        inspect=results,
        mean_raw=mean_raw,
        mean_cleaned=mean_cleaned,
        mean_removed=mean_removed,
        mean_predicted=mean_predicted,
        summary=summary,
        title="v4 seed-10",
        traces_title="Per-frame traces (seed-10, not the full stack)",
        means_title="Means of the 10 frames",
        shutter_subtitle=(
            "FOV std cliff. Seed-10 reused ChanA seed_compare indices on both PMTs "
            "for convenience; the two channels are not coupled."
        ),
        list_cover_frames=True,
    )


def write_v4_report(
    path: Path,
    *,
    tif_path: Path,
    channel: str,
    computer: str,
    shutter: dict,
    rows: list[dict],
    inspect: list[dict],
    mean_raw,
    mean_cleaned,
    mean_removed,
    mean_predicted,
    summary: dict,
    title: str = "v4 defringe",
    traces_title: str = "Per-frame traces (full stack)",
    means_title: str = "Means of the stack",
    shutter_subtitle: str | None = None,
    list_cover_frames: bool = False,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import TwoSlopeNorm

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.69, 8.27))

    def _imshow(ax, img, *, signed=False, title=""):
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if img is None:
            ax.set_facecolor("0.92")
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes, color="0.4")
            return
        arr = np.asarray(img, dtype=float)
        if signed:
            lim = _signed_limit(arr)
            ax.imshow(arr, cmap="gray", norm=TwoSlopeNorm(0.0, vmin=-lim, vmax=lim), interpolation="nearest")
        else:
            lo, hi = _percentile_limits(arr)
            ax.imshow(arr, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")

    with PdfPages(path) as pdf:
        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.97, f"{title}  ·  {channel}", fontsize=14, fontweight="bold", va="top")
        fig.text(0.06, 0.935, f"{tif_path}  ·  {computer}", fontsize=8, va="top", color="0.35")
        sm = summary
        frac = sm.get("frac_frames_any_active")
        lines = [
            f"shutter: {format_shutter_span(shutter)}",
            f"n={sm.get('n_frames')}  empty {sm.get('n_empty')}  core-only {sm.get('n_core_only')}  "
            f"biology-brake {sm.get('n_brake')}"
            + (f"  active {100 * frac:.1f}%" if frac is not None else ""),
            f"median removed RMS {sm.get('median_removed_rms', 0):.3g}   "
            f"median predicted↔removed {sm.get('median_agree', 0):.2f}",
            f"catalog qs={sm.get('catalog_qs') or 'none'}  "
            f"shutter-learn qs={sm.get('shutter_learn_qs') or 'none'}  "
            f"branch={sm.get('catalog_branch')}",
        ]
        inspect_note = sm.get("inspect")
        if inspect_note:
            lines.append(
                "inspect: "
                + ", ".join(f"{d.get('frame')}={d.get('why')}" for d in inspect_note)
            )
        lines.extend(["", "frame  role         lines  α     RMS      agree  brake  seed"])
        cover_rows = inspect if not list_cover_frames else rows
        for r in cover_rows[:18]:
            seed = r.get("seed") or {}
            win = seed.get("winner") or r.get("seed_winner") or "none"
            lines.append(
                f"{int(r['frame']):5d}  {str(r['role']):<12} {int(r['n_lines']):3d}  "
                f"{float(r['max_alpha']):.2f}  {float(r['removed_rms']):8.3g}  "
                f"{float(r['agree']):.2f}   {int(bool(r['brake']))}    {win}"
            )
        fig.text(0.06, 0.90, "\n".join(lines), fontsize=8, va="top", family="monospace")
        pdf.savefig(fig, dpi=140)

        draw_shutter_page(
            fig,
            title=f"Shutter detect  ·  {format_shutter_span(shutter)}",
            subtitle=shutter_subtitle
            or (
                "FOV std cliff. Quiet window is this experiment's shutter. "
                "ChanA and ChanB fringes are not coupled; inspect frames are per channel."
            ),
            det=shutter,
        )
        pdf.savefig(fig, dpi=140)

        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.97, traces_title, fontsize=13, fontweight="bold", va="top")
        trace = rows if rows else inspect
        xs = np.array([int(r["frame"]) for r in trace], dtype=int)
        gs = fig.add_gridspec(3, 1, left=0.08, right=0.98, top=0.90, bottom=0.08, hspace=0.22)
        ax = fig.add_subplot(gs[0])
        marker = "o-" if len(trace) <= 20 else "-"
        ax.plot(xs, [float(r["removed_rms"]) for r in trace], marker, color="0.15", lw=0.8)
        ax.set_ylabel("removed RMS")
        ax.tick_params(labelbottom=False)
        axg = fig.add_subplot(gs[1], sharex=ax)
        axg.plot(xs, [int(r["n_lines"]) for r in trace], marker, color="C0", lw=0.8, label="n lines")
        axg.plot(xs, [float(r["max_alpha"]) for r in trace], marker, color="C1", lw=0.8, label="max α")
        axg.legend(fontsize=7, loc="upper right", frameon=False)
        axg.tick_params(labelbottom=False)
        axa = fig.add_subplot(gs[2], sharex=ax)
        axa.plot(xs, [float(r["agree"]) for r in trace], marker, color="C2", lw=0.8)
        axa.set_ylabel("pred↔removed")
        axa.set_ylim(-0.05, 1.05)
        axa.set_xlabel("frame index")
        pdf.savefig(fig, dpi=140)

        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.97, means_title, fontsize=13, fontweight="bold", va="top")
        gs = fig.add_gridspec(1, 4, left=0.04, right=0.99, top=0.88, bottom=0.10, wspace=0.10)
        _imshow(fig.add_subplot(gs[0]), mean_raw, title="mean raw")
        _imshow(fig.add_subplot(gs[1]), mean_cleaned, title="mean cleaned")
        _imshow(fig.add_subplot(gs[2]), mean_removed, signed=True, title="mean removed")
        _imshow(fig.add_subplot(gs[3]), mean_predicted, signed=True, title="mean predicted")
        pdf.savefig(fig, dpi=140)

        for r in inspect:
            fi = int(r["frame"])
            raw = r["raw"]
            h, w = raw.shape[:2]
            seed = r.get("seed_full") or {}
            ranked = r.get("ranked") or []
            acc = r.get("accepted") or []
            fig.clear()
            fig.patch.set_facecolor("white")
            fig.text(
                0.06,
                0.975,
                f"Rung story  ·  frame {fi}  ·  {r['role']}",
                fontsize=12,
                fontweight="bold",
                va="top",
            )
            acc_s = (
                ", ".join(
                    f"{a['tier']}:{a['source']}/{a.get('kind', 'ridge')}/{a['axis']} q={a['q']:.1f}"
                    for a in acc
                )
                or "(none)"
            )
            fig.text(0.06, 0.945, f"accepted: {acc_s}", fontsize=7.5, va="top", family="monospace", color="0.3")
            fig.text(
                0.06,
                0.922,
                f"linescan {seed.get('winner') or 'none'}  qy={seed.get('qy')} qx={seed.get('qx')}  "
                f"brake={r['brake']}  α={r['max_alpha']:.2f}  agree={r['agree']:.2f}  rms={r['removed_rms']:.3g}",
                fontsize=8,
                va="top",
                color="0.35",
            )
            rank_txt = "ranked: " + " | ".join(
                f"{ln['tier']} {ln['source']}/{ln.get('kind', 'ridge')}/{ln['axis']} q={ln['q']:.1f}"
                for ln in ranked
            )
            fig.text(0.06, 0.898, rank_txt[:280], fontsize=7, va="top", color="0.4")
            gs = fig.add_gridspec(2, 4, left=0.03, right=0.99, top=0.87, bottom=0.05, hspace=0.18, wspace=0.08)
            _imshow(fig.add_subplot(gs[0, 0]), seed_peak_mask(h, w, seed) if seed else None, title="linescan winner mask")
            _imshow(fig.add_subplot(gs[0, 1]), _lines_mask(h, w, acc), title="accepted lines (FFT peaks)")
            _imshow(fig.add_subplot(gs[0, 2]), r.get("applied"), title="final applied mask")
            _imshow(fig.add_subplot(gs[0, 3]), r.get("predicted"), signed=True, title="final predicted")
            _imshow(fig.add_subplot(gs[1, 0]), raw, title="original")
            _imshow(fig.add_subplot(gs[1, 1]), r.get("removed"), signed=True, title="final removed")
            _imshow(fig.add_subplot(gs[1, 2]), r.get("cleaned"), title="cleaned")
            leftover = np.asarray(r.get("cleaned"), dtype=float)  # leftover ≈ cleaned
            _imshow(fig.add_subplot(gs[1, 3]), leftover, title="leftover (cleaned)")
            pdf.savefig(fig, dpi=140)

            for step in (r.get("steps") or [])[:8]:
                fig.clear()
                fig.patch.set_facecolor("white")
                tag = "KEPT"
                fig.text(
                    0.06,
                    0.97,
                    f"{tag}  ·  frame {fi}  ·  {step['why']}",
                    fontsize=12,
                    fontweight="bold",
                    va="top",
                )
                fig.text(
                    0.06,
                    0.935,
                    f"α={step['alpha']:.2f}  n_lines={step['n_lines']}  rms={step['removed_rms']:.3g}  "
                    f"agree={step['agree']:.2f}  {step['reason']}",
                    fontsize=8,
                    va="top",
                    color="0.35",
                )
                gs = fig.add_gridspec(1, 4, left=0.03, right=0.99, top=0.88, bottom=0.08, wspace=0.08)
                _imshow(fig.add_subplot(gs[0]), step.get("applied"), title="applied mask this step")
                _imshow(fig.add_subplot(gs[1]), step.get("predicted"), signed=True, title="predicted")
                _imshow(fig.add_subplot(gs[2]), step.get("removed"), signed=True, title="removed")
                _imshow(fig.add_subplot(gs[3]), step.get("cleaned"), title="cleaned / leftover")
                pdf.savefig(fig, dpi=140)

            for step in (r.get("undone") or [])[:4]:
                fig.clear()
                fig.patch.set_facecolor("white")
                fig.text(
                    0.06,
                    0.97,
                    f"UNDONE  ·  frame {fi}  ·  {step['why']}",
                    fontsize=12,
                    fontweight="bold",
                    va="top",
                    color="C3",
                )
                fig.text(
                    0.06,
                    0.935,
                    f"α={step['alpha']:.2f}  rms={step['removed_rms']:.3g}  agree={step['agree']:.2f}  "
                    f"{step['reason']}  — last increment rolled back",
                    fontsize=8,
                    va="top",
                    color="0.35",
                )
                gs = fig.add_gridspec(1, 4, left=0.03, right=0.99, top=0.88, bottom=0.08, wspace=0.08)
                _imshow(fig.add_subplot(gs[0]), step.get("applied"), title="trial applied")
                _imshow(fig.add_subplot(gs[1]), step.get("predicted"), signed=True, title="trial predicted")
                _imshow(fig.add_subplot(gs[2]), step.get("removed"), signed=True, title="trial removed")
                _imshow(fig.add_subplot(gs[3]), step.get("cleaned"), title="trial cleaned (not kept)")
                pdf.savefig(fig, dpi=140)

            fig.clear()
            fig.patch.set_facecolor("white")
            fig.text(0.06, 0.97, f"Linescan vs FFT  ·  frame {fi}", fontsize=12, fontweight="bold", va="top")
            fig.text(
                0.06,
                0.935,
                "Orange = rloess. Congruence none is honest on shutter 2-D ridges.",
                fontsize=8,
                va="top",
                color="0.35",
            )
            gs = fig.add_gridspec(2, 2, left=0.06, right=0.98, top=0.90, bottom=0.07, hspace=0.28, wspace=0.18)
            _plot_linescan(fig.add_subplot(gs[0, 0]), raw, seed, "horizontal", "H → qx")
            _plot_linescan(fig.add_subplot(gs[0, 1]), raw, seed, "vertical", "V → qy")
            _plot_linescan(fig.add_subplot(gs[1, 0]), raw, seed, "main", "TL–BR")
            _plot_linescan(fig.add_subplot(gs[1, 1]), raw, seed, "anti", "TR–BL")
            pdf.savefig(fig, dpi=140)

    plt.close(fig)
    return path
