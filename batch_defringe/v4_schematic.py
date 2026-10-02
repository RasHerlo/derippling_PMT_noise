"""Write the v4 pipeline schematic (matches coded ``process_v4``).

``python -m batch_defringe.v4_schematic``
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_PATH = _REPO / "v4_pipeline_schematic.pdf"


def write_v4_schematic_pdf(path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    path = Path(path) if path is not None else DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    def _box(ax, x, y, w, h, txt, *, fc="0.96", fs=7.2):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008",
                facecolor=fc,
                edgecolor="0.28",
                linewidth=0.9,
            )
        )
        ax.text(x + 0.012, y + h - 0.012, txt, fontsize=fs, va="top", ha="left", family="sans-serif")

    def _arrow(ax, x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=11,
                lw=0.9,
                color="0.35",
            )
        )

    def _page_text(fig, title: str, body: str, *, fs: float = 9.5) -> None:
        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.96, title, fontsize=14, fontweight="bold", va="top")
        fig.text(0.06, 0.90, body, fontsize=fs, va="top", family="sans-serif")

    fig = plt.figure(figsize=(11.69, 8.27))
    with PdfPages(path) as pdf:
        # --- page 1: flow (as coded) ---
        fig.clear()
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.06, 0.97, "v4 defringe  ·  one mask per frame", fontsize=16, fontweight="bold", va="top")
        ax.text(
            0.06,
            0.935,
            "Coded in batch_defringe/process_v4.py. Does not overwrite defringe_v22 or v3.  notes/V4_PIPELINE.md",
            fontsize=8,
            va="top",
            color="0.35",
        )
        _box(
            ax,
            0.06,
            0.82,
            0.88,
            0.09,
            "1. Stack hints (not locked q)  ·  Experiment.xml catalog  ·  shutter FOV-std cliff  ·  shutter-learn qs\n"
            "   Each frame re-seeds. Last-frame q is not tracked. Empty mask allowed.",
        )
        _arrow(ax, 0.50, 0.82, 0.50, 0.795)
        _box(
            ax,
            0.06,
            0.64,
            0.42,
            0.15,
            "2a. Collect lines (this order)\n"
            "• catalog q  ·  shutter-learn q\n"
            "• linescan center peak (if congruent)\n"
            "• same-trace edge / segment qs (chirp)\n"
            "• leftover FFT is later, after a mask exists",
            fc="#fff8f0",
        )
        _box(
            ax,
            0.52,
            0.64,
            0.42,
            0.15,
            "2b. Linescan congruence\n"
            "• H, V, both diagonals always\n"
            "• One (qy, qx): fx / fy / tilted — or none\n"
            "• none → skip linescan pieces; still catalog,\n"
            "  shutter-learn, leftover FFT",
            fc="#f0fff4",
        )
        _arrow(ax, 0.27, 0.64, 0.40, 0.615)
        _arrow(ax, 0.73, 0.64, 0.60, 0.615)
        _box(
            ax,
            0.06,
            0.46,
            0.88,
            0.15,
            "3. Add lines into ONE mask at low α (0.28 ≈ ⅓ of pack_D full 0.85)\n"
            "Each trial: FFT notch → predicted (IFFT of taken-out energy) and removed (raw − cleaned).\n"
            "Keep only if removed still looks like fringe and tracks predicted. Live: undo if cells. Shutter: push.\n"
            "Then leftover FFT on cleaned; try those lines at 0.28 too (dubious extras, not a second family).",
            fc="#f4f8ff",
        )
        _arrow(ax, 0.50, 0.46, 0.50, 0.435)
        _box(
            ax,
            0.06,
            0.26,
            0.88,
            0.17,
            "4. Raise α collectively on the whole accepted set: 0.55 → 0.85 (shutter also 1.00). Stop at first reject.\n"
            "One shared max_alpha — not per-line. Peak pieces: att = α × peak_mask. Ridge pieces: pack_D local excess\n"
            "with that same α, and a per-line spectral gate (gate=0 skips that ridge). Combined by max.\n"
            "Live search ±10 bins around this frame’s own hint; shutter ±2. No stack q-tracker.",
            fc="#fff4f4",
        )
        _arrow(ax, 0.50, 0.26, 0.50, 0.235)
        _box(
            ax,
            0.06,
            0.07,
            0.88,
            0.16,
            "5. Write  ·  dest --out: cleaned TIFF, slim overview.pdf, per_frame.csv, families.json,\n"
            "   mask_recipe.json (rung story), mask_patterns.npz (sparse final + per-rung FFT masks).\n"
            "   E:\\Rasmus-Guillermo is write-guarded. Removed/mean TIFFs not stored on dest; rebuild from npz + raw.",
            fc="#f4fff8",
        )
        pdf.savefig(fig, dpi=140)

        _page_text(
            fig,
            "Congruence  ·  one (qy, qx) or none",
            "All four cuts always run: several horizontal rows, several vertical columns, both diagonals.\n"
            "A fringe is one plane-wave, so the cuts must describe the same 2-D frequency — allowing for angle.\n\n"
            "Hypotheses (scored against measured diagonal periods):\n"
            "  • fx  — vertical stripes (intensity varies along x). qx from the HORIZONTAL cut; qy = 0.\n"
            "           The vertical cut can be quiet; that is expected, not a failure.\n"
            "  • fy  — horizontal bands. qy from the VERTICAL cut; qx = 0.\n"
            "  • tilted — both qx and qy. Diagonals must match the predicted period at (qy, qx).\n\n"
            "Score = mean relative error of predicted vs measured main/anti diagonal periods.\n"
            "Winner = lowest score. If no hypothesis, or best score > 0.25 → congruence is none.\n"
            "none is honest on shutter 2-D ridges (not a 1-D stripe). Then skip linescan center/edges.\n\n"
            "This ‘agree’ is not the predicted↔removed agree later. It is only ‘do the four cuts\n"
            "describe one orientation?’",
            fs=10,
        )
        pdf.savefig(fig, dpi=140)

        _page_text(
            fig,
            "α  ·  what 0.28 is  ·  one mask, not per-family α",
            "α is the attenuation cap on the FFT support (how hard we notch), not a spatial pixel mask.\n\n"
            "Rungs (chosen, not measured from the data):\n"
            "  live     0.28 → 0.55 → 0.85     (0.28 ≈ ⅓ of pack_D full 0.85)\n"
            "  shutter  0.28 → 0.55 → 0.85 → 1.00\n\n"
            "Adding a line: fold it into the same mask, apply the current set at 0.28. If the trial fails,\n"
            "that line is not kept. After leftover lines are considered, α is raised on the WHOLE accepted\n"
            "set together. Families do not get separate α schedules.\n\n"
            "v2.2 / pack_D DID use per-family gate and eff_max_alpha (per_frame.csv family{i}_*).\n"
            "v4 dropped independent families on purpose: leftover / chirp / harmonics are extra support of\n"
            "the same fringe, not a second cleaner. Ridge lines still have pack_D gate from this frame’s\n"
            "spectral strength (gate=0 skips that line). Peak (linescan) pieces have no gate — only shared α.\n"
            "Open: per-line α later, so a dubious leftover is not pushed as hard as the core. Not implemented.",
            fs=10,
        )
        pdf.savefig(fig, dpi=140)

        _page_text(
            fig,
            "Predicted vs removed  ·  RMSE / L2 / agree",
            "predicted  = IFFT of the FFT energy the current mask attenuates (what we intended to take out).\n"
            "removed    = raw − cleaned  (what actually left the image).\n"
            "They should look like the same striped fringe. Cells in removed that predicted does not have\n"
            "means those bins held biology — undo the last step only.\n\n"
            "All scalars are over every pixel of that one frame (not tiles, not a mean of row-RMSs):\n"
            "  RMS(pred)     = sqrt( mean( pred² ) )\n"
            "  RMS(removed)  = sqrt( mean( removed² ) )\n"
            "  RMSE          = sqrt( mean( (removed − pred)² ) )\n"
            "That RMSE is the L2 (Euclidean / sum-of-squares) distance between the two images.\n"
            "L1 would have used |removed − pred| instead.\n\n"
            "agree = 1 − RMSE / (RMS(pred) + RMS(removed))\n"
            "  1 = identical images. Live keep if agree ≥ 0.40 and removed still looks like fringe\n"
            "  (coverage / even, and blob-or-ridges). Reject if agree < 0.40, removed RMS > 0.5,\n"
            "  and the increment looks like cells. Shutter skips the biology brake.\n"
            "Removed RMS vs 0 is a heaviness trace in the PDF/CSV, not this success test.",
            fs=10,
        )
        pdf.savefig(fig, dpi=140)

        _page_text(
            fig,
            "(fx, fy) between frames  ·  no stack tracker",
            "v4 does not lock q across time. Each frame re-collects catalog / shutter-learn / linescan.\n"
            "There is no ±N bin bound on how much the mask location may jump from frame n to n+1.\n\n"
            "The only snap is INSIDE a frame, and only for ridge pieces:\n"
            "  live     search ±10 bins around that frame’s own hint  (TRACK_SEARCH)\n"
            "  shutter  search ±2 bins\n"
            "Q_CLUSTER_TOL = 3 bins only merges duplicate lines on the SAME frame (not tracking).\n"
            "Peak pieces use this frame’s linescan (qy, qx) with no inter-frame bound.\n\n"
            "Do not carry forward from v2.2/v3: stack q-tracker (±10 / apply ±2) as the product.\n"
            "Do not union-apply frozen pack_D families. Do not treat edge-q as family 2.\n"
            "Do not IFFT whole FFT rows/columns. Do not overwrite defringe_v22.",
            fs=10,
        )
        pdf.savefig(fig, dpi=140)

        _page_text(
            fig,
            "What ‘better’ looks like  ·  dest readout",
            "At each rung, predicted should get more complete and still stripe-like / conjugate-symmetric.\n"
            "Removed should track it. Leftover should lose fringe, not cells. Undo the increment if not.\n\n"
            "overview.pdf (slim dest run): cover, shutter cliff, traces (RMS / n lines / α / agree / brake),\n"
            "means of raw / cleaned / removed / predicted. Full rung stories live in mask_recipe.json +\n"
            "mask_patterns.npz (sparse applied FFT after every kept and undone step).\n\n"
            "per_frame.csv: frame, role, n_lines, max_alpha, removed_rms, removed_mean_abs, removed_p99_abs,\n"
            "agree, brake, empty, core_only, seed_winner, qx, qy, accepted, n_history, n_undone.",
            fs=10,
        )
        pdf.savefig(fig, dpi=140)

        _page_text(
            fig,
            "Dest batch  ·  USB keepalive  ·  progress",
            "python -m batch_defringe.process_v4 --root E:\\Rasmus-Guillermo\\ECF1 --out F:\\CollectedData\n\n"
            "E:\\Rasmus-Guillermo is write-guarded. Dest gets folder structure, Experiment.xml, cleaned TIFF,\n"
            "slim PDF, per_frame.csv, families.json, mask_recipe.json, mask_patterns.npz. Resume skips complete dest folders.\n\n"
            "USB: keepalive every 20s (read source, write dest .defringe_v4_runs/keepalive.txt). Retry a stack 3× on\n"
            "Permission denied / 0 written / vanished drive (wait up to 3 min). Stop after 3 consecutive drive failures.\n"
            "The script does not change Windows power options. USB selective suspend Off is still the stronger OS fix.\n\n"
            "Progress: console ETA, plus .defringe_progress.txt in the repo and\n"
            "F:\\CollectedData\\.defringe_v4_runs\\<utc>\\progress.txt",
            fs=10,
        )
        pdf.savefig(fig, dpi=140)

    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args(argv)
    out = write_v4_schematic_pdf(args.out)
    print(f"schematic: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
