"""Write the agreed v4 pipeline schematic (design; cleaner not coded yet).

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

    def _box(ax, x, y, w, h, txt, *, fc="0.96", fs=7.4):
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

    fig = plt.figure(figsize=(11.69, 8.27))
    with PdfPages(path) as pdf:
        # --- page 1: flow ---
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
            "Agreed design — not coded yet. Does not overwrite defringe_v22 or v3.  notes/V4_PIPELINE.md",
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
            "1. Raw frame  ·  hints only (shutter span, catalog/PMT, last-frame mask). Never a locked stack q.",
        )
        _arrow(ax, 0.50, 0.82, 0.50, 0.795)
        _box(
            ax,
            0.06,
            0.64,
            0.42,
            0.15,
            "2a. FFT lines\n"
            "• fy ridges + fx columns (fy=0 OK)\n"
            "• harmonics as extras\n"
            "• leftover peaks after a notch\n"
            "• chirp = nearby edge bins, not family 2",
            fc="#fff8f0",
        )
        _box(
            ax,
            0.52,
            0.64,
            0.42,
            0.15,
            "2b. Linescans\n"
            "• H, V, both diagonals always\n"
            "• One (qy, qx) or none\n"
            "• none is honest (shutter 2-D ridge)\n"
            "• P(x) / L–C–R = chirp extras",
            fc="#f0fff4",
        )
        _arrow(ax, 0.27, 0.64, 0.40, 0.615)
        _arrow(ax, 0.73, 0.64, 0.60, 0.615)
        _box(
            ax,
            0.06,
            0.48,
            0.88,
            0.13,
            "3. Rank into one list  ·  core (both views agree) → agreed extras (harmonics, chirp edges) → dubious (one view / leftover)\n"
            "If congruence is none: core = loudest symmetric FFT ridge. Do not blend fy and fx into one gate.",
            fc="#f4f8ff",
        )
        _arrow(ax, 0.50, 0.48, 0.50, 0.455)
        _box(
            ax,
            0.06,
            0.26,
            0.88,
            0.19,
            "4. Soft recursion on ONE mask  (support + strength)\n"
            "Start: core only, low α. Each step: add the next line and/or raise α a little. Notch.\n"
            "Look at predicted (IFFT of attenuated bins), removed (raw−cleaned), leftover, and the increment.\n"
            "Live: if removed looks like cells OR removed diverges from predicted → undo this step only.\n"
            "Shutter: no biology brake; push until leftover is not fringe. Predicted must still look like fringe.",
            fc="#fff4f4",
        )
        _arrow(ax, 0.50, 0.26, 0.50, 0.235)
        _box(
            ax,
            0.06,
            0.08,
            0.88,
            0.15,
            "5. Write frame  ·  cleaned + removed + predicted + applied heatmap + rung log\n"
            "This mask may hint the next frame. It does not walk q for the rest of the stack.\n"
            "Outputs: <channel>/defringe_v4/  stacks, per_frame.csv, overview.pdf",
            fc="#f4fff8",
        )
        pdf.savefig(fig, dpi=140)

        # --- page 2: checks ---
        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.96, "What ‘better’ looks like at each rung", fontsize=14, fontweight="bold", va="top")
        body = (
            "Predicted fringe = IFFT of energy the current mask attenuates.\n"
            "As lines are added it should look more complete, still striped, still conjugate-symmetric.\n"
            "Removed should track that image. Leftover should lose fringe, not cells.\n\n"
            "Internal checks (every step, not only the PDF):\n"
            "  • predicted looks like fringe (not soma, not a full-FOV bandpass)\n"
            "  • removed ≈ predicted  (biology sneaking in shows up as extra blobs in removed)\n"
            "  • increment = removed_k − removed_{k−1}  is the thing we undo if it fails\n\n"
            "Do not IFFT whole FFT rows/columns — that bandpasses the FOV and looks like ‘predicted cells’.\n"
            "Steal pack_D’s thin ridges and local excess attenuation; do not steal union-of-families apply.\n\n"
            "α rungs: few (≈3 live, ≈4 shutter). Recursion budget goes to adding lines in certainty order."
        )
        fig.text(0.06, 0.90, body, fontsize=10, va="top")
        pdf.savefig(fig, dpi=140)

        # --- page 3: overview PDF ---
        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.96, "overview.pdf  ·  how we evaluate a run", fontsize=14, fontweight="bold", va="top")
        pdf_pages = (
            "1. Cover — shutter span, empty vs core-only vs full masks, RMS, n biology-brakes undone\n"
            "2. Shutter detect — FOV std cliff (same method as v3)\n"
            "3. Stack traces — lines in mask, max α, removed RMS, predicted↔removed agreement, brake\n"
            "4. Means — raw / cleaned / removed / predicted\n"
            "5. Rung story — shutter mid, anchors 160 / 700 / 1061, strong / empty / brake / chirp frames:\n"
            "      ranked lines  ·  mask after each kept step  ·  predicted | removed | leftover | cleaned\n"
            "      plus any undone increment (pushed too hard)\n"
            "6. Undone / discarded gallery\n"
            "7. Linescan vs FFT on those frames (four traces + congruent peaks vs FFT core)\n\n"
            "Judge: predicted more complete and still a fringe; removed follows it;\n"
            "if removed grows cells that predicted does not have, that step failed."
        )
        fig.text(0.06, 0.88, pdf_pages, fontsize=10, va="top", family="sans-serif")
        pdf.savefig(fig, dpi=140)

        # --- page 4: do not ---
        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.96, "Do not carry these forward from v3", fontsize=14, fontweight="bold", va="top")
        dont = (
            "• Stack q-tracker (±10 / 50 frames, apply ±2) dominating linescan/FFT\n"
            "• Union-apply of frozen pack_D families as the product\n"
            "• Recursion = next unused (axis, q) instead of growing one mask\n"
            "• Starting at full α so the first notch can eat cells and ‘cancel’ the clean\n"
            "• Independent tile cleans  ·  IFFT of whole rows/columns\n"
            "• Treating edge-q as a second family  ·  blending fy/fx into one gate\n"
            "• Flattening shutter 756–760 as the definition of success\n"
            "• Overwriting defringe_v22 or promoting Haj Grant ChanA until 160 cleans\n"
            "  from this frame’s core and 700 is either a real family or judged empty"
        )
        fig.text(0.06, 0.88, dont, fontsize=11, va="top")
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
