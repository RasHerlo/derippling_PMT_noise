"""Assemble SUPPORT block-comparison figures into a PDF report."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

OUT = Path(
    r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\SUPPORT_ChanB\to build FFT deripple\cursor tests\support_block_compare"
)
PDF_PATH = OUT / "SUPPORT_block_compare_report.pdf"
CHANNELS = ("ChanA", "ChanB")


def add_text_page(pdf: PdfPages, title: str, body: str, fontsize: float = 10):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.93, title, fontsize=16, fontweight="bold", va="top")
    wrapped = []
    for para in body.split("\n"):
        if para.strip() == "":
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(para, width=110) or [""])
    fig.text(
        0.06,
        0.86,
        "\n".join(wrapped),
        fontsize=fontsize,
        va="top",
        family="monospace",
        linespacing=1.35,
    )
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path, caption: str = ""):
    if not image_path.exists():
        add_text_page(pdf, title, f"[Missing figure]\n{image_path}")
        return
    img = np.asarray(Image.open(image_path).convert("RGB"))
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.96, title, fontsize=12, fontweight="bold", va="top")
    if caption:
        fig.text(0.06, 0.92, caption, fontsize=9, va="top")
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.82])
    ax.imshow(img)
    ax.axis("off")
    pdf.savefig(fig, dpi=160)
    plt.close(fig)


def main():
    summary = (OUT / "SUMMARY_support_block_compare.md").read_text(
        encoding="utf-8", errors="replace"
    )

    with PdfPages(PDF_PATH) as pdf:
        add_text_page(
            pdf,
            "SUPPORT block-artifact comparison",
            textwrap.dedent(
                """
                Question
                Does v2 defringe before SUPPORT reduce the rectangular / blotchy
                artifacts that SUPPORT invents on fringed raw data?

                Pairs compared
                - Original:  SUPPORT_ChanX/denoised_cut.tif
                             (SUPPORT / model_10 on untouched raw)
                - New:       ChanX_defringe/ChanX_stk_defringed_denoised.tif
                             (v2 defringe -> same SUPPORT / model_10)

                Display layout in figures
                columns = SUPPORT(raw) | SUPPORT(defringed) | difference

                Caveat
                model_10 was trained on the non-defringed distribution. Softness or
                odd texture on defringed inputs may reflect domain shift and argue
                for retraining on defringed stacks, not for abandoning defringe.
                """
            ).strip(),
        )

        add_text_page(pdf, "Summary metrics", summary, fontsize=9)

        add_text_page(
            pdf,
            "Interpretation",
            textwrap.dedent(
                """
                ChanB
                - Clearest benefit. On strong-fringe frames, block-mean CV drops
                  substantially (~0.024-0.028 -> ~0.015-0.017; mean ~18% reduction).
                - Visually, SUPPORT(raw) still shows fringe + patchy tiles;
                  SUPPORT(defringed) is cleaner with sharper puncta.
                - Weak-fringe frames look nearly identical left/right (v2 barely
                  changed those raw frames).

                ChanA
                - Milder / mixed (~3-4% average metric improvement).
                - Some stronger frames improve seams/CV, but square tiling can
                  remain in both.
                - Defringed+denoised frames sometimes look softer/hazier — consistent
                  with training on non-defringed data.

                Bottom line
                Defringe -> SUPPORT helps most where fringe was strong (esp. ChanB).
                It does not fully erase SUPPORT tiling everywhere. Remaining issues
                motivate retraining model_10 on defringed stacks.
                """
            ).strip(),
        )

        for ch in CHANNELS:
            ch_dir = OUT / ch
            add_image_page(
                pdf,
                f"{ch} — montage (selected frames)",
                ch_dir / "montage_compare.png",
                "Columns: SUPPORT(raw) | SUPPORT(defringed) | difference",
            )

            # per-frame compares and highpass
            for p in sorted(ch_dir.glob("frame_*_rawSUPPORT_vs_defringeSUPPORT.png")):
                add_image_page(
                    pdf,
                    f"{ch} — {p.stem}",
                    p,
                    "Columns: SUPPORT(raw) | SUPPORT(defringed) | difference",
                )
            for p in sorted(ch_dir.glob("frame_*_highpass.png")):
                add_image_page(
                    pdf,
                    f"{ch} — high-pass {p.stem}",
                    p,
                    "Left: SUPPORT(raw). Right: SUPPORT(defringed). High-pass emphasizes tiles/fringe.",
                )

        add_text_page(
            pdf,
            "Next steps",
            textwrap.dedent(
                """
                1. Keep v2 as the raw preprocess (especially valuable for ChanB).
                2. Retrain / fine-tune SUPPORT model_10 on defringed stacks if
                   residual softness or tiling on defringed inputs matters.
                3. Optionally re-run this compare after retraining with identical
                   frame indices for a clean before/after on the denoiser itself.
                """
            ).strip(),
        )

    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
