"""Assemble cursor-tests figures + summaries into one PDF report."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

OUT_ROOT = Path(
    r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\SUPPORT_ChanB\to build FFT deripple\cursor tests"
)
PDF_PATH = OUT_ROOT / "PMT_fringe_deripple_test_report.pdf"
STACKS = ("ChanA_raw_500fr", "ChanB_raw_500fr")


def add_text_page(pdf: PdfPages, title: str, body: str, fontsize=10):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
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
    fig.text(0.06, 0.96, title, fontsize=13, fontweight="bold", va="top")
    if caption:
        fig.text(0.06, 0.92, caption, fontsize=9, va="top")
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.80])
    ax.imshow(img)
    ax.axis("off")
    pdf.savefig(fig, dpi=160)
    plt.close(fig)


def injection_table(stem: str) -> str:
    csv_path = OUT_ROOT / stem / "stress_test_v2" / "injection_results.csv"
    if not csv_path.exists():
        return "No injection CSV."
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    lines = [
        f"{'noise':16} {'alpha':>5} {'E_rec':>8} {'injRMS':>8} {'E/inj':>7} {'remain':>7}",
        "-" * 58,
    ]
    for noise in ("v2_residual", "broad_spectral"):
        for a in (0.25, 0.5, 1.0, 1.5):
            sub = [r for r in rows if r["noise"] == noise and float(r["alpha"]) == a]
            if not sub:
                continue
            er = float(np.median([float(r["E_recovery"]) for r in sub]))
            inj = float(np.median([float(r["injected_rms"]) for r in sub]))
            rem = float(np.median([float(r["remaining_fringe_band_frac"]) for r in sub]))
            lines.append(
                f"{noise:16} {a:5.2f} {er:8.3f} {inj:8.3f} {er/max(inj,1e-9):7.3f} {rem:7.3f}"
            )
    return "\n".join(lines)


def main():
    with PdfPages(PDF_PATH) as pdf:
        add_text_page(
            pdf,
            "PMT fringe deripple — test report",
            textwrap.dedent(
                """
                Dataset
                - ChanA_raw_500fr.tif and ChanB_raw_500fr.tif (500 x 512 x 512, uint16)
                - Goal: raw-only adaptive fringe removal before SUPPORT, minimizing biological artifacts

                Methods compared
                1) gpt_adaptive_raw          earlier narrow-peak adaptive filter
                2) rowband_raw               full FFT-row harmonic notches (detect on raw)
                3) rowband_support2raw       row notches detected on SUPPORT, applied to raw (ChanB)
                4) gpt_raw_adaptive_v2       ridge-segment adaptive filter with gating (leading)

                Stress tests on v2
                - Temporal continuity of removed RMS / gate / tracked q
                - Pseudo-ground-truth injection: I_test = I0 + alpha * N
                  N from (a) v2 residual or (b) broader spectral extract
                - Metrics: E_recovery = RMS(I_rec-I0); remaining fringe-band power fraction

                Hierarchy for raw -> defringe -> SUPPORT
                gpt_raw_adaptive_v2  >  old point-notch  >  whole-row band
                """
            ).strip(),
            fontsize=10,
        )

        # Bake-off summary page
        bake = (OUT_ROOT / "SUMMARY.md").read_text(encoding="utf-8", errors="replace")
        add_text_page(pdf, "Bake-off summary (SUMMARY.md)", bake, fontsize=8)

        v2sum = (OUT_ROOT / "SUMMARY_gpt_raw_adaptive_v2.md").read_text(
            encoding="utf-8", errors="replace"
        )
        add_text_page(pdf, "GPT raw-adaptive v2 vs prior methods", v2sum, fontsize=8)

        stress = (OUT_ROOT / "SUMMARY_stress_test_v2.md").read_text(
            encoding="utf-8", errors="replace"
        )
        add_text_page(pdf, "Stress-test summary", stress, fontsize=8)

        for stem in STACKS:
            # Detection diagnostics
            add_image_page(
                pdf,
                f"{stem} — v2 detected spectrum",
                OUT_ROOT / stem / "gpt_raw_adaptive_v2" / "diagnostics" / "detected_spectrum.png",
                "Median raw Fourier spectrum with detected fringe rows.",
            )
            add_image_page(
                pdf,
                f"{stem} — v2 row anomaly score",
                OUT_ROOT / stem / "gpt_raw_adaptive_v2" / "diagnostics" / "row_anomaly_score.png",
                "Recurrent row/ridge anomaly z-score used for family detection.",
            )
            add_image_page(
                pdf,
                f"{stem} — v2 strong vs weak frames",
                OUT_ROOT / stem / "gpt_raw_adaptive_v2" / f"{stem}_gpt_raw_adaptive_v2_strong_weak.png",
                "Rows: strong-fringe frame, weak-fringe frame. Columns: raw | cleaned | removed.",
            )
            add_image_page(
                pdf,
                f"{stem} — method compare (GPT v1)",
                OUT_ROOT / stem / "gpt_adaptive_raw" / f"{stem}_gpt_adaptive_compare.png",
                "Earlier GPT adaptive. Columns: raw | cleaned | removed.",
            )
            add_image_page(
                pdf,
                f"{stem} — method compare (rowband on raw)",
                OUT_ROOT / stem / "rowband_raw" / f"{stem}_rowband_raw_compare.png",
                "Full-row harmonic notches detected on raw.",
            )
            support_cmp = (
                OUT_ROOT / stem / "rowband_support2raw" / f"{stem}_rowband_support2raw_compare.png"
            )
            if support_cmp.exists():
                add_image_page(
                    pdf,
                    f"{stem} — method compare (SUPPORT detect -> raw)",
                    support_cmp,
                    "Row notches detected on SUPPORT Frame1t10, applied to raw.",
                )

            add_image_page(
                pdf,
                f"{stem} — v2 temporal continuity",
                OUT_ROOT / stem / "stress_test_v2" / "continuity.png",
                "removed RMS, max gate, tracked q across 500 frames.",
            )
            add_image_page(
                pdf,
                f"{stem} — injection recovery curves",
                OUT_ROOT / stem / "stress_test_v2" / "recovery_curves.png",
                "Left: biological distortion. Right: remaining fringe-band fraction vs alpha.",
            )
            add_image_page(
                pdf,
                f"{stem} — injection examples (alpha=1)",
                OUT_ROOT / stem / "stress_test_v2" / "injection_examples.png",
                "Columns: I0 | I_test | I_rec | I_rec-I0 | N",
            )
            add_text_page(
                pdf,
                f"{stem} — injection metrics (median over pairs)",
                injection_table(stem) + "\n\nE/inj << 1 with low remain = good operating point.",
                fontsize=9,
            )

        add_text_page(
            pdf,
            "Conclusions",
            textwrap.dedent(
                """
                1. Automatic raw-only detection finds distinct PMT signatures
                   (ChanA q~14/242; ChanB q~60/196) with similar fx ridge support (~+/-10-40).

                2. gpt_raw_adaptive_v2 is the leading method because weak-fringe frames are left
                   nearly untouched (removed RMS ~ 0), while strong-frame residuals look like
                   pure electronic fringe.

                3. Whole-row banding remains too aggressive on weak frames and is not recommended
                   as the default for bulk biological preprocessing.

                4. Stress tests support current v2 defaults for natural-strength residual injection
                   (especially ChanB: E/inj ~0.26 with ~7% remaining fringe-band power at alpha=1).
                   Broad-spectral injection is harder and suggests optional fine-tuning, not redesign.

                5. Continuity is mostly smooth; ChanB has more gate transitions around weak periods.
                   Consider light temporal smoothing of gate/q before locking production defaults.

                6. Suggested next practical check: identical SUPPORT settings on untouched raw vs
                   v2-cleaned raw, to confirm upstream defringe reduces SUPPORT block artifacts.
                """
            ).strip(),
            fontsize=10,
        )

    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
