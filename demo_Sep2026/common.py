"""Shared loading and saving for the demo figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile

from .paths import CHANA_TIF, DEMO_DIR, FIG_DPI, FIG_SIZE

_GPT = Path(__file__).resolve().parents[1] / "reference" / "gpt"
if str(_GPT) not in sys.path:
    sys.path.insert(0, str(_GPT))

from pmt_fringe_raw_adaptive import fft_log_amp, search_q  # noqa: E402

from batch_defringe.readout import _jsonable, _percentile_limits  # noqa: E402

__all__ = [
    "boxed_text",
    "fft_log_amp",
    "iter_frames",
    "load_frame",
    "new_figure",
    "save_figure",
    "search_q",
    "_percentile_limits",
]


def load_frame(frame: int, tif: Path = CHANA_TIF) -> tuple[np.ndarray, int]:
    """Return one raw frame (0-based index) and the stack length."""
    with tifffile.TiffFile(tif) as tf:
        n = len(tf.pages)
        if not 0 <= frame < n:
            raise IndexError(f"frame {frame} outside 0..{n - 1}")
        return np.asarray(tf.pages[int(frame)].asarray()), n


def iter_frames(tif: Path = CHANA_TIF):
    """Yield ``(index, frame, n_frames)`` without loading the whole stack."""
    with tifffile.TiffFile(tif) as tf:
        n = len(tf.pages)
        for i, page in enumerate(tf.pages):
            yield i, np.asarray(page.asarray()), n


def boxed_text(fig, box, text, *, fontsize=8.5, color="0.3", weight="normal", style="normal"):
    """Place wrapped text in figure-fraction box ``(left, bottom, width, height)``.

    Wrap width is taken from the box in inches so lines stay inside the figure.
    ``$...$`` spans are kept whole.
    """
    ax = fig.add_axes(box)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    inches = float(box[2]) * float(fig.get_figwidth())
    width = max(24, int(inches * 72.0 / (0.58 * float(fontsize))) - 2)

    def _tokens(paragraph: str) -> list[str]:
        out: list[str] = []
        buf: list[str] = []
        in_math = False
        for ch in paragraph:
            if ch == "$":
                buf.append(ch)
                if in_math:
                    out.append("".join(buf))
                    buf = []
                    in_math = False
                else:
                    in_math = True
                continue
            if in_math:
                buf.append(ch)
                continue
            if ch.isspace():
                if buf:
                    out.append("".join(buf))
                    buf = []
            else:
                buf.append(ch)
        if buf:
            out.append("".join(buf))
        return out

    lines: list[str] = []
    for para in text.replace("\r\n", "\n").split("\n"):
        if not para.strip():
            lines.append("")
            continue
        current: list[str] = []
        length = 0
        for tok in _tokens(para):
            add = len(tok) if not current else len(tok) + 1
            if current and length + add > width:
                lines.append(" ".join(current))
                current = [tok]
                length = len(tok)
            else:
                current.append(tok)
                length += add
        if current:
            lines.append(" ".join(current))
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        fontstyle=style,
        transform=ax.transAxes,
        linespacing=1.35,
        clip_on=True,
    )
    return ax


def new_figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=FIG_SIZE)
    fig.patch.set_facecolor("white")
    return fig


def save_figure(fig, name: str, numbers: dict, out_dir: Path = DEMO_DIR) -> list[Path]:
    """Write ``name``.png, ``name``.pdf and ``name``.json into the demo folder."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf", out_dir / f"{name}.json"]
    fig.savefig(paths[0], dpi=FIG_DPI)
    fig.savefig(paths[1], dpi=FIG_DPI)
    paths[2].write_text(json.dumps(_jsonable(numbers), indent=2), encoding="utf-8")
    plt.close(fig)
    return paths
