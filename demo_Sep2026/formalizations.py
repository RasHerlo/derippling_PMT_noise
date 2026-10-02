"""Formalizations and definitions PDF for the Sep 2026 demo.

``python -m demo_Sep2026.formalizations``
writes ``formalizations.pdf`` into the demo folder. Add a page function
and append it to ``PAGES`` when a new definition is agreed.
"""

from __future__ import annotations

import numpy as np

from .common import _percentile_limits, fft_log_amp, load_frame
from .fig02_catalog_entry import source_stack
from .paths import DEMO_DIR
from batch_defringe.library import load_catalog

A4 = (8.27, 11.69)
OUT = DEMO_DIR / "formalizations.pdf"
MARGIN = 0.11
TEXT_W = 1.0 - 2 * MARGIN
SECTION_MASK = (
    "Generating Mask-ridges from peak-scans of seeded rows in Fourier Space"
)
SECTION_SHUTTER = "In-stack shutter-family seeds"


def _new_page():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor("white")
    return fig, plt


def _tokens(paragraph: str) -> list[str]:
    """Split on spaces; keep each $...$ math span as one token."""
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


def _split_wide_math(token: str) -> list[str]:
    """Break a long $...$ span at \\qquad so each piece can wrap."""
    if not (token.startswith("$") and token.endswith("$")):
        return [token]
    inner = token[1:-1]
    if r"\qquad" not in inner:
        return [token]
    parts = [p.strip(" ,") for p in inner.split(r"\qquad") if p.strip(" ,")]
    return [f"${p}$" for p in parts] if len(parts) > 1 else [token]


def _measure_width(ax, s: str, *, fontsize: float, weight: str, style: str) -> float:
    tmp = ax.text(0.0, 0.0, s, fontsize=fontsize, fontweight=weight, fontstyle=style, transform=ax.transAxes)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    width = tmp.get_window_extent(renderer=renderer).width
    tmp.remove()
    return width


def _wrap_to_pixels(ax, text: str, max_px: float, *, fontsize: float, weight: str, style: str) -> str:
    """Wrap prose so each line's rendered width stays inside ``max_px``."""
    paragraphs = text.replace("\r\n", "\n").split("\n")
    lines: list[str] = []
    for para in paragraphs:
        if not para.strip():
            lines.append("")
            continue
        words: list[str] = []
        for tok in _tokens(para):
            words.extend(_split_wide_math(tok))
        current: list[str] = []
        for w in words:
            trial = w if not current else " ".join(current + [w])
            if current and _measure_width(ax, trial, fontsize=fontsize, weight=weight, style=style) > max_px:
                lines.append(" ".join(current))
                current = [w]
            else:
                current.append(w)
        if current:
            lines.append(" ".join(current))
    return "\n".join(lines)


def _flow_text(fig, box, text, *, fontsize=10, color="0.1", weight="normal", style="normal", width=None):
    """Place text in figure-fraction box ``(l, b, w, h)``, wrapping to the box width."""
    del width
    ax = fig.add_axes(box)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    max_px = 0.96 * box[2] * fig.bbox.width
    wrapped = _wrap_to_pixels(ax, text, max_px, fontsize=fontsize, weight=weight, style=style)
    ax.text(
        0.0,
        1.0,
        wrapped,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        fontstyle=style,
        linespacing=1.45,
        transform=ax.transAxes,
        clip_on=True,
    )
    return ax


def page_cover(pdf) -> None:
    fig, plt = _new_page()
    fig.text(0.5, 0.56, "Formalizations and definitions", fontsize=22, fontweight="bold", ha="center", va="center")
    fig.text(0.5, 0.48, "v4 defringe  ·  September 2026 demo", fontsize=12, ha="center", va="center", color="0.35")
    pdf.savefig(fig)
    plt.close(fig)


def page_contents(pdf) -> None:
    fig, plt = _new_page()
    _flow_text(fig, [MARGIN, 0.88, TEXT_W, 0.06], "Contents", fontsize=16, weight="bold")
    contents = (
        "1.  Centred Fourier coordinates\n\n"
        "2.  Generating Mask-ridges from peak-scans of seeded rows\n"
        "    in Fourier Space (catalog path)\n\n"
        "    2a.  Ridge-support leftover and Z-score\n\n"
        "    2b.  Mask weights and fading additions\n\n"
        "3.  In-stack shutter-family seeds\n\n"
        "    3a.  Median spectrum, row profile, and row Z\n\n"
        "    3b.  Pairing, ranking, and harmonics\n\n"
        "    3c.  Leftover Z, hydrate band, peak fx, and stored weight"
    )
    _flow_text(fig, [MARGIN, 0.18, TEXT_W, 0.64], contents, fontsize=12)
    pdf.savefig(fig)
    fig.savefig(DEMO_DIR / "_preview_contents.png", dpi=120)
    plt.close(fig)


def page_coordinates(pdf) -> None:
    fig, plt = _new_page()
    _flow_text(fig, [MARGIN, 0.92, TEXT_W, 0.05], "1.  Centred Fourier coordinates", fontsize=13, weight="bold")
    body = (
        r"The 2-D DFT of a $512\times 512$ frame is stored after $\mathrm{fftshift}$. "
        r"Array row $y\in\{0,\ldots,511\}$ is not $f_y$. The DC bin is at "
        r"$c_y=c_x=256$, and"
        "\n\n"
        r"$f_y = y - 256$,   $f_x = x - 256$."
        "\n\n"
        r"$(f_y,f_x)\in\{-256,\ldots,+255\}^2$."
        "\n\n"
        r"A catalog band $q=6$ is the pair $f_y=\pm 6$. In array indices that is"
        "\n\n"
        r"$y_+ = 256+6=262$."
        "\n\n"
        r"$y_- = 256-6=250$."
        "\n\n"
        r"On a plot whose axes run from $-256$ to $+255$, row 262 is six bins "
        r"above the origin, not near the top of the image. The Nyquist partner "
        r"stored as $h_i=250$ is the offset $256-q$, i.e. $f_y=\pm 250$, which "
        r"is a different pair of rows ($y=6$ and $y=506$)."
    )
    _flow_text(fig, [MARGIN, 0.56, TEXT_W, 0.34], body, fontsize=10)

    records = load_catalog()["records"]
    entry = records[3]
    raw, _ = load_frame(667, source_stack(entry))
    logamp = fft_log_amp(raw)
    h, w = logamp.shape
    cy, cx = h // 2, w // 2
    slo, shi = _percentile_limits(logamp, (3.0, 99.7))
    extent = (-cx - 0.5, w - cx - 0.5, h - cy - 0.5, -cy - 0.5)

    ax = fig.add_axes([0.16, 0.14, 0.68, 0.39])
    ax.imshow(logamp, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest", extent=extent)
    ax.axhline(0, color="0.7", lw=0.6)
    ax.axvline(0, color="0.7", lw=0.6)
    ax.axhline(6, color=(0.85, 0.12, 0.12), lw=1.2, label=r"$f_y=+6$  (row $y=262$)")
    ax.axhline(-6, color=(0.85, 0.12, 0.12), lw=1.2, ls="--", label=r"$f_y=-6$  (row $y=250$)")
    ax.set_xlim(-256, 255)
    ax.set_ylim(255, -256)
    ax.set_xlabel(r"$f_x$")
    ax.set_ylabel(r"$f_y$")
    ax.set_title(r"Frame 667, $\log(1+|F|)$. Red: catalog band $q=6$.", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    _flow_text(
        fig,
        [MARGIN, 0.04, TEXT_W, 0.06],
        r"Source: catalog entry 3 (ChanA dark current, PC250), frame 667.  Index $y=c_y+f_y$.",
        fontsize=8,
        color="0.35",
    )
    pdf.savefig(fig)
    fig.savefig(DEMO_DIR / "_preview_coordinates.png", dpi=120)
    plt.close(fig)


def _section_header(fig, *, part: str) -> None:
    _flow_text(fig, [MARGIN, 0.89, TEXT_W, 0.08], SECTION_MASK, fontsize=11, weight="bold", style="italic")
    _flow_text(fig, [MARGIN, 0.85, TEXT_W, 0.04], part, fontsize=10, color="0.25")


def page_ridge_z(pdf) -> None:
    fig, plt = _new_page()
    _section_header(fig, part="2a.  Ridge-support leftover and Z-score")

    body = (
        r"Input: $S=\log(1+|F|)$ after $\mathrm{fftshift}$. Band row $y=c_y+q$."
        "\n\n"
        r"For each $f_x$, the background is the median over the 14 neighbouring "
        r"rows with offsets $\delta\in\{\pm 4,\ldots,\pm 10\}$:"
        "\n\n"
        r"$B(f_x)=\mathrm{median}_\delta\, S(y+\delta,f_x)$."
        "\n\n"
        r"$e(f_x)=S(y,f_x)-B(f_x)$."
        "\n\n"
        r"$e$ is the leftover (code: excess). It is not $S-\mu$ of the whole row. "
        r"Exclude the DC neighbourhood and the outer edge, "
        r"$\mathcal{V}=\{f_x:5<|f_x|<c_x-10\}$."
        "\n\n"
        r"Robust location and scale of $e$ on $\mathcal{V}$ (not the sample variance):"
        "\n\n"
        r"$m=\mathrm{median}_{f_x\in\mathcal{V}} e(f_x)$."
        "\n\n"
        r"$\mathrm{MAD}=\mathrm{median}_{f_x\in\mathcal{V}} |e(f_x)-m|$."
        "\n\n"
        r"$\hat\sigma=1.4826\cdot\mathrm{MAD}$."
        "\n\n"
        r"$Z(f_x)=(e(f_x)-m)/\hat\sigma$."
        "\n\n"
        r"The factor $1.4826$ is the normal consistency constant for the MAD. "
        r"A column is a support candidate if $Z(f_x)>3.5$ and "
        r"$f_x\in\mathcal{V}$. The cut $3.5$ is the constant SAFE_X_Z; it is "
        r"not estimated on the frame. $Z$ is evaluated on $\pm q$ and "
        r"$\pm h_i$ and the pointwise maximum is kept."
    )
    _flow_text(fig, [MARGIN, 0.40, TEXT_W, 0.44], body, fontsize=9.5)

    from pmt_fringe_raw_adaptive import ridge_z_at_row

    records = load_catalog()["records"]
    entry = records[3]
    fam = entry["families"][0]
    raw, _ = load_frame(667, source_stack(entry))
    logamp = fft_log_amp(raw)
    h, w = logamp.shape
    cy, cx = h // 2, w // 2
    fx = np.arange(w) - cx
    q = int(round(float(fam["q"])))
    hi = int(round(float(fam["hi"])))
    z = np.max(
        np.stack(
            [
                ridge_z_at_row(logamp, +q),
                ridge_z_at_row(logamp, -q),
                ridge_z_at_row(logamp, +hi),
                ridge_z_at_row(logamp, -hi),
            ]
        ),
        axis=0,
    )
    ax = fig.add_axes([0.14, 0.10, 0.74, 0.26])
    ax.plot(fx, z, color="0.15", lw=0.8)
    ax.axhline(3.5, color=(0.85, 0.12, 0.12), ls="--", lw=1.0, label=r"$Z=3.5$ (SAFE_X_Z)")
    ax.axvspan(-19, -11, color="0.85", lw=0)
    ax.axvspan(11, 19, color="0.85", lw=0, label=r"retained after $w>0.20$ (this frame)")
    ax.set_xlim(-256, 255)
    ax.set_xlabel(r"$f_x$")
    ax.set_ylabel(r"$Z(f_x)$")
    ax.set_title(r"Frame 667, $\max Z$ on $\pm 6,\pm 250$. Grey: hydrated support.", fontsize=9)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    _flow_text(
        fig,
        [MARGIN, 0.015, TEXT_W, 0.055],
        r"Candidates are dilated by 1, smoothed with a Gaussian of $\sigma=1$, "
        r"and retained where the weight exceeds $0.20$. Those intervals are the "
        r"grey bands. They are not the mask (see 2b).",
        fontsize=9,
    )
    pdf.savefig(fig)
    fig.savefig(DEMO_DIR / "_preview_ridge_z.png", dpi=120)
    plt.close(fig)


def _catalog_mask(frame_idx: int = 667, index: int = 3):
    from batch_defringe.process_v4 import TRACK_SEARCH, Y_RADIUS, _catalog_line, apply_lines
    from .fig02_catalog_entry import MAX_ALPHA

    entry = load_catalog()["records"][index]
    fam = entry["families"][0]
    raw, _ = load_frame(frame_idx, source_stack(entry))
    logamp = fft_log_amp(raw)
    line = _catalog_line(fam, logamp)
    trial = apply_lines(raw, [line], max_alpha=MAX_ALPHA, search=TRACK_SEARCH)
    applied = np.asarray(trial["applied"], dtype=np.float64)
    return raw, applied, int(round(float(fam["q"]))), Y_RADIUS, entry


def page_mask_shapes(pdf) -> None:
    fig, plt = _new_page()
    _section_header(fig, part="2b.  Mask weights and fading additions")

    body = (
        r"The grey bands of 2a are support, not the mask. On those columns the "
        r"attenuation uses linear amplitude $A=|F|$ (not $S$), a different "
        r"neighbourhood $\delta\in\{\pm 5,\ldots,\pm 9\}$, and"
        "\n\n"
        r"$B_\ell(f_x)=\mathrm{median}_\delta\, A(y+\delta,f_x)$."
        "\n\n"
        r"$\rho(f_x)=A(y,f_x)/B_\ell(f_x)$."
        "\n\n"
        r"$c(f_x)=\mathrm{clip}((\rho-1.4)/(3.5-1.4),\,0,1)$."
        "\n\n"
        r"$c=0$ at $\rho=1.4$ (ratio_start); $c=1$ for $\rho >= 3.5$ "
        r"(ratio_full). The left-right envelope $w_x$ is the hydrated weight "
        r"from 2a. The ridge is then copied to the five rows "
        r"$\delta\in\{-2,\ldots,+2\}$ with"
        "\n\n"
        r"$w_y(\delta)=\exp(-\frac{1}{2}\delta^2/\sigma^2)$, $\sigma=1$ "
        r"(Y_RADIUS $=2$)."
        "\n\n"
        r"The applied mask is"
        "\n\n"
        r"$M=\alpha_{\max}\, g\, w_y(\delta)\, w_x(f_x)\, c(f_x)$"
        "\n\n"
        r"(here $\alpha_{\max}=1$; $g$ is the strength gate). The same "
        r"construction is repeated at $f_y=\pm q$ and at the stored partner "
        r"$f_y=\pm h_i$."
    )
    _flow_text(fig, [MARGIN, 0.50, TEXT_W, 0.34], body, fontsize=9.5)

    raw, applied, q, y_radius, entry = _catalog_mask()
    h, w = applied.shape
    cy, cx = h // 2, w // 2
    extent = (-cx - 0.5, w - cx - 0.5, h - cy - 0.5, -cy - 0.5)

    ax = fig.add_axes([0.14, 0.14, 0.32, 0.32])
    ax.imshow(applied, cmap="gray", vmin=0, vmax=1, interpolation="nearest", extent=extent)
    ax.set_xlim(-256, 255)
    ax.set_ylim(255, -256)
    ax.set_xlabel(r"$f_x$")
    ax.set_ylabel(r"$f_y$")
    ax.set_title(r"Applied mask $M$ (full plane)", fontsize=10)

    mask_row = applied[cy + q]
    xs = np.where(mask_row > 1e-3)[0]
    fx_hit = xs - cx
    fx_m0, fx_m1 = int(fx_hit.min()) - 6, int(fx_hit.max()) + 6
    fy_m0, fy_m1 = -q - y_radius - 3, q + y_radius + 3
    y_a, y_b = max(0, cy + fy_m0), min(h, cy + fy_m1 + 1)
    x_a, x_b = max(0, cx + fx_m0), min(w, cx + fx_m1 + 1)
    patch = applied[y_a:y_b, x_a:x_b]
    zext = (x_a - cx - 0.5, x_b - cx - 0.5, y_b - cy - 0.5, y_a - cy - 0.5)
    ax2 = fig.add_axes([0.52, 0.14, 0.26, 0.32])
    im = ax2.imshow(patch, cmap="inferno", vmin=0, vmax=1, interpolation="nearest", extent=zext, aspect="equal")
    ax2.axhline(q, color=(0.85, 0.12, 0.12), lw=0.7)
    ax2.axhline(-q, color=(0.85, 0.12, 0.12), lw=0.7, ls="--")
    ax2.set_xlabel(r"$f_x$")
    ax2.set_ylabel(r"$f_y$")
    ax2.set_title(rf"Zoom: fade $\pm{y_radius}$ rows about $f_y=\pm{q}$", fontsize=9)
    cax = fig.add_axes([0.80, 0.14, 0.016, 0.32])
    fig.colorbar(im, cax=cax, label="weight")
    _flow_text(
        fig,
        [MARGIN, 0.025, TEXT_W, 0.08],
        rf"Left: $M$ on the full $\pm 256$ plane (fig. 2, upper right). "
        rf"Right: zoom on the $q=6$ spots, showing the $\pm 2$-row fade. "
        rf"Catalog entry 3, {entry.get('channel')}, frame 667.",
        fontsize=8,
        color="0.35",
    )
    pdf.savefig(fig)
    fig.savefig(DEMO_DIR / "_preview_mask_shapes.png", dpi=120)
    plt.close(fig)


def _shutter_header(fig, *, part: str) -> None:
    _flow_text(fig, [MARGIN, 0.89, TEXT_W, 0.08], SECTION_SHUTTER, fontsize=11, weight="bold", style="italic")
    _flow_text(fig, [MARGIN, 0.85, TEXT_W, 0.04], part, fontsize=10, color="0.25")


def _shutter_medspec():
    from .fig04b_median_fourier import _shutter_indices
    from .fig04c_row_profile import _median_spectrum, _row_collapse

    idxs = _shutter_indices()
    medspec = _median_spectrum(idxs)
    return idxs, medspec, _row_collapse(medspec)


def page_shutter_row(pdf) -> None:
    fig, plt = _new_page()
    _shutter_header(fig, part="3a.  Median spectrum, row profile, and row Z")
    body = (
        r"Quiet shutter frames $I_k$ (ChanA: 756--760). For each frame, "
        r"$S_k=\log(1+|F_k|)$ after subtracting the frame median and "
        r"$\mathrm{fftshift}$. The median spectrum is the pixelwise median "
        r"over those $S_k$, not the FFT of the median image:"
        "\n\n"
        r"$S_{\mathrm{med}}(f_y,f_x)=\mathrm{median}_k\, S_k(f_y,f_x)$."
        "\n\n"
        r"Exclude DC and the outer edge, "
        r"$\mathcal{V}=\{f_x: 5<|f_x|<c_x-10\}$."
        "\n\n"
        r"Each Fourier row is collapsed to one number, the 95th percentile "
        r"of $S_{\mathrm{med}}$ on $\mathcal{V}$. Those numbers, stacked "
        r"versus $f_y$, are the row profile $r(f_y)$:"
        "\n\n"
        r"$r(f_y)=\mathrm{percentile}_{95}\, S_{\mathrm{med}}(f_y,\mathcal{V})$."
        "\n\n"
        r"Row $Z$ is leftover-style robust $Z$, but of the row profile along "
        r"$f_y$, not of $e(f_x)$ along $f_x$. Window $\pm 8$, exclude the "
        r"bin itself:"
        "\n\n"
        r"$m_r(f_y)=\mathrm{median}_{|j-f_y|\in[2,8]}\, r(j)$."
        "\n\n"
        r"$\mathrm{MAD}_r(f_y)=\mathrm{median}\, |r(j)-m_r|$ on that window."
        "\n\n"
        r"$Z_{\mathrm{row}}(f_y)=(r(f_y)-m_r)/(1.4826\cdot\mathrm{MAD}_r)$."
        "\n\n"
        r"A first-pass proposed ridge row is a local maximum of $Z_{\mathrm{row}}$ "
        r"with $Z_{\mathrm{row}} >= 3.0$ and $d_y\in\{5,\ldots,c_y-5\}$ "
        r"(positive $f_y$ only). If that list is empty, a fallback uses $2.2$."
    )
    _flow_text(fig, [MARGIN, 0.42, TEXT_W, 0.42], body, fontsize=9)

    idxs, medspec, coll = _shutter_medspec()
    h, w = medspec.shape
    cy, cx = coll["cy"], coll["cx"]
    fy = np.arange(h) - cy
    slo, shi = _percentile_limits(medspec, (3.0, 99.7))
    extent = (-cx - 0.5, w - cx - 0.5, h - cy - 0.5, -cy - 0.5)

    ax = fig.add_axes([0.12, 0.08, 0.34, 0.30])
    ax.imshow(medspec, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest", extent=extent)
    for c in coll["candidates"]:
        if int(c["dy"]) in {10, 30, 128, 246}:
            ax.axhline(c["fy"], color=(1.0, 0.85, 0.15), lw=0.8)
    ax.set_xlim(-256, 255)
    ax.set_ylim(255, -256)
    ax.set_xlabel(r"$f_x$")
    ax.set_ylabel(r"$f_y$")
    ax.set_title(r"$S_{\mathrm{med}}$", fontsize=9)
    ax.tick_params(labelsize=7)

    axz = fig.add_axes([0.54, 0.08, 0.34, 0.30])
    axz.plot(coll["row_z"], fy, color="0.15", lw=0.7)
    axz.axvline(3.0, color=(0.85, 0.12, 0.12), ls="--", lw=0.9)
    axz.set_ylim(255, -256)
    axz.set_xlabel(r"row $Z$")
    axz.set_ylabel(r"$f_y$")
    axz.set_title(r"$Z_{\mathrm{row}}$ of the row profile", fontsize=9)
    axz.tick_params(labelsize=7)
    _flow_text(
        fig,
        [MARGIN, 0.015, TEXT_W, 0.05],
        rf"ChanA shutter frames {idxs[0]}--{idxs[-1]}. Yellow: proposed rows "
        r"$f_y=+10,+30,+128,+246$. Row $Z$ is not leftover $Z$ (see 2a and 3c).",
        fontsize=8,
        color="0.35",
    )
    pdf.savefig(fig)
    fig.savefig(DEMO_DIR / "_preview_shutter_row.png", dpi=120)
    plt.close(fig)


def page_shutter_pair(pdf) -> None:
    fig, plt = _new_page()
    _shutter_header(fig, part="3b.  Pairing, ranking, and harmonics")
    body = (
        r"Walk proposed ridge rows in row-$Z$ order. For a candidate $d_y$, "
        r"the partner search is a $\pm 2$ window about $c_y-d_y$. Take the "
        r"peak of $Z_{\mathrm{row}}$ there, $d_*$. If "
        r"$Z_{\mathrm{row}}(c_y+d_*) >= 2.0$, the two rows become one family:"
        "\n\n"
        r"$q=\min(d_y,d_*),\qquad h_i=\max(d_y,d_*)$."
        "\n\n"
        r"Unpaired rows are kept only if $Z_{\mathrm{row}} >= 9$. Families "
        r"whose $q$ lie within 3 bins are merged by first-wins. The walk "
        r"stops at 4 families."
        "\n\n"
        r"On ChanA shutter, $+30$ pairs with $+224$ (peak in the window "
        r"around $+226$), $+10$ with $+246$, $+128$ with itself "
        r"(Nyquist self-pair), and $+107$ with $+147$."
        "\n\n"
        r"After pairing, hydrate must leave nonempty $f_x$ ranges (3c). "
        r"Families are then ranked by the 99th percentile of $S_{\mathrm{med}}$ "
        r"on the four rows $\pm q,\pm h_i$, on $\mathcal{V}$ only --- not by "
        r"row $Z$:"
        "\n\n"
        r"$s=\max_{\mathrm{rows}}\,\mathrm{percentile}_{99}\, "
        r"S_{\mathrm{med}}(\mathrm{row},\mathcal{V})$."
        "\n\n"
        r"The strongest is the primary $q_0$ (here $q_0=10$, even though "
        r"$+30$ had the largest row $Z$). Drop Nyquist self-pairs "
        r"($q\approx 128$). Keep another family only if"
        "\n\n"
        r"$|q-2q_0|<2.5$ or $|q-3q_0|<2.5$"
        "\n\n"
        r"(or the inverse: $q_0$ is $2q$ or $3q$). So $30=3\times 10$ is "
        r"kept and $107$ is not. A $q$ within 16 bins of DC is skipped as "
        r"an extra, not as the primary."
    )
    _flow_text(fig, [MARGIN, 0.40, TEXT_W, 0.44], body, fontsize=9)

    idxs, medspec, coll = _shutter_medspec()
    h, w = medspec.shape
    cy, cx = coll["cy"], coll["cx"]
    slo, shi = _percentile_limits(medspec, (3.0, 99.7))
    extent = (-cx - 0.5, w - cx - 0.5, h - cy - 0.5, -cy - 0.5)
    pairs = [(10, 246, (0.10, 0.52, 0.38)), (30, 224, (0.85, 0.42, 0.08))]
    ax = fig.add_axes([0.22, 0.08, 0.56, 0.28])
    ax.imshow(medspec, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest", extent=extent)
    for q, hi, col in pairs:
        ax.axhline(q, color=col, lw=1.1)
        ax.axhline(hi, color=col, lw=1.1)
    ax.set_xlim(-256, 255)
    ax.set_ylim(255, -256)
    ax.set_xlabel(r"$f_x$")
    ax.set_ylabel(r"$f_y$")
    ax.set_title(r"$S_{\mathrm{med}}$: kept pairs $10\leftrightarrow 246$ and $30\leftrightarrow 224$", fontsize=9)
    ax.tick_params(labelsize=7)
    _flow_text(
        fig,
        [MARGIN, 0.015, TEXT_W, 0.05],
        rf"ChanA shutter {idxs[0]}--{idxs[-1]}. Green $q=10$, orange $q=30$. "
        r"Nyquist $q=128$ is paired then dropped at rank.",
        fontsize=8,
        color="0.35",
    )
    pdf.savefig(fig)
    fig.savefig(DEMO_DIR / "_preview_shutter_pair.png", dpi=120)
    plt.close(fig)


def page_shutter_peak(pdf) -> None:
    fig, plt = _new_page()
    _shutter_header(fig, part="3c.  Leftover Z, hydrate band, peak fx, and stored weight")
    body = (
        r"Leftover $e(f_x)$ and leftover $Z(f_x)$ are the same leftover as 2a, "
        r"now on $S_{\mathrm{med}}$. On row $y=c_y+q$,"
        "\n\n"
        r"$B=\mathrm{median}_{\delta\in\{\pm 4,\ldots,\pm 10\}} S_{\mathrm{med}}(y+\delta)$,   "
        r"$e=S_{\mathrm{med}}(y)-B$."
        "\n\n"
        r"$Z=(e-m)/(1.4826\cdot\mathrm{MAD})$ with $m$ and MAD of $e$ on "
        r"$\mathcal{V}$. So $Z$ is $e$ in units of the leftover's own robust "
        r"scale. Keep the pointwise max $Z_x$ of the four traces $\pm q,\pm h_i$."
        "\n\n"
        r"Hydrate band (gate only): $Z_x>2.5$ on $\mathcal{V}$, dilate, "
        r"smooth, keep weight $>0.20$. Empty band drops the family. Not the seed."
        "\n\n"
        r"Peak $f_x$ (stored $w_x$): local maxima of $Z_x$ with $Z_x >= 6$, "
        r"Gaussian half-width 3. Stored and passed on: $(q,h_i,w_x)$."
        "\n\n"
        r"$W=w_y(\delta)\,w_x(f_x)$,   $w_y=\exp(-0.5\,\delta^2)$, "
        r"$\delta\in\{-2,\ldots,+2\}$."
        "\n\n"
        r"$w_x$ is not re-hydrated later. On each frame, "
        r"$M=\alpha_{\max}\, g\, w_y\, w_x\, c$ with $c$ and $g$ from that "
        r"frame (same $c$ as 2b)."
    )
    _flow_text(fig, [MARGIN, 0.50, TEXT_W, 0.34], body, fontsize=9)

    from .fig04f_peak_fx import _kept_families, _pack
    from .fig04h_stored_weight import _draw_mask_zoom, stored_weight

    idxs, medspec, _coll = _shutter_medspec()
    packs = [_pack(medspec, fam) for fam in _kept_families(medspec)]
    p10 = next(p for p in packs if p["q"] == 10)
    h, w = medspec.shape
    cy, cx = h // 2, w // 2
    fx = np.arange(w) - cx

    ax = fig.add_axes([0.10, 0.16, 0.40, 0.30])
    zx = p10["zx"]
    for lo, hi in p10["hyd_ranges"]:
        ax.axvspan(lo, hi, color=(0.72, 0.72, 0.72), alpha=0.45, lw=0, zorder=0)
    for lo, hi in p10["peak_ranges"]:
        ax.axvspan(lo, hi, color=p10["color"], alpha=0.35, lw=0, zorder=1)
    ax.plot(fx, zx, color="0.15", lw=0.7, zorder=2)
    ax.axhline(2.5, color="0.45", ls="--", lw=0.8)
    ax.axhline(6.0, color=(0.75, 0.12, 0.12), ls="--", lw=0.8)
    ax.set_xlim(-160, 160)
    ax.set_xlabel(r"$f_x$")
    ax.set_ylabel(r"leftover $Z(f_x)$")
    ax.set_title(r"$q=10$: grey hydrate, colour = peak $f_x$", fontsize=8)
    ax.tick_params(labelsize=7)

    from .fig04g_shutter_mask import _learn

    families, _ = _learn()
    fam = next(f for f in families if int(round(float(f["q"]))) == 10)
    weight = stored_weight(fam, h, w)
    ax2 = fig.add_axes([0.56, 0.16, 0.30, 0.30])
    im = _draw_mask_zoom(ax2, weight, 10, cx, cy, title=r"Stored $W$ at $q=10$")
    ax2.tick_params(labelsize=6)
    cax = fig.add_axes([0.88, 0.16, 0.015, 0.30])
    fig.colorbar(im, cax=cax, ticks=[0, 1])
    _flow_text(
        fig,
        [MARGIN, 0.02, TEXT_W, 0.10],
        rf"Left: leftover $Z_x$ on $S_{{\mathrm{{med}}}}$ (frames {idxs[0]}--{idxs[-1]}). "
        r"Grey = hydrate band; colour = peak $f_x$. "
        r"Right: stored template $W=w_y w_x$ (figure 04h). $c$ is not in $W$.",
        fontsize=8,
        color="0.35",
    )
    pdf.savefig(fig)
    fig.savefig(DEMO_DIR / "_preview_shutter_peak.png", dpi=120)
    plt.close(fig)


PAGES = [
    page_cover,
    page_contents,
    page_coordinates,
    page_ridge_z,
    page_mask_shapes,
    page_shutter_row,
    page_shutter_pair,
    page_shutter_peak,
]


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT
    try:
        dest.open("ab").close()
    except OSError:
        dest = DEMO_DIR / "formalizations_new.pdf"
    with PdfPages(dest) as pdf:
        for page in PAGES:
            page(pdf)
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
