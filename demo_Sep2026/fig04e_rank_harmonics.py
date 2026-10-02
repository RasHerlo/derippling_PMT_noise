"""Figure 04e: rank paired families and keep near-harmonics of the primary.

``python -m demo_Sep2026.fig04e_rank_harmonics``
"""

from __future__ import annotations

import argparse

import numpy as np

from .common import _percentile_limits, boxed_text, new_figure, save_figure
from pmt_fringe_raw_adaptive import detect_families  # noqa: E402

from batch_defringe.seed import hydrate_families
from batch_defringe.shutter_seed_test import (
    _family_peak_score,
    _is_nyquist_self_pair,
    _near_dc,
)
from .fig04b_median_fourier import _shutter_indices
from .fig04c_row_profile import DC_FX, EDGE_MARGIN, _median_spectrum
from .fig04d_pairing import FAMILY_COLOR
from .paths import CHANA_TIF

X_Z = 2.5
HARMONIC_TOL = 2.5
MAX_FAMILIES = 3


def _rank_and_choose(medspec: np.ndarray) -> list[dict]:
    """Same rank / Nyquist / harmonic walk as ``learn_shutter_families``."""
    h = int(medspec.shape[0])
    raw, _, _ = detect_families(
        medspec,
        row_z_thresh=3.0,
        pair_z_min=2.0,
        x_z_thresh=X_Z,
        max_families=4,
        allow_standalone=True,
    )
    hyd = [f for f in hydrate_families(raw, medspec, x_z_thresh=X_Z) if f.get("fx_ranges")]
    ranked = sorted(hyd, key=lambda f: _family_peak_score(medspec, f), reverse=True)
    rows: list[dict] = []
    for fam in ranked:
        q = float(fam["q"])
        rows.append(
            {
                "q": q,
                "hi": None if fam.get("hi") is None else float(fam["hi"]),
                "score99": float(_family_peak_score(medspec, fam)),
                "row_score": float(fam.get("row_score", 0.0)),
                "nyquist": bool(_is_nyquist_self_pair(fam, h)),
                "near_dc": bool(_near_dc(q, h)),
                "n_fx_ranges": len(fam.get("fx_ranges") or []),
                "kept": False,
                "role": "",
            }
        )
    surviving = [r for r in rows if not r["nyquist"]] or list(rows)
    if not surviving:
        return rows
    primary = surviving[0]
    q0 = float(primary["q"])
    primary["kept"] = True
    primary["role"] = "primary"
    kept_n = 1
    for r in surviving[1:]:
        q = float(r["q"])
        if r["near_dc"]:
            r["role"] = "near DC"
            continue
        harmonic = any(abs(q - k * q0) < HARMONIC_TOL or abs(q0 - k * q) < HARMONIC_TOL for k in (2, 3))
        if harmonic and kept_n < MAX_FAMILIES:
            r["kept"] = True
            r["role"] = "3 x q0" if abs(q - 3 * q0) < HARMONIC_TOL or abs(q0 - 3 * q) < HARMONIC_TOL else "2 x q0"
            kept_n += 1
        elif harmonic:
            r["role"] = "harmonic, over cap"
        else:
            r["role"] = "not 2 q0 or 3 q0"
    for r in rows:
        if r["nyquist"] and not r["role"]:
            r["role"] = "Nyquist self-pair"
        if r["kept"] is False and r["role"] == "" and r["near_dc"]:
            r["role"] = "near DC"
    return rows


def _color_for(row: dict) -> tuple[float, float, float]:
    q = int(round(row["q"]))
    if row["kept"] and q in FAMILY_COLOR:
        return FAMILY_COLOR[q]
    return (0.55, 0.55, 0.55)


def draw() -> dict:
    idxs = _shutter_indices()
    medspec = _median_spectrum(idxs)
    h, w = medspec.shape
    cy, cx = h // 2, w // 2
    rows = _rank_and_choose(medspec)
    q0 = next(r["q"] for r in rows if r["role"] == "primary")
    extent = (-cx - 0.5, w - cx - 0.5, h - cy - 0.5, -cy - 0.5)
    slo, shi = _percentile_limits(medspec, (3.0, 99.7))
    edge = cx - EDGE_MARGIN

    fig = new_figure()
    fig.suptitle(
        "Figure 04e  ·  Rank families and keep harmonics  ·  ChanA",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    gs = fig.add_gridspec(
        1,
        3,
        left=0.05,
        right=0.985,
        top=0.88,
        bottom=0.40,
        wspace=0.24,
        width_ratios=[1.05, 1.05, 1.25],
    )

    ax = fig.add_subplot(gs[0])
    ax.imshow(medspec, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest", extent=extent)
    ax.axvspan(-DC_FX, DC_FX, color=(0.85, 0.15, 0.12), alpha=0.22, lw=0)
    ax.axvspan(-cx, -edge, color=(0.20, 0.35, 0.80), alpha=0.18, lw=0)
    ax.axvspan(edge, cx, color=(0.20, 0.35, 0.80), alpha=0.18, lw=0)
    for r in rows:
        color = _color_for(r)
        lw = 1.35 if r["kept"] else 0.8
        ls = "-" if r["kept"] else "--"
        ax.axhline(r["q"], color=color, lw=lw, ls=ls, alpha=0.95)
        if r["hi"] is not None:
            ax.axhline(r["hi"], color=color, lw=lw, ls=ls, alpha=0.95)
    ax.set_xlim(-256, 255)
    ax.set_ylim(255, -256)
    ax.set_xlabel(r"$f_x$")
    ax.set_ylabel(r"$f_y$")
    ax.set_title(r"$S_{\mathrm{med}}$  (kept solid)", fontsize=11)

    axb = fig.add_subplot(gs[1])
    labels = [f"q={int(r['q'])}" for r in rows]
    scores = [r["score99"] for r in rows]
    colors = [_color_for(r) for r in rows]
    ypos = np.arange(len(rows))[::-1]
    axb.barh(ypos, scores, color=colors, height=0.62, edgecolor="0.25", linewidth=0.4)
    for y, r in zip(ypos, rows):
        axb.text(r["score99"] + 0.12, y, f"{r['score99']:.2f}", va="center", fontsize=8, color="0.2")
    axb.set_yticks(ypos)
    axb.set_yticklabels(labels, fontsize=9)
    axb.set_xlabel("99th percentile on V")
    axb.set_xlim(0, max(scores) * 1.22)
    axb.set_title("Family score (rank order)", fontsize=11)
    axb.tick_params(axis="y", length=0)

    axt = fig.add_subplot(gs[2])
    axt.set_axis_off()
    axt.set_title("Keep / drop", fontsize=11, pad=8)
    cells = []
    for r in rows:
        hi = "—" if r["hi"] is None else str(int(r["hi"]))
        keep = "kept" if r["kept"] else "drop"
        role = {
            "primary": "primary",
            "3 x q0": "3 x q0",
            "2 x q0": "2 x q0",
            "not 2 q0 or 3 q0": "not 2q0/3q0",
            "Nyquist self-pair": "Nyquist",
            "near DC": "near DC",
        }.get(r["role"], r["role"])
        cells.append(
            [
                f"{int(r['q'])}",
                hi,
                f"{r['score99']:.2f}",
                f"{r['row_score']:.2f}",
                f"{keep}: {role}",
            ]
        )
    tbl = axt.table(
        cellText=cells,
        colLabels=["q", "hi", "score", "row Z", "decision"],
        loc="upper center",
        cellLoc="left",
        colLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.08, 1.55)
    for (ri, _ci), cell in tbl.get_celld().items():
        cell.set_edgecolor((0.85, 0.85, 0.85))
        cell.set_linewidth(0.4)
        if ri == 0:
            cell.set_facecolor((0.93, 0.93, 0.93))
            cell.set_text_props(weight="bold")
            continue
        r = rows[ri - 1]
        col = _color_for(r)
        if r["kept"]:
            cell.set_facecolor((col[0] * 0.18 + 0.82, col[1] * 0.18 + 0.82, col[2] * 0.18 + 0.82))
        else:
            cell.set_facecolor((0.94, 0.94, 0.94))

    boxed_text(
        fig,
        [0.05, 0.02, 0.90, 0.28],
        (
            "After pairing, hydrate must leave nonempty fx ranges (all four families do). "
            "Families are then ranked by the 99th percentile of S_med on their four rows "
            "(+/-q and +/-hi), on V only. That score is not row Z: +30 had the largest row Z, "
            f"but q0 = {int(q0)} is the primary because its 99th-percentile score is highest."
            "\n"
            "Nyquist self-pairs (q around 128) are dropped. Remaining extras are kept only if "
            "q is within 2.5 bins of 2 q0 or 3 q0. 30 is 3 x 10, so it is kept. 107 is not. "
            "q=10 sits near DC, but the near-DC skip applies only to extras, not to the primary. "
            "Peak fx weights (narrower than the hydrate band) are figure 04f."
        ),
        fontsize=8.5,
        color="0.3",
    )

    numbers = {
        "stack": str(CHANA_TIF),
        "shutter_frames": idxs,
        "q0": q0,
        "harmonic_tol": HARMONIC_TOL,
        "families": rows,
        "kept_q": [r["q"] for r in rows if r["kept"]],
        "note": "learn_shutter_families rank / Nyquist / 2q0-3q0 walk; no peak_fx yet",
    }
    try:
        written = save_figure(fig, "fig04e_rank_harmonics", numbers)
    except OSError:
        written = save_figure(fig, "fig04e_rank_harmonics_new", numbers)
    numbers["written"] = [str(p) for p in written]
    return numbers


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    numbers = draw()
    for p in numbers["written"]:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
