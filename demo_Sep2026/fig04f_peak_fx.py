"""Figure 04f: replace the hydrate band with peak-only fx weights.

``python -m demo_Sep2026.fig04f_peak_fx``
"""

from __future__ import annotations

import argparse

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .common import _percentile_limits, boxed_text, new_figure, save_figure
from batch_defringe.seed import hydrate_families
from batch_defringe.shutter_seed_test import FX_PEAK_Z, peak_fx_weight
from .fig04b_median_fourier import _shutter_indices
from .fig04c_row_profile import DC_FX, EDGE_MARGIN, _median_spectrum
from .fig04d_pairing import FAMILY_COLOR
from .fig04e_rank_harmonics import X_Z, _rank_and_choose
from .paths import CHANA_TIF

HYDRATE_Z = 2.5


def _kept_families(medspec: np.ndarray) -> list[dict]:
    rows = [r for r in _rank_and_choose(medspec) if r["kept"]]
    return [{"q": float(r["q"]), "hi": r["hi"], "paired": True} for r in rows]


def _pack(medspec: np.ndarray, fam: dict) -> dict:
    hyd = hydrate_families([fam], medspec, x_z_thresh=X_Z)[0]
    weight, peak_ranges = peak_fx_weight(medspec, hyd)
    zx = np.asarray(hyd["x_z"], dtype=np.float64)
    hyd_ranges = list(hyd.get("fx_ranges") or [])
    q = int(round(float(hyd["q"])))
    return {
        "q": q,
        "hi": None if hyd.get("hi") is None else int(round(float(hyd["hi"]))),
        "zx": zx,
        "hyd_ranges": hyd_ranges,
        "peak_ranges": [list(p) for p in peak_ranges],
        "n_hyd_bins": int(sum(hi - lo + 1 for lo, hi in hyd_ranges)),
        "n_peak_bins": int(np.sum(np.asarray(weight) > 0.20)),
        "n_peak_intervals": len(peak_ranges),
        "color": FAMILY_COLOR.get(q, (0.20, 0.20, 0.20)),
    }


def _draw_z(ax, pack: dict, fx: np.ndarray, xvalid: np.ndarray) -> None:
    zx = pack["zx"]
    color = pack["color"]
    zmax = max(float(np.nanmax(zx[xvalid])), FX_PEAK_Z) * 1.08
    zmin = min(-1.0, float(np.nanmin(zx[xvalid])) * 1.05)
    for lo, hi in pack["hyd_ranges"]:
        ax.axvspan(lo, hi, color=(0.72, 0.72, 0.72), alpha=0.45, lw=0, zorder=0)
    for lo, hi in pack["peak_ranges"]:
        ax.axvspan(lo, hi, color=color, alpha=0.35, lw=0, zorder=1)
    ax.plot(fx, zx, color="0.15", lw=0.85, zorder=2)
    ax.axhline(HYDRATE_Z, color="0.45", ls="--", lw=1.0, zorder=3)
    ax.axhline(FX_PEAK_Z, color=(0.75, 0.12, 0.12), ls="--", lw=1.05, zorder=3)
    peaks = [0.5 * (lo + hi) for lo, hi in pack["peak_ranges"]]
    if peaks:
        ax.plot(peaks, np.interp(peaks, fx, zx), "o", color=color, ms=4.5, zorder=4)
    ax.set_xlim(-160, 160)
    ax.set_ylim(zmin, zmax)
    ax.set_xlabel(r"$f_x$")
    ax.set_ylabel(r"leftover $Z(f_x)$")
    ax.set_title(rf"$q={pack['q']}$  leftover $Z(f_x)$", fontsize=11)


def draw() -> dict:
    idxs = _shutter_indices()
    medspec = _median_spectrum(idxs)
    h, w = medspec.shape
    cy, cx = h // 2, w // 2
    fx = np.arange(w) - cx
    xvalid = (np.abs(fx) > DC_FX) & (np.abs(fx) < cx - EDGE_MARGIN)
    packs = [_pack(medspec, fam) for fam in _kept_families(medspec)]
    extent = (-cx - 0.5, w - cx - 0.5, h - cy - 0.5, -cy - 0.5)
    slo, shi = _percentile_limits(medspec, (3.0, 99.7))
    edge = cx - EDGE_MARGIN

    fig = new_figure()
    fig.suptitle(
        r"Figure 04f  ·  Peak $f_x$ replaces the hydrate band  ·  ChanA",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    gs = fig.add_gridspec(
        1,
        3,
        left=0.05,
        right=0.975,
        top=0.88,
        bottom=0.40,
        wspace=0.26,
        width_ratios=[1.05, 1.1, 1.1],
    )

    ax = fig.add_subplot(gs[0])
    ax.imshow(medspec, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest", extent=extent)
    ax.axvspan(-DC_FX, DC_FX, color=(0.85, 0.15, 0.12), alpha=0.22, lw=0)
    ax.axvspan(-cx, -edge, color=(0.20, 0.35, 0.80), alpha=0.18, lw=0)
    ax.axvspan(edge, cx, color=(0.20, 0.35, 0.80), alpha=0.18, lw=0)
    for pack in packs:
        color = pack["color"]
        ax.axhline(pack["q"], color=color, lw=1.15, alpha=0.95)
        if pack["hi"] is not None:
            ax.axhline(pack["hi"], color=color, lw=1.15, alpha=0.95)
        for lo, hi in pack["peak_ranges"]:
            ax.axvspan(lo, hi, color=color, alpha=0.22, lw=0)
    ax.set_xlim(-256, 255)
    ax.set_ylim(255, -256)
    ax.set_xlabel(r"$f_x$")
    ax.set_ylabel(r"$f_y$")
    ax.set_title(r"$S_{\mathrm{med}}$  (shipped peak $f_x$)", fontsize=11)

    for i, pack in enumerate(packs):
        axz = fig.add_subplot(gs[i + 1])
        _draw_z(axz, pack, fx, xvalid)
        if i == 0:
            axz.legend(
                handles=[
                    Patch(facecolor=(0.72, 0.72, 0.72), alpha=0.45, label="hydrate Z > 2.5"),
                    Patch(facecolor=pack["color"], alpha=0.35, label="peak fx"),
                    Line2D([0], [0], color="0.45", ls="--", lw=1.0, label="Z = 2.5"),
                    Line2D([0], [0], color=(0.75, 0.12, 0.12), ls="--", lw=1.05, label="Z = 6"),
                ],
                fontsize=7,
                loc="lower left",
                frameon=False,
            )

    p10 = next(p for p in packs if p["q"] == 10)
    p30 = next(p for p in packs if p["q"] == 30)
    boxed_text(
        fig,
        [0.05, 0.02, 0.90, 0.28],
        (
            "Leftover Z(fx) is the pointwise max of ridge Z on the four rows +/-q and +/-hi. "
            "It is not row Z from 04c. Grey band: hydrate support, leftover Z > 2.5 on V "
            "(dilated, then weight > 0.20). That band is only a gate: empty fx ranges drop "
            "the family. It is not the shipped seed."
            "\n"
            f"Coloured intervals: peak_fx_weight. Local maxima of leftover Z in a 7-bin window, "
            f"keep peaks with Z >= {FX_PEAK_Z:.0f} (at most 12), each a Gaussian of half-width 3. "
            "fx_ranges are where that weight > 0.20. Those intervals are shutter_hint. "
            f"q=10: hydrate {p10['n_hyd_bins']} bins vs {p10['n_peak_intervals']} peak intervals "
            f"({p10['n_peak_bins']} bins). "
            f"q=30: hydrate {p30['n_hyd_bins']} bins vs {p30['n_peak_intervals']} peak intervals "
            f"({p30['n_peak_bins']} bins). Amplitude is not copied; each live frame re-measures support."
        ),
        fontsize=8.5,
        color="0.3",
    )

    numbers = {
        "stack": str(CHANA_TIF),
        "shutter_frames": idxs,
        "hydrate_z": HYDRATE_Z,
        "peak_z": FX_PEAK_Z,
        "families": [
            {
                "q": p["q"],
                "hi": p["hi"],
                "n_hyd_bins": p["n_hyd_bins"],
                "n_peak_bins": p["n_peak_bins"],
                "n_peak_intervals": p["n_peak_intervals"],
                "hyd_ranges": p["hyd_ranges"],
                "peak_ranges": p["peak_ranges"],
            }
            for p in packs
        ],
        "note": "hydrate Z>2.5 grey band vs shipped peak_fx_weight Z>=6",
    }
    try:
        written = save_figure(fig, "fig04f_peak_fx", numbers)
    except OSError:
        written = save_figure(fig, "fig04f_peak_fx_new", numbers)
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
