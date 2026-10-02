"""Figure 04d: pair proposed ridge rows into families.

``python -m demo_Sep2026.fig04d_pairing``
"""

from __future__ import annotations

import argparse

import numpy as np

from .common import _percentile_limits, boxed_text, new_figure, save_figure
from .fig04b_median_fourier import _shutter_indices
from .fig04c_row_profile import DC_FX, EDGE_MARGIN, ROW_Z_THRESH, _median_spectrum, _row_collapse
from .paths import CHANA_TIF

PAIR_Z_MIN = 2.0
MAX_FAMILIES = 4
STANDALONE_BAR = max(9.0, ROW_Z_THRESH + 2)
CLOSE_Q = 3.0

# q of the family that owns the pair (detect_families walk order).
FAMILY_COLOR = {
    10: (0.10, 0.52, 0.38),
    30: (0.85, 0.42, 0.08),
    107: (0.48, 0.28, 0.68),
    128: (0.42, 0.42, 0.46),
}


def pair_walk(coll: dict) -> list[dict]:
    """Pair first-pass proposals the same way ``detect_families`` does.

    Returns one record per proposed ridge row, in row-Z order.
    """
    cy = int(coll["cy"])
    h = int(coll["row_z"].shape[0])
    row_z = np.asarray(coll["row_z"], dtype=np.float64)
    records: list[dict] = []
    by_dy: dict[int, dict] = {}
    for c in coll["candidates"]:
        rec = {
            "dy": int(c["dy"]),
            "fy": float(c["fy"]),
            "row_z": float(c["row_z"]),
            "partner_nominal": int(cy - int(c["dy"])),
            "partner_dy": None,
            "partner_z": None,
            "fate": "not_reached",
            "q": None,
            "hi": None,
            "paired": False,
        }
        records.append(rec)
        by_dy[rec["dy"]] = rec

    used: set[int] = set()
    families: list[dict] = []
    for rec in records:
        dy = rec["dy"]
        if dy in used:
            rec["fate"] = "used_as_partner"
            continue
        if len(families) >= MAX_FAMILIES:
            rec["fate"] = "not_reached"
            continue
        comp = cy - dy
        lo = max(5, comp - 2)
        hi = min(cy - 5, comp + 2)
        if lo <= hi:
            partner = max(range(lo, hi + 1), key=lambda d: row_z[cy + d])
            partner_z = float(row_z[cy + partner])
        else:
            partner, partner_z = comp, float("-inf")
        rec["partner_dy"] = int(partner)
        rec["partner_z"] = partner_z
        if partner_z >= PAIR_Z_MIN:
            q, him = sorted([dy, partner])
            fam = {"q": float(q), "hi": float(him)}
            if any(abs(fam["q"] - f["q"]) < CLOSE_Q for f in families):
                rec["fate"] = "dropped_close_q"
                continue
            rec["fate"] = "paired"
            rec["paired"] = True
            rec["q"] = float(q)
            rec["hi"] = float(him)
            used.add(dy)
            used.add(int(partner))
            families.append(fam)
            if partner in by_dy and partner != dy:
                other = by_dy[partner]
                other["fate"] = "used_as_partner"
                other["partner_dy"] = dy
                other["partner_z"] = rec["row_z"]
                other["q"] = float(q)
                other["hi"] = float(him)
        elif rec["row_z"] >= STANDALONE_BAR:
            rec["fate"] = "standalone"
            rec["q"] = float(dy)
            used.add(dy)
            families.append({"q": float(dy), "hi": None})
        else:
            rec["fate"] = "dropped_unpaired"

    # Anything still marked not_reached after the walk stays that way.
    _ = h
    return records


def _fate_label(rec: dict) -> str:
    fate = rec["fate"]
    if fate == "paired":
        q, hi = rec["q"], rec["hi"]
        if q is not None and hi is not None and abs(float(q) - float(hi)) < 1.5:
            return f"q={int(q)} self-pair"
        return f"q={int(q)}, hi={int(hi)}"
    if fate == "used_as_partner":
        if rec["q"] is not None:
            return f"partner of q={int(rec['q'])}"
        return "used as partner"
    if fate == "not_reached":
        return "not reached"
    if fate == "dropped_unpaired":
        return "dropped unpaired"
    if fate == "dropped_close_q":
        return "dropped close q"
    if fate == "standalone":
        return f"standalone q={int(rec['q'])}"
    return fate


def _family_q(rec: dict) -> int | None:
    if rec["q"] is None:
        return None
    return int(round(float(rec["q"])))


def draw() -> dict:
    idxs = _shutter_indices()
    medspec = _median_spectrum(idxs)
    h, w = medspec.shape
    coll = _row_collapse(medspec)
    cy, cx = coll["cy"], coll["cx"]
    records = pair_walk(coll)
    fy = np.arange(h) - cy
    extent = (-cx - 0.5, w - cx - 0.5, h - cy - 0.5, -cy - 0.5)
    slo, shi = _percentile_limits(medspec, (3.0, 99.7))
    edge = cx - EDGE_MARGIN
    families = [r for r in records if r["fate"] == "paired"]

    fig = new_figure()
    fig.suptitle(
        "Figure 04d  ·  Pairing proposed ridge rows  ·  ChanA",
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
        wspace=0.22,
        width_ratios=[1.05, 0.95, 1.35],
    )

    ax = fig.add_subplot(gs[0])
    ax.imshow(medspec, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest", extent=extent)
    ax.axvspan(-DC_FX, DC_FX, color=(0.85, 0.15, 0.12), alpha=0.22, lw=0)
    ax.axvspan(-cx, -edge, color=(0.20, 0.35, 0.80), alpha=0.18, lw=0)
    ax.axvspan(edge, cx, color=(0.20, 0.35, 0.80), alpha=0.18, lw=0)
    for rec in families:
        color = FAMILY_COLOR.get(_family_q(rec), (0.85, 0.75, 0.15))
        ax.axhline(rec["fy"], color=color, lw=1.2, alpha=0.95)
        if rec["hi"] is not None:
            ax.axhline(float(rec["hi"]), color=color, lw=1.2, alpha=0.95)
    ax.set_xlim(-256, 255)
    ax.set_ylim(255, -256)
    ax.set_xlabel(r"$f_x$")
    ax.set_ylabel(r"$f_y$")
    ax.set_title(r"$S_{\mathrm{med}}$  (paired rows)", fontsize=11)

    axz = fig.add_subplot(gs[1])
    axz.plot(coll["row_z"], fy, color="0.15", lw=0.9)
    axz.axvline(PAIR_Z_MIN, color=(0.85, 0.12, 0.12), ls="--", lw=1.0, label=f"partner Z = {PAIR_Z_MIN:.1f}")
    axz.axhspan(-5, 5, color="0.85", alpha=0.45, lw=0, zorder=0)
    zmax = max(float(coll["row_z"].max()), PAIR_Z_MIN) * 1.08
    axz.set_xlim(-2.0, zmax + 11.0)
    # Stagger brackets so long pairs (10-246, 30-224) do not sit on one line.
    bracket_off = {10: 0.0, 30: 2.4, 107: 4.8, 128: 7.2}
    for rec in families:
        q = _family_q(rec)
        color = FAMILY_COLOR.get(q, (0.85, 0.75, 0.15))
        y1 = rec["fy"]
        y2 = float(rec["hi"]) if rec["hi"] is not None else rec["fy"]
        z1 = rec["row_z"]
        z2 = float(rec["partner_z"]) if rec["partner_z"] is not None else rec["row_z"]
        axz.plot([z1], [y1], "o", color=color, ms=5.5, zorder=3)
        axz.plot([z2], [y2], "o", color=color, ms=5.5, zorder=3)
        bx = zmax + 2.0 + bracket_off.get(q, 0.0)
        if abs(y1 - y2) < 1.5:
            axz.plot([z1, bx], [y1, y1], color=color, lw=1.05, zorder=2)
        else:
            axz.plot([z1, bx, bx, z2], [y1, y1, y2, y2], color=color, lw=1.05, zorder=2)
    axz.set_ylim(255, -256)
    axz.set_xlabel(r"row $Z$")
    axz.set_ylabel(r"$f_y$")
    axz.set_title("Partner links", fontsize=11)
    axz.legend(fontsize=7.5, loc="upper right", frameon=False)

    axt = fig.add_subplot(gs[2])
    axt.set_axis_off()
    axt.set_title("Pairing walk (row Z order)", fontsize=11, pad=8)
    col_labels = ["fy", "row Z", "partner", "p. Z", "outcome"]
    cells = []
    for rec in records:
        pdy = rec["partner_dy"]
        pz = rec["partner_z"]
        cells.append(
            [
                f"+{rec['dy']}",
                f"{rec['row_z']:.2f}",
                f"+{pdy}" if pdy is not None else "—",
                f"{pz:.2f}" if pz is not None else "—",
                _fate_label(rec),
            ]
        )
    tbl = axt.table(
        cellText=cells,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="left",
        colLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.2)
    tbl.scale(1.05, 1.38)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor((0.85, 0.85, 0.85))
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor((0.93, 0.93, 0.93))
            cell.set_text_props(weight="bold")
            continue
        rec = records[r - 1]
        q = _family_q(rec)
        if q in FAMILY_COLOR and rec["fate"] in {"paired", "used_as_partner"}:
            rcol, gcol, bcol = FAMILY_COLOR[q]
            cell.set_facecolor((rcol * 0.18 + 0.82, gcol * 0.18 + 0.82, bcol * 0.18 + 0.82))

    boxed_text(
        fig,
        [0.05, 0.02, 0.90, 0.28],
        (
            "Each proposed ridge row from 04c is walked in row-Z order. "
            "For a row dy, the partner row is the peak of row Z in a +/-2 window around cy-dy. "
            f"If that partner row Z >= {PAIR_Z_MIN:.1f}, the two rows become one family: "
            "q = min(dy, partner), hi = max. Unpaired rows would need row Z >= "
            f"{STANDALONE_BAR:.0f} to stand alone; none do here."
            "\n"
            "+30 pairs with +224 (peak in the window around +226). "
            "+10 pairs with +246. +128 pairs with itself (Nyquist self-pair). "
            "+107 pairs with +147. The walk stops at 4 families, so +49 and +70 are not reached. "
            "Ranking and harmonics are figure 04e."
        ),
        fontsize=8.5,
        color="0.3",
    )

    numbers = {
        "stack": str(CHANA_TIF),
        "shutter_frames": idxs,
        "pair_z_min": PAIR_Z_MIN,
        "standalone_bar": STANDALONE_BAR,
        "max_families": MAX_FAMILIES,
        "records": [
            {
                "fy": r["fy"],
                "row_z": r["row_z"],
                "partner_dy": r["partner_dy"],
                "partner_z": r["partner_z"],
                "fate": r["fate"],
                "q": r["q"],
                "hi": r["hi"],
            }
            for r in records
        ],
        "families": [
            {"q": r["q"], "hi": r["hi"], "row_z": r["row_z"], "partner_z": r["partner_z"]}
            for r in families
        ],
        "note": "detect_families pairing on S_med; no ranking yet",
    }
    try:
        written = save_figure(fig, "fig04d_pairing", numbers)
    except OSError:
        written = save_figure(fig, "fig04d_pairing_new", numbers)
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
