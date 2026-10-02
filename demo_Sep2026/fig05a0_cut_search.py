"""Figure 05a0: probe H rows and V columns scored against the diagonals.

Goes before 05a (the four cuts after the choice).

``python -m demo_Sep2026.fig05a0_cut_search``
"""

from __future__ import annotations

import argparse

import numpy as np

from batch_defringe.congruence import (
    REL_OK,
    _probe_indices,
    _strongest_index,
    choose_hypothesis,
    measure_four_cuts,
)

from .common import _percentile_limits, boxed_text, new_figure
from .fig05_common import LIVE_FRAME, _save, seed_for


def _fx_scores(meas: dict) -> list[dict]:
    h, w = meas["h"], meas["w"]
    p_main = meas["traces"]["main"]["period"]
    p_anti = meas["traces"]["anti"]["period"]
    out = []
    for hc in meas["h_cands"]:
        if hc.get("q") is None:
            continue
        picked = choose_hypothesis(h, w, qx_h=hc["q"], qy_v=None, p_main=p_main, p_anti=p_anti)
        fx = next(x for x in picked["hypotheses"] if x["name"] == "fx")
        out.append({"index": int(hc["index"]), "q": float(hc["q"]), "score": float(fx["score"])})
    out.sort(key=lambda d: d["score"])
    return out


def _fy_scores(meas: dict) -> list[dict]:
    h, w = meas["h"], meas["w"]
    p_main = meas["traces"]["main"]["period"]
    p_anti = meas["traces"]["anti"]["period"]
    out = []
    for vc in meas["v_cands"]:
        if vc.get("q") is None:
            continue
        picked = choose_hypothesis(h, w, qx_h=None, qy_v=vc["q"], p_main=p_main, p_anti=p_anti)
        fy = next(x for x in picked["hypotheses"] if x["name"] == "fy")
        out.append({"index": int(vc["index"]), "q": float(vc["q"]), "score": float(fy["score"])})
    out.sort(key=lambda d: d["score"])
    return out


def _bars(ax, rows: list[dict], *, xlabel: str, winner_idx: int | None, color) -> None:
    labels = [str(r["index"]) for r in rows]
    scores = [r["score"] for r in rows]
    ypos = np.arange(len(rows))[::-1]
    cols = []
    for r in rows:
        if winner_idx is not None and r["index"] == winner_idx:
            cols.append(color)
        else:
            cols.append((0.72, 0.72, 0.72))
    ax.barh(ypos, scores, color=cols, height=0.65, edgecolor="0.35", linewidth=0.3)
    ax.axvline(REL_OK, color="0.15", ls="--", lw=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel)
    xmax = max(REL_OK, max(scores, default=0.0)) * 1.18
    ax.set_xlim(0, max(0.3, xmax))
    ax.tick_params(axis="x", labelsize=7)


def draw() -> dict:
    raw, seed, _n = seed_for(LIVE_FRAME)
    arr = np.asarray(raw, dtype=np.float64)
    h, w = arr.shape
    meas = measure_four_cuts(arr)
    h_rows = _fx_scores(meas)
    v_cols = _fy_scores(meas)
    win_row = seed.get("cuts", {}).get("row") if seed.get("winner") == "fx" else None
    win_col = seed.get("cuts", {}).get("col") if seed.get("winner") == "fy" else None
    if seed.get("winner") == "tilted":
        win_row = seed.get("cuts", {}).get("row")
        win_col = seed.get("cuts", {}).get("col")
    strong_r = _strongest_index(arr, 1)
    strong_c = _strongest_index(arr, 0)
    probes_r = set(int(i) for i in _probe_indices(h))
    probes_c = set(int(i) for i in _probe_indices(w))
    lo, hi = _percentile_limits(raw)

    fig = new_figure()
    fig.suptitle(
        f"Figure 05a0  ·  Probe-cut search  ·  ChanA frame {LIVE_FRAME}",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    gs = fig.add_gridspec(1, 3, left=0.05, right=0.98, top=0.88, bottom=0.40, wspace=0.28, width_ratios=[1.15, 1.0, 1.0])

    ax = fig.add_subplot(gs[0])
    ax.imshow(raw, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    for r in meas["h_cands"]:
        idx = int(r["index"])
        lw = 1.8 if win_row is not None and idx == int(win_row) else 0.7
        ls = "--" if idx == strong_r and idx not in probes_r else "-"
        ax.axhline(idx, color=(0.80, 0.12, 0.12), lw=lw, ls=ls, alpha=0.95)
    for c in meas["v_cands"]:
        idx = int(c["index"])
        lw = 1.8 if win_col is not None and idx == int(win_col) else 0.7
        ls = "--" if idx == strong_c and idx not in probes_c else "-"
        ax.axvline(idx, color=(0.15, 0.35, 0.80), lw=lw, ls=ls, alpha=0.95)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("All probe H rows and V columns", fontsize=11)

    axh = fig.add_subplot(gs[1])
    _bars(axh, h_rows, xlabel="fx score vs diagonals", winner_idx=None if win_row is None else int(win_row), color=(0.80, 0.12, 0.12))
    axh.set_title("Each H row as fx", fontsize=11)
    axh.set_ylabel("row index")

    axv = fig.add_subplot(gs[2])
    _bars(axv, v_cols, xlabel="fy score vs diagonals", winner_idx=None if win_col is None else int(win_col), color=(0.15, 0.35, 0.80))
    axv.set_title("Each V column as fy", fontsize=11)
    axv.set_ylabel("column index")

    boxed_text(
        fig,
        [0.05, 0.02, 0.90, 0.28],
        (
            "Before 05a keeps four traces, congruence tries many cuts. "
            "Nine H rows and nine V columns are spaced through the inner 7/8 of the frame "
            "(outer n/16 skipped), plus the strongest-std row and column if they are not already in that set. "
            "Dashed = that extra strongest-std cut."
            "\n"
            "Each H row is scored as an fx hypothesis: its q must predict the two measured "
            "diagonal periods. Each V column is scored as fy. Lower score is better; "
            f"dashed line is REL_OK = {REL_OK}. Coloured bar = the cut that belongs to the "
            f"eventual winner ({seed.get('winner')}). Tilted (every H-V pair) is figure 05c. "
            "Figure 05a then shows only the kept H/V traces plus the two diagonals."
        ),
        fontsize=8.5,
        color="0.3",
    )
    return _save(
        fig,
        "fig05a0_cut_search",
        {
            "frame": LIVE_FRAME,
            "winner": seed.get("winner"),
            "win_row": win_row,
            "win_col": win_col,
            "strongest_row": strong_r,
            "strongest_col": strong_c,
            "probe_rows": sorted(probes_r),
            "probe_cols": sorted(probes_c),
            "h_fx_scores": h_rows,
            "v_fy_scores": v_cols,
        },
    )


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    numbers = draw()
    for p in numbers["written"]:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
