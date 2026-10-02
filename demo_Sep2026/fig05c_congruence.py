"""Figure 05c: diagonal veto, winner or none.

``python -m demo_Sep2026.fig05c_congruence``
"""

from __future__ import annotations

import argparse

import numpy as np

from batch_defringe.congruence import REL_OK

from .common import _percentile_limits, boxed_text, new_figure
from .fig05_common import LIVE_FRAME, SHUTTER_FRAME, _save, seed_for


def _best_by_name(hyps: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for hyp in hyps:
        name = hyp.get("name")
        if name not in {"fx", "fy", "tilted"}:
            continue
        if name not in out or float(hyp["score"]) < float(out[name]["score"]):
            out[name] = hyp
    return out


def _panel(fig, gs, raw, seed: dict, title: str):
    ax = fig.add_subplot(gs[0])
    lo, hi = _percentile_limits(raw)
    ax.imshow(raw, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)

    axb = fig.add_subplot(gs[1])
    names = ["fx", "fy", "tilted"]
    best = _best_by_name(list(seed.get("hypotheses") or []))
    scores = [float(best[n]["score"]) if n in best else np.nan for n in names]
    colors = [(0.80, 0.12, 0.12), (0.15, 0.35, 0.80), (0.85, 0.50, 0.08)]
    ypos = np.arange(len(names))[::-1]
    axb.barh(ypos, [0 if np.isnan(s) else s for s in scores], color=colors, height=0.55)
    axb.axvline(REL_OK, color="0.2", ls="--", lw=1.0, label=f"REL_OK = {REL_OK}")
    axb.set_yticks(ypos)
    axb.set_yticklabels(names, fontsize=9)
    axb.set_xlabel("diagonal rel-err score (lower is better)")
    win = seed.get("winner") or "none"
    axb.set_title(f"winner = {win}", fontsize=11)
    axb.set_xlim(0, max(0.4, max([s for s in scores if np.isfinite(s)] or [0.4]) * 1.15))
    axb.legend(fontsize=7, loc="lower right", frameon=False)
    for y, s in zip(ypos, scores):
        if np.isfinite(s):
            axb.text(s + 0.01, y, f"{s:.3f}", va="center", fontsize=8, color="0.25")
    return best


def draw() -> dict:
    raw_l, seed_l, _ = seed_for(LIVE_FRAME)
    raw_s, seed_s, _ = seed_for(SHUTTER_FRAME)

    fig = new_figure()
    fig.suptitle(
        "Figure 05c  ·  Congruence: one (qy, qx) or none  ·  ChanA",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    gs = fig.add_gridspec(1, 4, left=0.05, right=0.98, top=0.86, bottom=0.38, wspace=0.28)
    best_l = _panel(fig, [gs[0], gs[1]], raw_l, seed_l, f"Live frame {LIVE_FRAME}")
    best_s = _panel(fig, [gs[2], gs[3]], raw_s, seed_s, f"Shutter frame {SHUTTER_FRAME}")

    boxed_text(
        fig,
        [0.05, 0.02, 0.90, 0.26],
        (
            "Each H q proposes fx (qy=0). Each V q proposes fy (qx=0). "
            "Each (H, V) pair proposes tilted. For a hypothesis, predict the two "
            "diagonal periods from (qy, qx) and score the mean relative error "
            "against the measured main and anti periods. Lowest score wins, "
            f"but only if that score is at most {REL_OK}. Otherwise none."
            "\n"
            f"Frame {LIVE_FRAME}: winner = {seed_l.get('winner')}, "
            f"qy = {seed_l.get('qy')}, qx = {seed_l.get('qx')}, "
            f"score = {seed_l.get('score')}. "
            f"Frame {SHUTTER_FRAME}: winner = {seed_s.get('winner') or 'none'}. "
            "None skips linescan centre and edges; catalog and shutter-learn still run."
        ),
        fontsize=8.5,
        color="0.3",
    )
    return _save(
        fig,
        "fig05c_congruence",
        {
            "rel_ok": REL_OK,
            "live": {
                "frame": LIVE_FRAME,
                "winner": seed_l.get("winner"),
                "qy": seed_l.get("qy"),
                "qx": seed_l.get("qx"),
                "score": seed_l.get("score"),
                "best": {k: {"score": v["score"], "qy": v.get("qy"), "qx": v.get("qx")} for k, v in best_l.items()},
            },
            "shutter": {
                "frame": SHUTTER_FRAME,
                "winner": seed_s.get("winner"),
                "qy": seed_s.get("qy"),
                "qx": seed_s.get("qx"),
                "score": seed_s.get("score"),
                "best": {k: {"score": v["score"], "qy": v.get("qy"), "qx": v.get("qx")} for k, v in best_s.items()},
            },
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
