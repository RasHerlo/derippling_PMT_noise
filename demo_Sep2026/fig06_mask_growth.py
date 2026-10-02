"""Figure 06: one mask, grown by pixelwise max, then a shared alpha raise.

``python -m demo_Sep2026.fig06_mask_growth``
"""

from __future__ import annotations

import argparse

import numpy as np

from batch_defringe.process_v4 import ALPHA_LIVE, grow_frame

from .common import boxed_text, load_frame, new_figure
from .fig04g_shutter_mask import _learn
from .fig05_common import LIVE_FRAME, _save

FY_CROP = 40
FX_CROP = 90


def _pick_steps(steps: list[dict]) -> list[dict]:
    """First line, last core add, last leftover add, then the last alpha raise."""
    if not steps:
        return []
    a0 = float(ALPHA_LIVE[0])
    at_a0 = [s for s in steps if float(s["alpha"]) == a0]
    core = [s for s in at_a0 if not str(s["why"]).startswith("leftover")]
    leftover = [s for s in at_a0 if str(s["why"]).startswith("leftover")]
    raises = [s for s in steps if str(s["why"]).startswith("raise")]
    picked: list[dict] = []
    if core:
        picked.append(core[0])
        if len(core) > 1:
            picked.append(core[-1])
    if leftover:
        picked.append(leftover[-1])
    if raises:
        picked.append(raises[-1])
    elif steps and steps[-1] not in picked:
        picked.append(steps[-1])
    out: list[dict] = []
    seen: set[tuple] = set()
    for s in picked:
        key = (s["why"], round(float(s["alpha"]), 2), int(s["n_lines"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _panel_title(step: dict, index: int) -> str:
    why = str(step["why"])
    n = int(step["n_lines"])
    a = float(step["alpha"])
    lines = f"{n} line" + ("" if n == 1 else "s")
    if why.startswith("raise"):
        head = rf"raise $\alpha$ to {a:.2f}"
    elif why.startswith("leftover"):
        head = "after leftover"
    elif n == 1:
        head = "first line"
    else:
        head = "after shutter + linescan"
    return rf"{index + 1}.  {head}" + "\n" + rf"$\alpha={a:.2f}$,  {lines}"


def _crop_patch(mask: np.ndarray, cy: int, cx: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    h, w = mask.shape
    ya, yb = max(0, cy - FY_CROP), min(h, cy + FY_CROP + 1)
    xa, xb = max(0, cx - FX_CROP), min(w, cx + FX_CROP + 1)
    extent = (xa - cx - 0.5, xb - cx - 0.5, yb - cy - 0.5, ya - cy - 0.5)
    return mask[ya:yb, xa:xb], extent


def draw() -> dict:
    families, shutter_idxs = _learn()
    raw, _n = load_frame(LIVE_FRAME)
    rec = grow_frame(
        raw,
        role="live",
        catalog_families=None,
        shutter_families=families,
        keep_images=True,
    )
    steps = _pick_steps(list(rec.get("steps") or []))
    h, w = raw.shape
    cy, cx = h // 2, w // 2
    n_panel = max(1, len(steps))

    fig = new_figure()
    fig.suptitle(
        f"Figure 06  ·  Iterative mask  ·  ChanA frame {LIVE_FRAME}",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    gs = fig.add_gridspec(
        1,
        n_panel,
        left=0.05,
        right=0.93,
        top=0.88,
        bottom=0.40,
        wspace=0.18,
    )
    im = None
    shown: list[dict] = []
    for i, step in enumerate(steps):
        ax = fig.add_subplot(gs[i])
        applied = np.asarray(step["applied"], dtype=np.float64)
        patch, extent = _crop_patch(applied, cy, cx)
        im = ax.imshow(
            patch,
            cmap="inferno",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=extent,
            aspect="auto",
        )
        ax.set_xlabel(r"$f_x$")
        ax.set_ylabel(r"$f_y$" if i == 0 else "")
        ax.tick_params(labelsize=7)
        ax.set_title(_panel_title(step, i), fontsize=10)
        shown.append(
            {
                "why": step["why"],
                "alpha": float(step["alpha"]),
                "n_lines": int(step["n_lines"]),
                "n_bins": int(np.sum(applied > 1e-3)),
                "max": float(np.max(applied)),
            }
        )
    if im is not None:
        cax = fig.add_axes([0.945, 0.40, 0.012, 0.48])
        fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1], label="applied")

    n_kept = len(rec.get("steps") or [])
    n_undone = len(rec.get("undone") or [])
    a0, a1, a2 = ALPHA_LIVE
    boxed_text(
        fig,
        [0.05, 0.02, 0.90, 0.28],
        (
            "One mask, not a union of separate families. Each accepted line is a "
            r"weighted field $W$. At a bin the mask is $\max(M, \alpha W)$, not a sum: "
            "overlapping bins keep the larger weight."
            "\n"
            f"Ridge lines (shutter, leftover FFT) use pack_D local excess. Peak lines "
            r"(linescan centre / edges) use $\alpha$ times the blob weight. "
            f"Lines are tried in collect order at $\\alpha={a0:g}$. After the kept set "
            f"is fixed, $\\alpha$ is raised on that whole set ({a1:g}, then {a2:g}). "
            "If the increment makes removed look like cells, that step is undone. "
            f"This frame kept {n_kept} step{'s' if n_kept != 1 else ''}"
            f"{f' and undid {n_undone}' if n_undone else ''}. "
            f"Crop $|f_y|<{FY_CROP}$, $|f_x|<{FX_CROP}$; higher-$q$ shutter rows can "
            "sit outside. Catalog is first in collect order; this example starts at shutter."
        ),
        fontsize=8.5,
        color="0.3",
    )
    return _save(
        fig,
        "fig06_mask_growth",
        {
            "frame": LIVE_FRAME,
            "shutter_frames": shutter_idxs,
            "alpha_live": list(ALPHA_LIVE),
            "n_kept": n_kept,
            "n_undone": n_undone,
            "accepted": rec.get("accepted") or [],
            "panels": shown,
            "fy_crop": FY_CROP,
            "fx_crop": FX_CROP,
            "final_alpha": float(rec.get("max_alpha") or 0),
            "note": "pixelwise max; shared alpha; catalog omitted in this example",
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
