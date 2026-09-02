"""v3 schematic (repo) and per-stack inspection PDF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from .congruence import congruent_seed, seed_peak_mask
from .image_check import diagonal_sample
from .process_v3 import APPLY_SEARCH, KeptFamily, apply_frame
from .readout import Y_RADIUS, _percentile_limits, _signed_limit, fft_mask_image
from .shutter_detect import draw_shutter_page, format_shutter_span


def write_schematic_pdf(path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _box(ax, x, y, w, h, txt, *, fc="0.96"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008",
                facecolor=fc,
                edgecolor="0.28",
                linewidth=0.9,
            )
        )
        ax.text(x + 0.012, y + h - 0.012, txt, fontsize=7.6, va="top", ha="left", family="sans-serif")

    def _arrow(ax, x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=11,
                lw=0.9,
                color="0.35",
            )
        )

    fig = plt.figure(figsize=(11.69, 8.27))
    with PdfPages(path) as pdf:
        fig.clear()
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.06, 0.97, "v3 defringe pipeline", fontsize=16, fontweight="bold", va="top")
        ax.text(
            0.06,
            0.935,
            "Writes <channel>/defringe_v3/ only. Does not overwrite defringe_v22. Catalog is not appended.",
            fontsize=8,
            va="top",
            color="0.35",
        )
        _box(ax, 0.06, 0.82, 0.88, 0.09, "1. Raw stack  ·  ChanA_stk.tif / ChanB_stk.tif\nNever overwritten. All v3 products sit beside it in defringe_v3/.")
        _arrow(ax, 0.50, 0.82, 0.50, 0.79)
        _box(
            ax,
            0.06,
            0.68,
            0.88,
            0.11,
            "2. Shutter detect  —  FOV std cliff (not mean; PMT offset stays ~850/738 ADU)\n"
            "Quiet if std ≤ 0.40×p75(std) for ≥3 frames after a drop ≥0.35×live std.\n"
            "Logged on overview + families.json. Not a hard seed: image-test still vetoes shutter q on live.",
            fc="#f4f8ff",
        )
        _arrow(ax, 0.50, 0.68, 0.50, 0.655)
        _box(
            ax,
            0.06,
            0.50,
            0.42,
            0.15,
            "3a. FFT-domain proposals\n"
            "• Catalog A/B/C (q / fx support only)\n"
            "• Same-raster cache prior\n"
            "• Shutter-learn fy (incomplete)\n"
            "• fy-row detect on leftover",
            fc="#fff8f0",
        )
        _box(
            ax,
            0.52,
            0.50,
            0.42,
            0.15,
            "3b. Image-domain proposals\n"
            "• Congruence: H, V, both diagonals\n"
            "• One (qy, qx) or none\n"
            "• fx and fy stay separate cands\n"
            "• Never max(fy,fx) into one gate",
            fc="#f0fff4",
        )
        _arrow(ax, 0.27, 0.50, 0.27, 0.47)
        _arrow(ax, 0.73, 0.50, 0.73, 0.47)
        _box(
            ax,
            0.06,
            0.32,
            0.88,
            0.15,
            "4. Image-test leftover  ·  eval = shutter ∪ anchors 160 / 700 / 1061\n"
            "Order: catalog → shutter-learn → linescan → leftover FFT. Notch one candidate. Score removed\n"
            "(coverage, evenness, ridges). PASS → keep family, leftover that frame. FAIL → discard (PDF).\n"
            "Do not retune a rejected mask. Repeat ≤4 rounds. Two consecutive rejects stop.",
            fc="#fff4f4",
        )
        _arrow(ax, 0.50, 0.32, 0.50, 0.295)
        _box(
            ax,
            0.06,
            0.18,
            0.88,
            0.11,
            "5. Track kept families  ·  q walk ±10 bins / 50-frame block  ·  apply search ±2\n"
            "fx/fy support is the seed template and does not wander. Same-axis families cannot hop onto each other.",
        )
        _arrow(ax, 0.50, 0.18, 0.50, 0.155)
        _box(
            ax,
            0.06,
            0.04,
            0.88,
            0.11,
            "6. Union apply (pack_D) per frame  ·  one FFT → each family on its own axis → one IFFT\n"
            "Outputs: *_defringed_v3.tif  ·  *_removed_v3.tif  ·  per_frame.csv  ·  overview.pdf",
            fc="#f4fff8",
        )
        pdf.savefig(fig, dpi=140)

        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.96, "Leftover loop and apply  —  what is allowed to move", fontsize=14, fontweight="bold", va="top")
        body = (
            "Proposal uniqueness is (axis, q), not q alone. fy q=10 and fx q=10 can both be kept.\n"
            "A failed family is not reshaped. Recursion means: evaluate the next unused candidate on leftover.\n\n"
            "Keepers are union-applied. Gate and local column-excess scale how hard a family is notched;\n"
            "the applied heatmap can shrink inside the seed fx/fy support. q may drift a few bins from seed.\n\n"
            "Outputs under <channel>/defringe_v3/\n"
            "  *_defringed_v3.tif     cleaned stack (same dtype as raw)\n"
            "  *_removed_v3.tif       raw − cleaned (float32)\n"
            "  per_frame.csv          q / gate / drift / axis / source per family\n"
            "  families.json          shutter, leftover rounds, kept families, summary\n"
            "  mean_raw / mean_cleaned / mean_removed.tif\n"
            "  overview.pdf           shutter, metrics vs frame, discarded masks, inspect samples\n\n"
            "Inspect pages pick: shutter mid, anchors 160/700/1061, strongest-gate linescan vs FFT-seed\n"
            "frames, drifted-q frames (|q−q_seed|≥2 while gated), and strongest RMS. Each sample shows\n"
            "the four linescans + their mask contributions, FFT-family seed masks, the final applied mask,\n"
            "then original / predicted fringe / actual removed / cleaned."
        )
        fig.text(0.06, 0.90, body, fontsize=10, va="top", family="sans-serif")
        pdf.savefig(fig, dpi=140)
    plt.close(fig)
    return path


def _fx_seed_mask(h: int, w: int, family: dict, q: float) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    cx = w // 2
    yw = family.get("y_weight")
    if yw is None:
        yw = np.ones(h, dtype=np.float32)
    yw = np.asarray(yw, dtype=np.float32)
    if yw.size != h:
        yw = np.ones(h, dtype=np.float32)
    for sgn in (-1, +1):
        xc = cx + sgn * int(round(q))
        for off in range(-Y_RADIUS, Y_RADIUS + 1):
            x = xc + off
            if 0 <= x < w and abs(x - cx) >= 5:
                mask[:, x] = np.maximum(mask[:, x], yw)
    return mask


def _stamp_peaks(h: int, w: int, peaks: list[tuple[int, int]], radius: int = 4) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2
    sigma = max(0.6, radius / 1.5)
    for fy, fx in peaks:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                y = cy + fy + dy
                x = cx + fx + dx
                if not (0 <= y < h and 0 <= x < w) or (y == cy and x == cx):
                    continue
                wt = float(np.exp(-0.5 * (dy * dy + dx * dx) / (sigma * sigma)))
                if wt > mask[y, x]:
                    mask[y, x] = wt
    return mask


def _cut_contrib_mask(h: int, w: int, *, qx=None, qy=None, kind: str) -> np.ndarray:
    peaks: list[tuple[int, int]] = []
    if kind == "fx" and qx is not None:
        q = int(round(float(qx)))
        peaks = [(0, q), (0, -q)]
    elif kind == "fy" and qy is not None:
        q = int(round(float(qy)))
        peaks = [(q, 0), (-q, 0)]
    elif kind == "tilted" and qy is not None and qx is not None:
        iy, ix = int(round(float(qy))), int(round(float(qx)))
        peaks = [(fy, fx) for fy in (iy, -iy) for fx in (ix, -ix)]
    return _stamp_peaks(h, w, peaks)


def _plot_linescan(ax, frame: np.ndarray, seed: dict, name: str, title: str) -> None:
    arr = np.asarray(frame, dtype=np.float64)
    cuts = (seed or {}).get("cuts") or {}
    tr = (cuts.get("traces") or {}).get(name) or {}
    y = tr.get("smooth")
    if y is None:
        ax.set_title(title, fontsize=8)
        ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes, color="0.4")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    if name == "horizontal":
        row = cuts.get("row")
        raw = arr[int(row)] if row is not None else y
    elif name == "vertical":
        col = cuts.get("col")
        raw = arr[:, int(col)] if col is not None else y
    else:
        _, raw = diagonal_sample(arr, name)
    raw = np.asarray(raw, dtype=float)[:n]
    x = np.arange(n)
    ax.plot(x, raw, color="0.7", lw=0.5)
    marked = tr.get("marked")
    if marked is not None:
        marked = np.asarray(marked)
        if marked.dtype == bool and marked.size == n:
            ax.plot(x[marked], raw[marked], ".", color="0.1", ms=1.6)
    ax.plot(x, y, color="C1", lw=1.1)
    ax.set_xlim(0, n - 1)
    p = tr.get("period")
    q = tr.get("q")
    ptxt = "-" if p is None else f"{float(p):.1f}"
    qtxt = "-" if q is None else f"{float(q):.1f}"
    ax.set_title(f"{title}  P={ptxt}  q={qtxt}", fontsize=7.5)
    ax.tick_params(labelsize=6)


def _family_seed_mask(spec: KeptFamily, h: int, w: int, q: float) -> np.ndarray:
    if spec.axis == "fy":
        fam = dict(spec.family)
        fam["q"] = float(q)
        if fam.get("paired", True):
            fam["hi"] = float(h // 2) - float(q)
        return fft_mask_image((h, w), [fam])
    return _fx_seed_mask(h, w, spec.family, q)


def _pick_inspect(rows: list[dict], shutter: dict, kept: list[KeptFamily], n: int) -> list[dict]:
    chosen: list[dict] = []
    seen: set[int] = set()

    def add(idx: int, role: str) -> None:
        if idx in seen or not (0 <= idx < n):
            return
        seen.add(idx)
        row = next((r for r in rows if int(r["frame"]) == idx), None)
        chosen.append({"frame": idx, "role": role, "row": row})

    frames = shutter.get("frames") or []
    if frames:
        add(int(frames[len(frames) // 2]), "shutter")
    for a in (160, 700, 1061):
        add(int(a), "anchor")
    if rows and kept:
        for i, spec in enumerate(kept):
            key = f"family{i}_gate"
            ranked = sorted(rows, key=lambda r: float(r.get(key) or 0), reverse=True)
            if ranked and float(ranked[0].get(key) or 0) > 0:
                tag = "linescan" if "linescan" in spec.source else "fft_seed"
                add(int(ranked[0]["frame"]), f"{tag}_{spec.axis}")
            dkey = f"family{i}_drift"
            drifted = [
                r
                for r in rows
                if float(r.get(f"family{i}_gate") or 0) > 0 and float(r.get(dkey) or 0) >= 2.0
            ]
            if drifted:
                drifted.sort(key=lambda r: float(r.get(dkey) or 0), reverse=True)
                add(int(drifted[0]["frame"]), "drifted")
        strong = max(rows, key=lambda r: float(r.get("removed_rms") or 0))
        add(int(strong["frame"]), "strong")
    return chosen[:8]


def write_v3_report(
    path: Path,
    *,
    tif_path: Path,
    channel: str,
    computer: str,
    status: str,
    shutter: dict,
    kept: list,
    rounds: list[dict],
    discarded: list[dict],
    rows: list[dict],
    eval_idx: list[int],
    roles: dict,
    cleaned_tif: Path | None = None,
    removed_tif: Path | None = None,
    mean_raw: np.ndarray | None = None,
    mean_cleaned: np.ndarray | None = None,
    mean_removed: np.ndarray | None = None,
    summary: dict | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import TwoSlopeNorm

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int((summary or {}).get("n_frames") or (shutter.get("n_frames") or 0) or max((int(r["frame"]) for r in rows), default=-1) + 1)
    fig = plt.figure(figsize=(11.69, 8.27))

    def _imshow(ax, img, *, signed=False, title=""):
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if img is None:
            ax.set_facecolor("0.92")
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes, color="0.4")
            return
        arr = np.asarray(img, dtype=float)
        if signed:
            lim = _signed_limit(arr)
            ax.imshow(arr, cmap="gray", norm=TwoSlopeNorm(0.0, vmin=-lim, vmax=lim), interpolation="nearest")
        else:
            lo, hi = _percentile_limits(arr)
            ax.imshow(arr, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")

    with PdfPages(path) as pdf:
        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.97, f"v3 defringe  ·  {channel}  ·  {status}", fontsize=13, fontweight="bold", va="top")
        fig.text(0.06, 0.935, f"{tif_path}  ·  {computer}", fontsize=8, va="top", color="0.35")
        sm = summary or {}
        kept_line = ", ".join(
            f"{k.source}/{k.axis} q_seed={k.q_seed:.1f}" if isinstance(k, KeptFamily) else str(k)
            for k in kept
        ) or "(none)"
        lines = [
            f"shutter: {format_shutter_span(shutter)}",
            f"eval frames: {eval_idx}",
            f"kept: {kept_line}",
            f"active {100 * sm.get('frac_frames_any_active', 0):.1f}%   "
            f"median removed RMS {sm.get('median_removed_rms', 0):.3g}   "
            f"max {sm.get('max_removed_rms', 0):.3g}",
            "",
            "leftover rounds:",
        ]
        for rnd in rounds:
            lines.append(
                f"  r{rnd['round']} {rnd['source']}/{rnd['axis']} q={rnd['q']:.1f}  {rnd['verdict']}  {rnd['reason']}"
            )
        fig.text(0.06, 0.90, "\n".join(lines), fontsize=8, va="top", family="monospace")
        pdf.savefig(fig, dpi=140)

        draw_shutter_page(
            fig,
            title=f"Shutter detect  ·  {format_shutter_span(shutter)}",
            subtitle="Contrast cliff. Check the orange span against live vs shutter stills in the method PDF.",
            det=shutter,
        )
        pdf.savefig(fig, dpi=140)

        if rows and kept:
            fig.clear()
            fig.patch.set_facecolor("white")
            fig.text(0.06, 0.97, "Tracked q and gate across the stack", fontsize=13, fontweight="bold", va="top")
            fig.text(
                0.06,
                0.935,
                "q may walk ±10 bins per 50-frame block; apply is ±2. fx/fy support is the seed template.",
                fontsize=8,
                va="top",
                color="0.35",
            )
            frames = np.array([r["frame"] for r in rows], dtype=int)
            gs = fig.add_gridspec(3, 1, left=0.08, right=0.98, top=0.90, bottom=0.08, hspace=0.22)
            ax = fig.add_subplot(gs[0])
            ax.plot(frames, [r["removed_rms"] for r in rows], color="0.15", lw=0.8)
            ax.set_ylabel("removed RMS")
            ax.tick_params(labelbottom=False)
            ax.set_title("Cleaning heaviness", fontsize=9)
            axg = fig.add_subplot(gs[1], sharex=ax)
            axq = fig.add_subplot(gs[2], sharex=ax)
            for i, spec in enumerate(kept):
                axg.plot(frames, [r.get(f"family{i}_gate", 0) for r in rows], lw=0.8, label=f"{spec.source}/{spec.axis}")
                axq.plot(frames, [r.get(f"family{i}_q", spec.q_seed) for r in rows], lw=0.8, label=f"seed {spec.q_seed:.0f}")
            axg.set_ylabel("gate")
            axg.set_ylim(-0.05, 1.05)
            axg.legend(fontsize=7, loc="upper right", frameon=False)
            axq.set_ylabel("applied q")
            axq.set_xlabel("frame")
            axq.legend(fontsize=7, loc="upper right", frameon=False)
            pdf.savefig(fig, dpi=140)

        if mean_raw is not None:
            fig.clear()
            fig.patch.set_facecolor("white")
            fig.text(0.06, 0.97, "Stack means", fontsize=13, fontweight="bold", va="top")
            gs = fig.add_gridspec(1, 3, left=0.05, right=0.98, top=0.88, bottom=0.10, wspace=0.12)
            _imshow(fig.add_subplot(gs[0]), mean_raw, title="mean raw")
            _imshow(fig.add_subplot(gs[1]), mean_cleaned, title="mean cleaned")
            _imshow(fig.add_subplot(gs[2]), mean_removed, signed=True, title="mean removed")
            pdf.savefig(fig, dpi=140)

        for snap in discarded:
            fig.clear()
            fig.patch.set_facecolor("white")
            fig.text(
                0.06,
                0.97,
                f"DISCARDED  ·  frame {snap['frame']}  ·  {snap['source']}/{snap['axis']}  "
                f"q {snap['q_proposed']:.0f}→{snap['q']:.0f}  ·  {snap['verdict']}",
                fontsize=12,
                fontweight="bold",
                va="top",
            )
            fig.text(
                0.06,
                0.935,
                "Image-test rejected this notch. It is not in the union mask. Removed should look like cells or junk.",
                fontsize=8,
                va="top",
                color="0.35",
            )
            gs = fig.add_gridspec(2, 3, left=0.04, right=0.98, top=0.90, bottom=0.06, hspace=0.18, wspace=0.10)
            _imshow(fig.add_subplot(gs[0, 0]), snap.get("raw"), title="original")
            _imshow(fig.add_subplot(gs[0, 1]), snap.get("applied"), title="trial applied mask")
            _imshow(fig.add_subplot(gs[0, 2]), snap.get("removed"), signed=True, title="trial removed")
            _imshow(fig.add_subplot(gs[1, 0]), snap.get("cleaned"), title="trial cleaned (not kept)")
            ax = fig.add_subplot(gs[1, 1:])
            ax.axis("off")
            ax.text(0.0, 1.0, f"rms={snap['removed_rms']:.3g}  gate={snap['gate']:.2f}", va="top", family="monospace")
            pdf.savefig(fig, dpi=140)

        kept_specs = [k for k in kept if isinstance(k, KeptFamily)]
        inspect = _pick_inspect(rows, shutter, kept_specs, n)
        if inspect and cleaned_tif and removed_tif and Path(tif_path).is_file():
            with (
                tifffile.TiffFile(tif_path) as raw_tf,
                tifffile.TiffFile(cleaned_tif) as cl_tf,
                tifffile.TiffFile(removed_tif) as rm_tf,
            ):
                n_pages = min(len(raw_tf.pages), len(cl_tf.pages), len(rm_tf.pages))
                for spec in inspect:
                    fi = int(spec["frame"])
                    if fi >= n_pages:
                        continue
                    raw = np.asarray(raw_tf.pages[fi].asarray())
                    cleaned = np.asarray(cl_tf.pages[fi].asarray())
                    removed = np.asarray(rm_tf.pages[fi].asarray())
                    h, w = raw.shape[:2]
                    rec = None
                    q_preds: list[float] = []
                    if kept_specs:
                        row = spec.get("row") or {}
                        q_preds = [
                            float(row.get(f"family{i}_q_pred") or k.q_seed)
                            for i, k in enumerate(kept_specs)
                        ]
                        rec = apply_frame(raw, kept_specs, q_preds, search=APPLY_SEARCH)
                    seed = congruent_seed(raw)
                    measured = seed.get("measured") or {}
                    ls_mask = seed_peak_mask(h, w, seed)
                    h_mask = _cut_contrib_mask(h, w, qx=measured.get("qx_from_H"), kind="fx")
                    v_mask = _cut_contrib_mask(h, w, qy=measured.get("qy_from_V"), kind="fy")
                    fft_masks = [
                        _family_seed_mask(k, h, w, (q_preds[i] if q_preds else k.q_seed))
                        for i, k in enumerate(kept_specs)
                    ]
                    fft_combo = np.zeros((h, w), dtype=np.float32)
                    for m in fft_masks:
                        fft_combo = np.maximum(fft_combo, m)

                    src = ""
                    if spec.get("row") and kept_specs:
                        bits = []
                        for i, k in enumerate(kept_specs):
                            bits.append(
                                f"{k.source}/{k.axis} q {spec['row'].get(f'family{i}_q_seed', k.q_seed):.0f}→"
                                f"{spec['row'].get(f'family{i}_q', 0):.0f} "
                                f"gate={spec['row'].get(f'family{i}_gate', 0):.2f}"
                            )
                        src = "  ·  ".join(bits)
                    win = seed.get("winner") or "none"

                    fig.clear()
                    fig.patch.set_facecolor("white")
                    fig.text(
                        0.06,
                        0.975,
                        f"Inspect seeds  ·  frame {fi}  ·  {spec['role']}",
                        fontsize=12,
                        fontweight="bold",
                        va="top",
                    )
                    fig.text(0.06, 0.945, src or "(no kept families)", fontsize=7.5, va="top", family="monospace", color="0.3")
                    fig.text(
                        0.06,
                        0.922,
                        f"linescan congruence: {win}  qy={seed.get('qy')}  qx={seed.get('qx')}  "
                        f"score={seed.get('score')}   ·   orange = rloess, dots = rolling lowest-4",
                        fontsize=8,
                        va="top",
                        color="0.35",
                    )
                    gs = fig.add_gridspec(3, 4, left=0.04, right=0.99, top=0.90, bottom=0.04, hspace=0.28, wspace=0.12)
                    _plot_linescan(fig.add_subplot(gs[0, 0]), raw, seed, "horizontal", "H linescan → qx")
                    _plot_linescan(fig.add_subplot(gs[0, 1]), raw, seed, "vertical", "V linescan → qy")
                    _plot_linescan(fig.add_subplot(gs[0, 2]), raw, seed, "main", "TL–BR diagonal")
                    _plot_linescan(fig.add_subplot(gs[0, 3]), raw, seed, "anti", "TR–BL diagonal")
                    _imshow(fig.add_subplot(gs[1, 0]), h_mask, title="H contribution (fx peaks)")
                    _imshow(fig.add_subplot(gs[1, 1]), v_mask, title="V contribution (fy peaks)")
                    _imshow(fig.add_subplot(gs[1, 2]), ls_mask, title="congruent winner mask")
                    fam0 = fft_masks[0] if fft_masks else None
                    _imshow(
                        fig.add_subplot(gs[1, 3]),
                        fam0,
                        title=(
                            f"FFT family0 {kept_specs[0].source}/{kept_specs[0].axis}"
                            if kept_specs
                            else "FFT family0"
                        ),
                    )
                    fam1 = fft_masks[1] if len(fft_masks) > 1 else None
                    _imshow(
                        fig.add_subplot(gs[2, 0]),
                        fam1,
                        title=(
                            f"FFT family1 {kept_specs[1].source}/{kept_specs[1].axis}"
                            if len(kept_specs) > 1
                            else "FFT family1 (none)"
                        ),
                    )
                    _imshow(fig.add_subplot(gs[2, 1]), fft_combo, title="FFT-family union seed")
                    _imshow(fig.add_subplot(gs[2, 2]), None if rec is None else rec["applied"], title="final applied mask")
                    _imshow(
                        fig.add_subplot(gs[2, 3]),
                        None if rec is None else rec["predicted"],
                        signed=True,
                        title="predicted fringe (mask IFFT)",
                    )
                    pdf.savefig(fig, dpi=140)

                    fig.clear()
                    fig.patch.set_facecolor("white")
                    fig.text(
                        0.06,
                        0.97,
                        f"Inspect result  ·  frame {fi}  ·  {spec['role']}",
                        fontsize=12,
                        fontweight="bold",
                        va="top",
                    )
                    fig.text(
                        0.06,
                        0.935,
                        "Predicted = IFFT of attenuated FFT energy. Actual removed = raw − cleaned.",
                        fontsize=8,
                        va="top",
                        color="0.35",
                    )
                    gs = fig.add_gridspec(1, 4, left=0.03, right=0.99, top=0.88, bottom=0.08, wspace=0.08)
                    _imshow(fig.add_subplot(gs[0]), raw, title="original")
                    _imshow(
                        fig.add_subplot(gs[1]),
                        None if rec is None else rec["predicted"],
                        signed=True,
                        title="predicted fringe",
                    )
                    _imshow(fig.add_subplot(gs[2]), removed, signed=True, title="actual removed")
                    _imshow(fig.add_subplot(gs[3]), cleaned, title="cleaned")
                    pdf.savefig(fig, dpi=140)

    plt.close(fig)
    return path
