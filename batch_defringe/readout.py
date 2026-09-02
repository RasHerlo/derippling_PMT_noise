"""User-facing v2.2 defringe readout: removed stack, mask/metrics, overview PDF.

The filter is an FFT ridge-segment notch, not a spatial pixel mask. That is why
the readout stores:

- the spatial remainder ``raw - cleaned`` as a stack (what was taken out)
- the seed ridge support as a 2-D FFT-domain mask plus ``families.json``
- per-frame numbers (how heavily each family was applied)

Everything is written under ``<channel>/defringe_v22/`` next to the raw stack.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .library import format_catalog_line
from .seed import EVAL_ANCHOR_FRAMES as INSPECTION_ANCHOR_FRAMES

OUTPUT_DIRNAME = "defringe_v22"
Y_RADIUS = 2  # matches clean_frame_v21 default ridge half-width in fy
FRAMES_PER_PAGE = 3
N_STRONG = 2
N_WEAK = 2


def output_dir_for(tif_path: Path) -> Path:
    return Path(tif_path).parent / OUTPUT_DIRNAME


def cleaned_path_for(tif_path: Path) -> Path:
    return output_dir_for(tif_path) / f"{Path(tif_path).stem}_defringed_v22.tif"


def removed_path_for(tif_path: Path, out_dir: Path | None = None) -> Path:
    folder = Path(out_dir) if out_dir is not None else output_dir_for(tif_path)
    return folder / f"{Path(tif_path).stem}_removed_v22.tif"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if not np.isfinite(x):
            return None
        return x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def family_public(fam: dict) -> dict:
    """JSON-safe family record without bulky diagnostic arrays."""
    return {
        "q": float(fam["q"]),
        "hi": None if fam.get("hi") is None else float(fam["hi"]),
        "paired": bool(fam.get("paired", True)),
        "row_score": float(fam.get("row_score", 0.0)),
        "pair_score": None
        if fam.get("pair_score") is None
        else float(fam["pair_score"]),
        "fx_ranges": _jsonable(fam.get("fx_ranges", [])),
        "x_weight": np.asarray(fam["x_weight"], dtype=float).tolist()
        if fam.get("x_weight") is not None
        else None,
    }


def fft_mask_image(shape_hw: tuple[int, int], families: list[dict]) -> np.ndarray:
    """2-D FFT-domain support of seeded families (0–1), same size as one frame."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.float32)
    cy = h // 2
    for fam in families:
        xw = fam.get("x_weight")
        if xw is None:
            continue
        xw = np.asarray(xw, dtype=np.float32)
        if xw.shape[0] != w:
            continue
        qs = [float(fam["q"])]
        if fam.get("hi") is not None:
            qs.append(float(fam["hi"]))
        for d in qs:
            for sgn in (-1, +1):
                yc = cy + sgn * int(round(d))
                for off in range(-Y_RADIUS, Y_RADIUS + 1):
                    y = yc + off
                    if 0 <= y < h:
                        mask[y, :] = np.maximum(mask[y, :], xw)
    return mask


def per_frame_fieldnames(n_families: int) -> list[str]:
    fields = [
        "frame",
        "n_active_families",
        "n_residual_passes",
        "removed_rms",
        "removed_mean_abs",
        "removed_p99_abs",
        "max_gate",
        "max_eff_max_alpha",
    ]
    for i in range(n_families):
        fields += [
            f"family{i}_q",
            f"family{i}_strength",
            f"family{i}_gate",
            f"family{i}_eff_max_alpha",
            f"family{i}_residual_pass",
            f"family{i}_residual_strength",
            f"family{i}_active",
        ]
    return fields


def tracking_row(frame_index: int, removed: np.ndarray, tracking: list[dict]) -> dict:
    rem = np.asarray(removed, dtype=np.float64)
    abs_rem = np.abs(rem)
    gates = [float(t.get("gate", 0.0)) for t in tracking]
    alphas = [float(t.get("eff_max_alpha", 0.0)) for t in tracking]
    n_active = int(sum(g > 0.0 for g in gates))
    n_resid = int(sum(int(t.get("residual_pass", 0)) for t in tracking))
    row = {
        "frame": int(frame_index),
        "n_active_families": n_active,
        "n_residual_passes": n_resid,
        "removed_rms": float(np.sqrt(np.mean(rem * rem))),
        "removed_mean_abs": float(np.mean(abs_rem)),
        "removed_p99_abs": float(np.percentile(abs_rem, 99.0)),
        "max_gate": float(max(gates) if gates else 0.0),
        "max_eff_max_alpha": float(max(alphas) if alphas else 0.0),
    }
    for i, t in enumerate(tracking):
        gate = float(t.get("gate", 0.0))
        row[f"family{i}_q"] = float(t["q"])
        row[f"family{i}_strength"] = float(t.get("strength", 0.0))
        row[f"family{i}_gate"] = gate
        row[f"family{i}_eff_max_alpha"] = float(t.get("eff_max_alpha", 0.0))
        row[f"family{i}_residual_pass"] = int(t.get("residual_pass", 0))
        row[f"family{i}_residual_strength"] = float(t.get("residual_strength", 0.0))
        row[f"family{i}_active"] = int(gate > 0.0)
    return row


def write_mean_tif(path: Path, mean: np.ndarray) -> None:
    tifffile.imwrite(path, np.asarray(mean, dtype=np.float32), photometric="minisblack")


def write_families_json(
    path: Path,
    *,
    families: list[dict],
    params: dict,
    seed_info: dict,
    n_frames: int,
    shape: tuple[int, ...],
    source_tif: Path,
    computer: str,
    channel: str,
    fingerprint: dict,
    qc: str,
    used_prior: bool,
    reseeded: bool,
    rows: list[dict],
    status: str = "ok",
    prior_branch: str | None = None,
    ladder: list[dict] | None = None,
    failure_message: str | None = None,
    catalog: dict | None = None,
) -> dict:
    rms = np.array([r["removed_rms"] for r in rows], dtype=float) if rows else np.zeros(0)
    n_active = (
        np.array([r["n_active_families"] for r in rows], dtype=float) if rows else np.zeros(0)
    )
    summary = {
        "n_frames": int(n_frames),
        "n_families": len(families),
        "frac_frames_any_active": float(np.mean(n_active > 0)) if len(n_active) else 0.0,
        "median_removed_rms": float(np.median(rms)) if len(rms) else 0.0,
        "max_removed_rms": float(np.max(rms)) if len(rms) else 0.0,
        "median_n_active_families": float(np.median(n_active)) if len(n_active) else 0.0,
        "frac_residual_pass": float(
            np.mean([r["n_residual_passes"] > 0 for r in rows]) if rows else 0.0
        ),
    }
    payload = {
        "version": "v2.2",
        "config_id": "pack_D",
        "note": (
            "Mask is FFT ridge-segment support (q, fx_ranges, x_weight), not a "
            "spatial pixel mask. Per-frame application is per_frame.csv. The "
            "spatial remainder is *_removed_v22.tif (raw − cleaned)."
        ),
        "source_tif": str(source_tif),
        "computer": computer,
        "channel": channel,
        "fingerprint": _jsonable(fingerprint),
        "shape": [int(x) for x in shape],
        "params": _jsonable(params),
        "seed": _jsonable(seed_info),
        "qc": qc,
        "used_prior": bool(used_prior),
        "reseeded": bool(reseeded),
        "status": status,
        "prior_branch": prior_branch,
        "catalog": _jsonable(catalog if catalog is not None else seed_info.get("catalog")),
        "ladder": _jsonable(ladder) if ladder else None,
        "failure_message": failure_message,
        "families": [family_public(f) for f in families],
        "summary": summary,
        "shutter": _jsonable((seed_info or {}).get("shutter")),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary


def _percentile_limits(img: np.ndarray, p=(1.0, 99.5)) -> tuple[float, float]:
    lo, hi = np.percentile(img, p)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(img)), float(np.max(img))
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def _signed_limit(img: np.ndarray, p: float = 99.5) -> float:
    mag = float(np.percentile(np.abs(img), p))
    if not np.isfinite(mag) or mag <= 0:
        mag = float(np.max(np.abs(img)))
    if mag <= 0:
        mag = 1.0
    return mag


def _qs_from_row(row: dict) -> list[float]:
    qs: list[float] = []
    i = 0
    while f"family{i}_q" in row:
        val = row.get(f"family{i}_q")
        if val is not None and val != "":
            qs.append(float(val))
        i += 1
    return qs


def choose_inspection_frames(
    rows: list[dict],
    *,
    n_frames: int,
    anchors: tuple[int, ...] = INSPECTION_ANCHOR_FRAMES,
    n_strong: int = N_STRONG,
    n_weak: int = N_WEAK,
) -> list[dict]:
    """Pick anchors plus strongest/weakest removal for the overview PDF."""
    if not rows or n_frames <= 0:
        return []
    by_frame = {int(r["frame"]): r for r in rows}
    chosen: list[dict] = []
    seen: set[int] = set()

    def _add(idx: int, role: str) -> None:
        if idx in seen or not (0 <= idx < n_frames):
            return
        row = by_frame.get(idx)
        if row is None:
            return
        seen.add(idx)
        chosen.append(
            {
                "frame": idx,
                "role": role,
                "removed_rms": float(row.get("removed_rms", 0.0) or 0.0),
                "max_gate": float(row.get("max_gate", 0.0) or 0.0),
            }
        )

    for a in anchors:
        _add(int(a), "anchor")

    ranked = sorted(rows, key=lambda r: float(r.get("removed_rms", 0.0) or 0.0), reverse=True)
    for rec in ranked:
        if sum(1 for c in chosen if c["role"] == "strong") >= n_strong:
            break
        _add(int(rec["frame"]), "strong")
    for rec in reversed(ranked):
        if sum(1 for c in chosen if c["role"] == "weak") >= n_weak:
            break
        _add(int(rec["frame"]), "weak")

    order = {"anchor": 0, "strong": 1, "weak": 2}
    chosen.sort(key=lambda c: (order.get(c["role"], 9), c["frame"]))
    return chosen


def load_inspection_triples(
    *,
    raw_tif: Path,
    cleaned_tif: Path,
    removed_tif: Path,
    chosen: list[dict],
    rows: list[dict] | None = None,
) -> list[dict]:
    """Load original | cleaned | removed for the chosen frame indices."""
    if not chosen:
        return []
    by_frame = {int(r["frame"]): r for r in (rows or [])}
    triples: list[dict] = []
    with (
        tifffile.TiffFile(raw_tif) as raw_tf,
        tifffile.TiffFile(cleaned_tif) as clean_tf,
        tifffile.TiffFile(removed_tif) as rem_tf,
    ):
        n = min(len(raw_tf.pages), len(clean_tf.pages), len(rem_tf.pages))
        for spec in chosen:
            idx = int(spec["frame"])
            if not (0 <= idx < n):
                continue
            removed = np.asarray(rem_tf.pages[idx].asarray())
            row = by_frame.get(idx, {})
            rms = spec.get("removed_rms")
            if rms is None:
                rms = float(np.sqrt(np.mean(np.asarray(removed, dtype=np.float64) ** 2)))
            triples.append(
                {
                    "frame": idx,
                    "role": spec.get("role", ""),
                    "raw": np.asarray(raw_tf.pages[idx].asarray()),
                    "cleaned": np.asarray(clean_tf.pages[idx].asarray()),
                    "removed": removed,
                    "qs": _qs_from_row(row),
                    "removed_rms": float(rms),
                    "max_gate": float(spec.get("max_gate", row.get("max_gate", 0.0)) or 0.0),
                }
            )
    return triples


def draw_frame_inspection_page(
    fig,
    *,
    title: str,
    subtitle: str,
    triples: list[dict],
    cleaned_label: str = "cleaned",
) -> None:
    """One PDF page: original | cleaned | removed for up to FRAMES_PER_PAGE frames."""
    fig.clear()
    fig.patch.set_facecolor("white")
    n = max(len(triples), 1)
    fig.text(0.06, 0.97, title, fontsize=12, fontweight="bold", va="top")
    fig.text(0.06, 0.935, subtitle, fontsize=8, va="top", color="0.25")

    gs = fig.add_gridspec(
        n,
        3,
        left=0.06,
        right=0.98,
        top=0.88,
        bottom=0.05,
        hspace=0.28,
        wspace=0.12,
    )
    for i, trip in enumerate(triples):
        raw = np.asarray(trip["raw"])
        cleaned = np.asarray(trip["cleaned"])
        removed = np.asarray(trip["removed"])
        vmin, vmax = _percentile_limits(raw)
        rlim = _signed_limit(removed)
        rms = trip.get("removed_rms")
        if rms is None:
            rms = float(np.sqrt(np.mean(np.asarray(removed, dtype=np.float64) ** 2)))
        labels = (
            f"frame {trip['frame']}  ({trip['role']})  original",
            cleaned_label,
            f"removed  rms={float(rms):.3g}",
        )
        images = (raw, cleaned, removed)
        cmaps = ("gray", "gray", "RdBu_r")
        limits = ((vmin, vmax), (vmin, vmax), (-rlim, rlim))
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(images[j], cmap=cmaps[j], vmin=limits[j][0], vmax=limits[j][1], interpolation="nearest")
            ax.set_title(labels[j], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                extra = trip.get("qs") or []
                if extra:
                    ax.set_ylabel("q=" + ",".join(f"{q:.0f}" for q in extra), fontsize=7)


def write_overview_pdf(
    path: Path,
    *,
    title: str,
    subtitle: str,
    mean_raw: np.ndarray,
    mean_cleaned: np.ndarray,
    mean_removed: np.ndarray,
    medspec: np.ndarray | None,
    families: list[dict],
    rows: list[dict],
    summary: dict,
    status: str = "ok",
    ladder: list[dict] | None = None,
    failure_message: str | None = None,
    inspection_frames: list[dict] | None = None,
    example_triples: list[dict] | None = None,
    cleaned_label: str = "cleaned",
    catalog: dict | None = None,
    shutter: dict | None = None,
) -> None:
    """Summary page plus optional original | cleaned | removed inspection pages."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    mean_raw = np.asarray(mean_raw)
    mean_cleaned = np.asarray(mean_cleaned)
    mean_removed = np.asarray(mean_removed)
    vmin, vmax = _percentile_limits(mean_raw)
    rlim = _signed_limit(mean_removed)

    n_fam = len(families)
    frames = np.array([r["frame"] for r in rows], dtype=int) if rows else np.arange(0)
    rms = np.array([r["removed_rms"] for r in rows], dtype=float) if rows else np.zeros(0)
    gates = np.zeros((len(rows), max(n_fam, 1)))
    qs = np.zeros((len(rows), max(n_fam, 1)))
    for j in range(n_fam):
        gates[:, j] = [r.get(f"family{j}_gate", 0.0) for r in rows]
        qs[:, j] = [r.get(f"family{j}_q", 0.0) for r in rows]

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.05, 1.15],
        hspace=0.32,
        wspace=0.22,
        left=0.055,
        right=0.98,
        top=0.78 if inspection_frames else 0.82,
        bottom=0.07,
    )

    fig.text(0.055, 0.97, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.055, 0.935, subtitle, fontsize=8.5, va="top", color="0.25")
    fig.text(
        0.055,
        0.905,
        format_catalog_line(catalog),
        fontsize=8,
        va="top",
        color="0.2",
    )

    if status != "ok":
        fail = failure_message or status
        fig.text(0.055, 0.875, f"STATUS: {status}  —  {fail}", fontsize=8, va="top", color="0.6")
    else:
        stats = (
            f"frames {summary.get('n_frames', len(rows))}   "
            f"families {summary.get('n_families', n_fam)}   "
            f"active on {100 * summary.get('frac_frames_any_active', 0):.1f}% of frames   "
            f"median removed RMS {summary.get('median_removed_rms', 0):.3g}   "
            f"max {summary.get('max_removed_rms', 0):.3g}   "
            f"residual pass {100 * summary.get('frac_residual_pass', 0):.1f}%"
        )
        fig.text(0.055, 0.875, stats, fontsize=8, va="top")
        if inspection_frames:
            shown = ",  ".join(
                f"{c['frame']} ({c['role']})" for c in inspection_frames
            )
            fig.text(
                0.055,
                0.850,
                f"inspection frames (later pages): {shown}",
                fontsize=7.5,
                va="top",
                color="0.35",
            )

    panels = [
        (gs[0, 0], mean_raw, "Mean before (raw)", "gray", vmin, vmax),
        (gs[0, 1], mean_cleaned, "Mean after (cleaned)" if status == "ok" else "Not cleaned (needs review)", "gray", vmin, vmax),
        (gs[0, 2], mean_removed, "Mean removed (raw − cleaned)" if status == "ok" else "No remainder (not applied)", "RdBu_r", -rlim, rlim),
    ]
    for slot, img, label, cmap, lo, hi in panels:
        ax = fig.add_subplot(slot)
        ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_spec = fig.add_subplot(gs[1, 0])
    if medspec is not None:
        spec = np.asarray(medspec)
        slo, shi = _percentile_limits(spec, (3.0, 99.7))
        ax_spec.imshow(spec, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest")
        h, w = spec.shape
        cy, cx = h // 2, w // 2
        ax_spec.set_xlim(-0.5, w - 0.5)
        ax_spec.set_ylim(h - 0.5, -0.5)
        for i, fam in enumerate(families):
            color = f"C{i}"
            ranges = fam.get("fx_ranges") or []
            qs_draw = [float(fam["q"])]
            if fam.get("hi") is not None:
                qs_draw.append(float(fam["hi"]))
            for d in qs_draw:
                for sgn in (-1, +1):
                    y = cy + sgn * int(round(d))
                    if not (0 <= y < h):
                        continue
                    ax_spec.axhline(y, color=color, lw=0.6, alpha=0.85)
                    for pair in ranges:
                        if not pair:
                            continue
                        lo_fx, hi_fx = float(pair[0]), float(pair[1])
                        ax_spec.add_patch(
                            Rectangle(
                                (cx + lo_fx, y - Y_RADIUS - 0.5),
                                hi_fx - lo_fx + 1,
                                2 * Y_RADIUS + 1,
                                linewidth=0.6,
                                edgecolor=color,
                                facecolor=color,
                                alpha=0.25,
                            )
                        )
        ax_spec.set_title("FFT mask on seed spectrum", fontsize=9)
    else:
        ax_spec.text(0.5, 0.5, "No seed spectrum", ha="center", va="center")
        ax_spec.set_axis_off()
    ax_spec.set_xticks([])
    ax_spec.set_yticks([])

    inner = gs[1, 1:].subgridspec(2, 1, hspace=0.12)
    ax_tr = fig.add_subplot(inner[0])
    ax_q = fig.add_subplot(inner[1], sharex=ax_tr)
    ax_tr.set_title("Cleaning heaviness and tracked q across frames", fontsize=9)
    if status != "ok" and not len(frames):
        ax_tr.axis("off")
        ax_q.axis("off")
        lines = ["Inspect-only ladder (not applied):"]
        for rung in ladder or []:
            name = rung.get("rung", "?")
            fams = rung.get("families") or []
            qs = ", ".join(f"q={f.get('q')}" for f in fams) if fams else "(none)"
            lines.append(f"  {name}: {qs}")
        ax_tr.text(
            0.02,
            0.98,
            "\n".join(lines) or "No ladder candidates",
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            transform=ax_tr.transAxes,
        )
    elif len(frames):
        ax_tr.plot(frames, rms, color="0.15", lw=0.9, label="removed RMS")
        mark = {"anchor": "C3", "strong": "C1", "weak": "C0"}
        for spec in inspection_frames or []:
            ax_tr.axvline(
                spec["frame"],
                color=mark.get(spec.get("role", ""), "0.5"),
                lw=0.8,
                alpha=0.85,
            )
        ax_g = ax_tr.twinx()
        for j in range(n_fam):
            ax_g.plot(
                frames,
                gates[:, j],
                lw=0.8,
                alpha=0.85,
                color=f"C{j}",
                label=f"fam{j} gate",
            )
        ax_g.set_ylim(-0.05, 1.05)
        ax_g.set_ylabel("gate")
        ax_tr.set_ylabel("removed RMS")
        ax_tr.tick_params(labelbottom=False)
        h1, l1 = ax_tr.get_legend_handles_labels()
        h2, l2 = ax_g.get_legend_handles_labels()
        ax_tr.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right", frameon=False)

        for j in range(n_fam):
            ax_q.plot(
                frames,
                qs[:, j],
                lw=0.9,
                color=f"C{j}",
                label=f"fam{j} q (seed {families[j]['q']:.0f})",
            )
        ax_q.set_ylabel("tracked q  (±10 / 50-fr block; apply ±2)")
        ax_q.set_xlabel("frame  ·  fx support is the seed template; gate/q walk per frame")
        ax_q.legend(fontsize=7, loc="upper right", frameon=False)
    else:
        ax_tr.set_axis_off()
        ax_q.set_axis_off()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=130)
    with PdfPages(path) as pdf:
        pdf.savefig(fig, dpi=150)
        if shutter and (shutter.get("mean") or shutter.get("std") or shutter.get("frames") is not None):
            from .shutter_detect import draw_shutter_page, format_shutter_span

            draw_shutter_page(
                fig,
                title=f"Shutter detect  ·  {format_shutter_span(shutter)}",
                subtitle="Contrast cliff on FOV std. Not a hard seed. Check the orange span against the stills in shutter_detect_overview.pdf.",
                det=shutter,
            )
            pdf.savefig(fig, dpi=150)
        if example_triples:
            for start in range(0, len(example_triples), FRAMES_PER_PAGE):
                chunk = example_triples[start : start + FRAMES_PER_PAGE]
                draw_frame_inspection_page(
                    fig,
                    title="Frame inspection — original | cleaned | removed",
                    subtitle=(
                        "Same clean as the TIFF. Periodic stripes in 'removed' are fringes; "
                        "cells or structure means over-cleaning."
                    ),
                    triples=chunk,
                    cleaned_label=cleaned_label,
                )
                pdf.savefig(fig, dpi=150)
    plt.close(fig)


def write_readout(
    out_dir: Path,
    *,
    tif_path: Path,
    families: list[dict],
    params: dict,
    seed_info: dict,
    medspec: np.ndarray | None,
    n_frames: int,
    frame_hw: tuple[int, int],
    source_shape: tuple[int, ...],
    mean_raw: np.ndarray,
    mean_cleaned: np.ndarray,
    mean_removed: np.ndarray,
    rows: list[dict],
    computer: str,
    channel: str,
    fingerprint: dict,
    qc: str,
    used_prior: bool,
    reseeded: bool,
    prior_branch: str | None = None,
    catalog: dict | None = None,
) -> dict[str, Path | None]:
    """Write mask, per-frame table, mean TIFFs, and overview PDF into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "per_frame.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=per_frame_fieldnames(len(families)))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path = out_dir / "families.json"
    summary = write_families_json(
        json_path,
        families=families,
        params=params,
        seed_info=seed_info,
        n_frames=n_frames,
        shape=source_shape,
        source_tif=tif_path,
        computer=computer,
        channel=channel,
        fingerprint=fingerprint,
        qc=qc,
        used_prior=used_prior,
        reseeded=reseeded,
        rows=rows,
        prior_branch=prior_branch,
        catalog=catalog if catalog is not None else seed_info.get("catalog"),
    )

    mask = fft_mask_image(frame_hw, families)
    mask_path = out_dir / "mask_fft.tif"
    tifffile.imwrite(mask_path, mask, photometric="minisblack")

    mean_raw_path = out_dir / "mean_raw.tif"
    mean_cleaned_path = out_dir / "mean_cleaned.tif"
    mean_removed_path = out_dir / "mean_removed.tif"
    write_mean_tif(mean_raw_path, mean_raw)
    write_mean_tif(mean_cleaned_path, mean_cleaned)
    write_mean_tif(mean_removed_path, mean_removed)

    inspection_frames: list[dict] = []
    example_triples: list[dict] = []
    cleaned_tif = out_dir / f"{Path(tif_path).stem}_defringed_v22.tif"
    removed_tif = removed_path_for(tif_path, out_dir)
    if rows and Path(tif_path).is_file() and cleaned_tif.is_file() and removed_tif.is_file():
        inspection_frames = choose_inspection_frames(rows, n_frames=n_frames)
        try:
            example_triples = load_inspection_triples(
                raw_tif=Path(tif_path),
                cleaned_tif=cleaned_tif,
                removed_tif=removed_tif,
                chosen=inspection_frames,
                rows=rows,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"      frame inspection skipped: {exc}", flush=True)
            example_triples = []

    pdf_path = out_dir / "overview.pdf"
    qs = ",".join(f"{float(f['q']):.1f}" for f in families) if families else "—"
    subtitle = (
        f"{tif_path.name}  ·  {computer} / {channel}  ·  "
        f"seed q={qs}  ·  {qc}"
    )
    png_path: Path | None = pdf_path.with_suffix(".png")
    try:
        write_overview_pdf(
            pdf_path,
            title=f"v2.2 defringe readout — {channel}",
            subtitle=subtitle,
            mean_raw=mean_raw,
            mean_cleaned=mean_cleaned,
            mean_removed=mean_removed,
            medspec=medspec,
            families=families,
            rows=rows,
            summary=summary,
            status="ok",
            inspection_frames=inspection_frames,
            example_triples=example_triples,
            cleaned_label="cleaned",
            catalog=catalog if catalog is not None else seed_info.get("catalog"),
            shutter=seed_info.get("shutter"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"      overview PDF skipped: {exc}", flush=True)
        pdf_path = None
        png_path = None

    return {
        "per_frame_csv": csv_path,
        "families_json": json_path,
        "mask_fft": mask_path,
        "mean_raw": mean_raw_path,
        "mean_cleaned": mean_cleaned_path,
        "mean_removed": mean_removed_path,
        "overview_pdf": pdf_path,
        "overview_png": png_path,
    }


def write_failure_readout(
    out_dir: Path,
    *,
    tif_path: Path,
    families: list[dict],
    params: dict,
    seed_info: dict,
    medspec: np.ndarray | None,
    n_frames: int,
    frame_hw: tuple[int, int],
    source_shape: tuple[int, ...],
    mean_raw: np.ndarray,
    computer: str,
    channel: str,
    fingerprint: dict,
    qc: str,
    used_prior: bool,
    reseeded: bool,
    prior_branch: str | None = None,
    ladder: list[dict] | None = None,
    failure_message: str = "needs_review",
    catalog: dict | None = None,
) -> dict[str, Path | None]:
    """Write a needs_review overview (no cleaned/removed stacks)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zeros = np.zeros_like(mean_raw, dtype=np.float64)

    json_path = out_dir / "families.json"
    summary = write_families_json(
        json_path,
        families=families,
        params=params,
        seed_info=seed_info,
        n_frames=n_frames,
        shape=source_shape,
        source_tif=tif_path,
        computer=computer,
        channel=channel,
        fingerprint=fingerprint,
        qc=qc,
        used_prior=used_prior,
        reseeded=reseeded,
        rows=[],
        status="needs_review",
        prior_branch=prior_branch,
        ladder=ladder,
        failure_message=failure_message,
        catalog=catalog if catalog is not None else seed_info.get("catalog"),
    )

    ladder_path = out_dir / "ladder.json"
    ladder_path.write_text(
        json.dumps(_jsonable({"message": failure_message, "ladder": ladder or []}), indent=2),
        encoding="utf-8",
    )

    if families:
        mask = fft_mask_image(frame_hw, families)
        mask_path = out_dir / "mask_fft.tif"
        tifffile.imwrite(mask_path, mask, photometric="minisblack")
    else:
        mask_path = None

    mean_raw_path = out_dir / "mean_raw.tif"
    write_mean_tif(mean_raw_path, mean_raw)

    pdf_path = out_dir / "overview.pdf"
    subtitle = (
        f"{tif_path.name}  ·  {computer} / {channel}  ·  "
        f"{failure_message}"
    )
    png_path: Path | None = pdf_path.with_suffix(".png")
    try:
        write_overview_pdf(
            pdf_path,
            title=f"v2.2 defringe — NEEDS REVIEW — {channel}",
            subtitle=subtitle,
            mean_raw=mean_raw,
            mean_cleaned=zeros,
            mean_removed=zeros,
            medspec=medspec,
            families=families,
            rows=[],
            summary=summary,
            status="needs_review",
            ladder=ladder,
            failure_message=failure_message,
            catalog=catalog if catalog is not None else seed_info.get("catalog"),
            shutter=seed_info.get("shutter"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"      overview PDF skipped: {exc}", flush=True)
        pdf_path = None
        png_path = None

    return {
        "families_json": json_path,
        "ladder_json": ladder_path,
        "mask_fft": mask_path,
        "mean_raw": mean_raw_path,
        "overview_pdf": pdf_path,
        "overview_png": png_path,
    }


def _coerce_csv_row(row: dict) -> dict:
    out: dict = {}
    int_keys = {"frame", "n_active_families", "n_residual_passes"}
    for k, v in row.items():
        if v is None or v == "":
            out[k] = v
            continue
        as_int = k in int_keys or k.endswith("_active") or k.endswith("_residual_pass")
        try:
            out[k] = int(float(v)) if as_int else float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def rebuild_success_overview(out_dir: Path) -> Path:
    """Rewrite overview.pdf from an existing successful defringe_v22 folder."""
    out_dir = Path(out_dir)
    payload = json.loads((out_dir / "families.json").read_text(encoding="utf-8"))
    source_tif = Path(payload["source_tif"])
    families = payload.get("families") or []
    summary = payload.get("summary") or {}
    computer = str(payload.get("computer") or "")
    channel = str(payload.get("channel") or "")
    qc = str(payload.get("qc") or "")

    csv_path = out_dir / "per_frame.csv"
    rows: list[dict] = []
    if csv_path.is_file():
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = [_coerce_csv_row(r) for r in csv.DictReader(fh)]
    n_frames = int(summary.get("n_frames") or (max((int(r["frame"]) for r in rows), default=-1) + 1))

    mean_raw = tifffile.imread(out_dir / "mean_raw.tif")
    mean_cleaned = tifffile.imread(out_dir / "mean_cleaned.tif")
    mean_removed = tifffile.imread(out_dir / "mean_removed.tif")

    cleaned_hits = sorted(out_dir.glob("*_defringed_v22.tif"))
    removed_hits = sorted(out_dir.glob("*_removed_v22.tif"))
    if not cleaned_hits or not removed_hits:
        raise FileNotFoundError(f"No cleaned/removed stacks in {out_dir}")

    inspection_frames = choose_inspection_frames(rows, n_frames=n_frames) if rows else []
    example_triples = []
    if inspection_frames and source_tif.is_file():
        example_triples = load_inspection_triples(
            raw_tif=source_tif,
            cleaned_tif=cleaned_hits[0],
            removed_tif=removed_hits[0],
            chosen=inspection_frames,
            rows=rows,
        )

    qs = ",".join(f"{float(f['q']):.1f}" for f in families) if families else "—"
    pdf_path = out_dir / "overview.pdf"
    write_overview_pdf(
        pdf_path,
        title=f"v2.2 defringe readout — {channel}",
        subtitle=f"{source_tif.name}  ·  {computer} / {channel}  ·  seed q={qs}  ·  {qc}",
        mean_raw=mean_raw,
        mean_cleaned=mean_cleaned,
        mean_removed=mean_removed,
        medspec=None,
        families=families,
        rows=rows,
        summary=summary,
        status="ok",
        inspection_frames=inspection_frames,
        example_triples=example_triples,
        cleaned_label="cleaned",
        catalog=payload.get("catalog") or (payload.get("seed") or {}).get("catalog"),
        shutter=payload.get("shutter") or (payload.get("seed") or {}).get("shutter"),
    )
    return pdf_path
