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

OUTPUT_DIRNAME = "defringe_v22"
Y_RADIUS = 2  # matches clean_frame_v21 default ridge half-width in fy


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
        "families": [family_public(f) for f in families],
        "summary": summary,
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
) -> None:
    """One-page landscape overview. Raises if matplotlib is unavailable."""
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
        top=0.84,
        bottom=0.07,
    )

    fig.text(0.055, 0.97, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.055, 0.935, subtitle, fontsize=8.5, va="top", color="0.25")

    stats = (
        f"frames {summary.get('n_frames', len(rows))}   "
        f"families {summary.get('n_families', n_fam)}   "
        f"active on {100 * summary.get('frac_frames_any_active', 0):.1f}% of frames   "
        f"median removed RMS {summary.get('median_removed_rms', 0):.3g}   "
        f"max {summary.get('max_removed_rms', 0):.3g}   "
        f"residual pass {100 * summary.get('frac_residual_pass', 0):.1f}%"
    )
    fig.text(0.055, 0.90, stats, fontsize=8, va="top")

    panels = [
        (gs[0, 0], mean_raw, "Mean before (raw)", "gray", vmin, vmax),
        (gs[0, 1], mean_cleaned, "Mean after (cleaned)", "gray", vmin, vmax),
        (gs[0, 2], mean_removed, "Mean removed (raw − cleaned)", "RdBu_r", -rlim, rlim),
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
    if len(frames):
        ax_tr.plot(frames, rms, color="0.15", lw=0.9, label="removed RMS")
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
        ax_q.set_ylabel("tracked q")
        ax_q.set_xlabel("frame")
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
