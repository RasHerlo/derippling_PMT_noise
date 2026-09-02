"""Probe: seed q/fx from in-stack shutter frames, flatten them, track onto live frames.

Does not overwrite production defringe_v22 stacks. Writes a PDF + metrics.json
under <channel>/defringe_v22/shutter_seed_test/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

_REPO = Path(__file__).resolve().parents[1]
_GPT = _REPO / "reference" / "gpt"
if str(_GPT) not in sys.path:
    sys.path.insert(0, str(_GPT))

from pmt_fringe_raw_adaptive import (  # noqa: E402
    contiguous_ranges,
    detect_families,
    fft_log_amp,
    ridge_z_at_row,
    search_q,
)
from pmt_fringe_raw_adaptive_v21 import _attenuate_family_on_amp  # noqa: E402
from pmt_fringe_raw_adaptive_v22 import clean_frame_v22  # noqa: E402

from .process import PACK_D
from .readout import (
    FRAMES_PER_PAGE,
    _jsonable,
    _percentile_limits,
    draw_frame_inspection_page,
    fft_mask_image,
)
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from .seed import hydrate_families
from .shutter_detect import low_std_runs, scan_frame_stats

SHUTTER_DEFAULT = tuple(range(756, 761))  # plateau; 761 is already opening
LIVE_DEFAULT = (160, 700, 1061)
Q_TRACK = 8
FLAT_PASSES = 3
FX_PEAK_Z = 6.0
FX_HALFWIDTH = 3
MAX_FX_PEAKS = 12


def _xvalid(width: int) -> np.ndarray:
    cx = width // 2
    fx = np.arange(width) - cx
    return (np.abs(fx) > 5) & (np.abs(fx) < cx - 10)


def frame_std(frame: np.ndarray) -> float:
    return float(np.std(np.asarray(frame, dtype=np.float64)))


def _family_peak_score(medspec: np.ndarray, fam: dict) -> float:
    h, w = medspec.shape
    cy = h // 2
    xvalid = _xvalid(w)
    scores = []
    for d in (float(fam["q"]), fam.get("hi")):
        if d is None:
            continue
        for sgn in (-1, +1):
            y = cy + sgn * int(round(float(d)))
            if 0 <= y < h:
                scores.append(float(np.percentile(medspec[y, xvalid], 99.0)))
    return max(scores) if scores else -1.0


def _is_nyquist_self_pair(fam: dict, height: int) -> bool:
    q = float(fam["q"])
    hi = fam.get("hi")
    nyq = height / 4.0
    if hi is not None and abs(float(hi) - q) < 1.5 and abs(q - nyq) < 4:
        return True
    return abs(q - nyq) < 2


def peak_fx_weight(medspec: np.ndarray, fam: dict) -> tuple[np.ndarray, list]:
    """Keep only local fx peaks on the shutter ridge — not the whole row."""
    h, w = medspec.shape
    cy, cx = h // 2, w // 2
    fx = np.arange(w) - cx
    xvalid = _xvalid(w)
    components = [float(fam["q"])]
    if fam.get("hi") is not None:
        components.append(float(fam["hi"]))
    zx_rows = []
    for d in components:
        zx_rows.append(ridge_z_at_row(medspec, +int(round(d))))
        zx_rows.append(ridge_z_at_row(medspec, -int(round(d))))
    zx = np.max(np.stack(zx_rows), axis=0)
    locmax = zx == maximum_filter1d(zx, size=7)
    peak_idx = np.where(locmax & xvalid & (zx >= FX_PEAK_Z))[0]
    if peak_idx.size == 0:
        peak_idx = np.where(locmax & xvalid)[0]
        if peak_idx.size:
            peak_idx = peak_idx[np.argsort(zx[peak_idx])[-min(MAX_FX_PEAKS, peak_idx.size) :]]
    elif peak_idx.size > MAX_FX_PEAKS:
        peak_idx = peak_idx[np.argsort(zx[peak_idx])[-MAX_FX_PEAKS:]]
    weight = np.zeros(w, dtype=float)
    for p in peak_idx:
        lo = max(0, int(p) - FX_HALFWIDTH)
        hi = min(w, int(p) + FX_HALFWIDTH + 1)
        dist = np.abs(np.arange(lo, hi) - int(p))
        weight[lo:hi] = np.maximum(weight[lo:hi], np.exp(-0.5 * (dist / 1.2) ** 2))
    if weight.max() > 0:
        weight /= weight.max()
        weight = gaussian_filter1d(weight, sigma=0.6)
        weight /= max(float(weight.max()), 1e-12)
    return weight, contiguous_ranges(fx[weight > 0.20])


def _near_dc(q: float, height: int, floor: int = 16) -> bool:
    cy = height // 2
    return min(abs(q), abs(cy - abs(q))) < floor


def learn_shutter_families(
    frames: list[np.ndarray],
    *,
    x_z_thresh: float = 2.5,
    row_z_thresh: float = 3.0,
    max_families: int = 3,
) -> tuple[list[dict], np.ndarray]:
    specs = [fft_log_amp(f) for f in frames]
    medspec = np.median(np.stack(specs), axis=0)
    height = int(medspec.shape[0])
    raw, _, _ = detect_families(
        medspec,
        row_z_thresh=row_z_thresh,
        pair_z_min=2.0,
        x_z_thresh=x_z_thresh,
        max_families=max(max_families, 4),
        allow_standalone=True,
    )
    if not raw:
        raw, _, _ = detect_families(
            medspec,
            row_z_thresh=2.2,
            pair_z_min=1.5,
            x_z_thresh=x_z_thresh,
            max_families=max(max_families, 4),
            allow_standalone=True,
        )
    raw = hydrate_families(raw, medspec, x_z_thresh=x_z_thresh)
    raw = [f for f in raw if f.get("fx_ranges")]
    if not raw:
        return [], medspec

    ranked = sorted(raw, key=lambda f: _family_peak_score(medspec, f), reverse=True)
    ranked = [f for f in ranked if not _is_nyquist_self_pair(f, height)] or ranked
    primary = ranked[0]
    q0 = float(primary["q"])
    chosen = [primary]
    for fam in ranked[1:]:
        q = float(fam["q"])
        if _near_dc(q, height):
            continue
        if any(abs(q - k * q0) < 2.5 or abs(q0 - k * q) < 2.5 for k in (2, 3)):
            chosen.append(fam)
        if len(chosen) >= max_families:
            break
    out = []
    for fam in chosen:
        weight, ranges = peak_fx_weight(medspec, fam)
        if not ranges or float(np.max(weight)) < 0.20:
            continue
        fam = dict(fam)
        fam["x_weight"] = weight
        fam["fx_ranges"] = ranges
        out.append(fam)
    return out, medspec


def flatten_frame(frame: np.ndarray, families: list[dict], qs: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Replace ridge-bin amplitude with local spectral background (phase kept)."""
    orig_dtype = frame.dtype
    x = np.asarray(frame, dtype=np.float32)
    offset = float(np.median(x))
    x0 = x - offset
    F = np.fft.fftshift(np.fft.fft2(x0))
    amp = np.abs(F)
    phase = np.exp(1j * np.angle(F))
    newamp = amp.copy()
    src = amp
    for _ in range(FLAT_PASSES):
        dst = newamp.copy()
        for family, q in zip(families, qs):
            _attenuate_family_on_amp(
                src,
                dst,
                family,
                float(q),
                gate=1.0,
                max_alpha=1.0,
                ratio_start=1.02,
                ratio_full=1.25,
                y_sigma=1.0,
                y_radius=2,
            )
        newamp = dst
        src = newamp
    cleaned = np.real(np.fft.ifft2(np.fft.ifftshift(newamp * phase))) + offset
    removed = np.asarray(frame, dtype=np.float64) - cleaned
    if np.issubdtype(orig_dtype, np.integer):
        lim = np.iinfo(orig_dtype)
        cleaned_w = np.clip(np.rint(cleaned), lim.min, lim.max).astype(orig_dtype)
    else:
        cleaned_w = cleaned.astype(orig_dtype)
    return cleaned_w, removed.astype(np.float32)


def track_qs(frame: np.ndarray, families: list[dict], radius: int = Q_TRACK) -> tuple[list[float], list[float]]:
    logamp = fft_log_amp(frame)
    xvalid = _xvalid(logamp.shape[1])
    qs: list[float] = []
    scores: list[float] = []
    for i, fam in enumerate(families):
        forbidden = [qs[j] for j in range(len(qs))]
        q, sc = search_q(
            logamp,
            float(fam["q"]),
            bool(fam.get("paired", True)),
            xvalid,
            radius,
            forbidden_qs=forbidden or None,
            forbidden_radius=3,
        )
        qs.append(float(q))
        scores.append(float(sc))
    return qs, scores


def live_clean(frame: np.ndarray, families: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    qs, _ = track_qs(frame, families, radius=Q_TRACK)
    return clean_frame_v22(frame, families, qs, **{**PACK_D, "frame_search": 1})


def family_public_light(fam: dict) -> dict:
    return {
        "q": float(fam["q"]),
        "hi": None if fam.get("hi") is None else float(fam["hi"]),
        "paired": bool(fam.get("paired", True)),
        "row_score": float(fam.get("row_score", 0.0)),
        "fx_ranges": fam.get("fx_ranges"),
        "n_fx_bins": int(np.sum(np.asarray(fam.get("x_weight", [0])) > 0.20)),
    }


def _draw_cover(fig, *, title, subtitle, mean_shutter, medspec, families, stats, shutter_idx, runs, metrics_lines):
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle

    fig.clear()
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.98, top=0.82, bottom=0.08, hspace=0.32, wspace=0.22)
    fig.text(0.06, 0.97, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.06, 0.935, subtitle, fontsize=8.5, va="top", color="0.25")

    ax0 = fig.add_subplot(gs[0, 0])
    vmin, vmax = _percentile_limits(mean_shutter)
    ax0.imshow(mean_shutter, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax0.set_title("Mean shutter (seed)", fontsize=9)
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[0, 1])
    spec = np.asarray(medspec)
    slo, shi = _percentile_limits(spec, (3.0, 99.7))
    ax1.imshow(spec, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest")
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    if families:
        mask = fft_mask_image((h, w), families)
        if float(np.max(mask)) > 0.2:
            ax1.contour(mask, levels=[0.2], colors=["C3"], linewidths=0.6)
    for i, fam in enumerate(families):
        for pair in fam.get("fx_ranges") or []:
            lo_fx, hi_fx = float(pair[0]), float(pair[1])
            y = cy + int(round(float(fam["q"])))
            ax1.add_patch(
                Rectangle((cx + lo_fx, y - 2.5), hi_fx - lo_fx + 1, 5, linewidth=0.5, edgecolor=f"C{i}", facecolor="none")
            )
    ax1.set_title("Shutter FFT + learned fx comb", fontsize=9)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[1, 0])
    xs = [s["frame"] for s in stats]
    ax2.plot(xs, [s["std"] for s in stats], color="0.2", lw=0.7)
    for a, b in [(min(shutter_idx), max(shutter_idx))]:
        ax2.axvspan(a - 0.5, b + 0.5, color="C1", alpha=0.25)
    ax2.set_title("Per-frame spatial std (orange = seed stretch)", fontsize=9)
    ax2.set_xlabel("frame")
    ax2.set_ylabel("std")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    run_txt = ", ".join(f"{r[0]}-{r[-1]}" for r in runs) or "(none)"
    fam_txt = ", ".join(f"q={f['q']:.0f} ({f['n_fx_bins']} fx bins)" for f in (family_public_light(x) for x in families))
    if not families:
        fam_txt = "(none)"
    ax3.text(
        0.0,
        1.0,
        "\n".join(
            [
                f"low-std runs (auto): {run_txt}",
                f"seed frames: {list(shutter_idx)}",
                f"families: {fam_txt}",
                "",
                *metrics_lines,
            ]
        ),
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        transform=ax3.transAxes,
    )


def run_one(
    tif_path: Path,
    *,
    shutter: tuple[int, ...] = SHUTTER_DEFAULT,
    live: tuple[int, ...] = LIVE_DEFAULT,
    x_z_thresh: float = 2.5,
) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    tif_path = Path(tif_path)
    out_dir = tif_path.parent / "defringe_v22" / "shutter_seed_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(tif_path) as tf:
        n = int(tf.series[0].shape[0])
        shutter_idx = [i for i in shutter if 0 <= i < n]
        live_idx = [i for i in live if 0 <= i < n]
        print(f"  scanning {n} frame stats...", flush=True)
        stats = scan_frame_stats(tf)
        runs = low_std_runs(stats)
        shutter_frames = [np.asarray(tf.pages[i].asarray()) for i in shutter_idx]
        live_frames = {i: np.asarray(tf.pages[i].asarray()) for i in live_idx}

    families, medspec = learn_shutter_families(shutter_frames, x_z_thresh=x_z_thresh)
    mean_shutter = np.mean(np.stack([f.astype(np.float64) for f in shutter_frames]), axis=0)

    shutter_triples = []
    shutter_metrics = []
    seed_qs = [float(f["q"]) for f in families]
    for idx, frame in zip(shutter_idx, shutter_frames):
        std0 = frame_std(frame)
        if families:
            qs, scores = track_qs(frame, families, radius=2)
            cleaned, removed = flatten_frame(frame, families, qs)
        else:
            qs, scores = [], []
            cleaned, removed = frame, np.zeros(frame.shape, np.float32)
        std1 = frame_std(cleaned)
        rms = float(np.sqrt(np.mean(np.asarray(removed, dtype=np.float64) ** 2)))
        shutter_metrics.append(
            {"frame": idx, "std_raw": std0, "std_flat": std1, "std_ratio": std1 / max(std0, 1e-9), "removed_rms": rms, "qs": qs, "scores": scores}
        )
        shutter_triples.append(
            {
                "frame": idx,
                "role": "shutter",
                "raw": frame,
                "cleaned": cleaned,
                "removed": removed,
                "qs": qs,
                "removed_rms": rms,
            }
        )

    live_triples = []
    live_metrics = []
    for idx in live_idx:
        frame = live_frames[idx]
        std0 = frame_std(frame)
        if families:
            qs, scores = track_qs(frame, families, radius=Q_TRACK)
            cleaned, removed, tracking = live_clean(frame, families)
            gates = [float(t.get("gate", 0.0)) for t in tracking]
        else:
            qs, scores, gates = [], [], []
            cleaned, removed = frame, np.zeros(frame.shape, np.float32)
        std1 = frame_std(cleaned)
        rms = float(np.sqrt(np.mean(np.asarray(removed, dtype=np.float64) ** 2)))
        live_metrics.append(
            {
                "frame": idx,
                "std_raw": std0,
                "std_clean": std1,
                "removed_rms": rms,
                "qs": qs,
                "scores": scores,
                "gates": gates,
            }
        )
        live_triples.append(
            {
                "frame": idx,
                "role": "live",
                "raw": frame,
                "cleaned": cleaned,
                "removed": removed,
                "qs": qs,
                "removed_rms": rms,
            }
        )

    std_raw = float(np.mean([m["std_raw"] for m in shutter_metrics]))
    std_flat = float(np.mean([m["std_flat"] for m in shutter_metrics]))
    metrics_lines = [
        f"shutter std {std_raw:.2f} → {std_flat:.2f}  ({100 * std_flat / max(std_raw, 1e-9):.1f}% remains)",
        f"live q-track radius ±{Q_TRACK}",
        "shutter: ridge flattened to spectral background",
        "live: pack_D excess notch, same fx comb",
    ]

    payload = {
        "source_tif": str(tif_path),
        "shutter_frames": shutter_idx,
        "live_frames": live_idx,
        "q_track": Q_TRACK,
        "x_z_thresh": x_z_thresh,
        "low_std_runs": [[int(i) for i in r] for r in runs],
        "families": [family_public_light(f) for f in families],
        "seed_q": seed_qs,
        "shutter": shutter_metrics,
        "live": live_metrics,
        "shutter_std_raw": std_raw,
        "shutter_std_flat": std_flat,
    }
    (out_dir / "metrics.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")

    pdf_path = out_dir / "overview.pdf"
    png_path = out_dir / "overview.png"
    fig = plt.figure(figsize=(11.69, 8.27))
    with PdfPages(pdf_path) as pdf:
        _draw_cover(
            fig,
            title=f"Shutter-seed probe — {tif_path.parent.name}",
            subtitle=f"{tif_path.name}  ·  flatten shutter  ·  track onto live  ·  stack not overwritten",
            mean_shutter=mean_shutter,
            medspec=medspec,
            families=families,
            stats=stats,
            shutter_idx=shutter_idx,
            runs=runs,
            metrics_lines=metrics_lines,
        )
        fig.savefig(png_path, dpi=130)
        pdf.savefig(fig, dpi=140)
        for start in range(0, len(shutter_triples), FRAMES_PER_PAGE):
            draw_frame_inspection_page(
                fig,
                title="Shutter frames — flatten with learned fx comb",
                subtitle="Aim: spatially flat (dark + noise). Periodic leftover in original should move to 'removed'.",
                triples=shutter_triples[start : start + FRAMES_PER_PAGE],
                cleaned_label="flattened",
            )
            pdf.savefig(fig, dpi=140)
            if start == 0:
                fig.savefig(out_dir / "shutter_frames.png", dpi=130)
        for start in range(0, len(live_triples), FRAMES_PER_PAGE):
            draw_frame_inspection_page(
                fig,
                title=f"Live frames — same comb, q tracked ±{Q_TRACK}, pack_D notch",
                subtitle="Removed should look like the shutter pattern (possibly shifted), not cells.",
                triples=live_triples[start : start + FRAMES_PER_PAGE],
                cleaned_label="cleaned",
            )
            pdf.savefig(fig, dpi=140)
            if start == 0:
                fig.savefig(out_dir / "live_frames.png", dpi=130)
    plt.close(fig)

    payload["overview_pdf"] = str(pdf_path)
    print(f"  families q={[f['q'] for f in families]}", flush=True)
    print(f"  shutter std {std_raw:.2f} -> {std_flat:.2f}", flush=True)
    for m in live_metrics:
        print(
            f"  live {m['frame']}: q={m['qs']} gate={m['gates']} rms={m['removed_rms']:.3g}",
            flush=True,
        )
    print(f"  wrote {pdf_path}", flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tif",
        action="append",
        type=Path,
        help="Raw stack TIFF (repeatable). Default: Haj Grant ChanA and ChanB.",
    )
    ap.add_argument("--x-z", type=float, default=2.5, help="fx-support z threshold (higher = smaller mask)")
    args = ap.parse_args(argv)

    tifs = args.tif
    if not tifs:
        root = Path(r"F:\bPACNewData2026\Haj Grant Example\DATA")
        tifs = [root / "ChanA" / "ChanA_stk.tif", root / "ChanB" / "ChanB_stk.tif"]

    any_fail = False
    for tif in tifs:
        print(f"\n{tif}", flush=True)
        if not tif.is_file():
            print("  MISSING")
            any_fail = True
            continue
        run_one(tif, x_z_thresh=args.x_z)
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
