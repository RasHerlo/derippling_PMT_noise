"""
Bake-off: compare fringe cleaners on raw stacks.

Methods
-------
1) gpt_adaptive_raw     — detect + clean on raw (reference/gpt)
2) rowband_raw          — Claude-style row bands, detect + clean on raw
3) rowband_support2raw  — detect on SUPPORT reference, apply to raw (ChanB only if ref given)

All outputs go under --out-dir (default: cursor tests folder).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reference" / "gpt"))

import deripple as rowband  # noqa: E402
import pmt_fringe_adaptive as gpt  # noqa: E402


def to_u8(img: np.ndarray, p=(1.0, 99.5), lo=None, hi=None) -> np.ndarray:
    x = img.astype(np.float32)
    if lo is None or hi is None:
        lo, hi = np.percentile(x, p)
    if hi <= lo:
        hi = lo + 1
    return (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def ripple_score(frame: np.ndarray) -> float:
    """Proxy: peak vertical-axis FFT energy vs median spectrum."""
    f = frame.astype(np.float64)
    f = f - np.median(f)
    F = np.abs(np.fft.fftshift(np.fft.fft2(f)))
    cy, cx = np.array(F.shape) // 2
    axis = F[:, max(0, cx - 2) : cx + 3].mean(axis=1)
    axis[cy - 5 : cy + 6] = 0
    return float(axis.max() / (np.median(F) + 1e-8))


def save_compare_png(
    path: Path,
    raw: np.ndarray,
    cleaned: np.ndarray,
    title: str,
    frames=(0, 100, 249),
) -> None:
    panels = []
    for i in frames:
        if i >= raw.shape[0]:
            continue
        lo, hi = np.percentile(raw[i], (1, 99.5))
        r = to_u8(raw[i], lo=lo, hi=hi)
        c = to_u8(cleaned[i], lo=lo, hi=hi)
        resid = raw[i].astype(np.float32) - cleaned[i].astype(np.float32)
        rem = to_u8(resid, p=(5, 99.5))
        row = np.concatenate([r, c, rem], axis=1)
        panels.append(row)
    if not panels:
        return
    grid = np.concatenate(panels, axis=0)
    Image.fromarray(grid).save(path)
    # tiny caption sidecar
    path.with_suffix(".txt").write_text(
        f"{title}\ncolumns: raw | cleaned | removed(raw-cleaned)\nrows: frames {frames}\n",
        encoding="utf-8",
    )


def subsample(stack: np.ndarray, n: int) -> np.ndarray:
    if stack.shape[0] <= n:
        return stack
    idx = np.unique(np.linspace(0, stack.shape[0] - 1, n).round().astype(int))
    return stack[idx]


def run_gpt(raw: np.ndarray, out_tif: Path, diag_dir: Path, sample_frames: int) -> dict:
    diag_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    peaks, score, zmap, used = gpt.detect_recurring_peaks(
        raw, sample_frames=sample_frames, detect_z=8.0, max_peaks=24
    )
    if not peaks:
        # relax once for hard-to-see raw fringes
        peaks, score, zmap, used = gpt.detect_recurring_peaks(
            raw, sample_frames=sample_frames, detect_z=6.0, max_peaks=32
        )
    cleaned = np.empty_like(raw)
    tracking = []
    for i in range(raw.shape[0]):
        out, tr = gpt.adaptive_filter_frame(raw[i], peaks)
        cleaned[i] = gpt._cast_like(out, raw.dtype)
        for row in tr:
            row = dict(row)
            row["frame"] = i
            tracking.append(row)
    tifffile.imwrite(out_tif, cleaned, photometric="minisblack")
    gpt.save_detection_png(diag_dir / "detected_peaks.png", zmap, peaks)
    with open(diag_dir / "temporal_tracking.csv", "w", newline="", encoding="utf-8") as f:
        fields = [
            "frame",
            "nominal_dy",
            "nominal_dx",
            "actual_dy",
            "actual_dx",
            "peak_to_background",
            "applied",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in tracking:
            w.writerow(row)
    elapsed = time.perf_counter() - t0
    applied = float(np.mean([r["applied"] for r in tracking])) if tracking else 0.0
    return {
        "method": "gpt_adaptive_raw",
        "peaks": peaks,
        "n_peaks": len(peaks),
        "elapsed_s": elapsed,
        "filter_application_fraction": applied,
        "output": str(out_tif),
        "sampled_detect_frames": used.tolist(),
    }


def run_rowband(
    raw: np.ndarray,
    out_tif: Path,
    diag_path: Path,
    *,
    reference: np.ndarray | None,
    sample_frames: int,
    tag: str,
) -> dict:
    t0 = time.perf_counter()
    ref = reference if reference is not None else raw
    ref_det = subsample(np.asarray(ref), sample_frames)
    dy = rowband.detect_noise_harmonics(ref_det, height_sigma=2.5, dc_col_protect=6)
    cleaned, dy = rowband.clean_stack(
        raw, dy_harmonics=dy, sigma_y=2.2, dc_col_protect=6
    )
    tifffile.imwrite(out_tif, cleaned)
    mask = rowband.build_rowband_mask(raw.shape[1:], dy, sigma_y=2.2, dc_col_protect=6)
    rowband.save_diagnostics(diag_path, raw, cleaned, dy, mask)
    elapsed = time.perf_counter() - t0
    return {
        "method": tag,
        "dy_harmonics": dy,
        "n_harmonic_rows": len(dy),
        "elapsed_s": elapsed,
        "output": str(out_tif),
        "mask_fraction_gt_0.5": float(np.mean(mask > 0.5)),
    }


def summarize_quality(raw: np.ndarray, cleaned: np.ndarray, frames=(0, 100, 249)) -> dict:
    scores = []
    for i in frames:
        if i >= raw.shape[0]:
            continue
        scores.append(
            {
                "frame": i,
                "ripple_raw": ripple_score(raw[i]),
                "ripple_clean": ripple_score(cleaned[i]),
                "std_raw": float(np.std(raw[i])),
                "std_clean": float(np.std(cleaned[i])),
                "resid_rms": float(
                    np.sqrt(np.mean((raw[i].astype(np.float64) - cleaned[i].astype(np.float64)) ** 2))
                ),
            }
        )
    return {"per_frame": scores}


def process_stack(
    raw_path: Path,
    out_root: Path,
    *,
    support_ref: Path | None,
    sample_frames: int,
    max_frames: int | None,
) -> list[dict]:
    print(f"\n=== {raw_path.name} ===")
    raw = tifffile.imread(raw_path)
    if raw.ndim != 3:
        raise ValueError(f"Expected TYX stack, got {raw.shape}")
    if max_frames is not None:
        raw = raw[:max_frames]
        print(f"using first {raw.shape[0]} frames")
    else:
        print(f"shape={raw.shape} dtype={raw.dtype}")

    stem = raw_path.stem
    stack_dir = out_root / stem
    stack_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # 1) GPT adaptive on raw
    print("-> gpt_adaptive_raw")
    gpt_dir = stack_dir / "gpt_adaptive_raw"
    gpt_dir.mkdir(exist_ok=True)
    meta = run_gpt(
        raw,
        gpt_dir / f"{stem}_gpt_adaptive.tif",
        gpt_dir / "diagnostics",
        sample_frames=sample_frames,
    )
    cleaned = tifffile.imread(meta["output"])
    meta["quality"] = summarize_quality(raw, cleaned)
    save_compare_png(
        gpt_dir / f"{stem}_gpt_adaptive_compare.png",
        raw,
        cleaned,
        "gpt_adaptive_raw",
    )
    results.append(meta)
    print(f"   peaks={meta['n_peaks']}  time={meta['elapsed_s']:.1f}s")

    # 2) rowband detect+clean on raw
    print("-> rowband_raw")
    rb_dir = stack_dir / "rowband_raw"
    rb_dir.mkdir(exist_ok=True)
    meta = run_rowband(
        raw,
        rb_dir / f"{stem}_rowband_raw.tif",
        rb_dir / f"{stem}_rowband_raw_diagnostics.png",
        reference=None,
        sample_frames=sample_frames,
        tag="rowband_raw",
    )
    cleaned = tifffile.imread(meta["output"])
    meta["quality"] = summarize_quality(raw, cleaned)
    save_compare_png(
        rb_dir / f"{stem}_rowband_raw_compare.png",
        raw,
        cleaned,
        "rowband_raw",
    )
    results.append(meta)
    print(f"   rows={meta['n_harmonic_rows']}  time={meta['elapsed_s']:.1f}s")

    # 3) rowband detect on SUPPORT, apply to raw
    if support_ref is not None and support_ref.exists():
        print(f"-> rowband_support2raw  (ref={support_ref.name})")
        s_dir = stack_dir / "rowband_support2raw"
        s_dir.mkdir(exist_ok=True)
        ref = tifffile.imread(support_ref)
        meta = run_rowband(
            raw,
            s_dir / f"{stem}_rowband_support2raw.tif",
            s_dir / f"{stem}_rowband_support2raw_diagnostics.png",
            reference=ref,
            sample_frames=sample_frames,
            tag="rowband_support2raw",
        )
        cleaned = tifffile.imread(meta["output"])
        meta["quality"] = summarize_quality(raw, cleaned)
        meta["reference"] = str(support_ref)
        save_compare_png(
            s_dir / f"{stem}_rowband_support2raw_compare.png",
            raw,
            cleaned,
            "rowband_support2raw",
        )
        results.append(meta)
        print(f"   rows={meta['n_harmonic_rows']}  time={meta['elapsed_s']:.1f}s")
    else:
        print("-> rowband_support2raw skipped (no SUPPORT reference)")

    (stack_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def write_summary(out_root: Path, all_results: dict) -> None:
    lines = [
        "# Fringe cleaner bake-off",
        "",
        "Columns in `*_compare.png`: raw | cleaned | removed (raw-cleaned).",
        "",
    ]
    for stem, methods in all_results.items():
        lines.append(f"## {stem}")
        lines.append("")
        lines.append("| method | detect | elapsed_s | ripple f0 raw->clean | notes |")
        lines.append("|---|---|---:|---|---|")
        for m in methods:
            q0 = m.get("quality", {}).get("per_frame", [{}])[0]
            rr = q0.get("ripple_raw", float("nan"))
            rc = q0.get("ripple_clean", float("nan"))
            if m["method"] == "gpt_adaptive_raw":
                note = f"n_peaks={m.get('n_peaks')}, apply_frac={m.get('filter_application_fraction', 0):.2f}"
                detect = "raw"
            elif m["method"] == "rowband_raw":
                note = f"n_rows={m.get('n_harmonic_rows')}, mask%>0.5={m.get('mask_fraction_gt_0.5', 0):.3f}"
                detect = "raw"
            else:
                note = f"n_rows={m.get('n_harmonic_rows')}, ref={Path(m.get('reference','')).name}"
                detect = "SUPPORT"
            lines.append(
                f"| `{m['method']}` | {detect} | {m['elapsed_s']:.1f} | {rr:.2f}->{rc:.2f} | {note} |"
            )
        lines.append("")
    path = out_root / "SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(
            r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\SUPPORT_ChanB\to build FFT deripple\raw test files"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\SUPPORT_ChanB\to build FFT deripple\cursor tests"
        ),
    )
    ap.add_argument(
        "--support-ref",
        type=Path,
        default=Path(
            r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\SUPPORT_ChanB\to build FFT deripple\Initial probes\Frame1t10.tif"
        ),
        help="SUPPORT stack for ChanB support→raw baseline (optional)",
    )
    ap.add_argument("--sample-frames", type=int, default=64)
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="If set, only process the first N frames (smoke test)",
    )
    ap.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Optional filenames to process (default: all .tif in raw-dir)",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.raw_dir.glob("*.tif"))
    if args.channels:
        wanted = set(args.channels)
        files = [f for f in files if f.name in wanted]
    if not files:
        raise SystemExit(f"No TIFF files found in {args.raw_dir}")

    all_results = {}
    for f in files:
        # SUPPORT reference only for ChanB (same folder / channel context)
        support = args.support_ref if "ChanB" in f.name else None
        all_results[f.stem] = process_stack(
            f,
            args.out_dir,
            support_ref=support,
            sample_frames=args.sample_frames,
            max_frames=args.max_frames,
        )

    write_summary(args.out_dir, all_results)
    (args.out_dir / "results_all.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    print("done")


if __name__ == "__main__":
    main()
