"""Compare SUPPORT-denoised raw vs defringe→SUPPORT for block artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

BASE = Path(r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA")
OUT = BASE / "SUPPORT_ChanB" / "to build FFT deripple" / "cursor tests" / "support_block_compare"
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = {
    "ChanA": {
        "original_support": BASE / "SUPPORT_ChanA" / "denoised_cut.tif",
        "defringe_support": BASE / "ChanA_defringe" / "ChanA_stk_defringed_denoised.tif",
        "tracking": BASE / "ChanA_defringe" / "diagnostics" / "temporal_tracking.csv",
    },
    "ChanB": {
        "original_support": BASE / "SUPPORT_ChanB" / "denoised_cut.tif",
        "defringe_support": BASE / "ChanB_defringe" / "ChanB_stk_defringed_denoised.tif",
        "tracking": BASE / "ChanB_defringe" / "diagnostics" / "temporal_tracking.csv",
    },
}


def to_u8(img, p=(1, 99.5), lo=None, hi=None):
    x = np.asarray(img, dtype=np.float32)
    if lo is None or hi is None:
        lo, hi = np.percentile(x, p)
    if hi <= lo:
        hi = lo + 1
    return (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def blockiness_score(frame: np.ndarray, block: int = 32) -> dict:
    """
    Heuristics for rectangular tiling / patch artifacts.

    - block_mean_cv: coef. of variation of per-block means (higher => blotchy tiles)
    - seam_excess: mean |jump| across regular vertical/horizontal seams vs random
    """
    f = np.asarray(frame, dtype=np.float64)
    h, w = f.shape
    hb, wb = h // block, w // block
    if hb < 2 or wb < 2:
        return {"block_mean_cv": np.nan, "seam_excess": np.nan}

    crop = f[: hb * block, : wb * block]
    tiles = crop.reshape(hb, block, wb, block).mean(axis=(1, 3))
    block_mean_cv = float(tiles.std() / (np.abs(tiles.mean()) + 1e-8))

    # vertical seams every `block` columns
    seams_v = []
    for x in range(block, wb * block, block):
        seams_v.append(np.mean(np.abs(crop[:, x] - crop[:, x - 1])))
    # horizontal seams
    seams_h = []
    for y in range(block, hb * block, block):
        seams_h.append(np.mean(np.abs(crop[y, :] - crop[y - 1, :])))

    # random nearby differences as baseline
    rng = np.random.default_rng(0)
    xs = rng.integers(1, w - 1, size=200)
    ys = rng.integers(0, h, size=200)
    base_v = float(np.mean(np.abs(f[ys, xs] - f[ys, xs - 1])))
    xs = rng.integers(0, w, size=200)
    ys = rng.integers(1, h - 1, size=200)
    base_h = float(np.mean(np.abs(f[ys, xs] - f[ys - 1, xs])))

    seam = 0.5 * (float(np.mean(seams_v)) + float(np.mean(seams_h)))
    base = 0.5 * (base_v + base_h)
    seam_excess = float(seam / (base + 1e-8))
    return {
        "block_mean_cv": block_mean_cv,
        "seam_excess": seam_excess,
        "seam": seam,
        "base_grad": base,
    }


def highpass_u8(frame, sigma=8.0):
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return to_u8(frame)
    f = frame.astype(np.float32)
    hp = f - gaussian_filter(f, sigma=sigma)
    return to_u8(hp, p=(5, 99.5))


def pick_frames(tracking_csv: Path, n: int = 6) -> list[int]:
    import csv

    if not tracking_csv.exists():
        return [0, 100, 500, 1000, 2500, 5000]
    rows = list(csv.DictReader(tracking_csv.open(encoding="utf-8")))
    for r in rows:
        r["removed_rms"] = float(r["removed_rms"])
        r["frame"] = int(r["frame"])
    strong = sorted(rows, key=lambda r: r["removed_rms"], reverse=True)
    weak = sorted(rows, key=lambda r: r["removed_rms"])
    frames = []
    for r in strong[:2] + weak[:2] + rows[len(rows) // 3 : len(rows) // 3 + 1] + rows[2 * len(rows) // 3 : 2 * len(rows) // 3 + 1]:
        if r["frame"] not in frames:
            frames.append(r["frame"])
        if len(frames) >= n:
            break
    return sorted(frames)


def load_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    out = {}
    with tifffile.TiffFile(path) as tf:
        n = tf.series[0].shape[0]
        for i in indices:
            if i < 0 or i >= n:
                continue
            out[i] = tf.pages[i].asarray()
    return out


def process_channel(name: str, paths: dict) -> dict:
    print(f"=== {name} ===", flush=True)
    frames = pick_frames(paths["tracking"])
    print("frames", frames, flush=True)
    a = load_frames(paths["original_support"], frames)
    b = load_frames(paths["defringe_support"], frames)

    ch_dir = OUT / name
    ch_dir.mkdir(exist_ok=True)
    metrics = []

    for i in frames:
        if i not in a or i not in b:
            continue
        fa, fb = a[i], b[i]
        # shared display scale from original support
        lo, hi = np.percentile(fa, (1, 99.5))
        row = np.concatenate(
            [
                to_u8(fa, lo=lo, hi=hi),
                to_u8(fb, lo=lo, hi=hi),
                to_u8(fa.astype(np.float32) - fb.astype(np.float32), p=(5, 99.5)),
            ],
            axis=1,
        )
        Image.fromarray(row).save(ch_dir / f"frame_{i:04d}_rawSUPPORT_vs_defringeSUPPORT.png")

        # high-pass view emphasizes blocks/tiles
        hp = np.concatenate([highpass_u8(fa), highpass_u8(fb)], axis=1)
        Image.fromarray(hp).save(ch_dir / f"frame_{i:04d}_highpass.png")

        ma = blockiness_score(fa)
        mb = blockiness_score(fb)
        metrics.append(
            {
                "frame": i,
                "original_block_mean_cv": ma["block_mean_cv"],
                "defringe_block_mean_cv": mb["block_mean_cv"],
                "original_seam_excess": ma["seam_excess"],
                "defringe_seam_excess": mb["seam_excess"],
                "cv_reduction_frac": float(
                    (ma["block_mean_cv"] - mb["block_mean_cv"]) / (ma["block_mean_cv"] + 1e-8)
                ),
                "seam_reduction_frac": float(
                    (ma["seam_excess"] - mb["seam_excess"]) / (ma["seam_excess"] + 1e-8)
                ),
            }
        )
        print(
            f"  f{i}: cv {ma['block_mean_cv']:.4f}->{mb['block_mean_cv']:.4f} "
            f"seam {ma['seam_excess']:.3f}->{mb['seam_excess']:.3f}",
            flush=True,
        )

    # montage of first few compares
    mont = []
    for i in frames[:4]:
        p = ch_dir / f"frame_{i:04d}_rawSUPPORT_vs_defringeSUPPORT.png"
        if p.exists():
            mont.append(np.asarray(Image.open(p)))
    if mont:
        Image.fromarray(np.concatenate(mont, axis=0)).save(ch_dir / "montage_compare.png")
        (ch_dir / "montage_compare.txt").write_text(
            "columns: SUPPORT(raw) | SUPPORT(defringed) | difference\n"
            f"rows frames: {frames[:4]}\n",
            encoding="utf-8",
        )

    summary = {
        "channel": name,
        "frames": frames,
        "mean_cv_reduction_frac": float(np.mean([m["cv_reduction_frac"] for m in metrics])),
        "mean_seam_reduction_frac": float(np.mean([m["seam_reduction_frac"] for m in metrics])),
        "per_frame": metrics,
        "note": (
            "Positive reduction_frac means defringe→SUPPORT looks less blocky/seamy "
            "than SUPPORT on untouched raw. Models were trained on non-defringed data."
        ),
    }
    (ch_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    all_sum = {}
    for name, paths in PAIRS.items():
        all_sum[name] = process_channel(name, paths)

    lines = [
        "# SUPPORT block-artifact comparison",
        "",
        "Left/original: `SUPPORT_*/denoised_cut.tif` (SUPPORT on untouched raw)",
        "Middle: `*_defringed_denoised.tif` (v2 defringe then SUPPORT/model_10)",
        "Right: difference",
        "",
        "Caveat: denoiser was trained on non-defringed data; quality shifts may reflect",
        "distribution mismatch and motivate retraining on defringed stacks.",
        "",
    ]
    for name, s in all_sum.items():
        lines.append(f"## {name}")
        lines.append(
            f"- mean block-mean CV reduction: {100*s['mean_cv_reduction_frac']:.1f}%"
        )
        lines.append(
            f"- mean seam-excess reduction: {100*s['mean_seam_reduction_frac']:.1f}%"
        )
        lines.append("")
        lines.append("| frame | CV rawSUPPORT | CV defringeSUPPORT | seam raw | seam defringe |")
        lines.append("|---:|---:|---:|---:|---:|")
        for m in s["per_frame"]:
            lines.append(
                f"| {m['frame']} | {m['original_block_mean_cv']:.4f} | "
                f"{m['defringe_block_mean_cv']:.4f} | {m['original_seam_excess']:.3f} | "
                f"{m['defringe_seam_excess']:.3f} |"
            )
        lines.append("")
    path = OUT / "SUMMARY_support_block_compare.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(all_sum, indent=2), encoding="utf-8")
    print("Wrote", path)


if __name__ == "__main__":
    main()
