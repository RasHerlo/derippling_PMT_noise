"""Score sandbox v2.1 500fr defringe outputs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference" / "gpt"))
import pmt_fringe_raw_adaptive as v2  # noqa: E402

SANDBOX = Path(r"F:\bPACNewData2026\PreProcessing Optimization\Level3b copy")
OUT = SANDBOX / "defringe_runs" / "v21_500fr"


def to_u8(img, lo=None, hi=None, p=(1, 99.5)):
    x = img.astype(np.float32)
    if lo is None or hi is None:
        lo, hi = np.percentile(x, p)
    if hi <= lo:
        hi = lo + 1
    return (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def fft_amp(frame):
    x = frame.astype(np.float64)
    x = x - np.median(x)
    return np.abs(np.fft.fftshift(np.fft.fft2(x)))


def ridge_excess_power(amp, families, qs, y_radius=2, x_weight_thresh=0.05):
    h, w = amp.shape
    cy = h // 2
    total = 0.0
    for family, q in zip(families, qs):
        xw = np.asarray(family["x_weight"], float)
        x_sel = xw > x_weight_thresh
        if not np.any(x_sel):
            continue
        dys = [q] + ([cy - q] if family.get("paired", True) else [])
        for d in dys:
            for sgn in (-1, +1):
                yc = cy + sgn * int(round(d))
                for yp in range(yc - y_radius, yc + y_radius + 1):
                    if not (0 <= yp < h):
                        continue
                    bgrows = []
                    for boff in list(range(-9, -4)) + list(range(5, 10)):
                        yy = yp + boff
                        if 0 <= yy < h:
                            bgrows.append(amp[yy, x_sel])
                    if not bgrows:
                        continue
                    bg = np.median(np.stack(bgrows), 0)
                    excess = np.maximum(amp[yp, x_sel] - bg, 0.0) * xw[x_sel]
                    total += float(np.sum(excess * excess))
    return total


def load_track(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gates = [
                float(row[k])
                for k in row
                if k.endswith("_gate") and "residual" not in k
            ]
            qs = [
                float(row[k]) for k in row if k.endswith("_q") and "residual" not in k
            ]
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "removed_rms": float(row["removed_rms"]),
                    "max_gate": max(gates) if gates else 0.0,
                    "qs": qs,
                }
            )
    return rows


def detect_fams(path):
    with tifffile.TiffFile(path) as tf:
        med, _ = v2.learn_median_spectrum(tf, sample_n=80)
        fams, _, _ = v2.detect_families(
            med,
            row_z_thresh=5.5,
            pair_z_min=3.5,
            x_z_thresh=3.5,
            max_families=4,
            allow_standalone=False,
        )
    return fams


def pct(x):
    return float("nan") if x is None else 100.0 * float(x)


def score_one(chan):
    raw_path = SANDBOX / "inputs" / "slices_500fr" / "raw" / f"{chan}_raw_500fr.tif"
    cln_path = OUT / chan / f"{chan}_raw_500fr_v21.tif"
    track = load_track(OUT / chan / "diagnostics" / "temporal_tracking.csv")
    fams = detect_fams(raw_path)
    raw = tifffile.imread(raw_path)
    cln = tifffile.imread(cln_path)
    n = raw.shape[0]
    fracs = np.zeros(n)
    raw_p = np.zeros(n)
    gates = np.array([r["max_gate"] for r in track])
    for i in range(n):
        qs = track[i]["qs"]
        pr = ridge_excess_power(fft_amp(raw[i]), fams, qs)
        pc = ridge_excess_power(fft_amp(cln[i]), fams, qs)
        raw_p[i] = pr
        fracs[i] = pc / (pr + 1e-12)
    strong25 = raw_p >= np.percentile(raw_p, 75)
    zero = np.where(gates <= 1e-12)[0]
    if len(zero):
        gate0_rms = float(
            np.median(
                [
                    np.sqrt(
                        np.mean(
                            (raw[i].astype(float) - cln[i].astype(float)) ** 2
                        )
                    )
                    for i in zero[:20]
                ]
            )
        )
    else:
        gate0_rms = float("nan")

    by = sorted(track, key=lambda r: r["max_gate"], reverse=True)
    strong, weak = by[0]["frame"], by[-1]["frame"]
    panels = []
    for fr in (strong, weak):
        lo, hi = np.percentile(raw[fr], (1, 99.5))
        panels.append(
            np.concatenate(
                [
                    to_u8(raw[fr], lo, hi),
                    to_u8(cln[fr], lo, hi),
                    to_u8(
                        raw[fr].astype(np.float32) - cln[fr].astype(np.float32),
                        p=(5, 99.5),
                    ),
                ],
                1,
            )
        )
    Image.fromarray(np.concatenate(panels, 0)).save(OUT / chan / f"{chan}_strong_weak.png")

    sig = json.loads(
        (OUT / chan / "diagnostics" / "signature.json").read_text(encoding="utf-8")
    )
    return {
        "channel": chan,
        "families": sig.get("families", []),
        "remaining_all_median": float(np.median(fracs)),
        "remaining_gate_gt_0.5_median": float(np.median(fracs[gates > 0.5]))
        if np.any(gates > 0.5)
        else None,
        "remaining_strongest25_median": float(np.median(fracs[strong25])),
        "frac_gate_gt_0": float(np.mean(gates > 0)),
        "frac_gate_gt_0.5": float(np.mean(gates > 0.5)),
        "gate0_removed_rms_median": gate0_rms,
        "strong_frame": strong,
        "weak_frame": weak,
        "strong_removed_rms": by[0]["removed_rms"],
        "weak_removed_rms": by[-1]["removed_rms"],
    }


def main():
    all_sum = {c: score_one(c) for c in ("ChanA", "ChanB")}
    (OUT / "summary.json").write_text(json.dumps(all_sum, indent=2), encoding="utf-8")
    lines = [
        "# v2.1 baseline on sandbox 500fr",
        "",
        "Outputs: `defringe_runs/v21_500fr/`",
        "",
    ]
    for c, s in all_sum.items():
        fam = s["families"][0] if s["families"] else {}
        lines += [
            f"## {c}",
            "",
            f"- family: q={fam.get('q')}, hi={fam.get('hi')}, "
            f"fx={fam.get('fx_ranges_weight_gt_0.20')}",
            (
                f"- remaining all / gate>0.5 / strongest25: "
                f"{pct(s['remaining_all_median']):.1f}% / "
                f"{pct(s['remaining_gate_gt_0.5_median']):.1f}% / "
                f"{pct(s['remaining_strongest25_median']):.1f}%"
            ),
            (
                f"- gate>0 / >0.5: {pct(s['frac_gate_gt_0']):.1f}% / "
                f"{pct(s['frac_gate_gt_0.5']):.1f}%"
            ),
            f"- gate0 removed RMS: {s['gate0_removed_rms_median']:.4f}",
            (
                f"- strong/weak frames: {s['strong_frame']} "
                f"(rms {s['strong_removed_rms']:.2f}) / {s['weak_frame']} "
                f"(rms {s['weak_removed_rms']:.2f})"
            ),
            "",
        ]
    text = "\n".join(lines)
    (OUT / "SUMMARY.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
