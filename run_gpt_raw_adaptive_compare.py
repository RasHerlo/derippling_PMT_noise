"""Run GPT pmt_fringe_raw_adaptive on raw test stacks and compare residuals."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

RAW_DIR = Path(
    r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\SUPPORT_ChanB\to build FFT deripple\raw test files"
)
OUT_ROOT = Path(
    r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\SUPPORT_ChanB\to build FFT deripple\cursor tests"
)
SCRIPT = Path(__file__).resolve().parent / "reference" / "gpt" / "pmt_fringe_raw_adaptive.py"
METHOD = "gpt_raw_adaptive_v2"


def to_u8(img, lo=None, hi=None, p=(1, 99.5)):
    x = img.astype(np.float32)
    if lo is None or hi is None:
        lo, hi = np.percentile(x, p)
    if hi <= lo:
        hi = lo + 1
    return (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def load_gates(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gates = [float(row[k]) for k in row if k.endswith("_gate")]
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "removed_rms": float(row["removed_rms"]),
                    "max_gate": max(gates) if gates else 0.0,
                    "mean_gate": float(np.mean(gates)) if gates else 0.0,
                }
            )
    return rows


def pick_frames(gate_rows, n_each=2):
    by_gate = sorted(gate_rows, key=lambda r: r["max_gate"], reverse=True)
    strong = [r["frame"] for r in by_gate[:n_each]]
    weak = [r["frame"] for r in by_gate[-n_each:]]
    # also a mid example
    mid = by_gate[len(by_gate) // 2]["frame"]
    frames = []
    for f in strong + [mid] + weak:
        if f not in frames:
            frames.append(f)
    return frames


def save_compare(path: Path, raw, cleaned, frames, labels=None):
    panels = []
    for i, fr in enumerate(frames):
        lo, hi = np.percentile(raw[fr], (1, 99.5))
        r = to_u8(raw[fr], lo, hi)
        c = to_u8(cleaned[fr], lo, hi)
        rem = to_u8(raw[fr].astype(np.float32) - cleaned[fr].astype(np.float32), p=(5, 99.5))
        panels.append(np.concatenate([r, c, rem], axis=1))
    Image.fromarray(np.concatenate(panels, axis=0)).save(path)
    txt = ["columns: raw | cleaned | removed(raw-cleaned)", f"rows frames: {frames}"]
    if labels:
        txt.append("labels: " + ", ".join(labels))
    path.with_suffix(".txt").write_text("\n".join(txt), encoding="utf-8")


def residual_energy_ratio(raw, cleaned, frames):
    out = []
    for fr in frames:
        a = raw[fr].astype(np.float64)
        b = cleaned[fr].astype(np.float64)
        resid = a - b
        out.append(
            {
                "frame": fr,
                "resid_rms": float(np.sqrt(np.mean(resid**2))),
                "resid_std": float(resid.std()),
                "raw_std": float(a.std()),
                "clean_std": float(b.std()),
                "corr_raw_clean": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
            }
        )
    return out


def compare_to_prior(stem: Path, frames, raw, new_cleaned):
    """Compare residual RMS of new method vs prior bakeoff methods on same frames."""
    prior_names = [
        ("gpt_adaptive_raw", f"{stem}_gpt_adaptive.tif"),
        ("rowband_raw", f"{stem}_rowband_raw.tif"),
        ("rowband_support2raw", f"{stem}_rowband_support2raw.tif"),
    ]
    rows = []
    for method, fname in prior_names:
        p = OUT_ROOT / stem / method / fname
        if not p.exists():
            continue
        prev = tifffile.imread(p)
        for fr in frames:
            r_new = raw[fr].astype(np.float64) - new_cleaned[fr].astype(np.float64)
            r_old = raw[fr].astype(np.float64) - prev[fr].astype(np.float64)
            rows.append(
                {
                    "frame": fr,
                    "prior_method": method,
                    "prior_resid_rms": float(np.sqrt(np.mean(r_old**2))),
                    "new_resid_rms": float(np.sqrt(np.mean(r_new**2))),
                    "prior_vs_new_resid_corr": float(
                        np.corrcoef(r_old.ravel(), r_new.ravel())[0, 1]
                    ),
                }
            )
    return rows


def run_one(raw_path: Path) -> dict:
    stem = raw_path.stem
    out_dir = OUT_ROOT / stem / METHOD
    diag_dir = out_dir / "diagnostics"
    out_tif = out_dir / f"{stem}_{METHOD}.tif"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT),
        str(raw_path),
        "-o",
        str(out_tif),
        "--diagnostics",
        str(diag_dir),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    raw = tifffile.imread(raw_path)
    cleaned = tifffile.imread(out_tif)
    gates = load_gates(diag_dir / "temporal_tracking.csv")
    frames = pick_frames(gates, n_each=2)
    labels = []
    gmap = {r["frame"]: r for r in gates}
    for fr in frames:
        labels.append(f"f{fr} gate={gmap[fr]['max_gate']:.2f} rms={gmap[fr]['removed_rms']:.2f}")

    save_compare(out_dir / f"{stem}_{METHOD}_compare.png", raw, cleaned, frames, labels)

    # dedicated strong/weak panels
    strong = sorted(gates, key=lambda r: r["max_gate"], reverse=True)[0]["frame"]
    weak = sorted(gates, key=lambda r: r["max_gate"])[0]["frame"]
    save_compare(
        out_dir / f"{stem}_{METHOD}_strong_weak.png",
        raw,
        cleaned,
        [strong, weak],
        [f"STRONG f{strong}", f"WEAK f{weak}"],
    )

    sig = json.loads((diag_dir / "signature.json").read_text(encoding="utf-8"))
    gate_arr = np.array([r["max_gate"] for r in gates], dtype=float)
    rms_arr = np.array([r["removed_rms"] for r in gates], dtype=float)

    summary = {
        "method": METHOD,
        "input": str(raw_path),
        "output": str(out_tif),
        "families": sig.get("families", []),
        "n_frames": int(raw.shape[0]),
        "frac_gated_gt_0": float(np.mean(gate_arr > 0)),
        "frac_gated_gt_0.5": float(np.mean(gate_arr > 0.5)),
        "median_removed_rms": float(np.median(rms_arr)),
        "max_removed_rms": float(np.max(rms_arr)),
        "example_frames": frames,
        "quality_examples": residual_energy_ratio(raw, cleaned, frames),
        "vs_prior": compare_to_prior(stem, frames, raw, cleaned),
        "strong_frame": strong,
        "weak_frame": weak,
        "weak_removed_rms": float(gmap[weak]["removed_rms"]),
        "strong_removed_rms": float(gmap[strong]["removed_rms"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_md(all_sum: dict) -> None:
    lines = [
        "# GPT raw-adaptive v2 vs prior bake-off",
        "",
        "New method folder: `<stack>/gpt_raw_adaptive_v2/`",
        "",
        "Compare PNGs: raw | cleaned | removed. Strong/weak panels use max/min gate.",
        "",
    ]
    for stem, s in all_sum.items():
        lines.append(f"## {stem}")
        lines.append("")
        lines.append("Detected families:")
        for i, fam in enumerate(s["families"], 1):
            lines.append(
                f"- family {i}: q={fam['q']}, hi={fam['hi']}, "
                f"row_z={fam['row_score']:.1f}, fx={fam['fx_ranges_weight_gt_0.20']}"
            )
        lines.append("")
        lines.append(
            f"Gate >0 on {100*s['frac_gated_gt_0']:.1f}% frames; "
            f">0.5 on {100*s['frac_gated_gt_0.5']:.1f}%. "
            f"Removed RMS strong/weak frames: {s['strong_removed_rms']:.3f} / {s['weak_removed_rms']:.3f}."
        )
        lines.append("")
        lines.append("| frame | new resid RMS | vs gpt_v1 | vs rowband_raw | vs support2raw |")
        lines.append("|---:|---:|---:|---:|---:|")
        # pivot vs_prior
        by_frame = {}
        for row in s["vs_prior"]:
            by_frame.setdefault(row["frame"], {})[row["prior_method"]] = row
        for fr in s["example_frames"]:
            d = by_frame.get(fr, {})
            new_rms = next(
                (q["resid_rms"] for q in s["quality_examples"] if q["frame"] == fr),
                float("nan"),
            )

            def cell(m):
                if m not in d:
                    return "-"
                return f"{d[m]['prior_resid_rms']:.2f} (corr={d[m]['prior_vs_new_resid_corr']:.2f})"

            lines.append(
                f"| {fr} | {new_rms:.2f} | {cell('gpt_adaptive_raw')} | "
                f"{cell('rowband_raw')} | {cell('rowband_support2raw')} |"
            )
        lines.append("")
        lines.append(
            "Interpretation tip: for weak-fringe frames, lower new resid RMS than priors "
            "is good (less over-filtering). For strong frames, similar residual structure "
            "with high correlation means comparable fringe isolation."
        )
        lines.append("")

    path = OUT_ROOT / "SUMMARY_gpt_raw_adaptive_v2.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", path)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("*_raw_500fr.tif"))
    if not files:
        raise SystemExit(f"No stacks in {RAW_DIR}")
    all_sum = {}
    for f in files:
        all_sum[f.stem] = run_one(f)
    (OUT_ROOT / "results_gpt_raw_adaptive_v2.json").write_text(
        json.dumps(all_sum, indent=2), encoding="utf-8"
    )
    write_md(all_sum)
    print("done")


if __name__ == "__main__":
    main()
