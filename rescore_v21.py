"""
Run GPT raw-adaptive v2.1 on 500-frame test stacks and re-score residual
fringe-specific excess power vs existing v2 outputs.

Metric (matches prior GPT remeasure):
  For each frame, measure Fourier power of excess above local spectral
  background inside the tracked PMT ridge segments (x_weight support).
  remaining_frac = excess_power(cleaned) / excess_power(raw)
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference" / "gpt"))
import pmt_fringe_raw_adaptive as v2  # noqa: E402

RAW_DIR = Path(
    r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\SUPPORT_ChanB\to build FFT deripple\raw test files"
)
OUT_ROOT = Path(
    r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\SUPPORT_ChanB\to build FFT deripple\cursor tests"
)
DATA_ROOT = Path(r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA")
SCRIPT_V21 = ROOT / "reference" / "gpt" / "pmt_fringe_raw_adaptive_v21.py"
METHOD = "gpt_raw_adaptive_v21"
STACKS = ("ChanA_raw_500fr", "ChanB_raw_500fr")


def to_u8(img, lo=None, hi=None, p=(1, 99.5)):
    x = img.astype(np.float32)
    if lo is None or hi is None:
        lo, hi = np.percentile(x, p)
    if hi <= lo:
        hi = lo + 1
    return (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def save_compare(path: Path, raw, cleaned, frames, labels=None):
    panels = []
    for fr in frames:
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


def load_tracking(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gates = [float(row[k]) for k in row if k.endswith("_gate") and "residual" not in k]
            strengths = [
                float(row[k]) for k in row if k.endswith("_strength") and "residual" not in k
            ]
            qs = [float(row[k]) for k in row if k.endswith("_q") and "residual" not in k]
            residual_passes = [
                int(float(row[k])) for k in row if k.endswith("_residual_pass")
            ]
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "removed_rms": float(row["removed_rms"]),
                    "max_gate": max(gates) if gates else 0.0,
                    "max_strength": max(strengths) if strengths else 0.0,
                    "qs": qs,
                    "residual_pass_any": int(any(residual_passes)) if residual_passes else 0,
                }
            )
    return rows


def fft_amp(frame: np.ndarray) -> np.ndarray:
    x = frame.astype(np.float64)
    x = x - np.median(x)
    return np.abs(np.fft.fftshift(np.fft.fft2(x)))


def ridge_excess_power(
    amp: np.ndarray,
    families: list,
    qs: list[float],
    *,
    y_radius: int = 2,
    x_weight_thresh: float = 0.05,
) -> float:
    """Sum of squared excess amplitude in ridge support vs local spectral background."""
    h, w = amp.shape
    cy = h // 2
    total = 0.0
    for family, q in zip(families, qs):
        xw = np.asarray(family["x_weight"], dtype=float)
        x_sel = xw > x_weight_thresh
        if not np.any(x_sel):
            continue
        dys = [q]
        if family.get("paired", True):
            dys.append(cy - q)
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
                    bg = np.median(np.stack(bgrows), axis=0)
                    excess = np.maximum(amp[yp, x_sel] - bg, 0.0) * xw[x_sel]
                    total += float(np.sum(excess * excess))
    return total


def summarize_remaining(fracs: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.median(fracs[mask]))


def score_stack(
    raw: np.ndarray,
    cleaned: np.ndarray,
    families: list,
    track_rows: list,
) -> dict:
    n = raw.shape[0]
    fracs = np.zeros(n, dtype=float)
    raw_powers = np.zeros(n, dtype=float)
    clean_powers = np.zeros(n, dtype=float)
    gates = np.array([r["max_gate"] for r in track_rows], dtype=float)
    strengths = np.array([r["max_strength"] for r in track_rows], dtype=float)

    for i in range(n):
        qs = track_rows[i]["qs"]
        if len(qs) < len(families):
            qs = qs + [families[j]["q"] for j in range(len(qs), len(families))]
        amp_r = fft_amp(raw[i])
        amp_c = fft_amp(cleaned[i])
        pr = ridge_excess_power(amp_r, families, qs)
        pc = ridge_excess_power(amp_c, families, qs)
        raw_powers[i] = pr
        clean_powers[i] = pc
        fracs[i] = pc / (pr + 1e-12)

    # strongest 25% by raw ridge excess power
    thr = np.percentile(raw_powers, 75)
    strong25 = raw_powers >= thr
    gate_gt_0 = gates > 0
    gate_gt_05 = gates > 0.5

    # gate-zero frames: cleaned should nearly match raw
    zero_idx = np.where(gates <= 1e-12)[0]
    if len(zero_idx):
        diffs = []
        for i in zero_idx[: min(20, len(zero_idx))]:
            d = raw[i].astype(np.float64) - cleaned[i].astype(np.float64)
            diffs.append(float(np.sqrt(np.mean(d * d))))
        gate0_rms = float(np.median(diffs))
    else:
        gate0_rms = float("nan")

    return {
        "remaining_all_median": summarize_remaining(fracs, np.ones(n, dtype=bool)),
        "remaining_gate_gt_0_median": summarize_remaining(fracs, gate_gt_0),
        "remaining_gate_gt_0.5_median": summarize_remaining(fracs, gate_gt_05),
        "remaining_strongest25_median": summarize_remaining(fracs, strong25),
        "frac_gate_gt_0": float(np.mean(gate_gt_0)),
        "frac_gate_gt_0.5": float(np.mean(gate_gt_05)),
        "gate0_removed_rms_median": gate0_rms,
        "per_frame_remaining_frac": fracs.tolist(),
        "per_frame_raw_excess_power": raw_powers.tolist(),
        "per_frame_clean_excess_power": clean_powers.tolist(),
        "per_frame_max_gate": gates.tolist(),
        "per_frame_max_strength": strengths.tolist(),
    }


def detect_families_from_raw(raw_path: Path) -> list:
    """Re-detect families (includes x_weight) using the same defaults as the cleaner."""
    with tifffile.TiffFile(raw_path) as tf:
        medspec, _ = v2.learn_median_spectrum(tf, sample_n=80)
        families, _, _ = v2.detect_families(
            medspec,
            row_z_thresh=5.5,
            pair_z_min=3.5,
            x_z_thresh=3.5,
            max_families=4,
            allow_standalone=False,
        )
    return families


def run_v21(raw_path: Path, *, force: bool = False) -> Path:
    stem = raw_path.stem
    out_dir = OUT_ROOT / stem / METHOD
    diag_dir = out_dir / "diagnostics"
    out_tif = out_dir / f"{stem}_{METHOD}.tif"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = diag_dir / "temporal_tracking.csv"
    settings_path = diag_dir / "v21_settings.json"
    settings_ok = False
    if settings_path.exists():
        try:
            st = json.loads(settings_path.read_text(encoding="utf-8"))
            settings_ok = (
                abs(float(st.get("high_strength", -1)) - 0.22) < 1e-9
                and abs(float(st.get("residual_strength_min", -1)) - 0.08) < 1e-9
            )
        except Exception:
            settings_ok = False
    if (
        not force
        and settings_ok
        and out_tif.exists()
        and csv_path.exists()
        and out_tif.stat().st_size > 1_000_000
    ):
        print(f"Reusing existing v2.1 output: {out_tif}", flush=True)
        return out_tif
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_V21),
        str(raw_path),
        "-o",
        str(out_tif),
        "--diagnostics",
        str(diag_dir),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(SCRIPT_V21.parent))
    return out_tif


def score_support_chain(stem: str, raw: np.ndarray, families: list, track_rows: list) -> dict | None:
    """Score raw -> v2-defringed -> SUPPORT on first N frames if stacks match."""
    chan = "ChanA" if "ChanA" in stem else "ChanB"
    raw_full = DATA_ROOT / chan / f"{chan}_stk.tif"
    defr = DATA_ROOT / f"{chan}_defringe" / f"{chan}_stk_defringed.tif"
    den_candidates = [
        DATA_ROOT / f"{chan}_defringe" / "SUPPORT" / f"{chan}_stk_defringed_denoised.tif",
        DATA_ROOT / f"{chan}_defringe" / f"{chan}_stk_defringed_denoised.tif",
        DATA_ROOT / f"SUPPORT_{chan}" / "denoised_cut.tif",
    ]
    den = next((p for p in den_candidates if p.exists()), None)
    if not (raw_full.exists() and defr.exists()):
        return {
            "matched_prefix": False,
            "note": f"missing raw/defringe full stacks for {chan}",
        }

    n = raw.shape[0]
    with tifffile.TiffFile(raw_full) as tf:
        probe = tf.pages[0].asarray()
        mid = tf.pages[min(100, n - 1)].asarray()
    if not np.array_equal(probe, raw[0]) or not np.array_equal(mid, raw[min(100, n - 1)]):
        return {
            "matched_prefix": False,
            "note": "500fr test stack is not a prefix of full raw; skipped SUPPORT chain",
        }

    with tifffile.TiffFile(defr) as tf:
        cleaned = np.stack([tf.pages[i].asarray() for i in range(n)])

    # Score with the family actually used for the full-stack defringe (may differ
    # from the 500fr test-stack detection, especially ChanA q=6 vs q=14).
    full_sig = DATA_ROOT / f"{chan}_defringe" / "diagnostics" / "signature.json"
    full_csv = DATA_ROOT / f"{chan}_defringe" / "diagnostics" / "temporal_tracking.csv"
    if full_sig.exists() and full_csv.exists():
        with tifffile.TiffFile(raw_full) as tf:
            medspec, _ = v2.learn_median_spectrum(tf, sample_n=80)
            fam_full, _, _ = v2.detect_families(
                medspec,
                row_z_thresh=5.5,
                pair_z_min=3.5,
                x_z_thresh=3.5,
                max_families=4,
                allow_standalone=False,
            )
        tr_full = load_tracking(full_csv)[:n]
        ridge_fams, ridge_tr = fam_full, tr_full
        out_note_q = (
            f"full-stack family q={[f['q'] for f in fam_full]}; "
            f"500fr test family q={[f['q'] for f in families]}"
        )
    else:
        v2_csv = OUT_ROOT / stem / "gpt_raw_adaptive_v2" / "diagnostics" / "temporal_tracking.csv"
        ridge_fams, ridge_tr = families, (load_tracking(v2_csv) if v2_csv.exists() else track_rows)
        out_note_q = "used 500fr families (full-stack diagnostics missing)"

    out = {
        "matched_prefix": True,
        "note": out_note_q,
        "v2_defringed_fullstack_family": score_stack(raw, cleaned, ridge_fams, ridge_tr),
        "support_path": str(den) if den else None,
    }
    # Also report residual at the 500fr test family (shows if full-stack missed it)
    v2_csv = OUT_ROOT / stem / "gpt_raw_adaptive_v2" / "diagnostics" / "temporal_tracking.csv"
    tr_500 = load_tracking(v2_csv) if v2_csv.exists() else track_rows
    out["v2_defringed_at_500fr_family"] = score_stack(raw, cleaned, families, tr_500)

    for key in ("v2_defringed_fullstack_family", "v2_defringed_at_500fr_family"):
        s = out[key]
        out[key] = {
            "remaining_all_median": s["remaining_all_median"],
            "remaining_gate_gt_0.5_median": s["remaining_gate_gt_0.5_median"],
            "remaining_strongest25_median": s["remaining_strongest25_median"],
        }

    if den is not None:
        label = "v2_then_SUPPORT_fullstack_family"
        with tifffile.TiffFile(den) as tf:
            support = np.stack([tf.pages[i].asarray() for i in range(n)])
        s = score_stack(raw, support, ridge_fams, ridge_tr)
        out[label] = {
            "remaining_all_median": s["remaining_all_median"],
            "remaining_gate_gt_0.5_median": s["remaining_gate_gt_0.5_median"],
            "remaining_strongest25_median": s["remaining_strongest25_median"],
        }
        s2 = score_stack(raw, support, families, tr_500)
        out["v2_then_SUPPORT_at_500fr_family"] = {
            "remaining_all_median": s2["remaining_all_median"],
            "remaining_gate_gt_0.5_median": s2["remaining_gate_gt_0.5_median"],
            "remaining_strongest25_median": s2["remaining_strongest25_median"],
        }
    else:
        out["note"] += "; SUPPORT/denoised full stack not found on disk"
    return out


def run_one(stem: str) -> dict:
    raw_path = RAW_DIR / f"{stem}.tif"
    out_tif = run_v21(raw_path)
    out_dir = OUT_ROOT / stem / METHOD
    diag_dir = out_dir / "diagnostics"

    raw = tifffile.imread(raw_path)
    cleaned_v21 = tifffile.imread(out_tif)
    families = detect_families_from_raw(raw_path)
    track_v21 = load_tracking(diag_dir / "temporal_tracking.csv")
    sig_families = json.loads((diag_dir / "signature.json").read_text(encoding="utf-8")).get(
        "families", []
    )

    # existing v2
    v2_dir = OUT_ROOT / stem / "gpt_raw_adaptive_v2"
    v2_tif = v2_dir / f"{stem}_gpt_raw_adaptive_v2.tif"
    track_v2 = load_tracking(v2_dir / "diagnostics" / "temporal_tracking.csv")
    cleaned_v2 = tifffile.imread(v2_tif)

    # Stratify with a common gate from raw strength: use v2 gates for frame sets,
    # but report each method's own remaining fractions on those frames.
    gates_v2 = np.array([r["max_gate"] for r in track_v2], dtype=float)
    raw_powers = np.array(
        [
            ridge_excess_power(fft_amp(raw[i]), families, track_v2[i]["qs"])
            for i in range(raw.shape[0])
        ],
        dtype=float,
    )
    strong25 = raw_powers >= np.percentile(raw_powers, 75)
    common_masks = {
        "all": np.ones(raw.shape[0], dtype=bool),
        "gate_v2_gt_0.5": gates_v2 > 0.5,
        "strongest25_raw_excess": strong25,
    }

    def remaining_on(cleaned, track):
        fracs = []
        for i in range(raw.shape[0]):
            qs = track[i]["qs"]
            pr = ridge_excess_power(fft_amp(raw[i]), families, qs)
            pc = ridge_excess_power(fft_amp(cleaned[i]), families, qs)
            fracs.append(pc / (pr + 1e-12))
        return np.asarray(fracs, dtype=float)

    frac_v2 = remaining_on(cleaned_v2, track_v2)
    frac_v21 = remaining_on(cleaned_v21, track_v21)

    table = {}
    for name, mask in common_masks.items():
        table[name] = {
            "v2_median_remaining": summarize_remaining(frac_v2, mask),
            "v21_median_remaining": summarize_remaining(frac_v21, mask),
            "n_frames": int(np.sum(mask)),
        }

    score_v2 = score_stack(raw, cleaned_v2, families, track_v2)
    score_v21 = score_stack(raw, cleaned_v21, families, track_v21)

    # example frames
    by_gate = sorted(track_v21, key=lambda r: r["max_gate"], reverse=True)
    strong = by_gate[0]["frame"]
    weak = by_gate[-1]["frame"]
    mid = by_gate[len(by_gate) // 2]["frame"]
    frames = []
    for f in (strong, by_gate[1]["frame"], mid, weak):
        if f not in frames:
            frames.append(f)
    labels = [
        f"f{fr} gate={track_v21[fr]['max_gate']:.2f} rms={track_v21[fr]['removed_rms']:.2f}"
        for fr in frames
    ]
    save_compare(out_dir / f"{stem}_{METHOD}_compare.png", raw, cleaned_v21, frames, labels)
    save_compare(
        out_dir / f"{stem}_{METHOD}_strong_weak.png",
        raw,
        cleaned_v21,
        [strong, weak],
        [f"STRONG f{strong}", f"WEAK f{weak}"],
    )

    residual_pass_frac = float(np.mean([r["residual_pass_any"] for r in track_v21]))
    high_alpha_frac = 0.0
    with open(diag_dir / "temporal_tracking.csv", newline="", encoding="utf-8") as f:
        alphas = []
        for row in csv.DictReader(f):
            for k, v in row.items():
                if k.endswith("_eff_max_alpha"):
                    alphas.append(float(v))
        if alphas:
            high_alpha_frac = float(np.mean(np.asarray(alphas) > 0.86))

    support_chain = score_support_chain(stem, raw, families, track_v2)

    summary = {
        "method": METHOD,
        "input": str(raw_path),
        "output": str(out_tif),
        "families": sig_families,
        "common_frame_comparison": table,
        "v2_own_gate_metrics": {
            k: score_v2[k]
            for k in (
                "remaining_all_median",
                "remaining_gate_gt_0.5_median",
                "remaining_strongest25_median",
                "frac_gate_gt_0",
                "frac_gate_gt_0.5",
                "gate0_removed_rms_median",
            )
        },
        "v21_own_gate_metrics": {
            k: score_v21[k]
            for k in (
                "remaining_all_median",
                "remaining_gate_gt_0.5_median",
                "remaining_strongest25_median",
                "frac_gate_gt_0",
                "frac_gate_gt_0.5",
                "gate0_removed_rms_median",
            )
        },
        "v21_residual_pass_frac": residual_pass_frac,
        "v21_frac_eff_alpha_gt_0.86": high_alpha_frac,
        "support_chain_500fr_prefix": support_chain,
        "strong_frame": strong,
        "weak_frame": weak,
    }
    slim_fams = []
    for f in summary["families"]:
        slim_fams.append(
            {
                "q": f.get("q"),
                "hi": f.get("hi"),
                "row_score": f.get("row_score"),
                "paired": f.get("paired"),
                "fx_ranges": f.get("fx_ranges") or f.get("fx_ranges_weight_gt_0.20"),
            }
        )
    summary["families"] = slim_fams
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_md(all_sum: dict) -> None:
    lines = [
        "# GPT raw-adaptive v2.1 re-score",
        "",
        "Confidence-scaled `max_alpha` (0.85→0.97 at high gate) + residual second pass.",
        "",
        "Remaining = median fringe-specific excess power(cleaned) / excess power(raw)",
        "inside tracked PMT ridge segments.",
        "",
    ]
    for stem, s in all_sum.items():
        lines += [f"## {stem}", ""]
        lines.append("Detected families:")
        for i, f in enumerate(s["families"], 1):
            lines.append(
                f"- family {i}: q={f['q']}, hi={f['hi']}, row_z={f.get('row_score')}, fx={f.get('fx_ranges')}"
            )
        lines.append("")
        lines.append("| Frame set | n | v2 remaining | v2.1 remaining |")
        lines.append("|---|---:|---:|---:|")
        for name, row in s["common_frame_comparison"].items():
            lines.append(
                f"| {name} | {row['n_frames']} | "
                f"{100*row['v2_median_remaining']:.1f}% | "
                f"{100*row['v21_median_remaining']:.1f}% |"
            )
        m = s["v21_own_gate_metrics"]
        lines += [
            "",
            f"- v2.1 gate>0: {100*m['frac_gate_gt_0']:.1f}%; gate>0.5: {100*m['frac_gate_gt_0.5']:.1f}%",
            f"- residual second pass used on {100*s['v21_residual_pass_frac']:.1f}% of frames",
            f"- eff_max_alpha > 0.86 on {100*s['v21_frac_eff_alpha_gt_0.86']:.1f}% of family-frame entries",
            f"- gate≈0 removed RMS (should be ~0): {m['gate0_removed_rms_median']:.4f}",
            "",
        ]
        sc = s.get("support_chain_500fr_prefix")
        if sc and sc.get("matched_prefix"):
            lines += [
                "Fourier ridge remaining on matching 500-frame prefix:",
                "",
                "| Stage | all | gate>0.5 | strongest 25% |",
                "|---|---:|---:|---:|",
            ]
            def pct(x):
                return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:.1f}%"

            for stage, r in sc.items():
                if not isinstance(r, dict) or "remaining_all_median" not in r:
                    continue
                lines.append(
                    f"| {stage} | {pct(r['remaining_all_median'])} | "
                    f"{pct(r['remaining_gate_gt_0.5_median'])} | "
                    f"{pct(r['remaining_strongest25_median'])} |"
                )
            if sc.get("support_path"):
                lines.append(f"- SUPPORT file: `{sc['support_path']}`")
            if sc.get("note"):
                lines.append(f"- note: {sc['note']}")
            lines.append("")
        elif sc:
            lines += [f"- SUPPORT chain: {sc.get('note', 'skipped')}", ""]

    path = OUT_ROOT / "SUMMARY_gpt_raw_adaptive_v21.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def main():
    all_sum = {}
    for stem in STACKS:
        print(f"\n=== {stem} ===", flush=True)
        all_sum[stem] = run_one(stem)
    (OUT_ROOT / "results_gpt_raw_adaptive_v21.json").write_text(
        json.dumps(all_sum, indent=2), encoding="utf-8"
    )
    write_md(all_sum)
    print("Done.")


if __name__ == "__main__":
    main()
