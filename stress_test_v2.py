"""
Stress-test GPT v2 raw-adaptive defaults on 500-frame raw stacks.

1) Gate/q/removed-RMS continuity over time
2) Pseudo-ground-truth fringe injection with two noise sources:
   - v2 residual: N = raw - v2_cleaned
   - broad spectral extract from strong frames (wider than v2 support)

Writes under:
  .../cursor tests/<stack>/stress_test_v2/
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference" / "gpt"))
import pmt_fringe_raw_adaptive as v2  # noqa: E402

RAW_DIR = Path(
    r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\SUPPORT_ChanB\to build FFT deripple\raw test files"
)
OUT_ROOT = Path(
    r"F:\bPACNewData2026\260511\C1_RLV_LW_maybe\LED_x15_Level3b\DATA\SUPPORT_ChanB\to build FFT deripple\cursor tests"
)
ALPHAS = (0.25, 0.5, 1.0, 1.5)
N_WEAK = 3
N_STRONG = 3


def load_tracking(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gates = [float(row[k]) for k in row if k.endswith("_gate")]
            qs = [float(row[k]) for k in row if k.endswith("_q")]
            strengths = [float(row[k]) for k in row if k.endswith("_strength")]
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "removed_rms": float(row["removed_rms"]),
                    "max_gate": max(gates) if gates else 0.0,
                    "mean_gate": float(np.mean(gates)) if gates else 0.0,
                    "q": float(qs[0]) if qs else np.nan,
                    "strength": float(strengths[0]) if strengths else np.nan,
                }
            )
    return rows


def continuity_metrics(track):
    rms = np.array([r["removed_rms"] for r in track], float)
    gate = np.array([r["max_gate"] for r in track], float)
    q = np.array([r["q"] for r in track], float)
    d_rms = np.diff(rms)
    d_gate = np.diff(gate)
    d_q = np.diff(q)

    # jumps near partially-open gate
    mid = (gate[1:] > 0.05) & (gate[1:] < 0.95) | (gate[:-1] > 0.05) & (gate[:-1] < 0.95)
    return {
        "removed_rms_median_abs_diff": float(np.median(np.abs(d_rms))),
        "removed_rms_p99_abs_diff": float(np.percentile(np.abs(d_rms), 99)),
        "gate_median_abs_diff": float(np.median(np.abs(d_gate))),
        "gate_p99_abs_diff": float(np.percentile(np.abs(d_gate), 99)),
        "q_median_abs_diff": float(np.nanmedian(np.abs(d_q))),
        "q_p99_abs_diff": float(np.nanpercentile(np.abs(d_q), 99)),
        "n_gate_flip_gt_0.5": int(np.sum(np.abs(d_gate) >= 0.5)),
        "n_rms_jump_gt_5_near_partial_gate": int(np.sum((np.abs(d_rms) > 5.0) & mid)),
        "frac_gate_gt_0": float(np.mean(gate > 0)),
        "frac_gate_gt_0.5": float(np.mean(gate > 0.5)),
        "frac_gate_eq_0": float(np.mean(gate <= 1e-9)),
    }


def save_continuity_plots(path: Path, track):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = [r["frame"] for r in track]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(frames, [r["removed_rms"] for r in track], lw=0.9)
    axes[0].set_ylabel("removed RMS")
    axes[0].set_title("v2 temporal continuity")
    axes[1].plot(frames, [r["max_gate"] for r in track], lw=0.9, color="C1")
    axes[1].set_ylabel("max gate")
    axes[2].plot(frames, [r["q"] for r in track], lw=0.9, color="C2")
    axes[2].set_ylabel("tracked q")
    axes[2].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def family_mask(shape, families, y_pad=4, fx_pad=12, x_weight_thresh=0.05):
    """Boolean FFT mask covering fringe bands (optionally broadened)."""
    h, w = shape
    cy, cx = h // 2, w // 2
    fx = np.arange(w) - cx
    mask = np.zeros((h, w), dtype=bool)
    for fam in families:
        dys = [fam["q"]]
        if fam.get("hi") is not None:
            dys.append(fam["hi"])
        # broaden fx support
        xw = np.asarray(fam["x_weight"], dtype=float)
        if fx_pad > 0:
            # dilate support in fx
            support = xw > x_weight_thresh
            idx = np.where(support)[0]
            broad = support.copy()
            for i in idx:
                broad[max(0, i - fx_pad) : i + fx_pad + 1] = True
            x_sel = broad
        else:
            x_sel = xw > max(x_weight_thresh, 0.2)

        for d in dys:
            for sgn in (-1, +1):
                yc = cy + sgn * int(round(d))
                for yp in range(yc - y_pad, yc + y_pad + 1):
                    if 0 <= yp < h:
                        mask[yp, x_sel] = True
        # never touch DC neighborhood
    yy, xx = np.ogrid[:h, :w]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 < 8**2] = False
    return mask


def fringe_band_power(frame, mask):
    x = frame.astype(np.float64)
    x = x - np.median(x)
    F = np.fft.fftshift(np.fft.fft2(x))
    p = (np.abs(F) ** 2)
    return float(p[mask].sum()), float(p.sum())


def extract_broad_fringe(strong_frame, families, y_pad=5, fx_pad=18):
    """Spectral extract of fringe using a broader mask than v2 defaults."""
    x = strong_frame.astype(np.float64)
    med = np.median(x)
    x0 = x - med
    F = np.fft.fftshift(np.fft.fft2(x0))
    mask = family_mask(x.shape, families, y_pad=y_pad, fx_pad=fx_pad)
    F2 = np.zeros_like(F)
    F2[mask] = F[mask]
    n = np.fft.ifft2(np.fft.ifftshift(F2)).real
    # zero-mean fringe component
    n = n - np.mean(n)
    return n.astype(np.float32), mask


def rms(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a)))


def pick_frames(track, n_weak=3, n_strong=3):
    # weak: lowest removed rms with gate~0 preferred
    weak = sorted(track, key=lambda r: (r["max_gate"], r["removed_rms"]))[: max(n_weak * 3, n_weak)]
    weak_frames = []
    for r in weak:
        if r["frame"] not in weak_frames:
            weak_frames.append(r["frame"])
        if len(weak_frames) >= n_weak:
            break

    strong = sorted(track, key=lambda r: (r["removed_rms"], r["max_gate"]), reverse=True)
    strong_frames = []
    for r in strong:
        if r["frame"] not in strong_frames and r["frame"] not in weak_frames:
            strong_frames.append(r["frame"])
        if len(strong_frames) >= n_strong:
            break
    return weak_frames, strong_frames


def rebuild_families_from_stack(raw_path: Path):
    with tifffile.TiffFile(raw_path) as tf:
        medspec, _ = v2.learn_median_spectrum(tf, sample_n=80)
        families, row_profile, row_z = v2.detect_families(
            medspec,
            row_z_thresh=5.5,
            pair_z_min=3.5,
            x_z_thresh=3.5,
            max_families=4,
            allow_standalone=False,
        )
        trajectories = []
        for fam in families:
            qtraj, _ = v2.track_family_blocks(tf, fam)
            trajectories.append(qtraj)
    return families, trajectories, medspec


def run_injection_suite(raw, families, trajectories, track, out_dir: Path):
    weak_frames, strong_frames = pick_frames(track, N_WEAK, N_STRONG)
    # v2-tight mask for remaining-fringe score
    tight_mask = family_mask(raw.shape[1:], families, y_pad=2, fx_pad=0, x_weight_thresh=0.2)

    rows = []
    examples = []

    # compute N sources from strong frames
    noise_bank = {}
    for s_idx in strong_frames:
        preds_s = [traj[s_idx] for traj in trajectories]
        cleaned_s, _, _ = v2.clean_frame(raw[s_idx], families, preds_s)
        N_v2 = raw[s_idx].astype(np.float32) - cleaned_s.astype(np.float32)
        N_broad, broad_mask = extract_broad_fringe(raw[s_idx], families)
        noise_bank[s_idx] = {
            "v2_residual": N_v2,
            "broad_spectral": N_broad.astype(np.float32),
            "broad_mask": broad_mask,
            "preds": preds_s,
            "N_v2_rms": rms(N_v2),
            "N_broad_rms": rms(N_broad),
        }

    for w_idx in weak_frames:
        I0 = raw[w_idx].astype(np.float32)
        # sanity: cleaning I0 alone should barely change it
        preds_w = [traj[w_idx] for traj in trajectories]
        _, rem0, tr0 = v2.clean_frame(I0, families, preds_w)
        baseline_gate = float(max(t["gate"] for t in tr0)) if tr0 else 0.0

        for s_idx, noises in noise_bank.items():
            # Injected fringe carries the strong-frame frequency; use that q prior.
            preds_clean = noises["preds"]
            for noise_name, N in (
                ("v2_residual", noises["v2_residual"]),
                ("broad_spectral", noises["broad_spectral"]),
            ):
                N_use = N
                for alpha in ALPHAS:
                    I_test = I0 + np.float32(alpha) * N_use
                    cleaned, _, tracking = v2.clean_frame(
                        I_test.astype(np.float32), families, preds_clean
                    )
                    I_rec = cleaned.astype(np.float32)
                    E_recovery = rms(I_rec - I0)

                    inj = (alpha * N_use).astype(np.float64)
                    resid = (I_rec - I0).astype(np.float64)
                    x_res = resid - np.median(resid)
                    x_inj = inj - np.mean(inj)
                    F_res = np.fft.fftshift(np.fft.fft2(x_res))
                    F_inj = np.fft.fftshift(np.fft.fft2(x_inj))
                    p_res_band = float((np.abs(F_res)[tight_mask] ** 2).sum())
                    p_inj_band = float((np.abs(F_inj)[tight_mask] ** 2).sum())
                    remaining_frac = p_res_band / (p_inj_band + 1e-12)

                    gate = float(max(t["gate"] for t in tracking)) if tracking else 0.0
                    row = {
                        "weak_frame": w_idx,
                        "strong_frame": s_idx,
                        "noise": noise_name,
                        "alpha": alpha,
                        "baseline_gate_on_I0": baseline_gate,
                        "baseline_rms_on_I0": rms(rem0),
                        "E_recovery": E_recovery,
                        "E_remaining_total": rms(resid),
                        "remaining_fringe_band_frac": remaining_frac,
                        "injected_rms": rms(alpha * N_use),
                        "gate_on_test": gate,
                        "N_source_rms": float(
                            noises["N_v2_rms"]
                            if noise_name == "v2_residual"
                            else noises["N_broad_rms"]
                        ),
                    }
                    rows.append(row)

                    if alpha == 1.0 and len(examples) < 8:
                        examples.append(
                            {
                                "meta": row,
                                "I0": I0,
                                "I_test": I_test,
                                "I_rec": I_rec,
                                "N": N_use,
                            }
                        )

    # write CSV
    csv_path = out_dir / "injection_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # aggregate curves
    curves = {}
    for noise_name in ("v2_residual", "broad_spectral"):
        curves[noise_name] = []
        for alpha in ALPHAS:
            sub = [r for r in rows if r["noise"] == noise_name and r["alpha"] == alpha]
            curves[noise_name].append(
                {
                    "alpha": alpha,
                    "E_recovery_median": float(np.median([r["E_recovery"] for r in sub])),
                    "E_recovery_mean": float(np.mean([r["E_recovery"] for r in sub])),
                    "remaining_fringe_band_frac_median": float(
                        np.median([r["remaining_fringe_band_frac"] for r in sub])
                    ),
                    "gate_median": float(np.median([r["gate_on_test"] for r in sub])),
                    "n": len(sub),
                }
            )

    return {
        "weak_frames": weak_frames,
        "strong_frames": strong_frames,
        "curves": curves,
        "n_rows": len(rows),
        "examples": examples,
        "noise_bank_rms": {
            str(k): {"v2_residual": v["N_v2_rms"], "broad_spectral": v["N_broad_rms"]}
            for k, v in noise_bank.items()
        },
    }, rows


def save_curve_plot(path: Path, curves: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for noise_name, style in (("v2_residual", "-o"), ("broad_spectral", "--s")):
        c = curves[noise_name]
        a = [x["alpha"] for x in c]
        axes[0].plot(a, [x["E_recovery_median"] for x in c], style, label=noise_name)
        axes[1].plot(
            a, [x["remaining_fringe_band_frac_median"] for x in c], style, label=noise_name
        )
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("median E_recovery = RMS(I_rec - I0)")
    axes[0].set_title("Biological distortion")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("median remaining fringe-band power frac")
    axes[1].set_title("Residual PMT fringe in bands")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_example_panels(path: Path, examples):
    from PIL import Image

    def u8(img, p=(1, 99.5)):
        lo, hi = np.percentile(img, p)
        return (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)

    panels = []
    for ex in examples[:4]:
        I0, It, Ir, N = ex["I0"], ex["I_test"], ex["I_rec"], ex["N"]
        lo, hi = np.percentile(It, (1, 99.5))
        row = np.concatenate(
            [
                u8(I0, (1, 99.5)),
                u8(It, (1, 99.5)),
                u8(Ir, (1, 99.5)),
                u8(Ir - I0, (5, 99.5)),
                u8(N, (5, 99.5)),
            ],
            axis=1,
        )
        panels.append(row)
        _ = lo, hi
    if panels:
        Image.fromarray(np.concatenate(panels, axis=0)).save(path)
        path.with_suffix(".txt").write_text(
            "columns: I0 | I_test | I_rec | I_rec-I0 | N\n"
            + "\n".join(
                f"weak={e['meta']['weak_frame']} strong={e['meta']['strong_frame']} "
                f"noise={e['meta']['noise']} alpha={e['meta']['alpha']} "
                f"E_rec={e['meta']['E_recovery']:.3f} remain_frac={e['meta']['remaining_fringe_band_frac']:.3f}"
                for e in examples[:4]
            ),
            encoding="utf-8",
        )


def process_stack(raw_path: Path) -> dict:
    stem = raw_path.stem
    v2_dir = OUT_ROOT / stem / "gpt_raw_adaptive_v2"
    track_csv = v2_dir / "diagnostics" / "temporal_tracking.csv"
    if not track_csv.exists():
        raise FileNotFoundError(f"Missing v2 tracking CSV: {track_csv}")

    out_dir = OUT_ROOT / stem / "stress_test_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Stress test {stem} ===", flush=True)
    track = load_tracking(track_csv)
    cont = continuity_metrics(track)
    save_continuity_plots(out_dir / "continuity.png", track)
    with open(out_dir / "continuity_metrics.json", "w", encoding="utf-8") as f:
        json.dump(cont, f, indent=2)
    print("continuity:", json.dumps(cont), flush=True)

    print("relearning families/trajectories from raw...", flush=True)
    families, trajectories, _ = rebuild_families_from_stack(raw_path)
    if not families:
        raise RuntimeError("No families detected for stress test")
    print("families:", [(f["q"], f["hi"], f["fx_ranges"]) for f in families], flush=True)

    print("loading raw stack...", flush=True)
    raw = tifffile.imread(raw_path)
    inj_summary, rows = run_injection_suite(raw, families, trajectories, track, out_dir)
    examples = inj_summary.pop("examples")
    save_curve_plot(out_dir / "recovery_curves.png", inj_summary["curves"])
    save_example_panels(out_dir / "injection_examples.png", examples)

    result = {
        "stack": stem,
        "continuity": cont,
        "injection": inj_summary,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # drop heavy examples from return
    return result


def write_global_summary(all_results: dict):
    lines = [
        "# v2 stress test: continuity + pseudo-ground-truth injection",
        "",
        "Injection: I_test = I0 + alpha * N, then clean with v2 defaults (learned families/q).",
        "",
        "- I0 = naturally weak-fringe raw frames",
        "- N = `v2_residual` (raw-v2) or `broad_spectral` (wider FFT extract)",
        "- E_recovery = RMS(I_rec - I0)",
        "- remaining_fringe_band_frac = fringe-band power of (I_rec-I0) / power of injected N",
        "",
    ]
    for stem, res in all_results.items():
        c = res["continuity"]
        lines.append(f"## {stem}")
        lines.append("")
        lines.append("### Continuity")
        lines.append(
            f"- gate>0 on {100*c['frac_gate_gt_0']:.1f}% frames; "
            f"gate==0 on {100*c['frac_gate_eq_0']:.1f}%"
        )
        lines.append(
            f"- median |d(removed_rms)|={c['removed_rms_median_abs_diff']:.3f}, "
            f"p99={c['removed_rms_p99_abs_diff']:.3f}"
        )
        lines.append(
            f"- median |d(gate)|={c['gate_median_abs_diff']:.3f}, "
            f"p99={c['gate_p99_abs_diff']:.3f}, "
            f"flips>=0.5: {c['n_gate_flip_gt_0.5']}"
        )
        lines.append(
            f"- median |d(q)|={c['q_median_abs_diff']:.3f}, "
            f"p99={c['q_p99_abs_diff']:.3f}"
        )
        lines.append(
            f"- large RMS jumps near partial gate: {c['n_rms_jump_gt_5_near_partial_gate']}"
        )
        lines.append("")
        lines.append("### Injection curves (median over weak x strong pairs)")
        lines.append("")
        lines.append("| noise | alpha | E_recovery | remain_band_frac | gate |")
        lines.append("|---|---:|---:|---:|---:|")
        for noise_name, curve in res["injection"]["curves"].items():
            for pt in curve:
                lines.append(
                    f"| {noise_name} | {pt['alpha']} | {pt['E_recovery_median']:.3f} | "
                    f"{pt['remaining_fringe_band_frac_median']:.3f} | {pt['gate_median']:.2f} |"
                )
        lines.append("")
        lines.append(
            f"Weak frames: {res['injection']['weak_frames']}; "
            f"strong frames: {res['injection']['strong_frames']}"
        )
        lines.append("")
        lines.append(f"Artifacts: `{stem}/stress_test_v2/`")
        lines.append("")

    # overall recommendation stub from numbers
    lines.append("## Quick read")
    lines.append("")
    lines.append(
        "Prefer defaults where remain_band_frac drops with alpha while E_recovery stays "
        "small relative to injected_rms (ideally E_recovery << injected_rms at alpha=1)."
    )
    path = OUT_ROOT / "SUMMARY_stress_test_v2.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", path)


def main():
    files = sorted(RAW_DIR.glob("*_raw_500fr.tif"))
    if not files:
        raise SystemExit(f"No stacks in {RAW_DIR}")
    all_results = {}
    for f in files:
        all_results[f.stem] = process_stack(f)
    (OUT_ROOT / "results_stress_test_v2.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    write_global_summary(all_results)
    print("done")


if __name__ == "__main__":
    main()
