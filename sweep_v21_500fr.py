"""
ChanA-focused high-confidence parameter sweep for v2.1.

Sweeps only aggression knobs (no wider fx / full-row masks). Detect+track once
per channel, clean in-memory per config, score residual ridge excess.

Writes under:
  .../Level3b copy/defringe_runs/v21_sweep_500fr/
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference" / "gpt"))
import pmt_fringe_raw_adaptive as v2  # noqa: E402
from pmt_fringe_raw_adaptive_v21 import clean_frame_v21  # noqa: E402

SANDBOX = Path(r"F:\bPACNewData2026\PreProcessing Optimization\Level3b copy")
OUT = SANDBOX / "defringe_runs" / "v21_sweep_500fr"

# Baseline = current v2.1 defaults. Other configs only raise high-confidence aggression.
BASE = dict(
    max_alpha=0.85,
    max_alpha_high=0.97,
    high_gate=0.95,
    high_strength=0.22,
    strength_span=0.15,
    residual_pass=True,
    residual_strength_min=0.08,
    residual_alpha=0.70,
)

CONFIGS = [
    ("baseline", dict(BASE)),
    ("high99", {**BASE, "max_alpha_high": 0.99}),
    ("high100", {**BASE, "max_alpha_high": 1.00}),
    ("resid85", {**BASE, "residual_alpha": 0.85}),
    ("resid95", {**BASE, "residual_alpha": 0.95}),
    ("resid_min05", {**BASE, "residual_strength_min": 0.05}),
    ("hs18", {**BASE, "high_strength": 0.18}),
    (
        "pack_A",
        {
            **BASE,
            "max_alpha_high": 0.99,
            "residual_alpha": 0.85,
            "residual_strength_min": 0.05,
            "high_strength": 0.18,
        },
    ),
    (
        "pack_B",
        {
            **BASE,
            "max_alpha_high": 1.00,
            "residual_alpha": 0.95,
            "residual_strength_min": 0.05,
            "high_strength": 0.18,
            "strength_span": 0.12,
        },
    ),
]


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


def to_u8(img, lo=None, hi=None, p=(1, 99.5)):
    x = img.astype(np.float32)
    if lo is None or hi is None:
        lo, hi = np.percentile(x, p)
    if hi <= lo:
        hi = lo + 1
    return (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def prepare_channel(chan: str):
    raw_path = SANDBOX / "inputs" / "slices_500fr" / "raw" / f"{chan}_raw_500fr.tif"
    print(f"Preparing {chan}: {raw_path}", flush=True)
    with tifffile.TiffFile(raw_path) as tf:
        med, _ = v2.learn_median_spectrum(tf, sample_n=80)
        families, _, _ = v2.detect_families(
            med,
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
        raw = np.stack([tf.pages[i].asarray() for i in range(tf.series[0].shape[0])])
    print(
        f"  families: {[ (f['q'], f['hi'], f['fx_ranges']) for f in families ]}",
        flush=True,
    )
    return raw_path, raw, families, trajectories


def run_config(raw, families, trajectories, params: dict):
    n = raw.shape[0]
    cleaned = np.empty_like(raw)
    gates = np.zeros(n, dtype=float)
    rms = np.zeros(n, dtype=float)
    qs_per = []
    resid_pass = np.zeros(n, dtype=int)
    for i in range(n):
        preds = [traj[i] for traj in trajectories]
        out, removed, tracking = clean_frame_v21(
            raw[i], families, preds, **params
        )
        cleaned[i] = out
        gates[i] = max(t["gate"] for t in tracking) if tracking else 0.0
        rms[i] = float(np.sqrt(np.mean(removed.astype(float) ** 2)))
        qs_per.append([t["q"] for t in tracking])
        resid_pass[i] = int(any(t.get("residual_pass", 0) for t in tracking))
        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"    cleaned {i+1}/{n}", flush=True)
    return cleaned, gates, rms, qs_per, resid_pass


def score(raw, cleaned, families, gates, rms, qs_per, resid_pass):
    n = raw.shape[0]
    fracs = np.zeros(n)
    raw_p = np.zeros(n)
    for i in range(n):
        pr = ridge_excess_power(fft_amp(raw[i]), families, qs_per[i])
        pc = ridge_excess_power(fft_amp(cleaned[i]), families, qs_per[i])
        raw_p[i] = pr
        fracs[i] = pc / (pr + 1e-12)
    strong25 = raw_p >= np.percentile(raw_p, 75)
    zero = gates <= 1e-12
    if np.any(zero):
        gate0_rms = float(np.median(rms[zero]))
    else:
        gate0_rms = float("nan")
    return {
        "remaining_all_median": float(np.median(fracs)),
        "remaining_gate_gt_0.5_median": float(np.median(fracs[gates > 0.5]))
        if np.any(gates > 0.5)
        else None,
        "remaining_strongest25_median": float(np.median(fracs[strong25])),
        "frac_gate_gt_0": float(np.mean(gates > 0)),
        "frac_gate_gt_0.5": float(np.mean(gates > 0.5)),
        "gate0_removed_rms_median": gate0_rms,
        "residual_pass_frac": float(np.mean(resid_pass)),
        "median_removed_rms": float(np.median(rms)),
        "max_removed_rms": float(np.max(rms)),
        "fracs": fracs,
        "gates": gates,
        "rms": rms,
        "raw_p": raw_p,
    }


def save_compare(path: Path, raw, cleaned, gates):
    order = np.argsort(gates)
    strong = int(order[-1])
    weak = int(order[0])
    panels = []
    for fr in (strong, weak):
        lo, hi = np.percentile(raw[fr], (1, 99.5))
        panels.append(
            np.concatenate(
                [
                    to_u8(raw[fr], lo, hi),
                    to_u8(cleaned[fr], lo, hi),
                    to_u8(
                        raw[fr].astype(np.float32) - cleaned[fr].astype(np.float32),
                        p=(5, 99.5),
                    ),
                ],
                1,
            )
        )
    Image.fromarray(np.concatenate(panels, 0)).save(path)


def slim(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k not in ("fracs", "gates", "rms", "raw_p")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # --- ChanA full sweep ---
    _, raw_a, fams_a, traj_a = prepare_channel("ChanA")
    results_a = []
    cleaned_cache = {}

    for name, params in CONFIGS:
        print(f"\n=== ChanA config: {name} ===", flush=True)
        t1 = time.perf_counter()
        cleaned, gates, rms, qs_per, resid_pass = run_config(
            raw_a, fams_a, traj_a, params
        )
        metrics = score(raw_a, cleaned, fams_a, gates, rms, qs_per, resid_pass)
        row = {
            "name": name,
            "params": params,
            "metrics": slim(metrics),
            "seconds": time.perf_counter() - t1,
        }
        results_a.append(row)
        cleaned_cache[name] = (cleaned, metrics)
        print(
            f"  remain gate>0.5={100*metrics['remaining_gate_gt_0.5_median']:.2f}% "
            f"strong25={100*metrics['remaining_strongest25_median']:.2f}% "
            f"gate0_rms={metrics['gate0_removed_rms_median']:.4f} "
            f"resid_pass={100*metrics['residual_pass_frac']:.1f}%",
            flush=True,
        )

    # Rank: minimize strong25 residual, hard constraint gate0_rms ~ 0
    def rank_key(r):
        m = r["metrics"]
        g0 = m["gate0_removed_rms_median"]
        penalty = 0.0 if (g0 is not None and g0 < 1e-6) else 10.0
        return (
            penalty,
            m["remaining_strongest25_median"],
            m["remaining_gate_gt_0.5_median"],
        )

    ranked = sorted(results_a, key=rank_key)
    baseline_m = next(r for r in results_a if r["name"] == "baseline")["metrics"]
    winner = ranked[0]
    # Prefer a clear improvement over baseline if available
    improved = [
        r
        for r in ranked
        if r["metrics"]["gate0_removed_rms_median"] < 1e-6
        and r["metrics"]["remaining_strongest25_median"]
        < baseline_m["remaining_strongest25_median"] - 0.005
    ]
    if improved:
        winner = improved[0]

    print(f"\nWinner on ChanA: {winner['name']}", flush=True)

    # --- Verify top 3 (incl winner & baseline) on ChanB ---
    verify_names = []
    for r in ranked:
        if r["name"] not in verify_names:
            verify_names.append(r["name"])
        if len(verify_names) >= 3:
            break
    if "baseline" not in verify_names:
        verify_names.append("baseline")

    _, raw_b, fams_b, traj_b = prepare_channel("ChanB")
    results_b = []
    cleaned_b_cache = {}
    for name in verify_names:
        params = dict(next(c for c in CONFIGS if c[0] == name)[1])
        print(f"\n=== ChanB verify: {name} ===", flush=True)
        cleaned, gates, rms, qs_per, resid_pass = run_config(
            raw_b, fams_b, traj_b, params
        )
        metrics = score(raw_b, cleaned, fams_b, gates, rms, qs_per, resid_pass)
        results_b.append(
            {"name": name, "params": params, "metrics": slim(metrics)}
        )
        cleaned_b_cache[name] = (cleaned, metrics)
        print(
            f"  remain gate>0.5={100*metrics['remaining_gate_gt_0.5_median']:.2f}% "
            f"strong25={100*metrics['remaining_strongest25_median']:.2f}% "
            f"gate0_rms={metrics['gate0_removed_rms_median']:.4f}",
            flush=True,
        )

    # Regression check: winner must not raise ChanB strong25 by >1 pp vs baseline
    b_base = next(r for r in results_b if r["name"] == "baseline")["metrics"]
    b_win = next((r for r in results_b if r["name"] == winner["name"]), None)
    accepted = winner
    if b_win is not None:
        delta_b = (
            b_win["metrics"]["remaining_strongest25_median"]
            - b_base["remaining_strongest25_median"]
        )
        if delta_b > 0.01 or b_win["metrics"]["gate0_removed_rms_median"] > 1e-6:
            print(
                f"Winner {winner['name']} regresses ChanB (delta_strong25={100*delta_b:.2f} pp); "
                f"falling back to best safe config.",
                flush=True,
            )
            safe = [
                r
                for r in ranked
                if r["name"] in cleaned_b_cache
                and cleaned_b_cache[r["name"]][1]["gate0_removed_rms_median"] < 1e-6
                and (
                    cleaned_b_cache[r["name"]][1]["remaining_strongest25_median"]
                    <= b_base["remaining_strongest25_median"] + 0.01
                )
            ]
            accepted = safe[0] if safe else next(r for r in ranked if r["name"] == "baseline")

    print(f"Accepted config: {accepted['name']}", flush=True)

    # Write accepted TIFFs + compares
    win_dir = OUT / "accepted" / accepted["name"]
    win_dir.mkdir(parents=True, exist_ok=True)
    cln_a = cleaned_cache[accepted["name"]][0]
    tifffile.imwrite(win_dir / "ChanA_raw_500fr_v21.tif", cln_a, photometric="minisblack")
    save_compare(
        win_dir / "ChanA_strong_weak.png",
        raw_a,
        cln_a,
        cleaned_cache[accepted["name"]][1]["gates"],
    )
    if accepted["name"] in cleaned_b_cache:
        cln_b = cleaned_b_cache[accepted["name"]][0]
        tifffile.imwrite(
            win_dir / "ChanB_raw_500fr_v21.tif", cln_b, photometric="minisblack"
        )
        save_compare(
            win_dir / "ChanB_strong_weak.png",
            raw_b,
            cln_b,
            cleaned_b_cache[accepted["name"]][1]["gates"],
        )

    # Persist settings used by accepted
    (win_dir / "params.json").write_text(
        json.dumps(accepted["params"], indent=2), encoding="utf-8"
    )

    summary = {
        "chanA_sweep": [
            {"name": r["name"], "params": r["params"], "metrics": r["metrics"], "seconds": r.get("seconds")}
            for r in results_a
        ],
        "chanA_ranked": [r["name"] for r in ranked],
        "chanB_verify": results_b,
        "winner_chanA": winner["name"],
        "accepted": accepted["name"],
        "accepted_params": accepted["params"],
        "elapsed_sec": time.perf_counter() - t0,
    }
    (OUT / "sweep_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown table
    lines = [
        "# v2.1 high-confidence sweep (500fr sandbox)",
        "",
        "Knobs swept: `max_alpha_high`, `residual_alpha`, `residual_strength_min`, `high_strength`, `strength_span`.",
        "No wider fx / full-row masks.",
        "",
        f"**Accepted config: `{accepted['name']}`**",
        "",
        "## ChanA",
        "",
        "| config | gate>0.5 | strong25 | all | gate0 RMS | resid_pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in ranked:
        m = r["metrics"]
        lines.append(
            f"| {r['name']} | {100*m['remaining_gate_gt_0.5_median']:.2f}% | "
            f"{100*m['remaining_strongest25_median']:.2f}% | "
            f"{100*m['remaining_all_median']:.2f}% | "
            f"{m['gate0_removed_rms_median']:.4f} | "
            f"{100*m['residual_pass_frac']:.1f}% |"
        )
    lines += ["", "## ChanB verify (subset)", "", "| config | gate>0.5 | strong25 | gate0 RMS |", "|---|---:|---:|---:|"]
    for r in results_b:
        m = r["metrics"]
        lines.append(
            f"| {r['name']} | {100*m['remaining_gate_gt_0.5_median']:.2f}% | "
            f"{100*m['remaining_strongest25_median']:.2f}% | "
            f"{m['gate0_removed_rms_median']:.4f} |"
        )
    lines += [
        "",
        f"Accepted params: `{json.dumps(accepted['params'])}`",
        "",
        f"Outputs: `{win_dir}`",
        "",
    ]
    text = "\n".join(lines)
    (OUT / "SUMMARY.md").write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
