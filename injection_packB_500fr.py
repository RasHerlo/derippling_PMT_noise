"""
Injection stress test for v2.1 pack_B defaults on sandbox 500fr stacks.

Pseudo-ground-truth:
  I0 = weak/absent-fringe frame (gate≈0)
  N  = fringe from strong frame (pack_B residual OR broad spectral extract)
  I_test = I0 + alpha * N
  I_rec  = clean_frame_v21(I_test)
  Report E_recovery = RMS(I_rec - I0) and remaining fringe-band power fraction.

Writes under:
  .../Level3b copy/defringe_runs/v21_packB_injection/
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reference" / "gpt"))
import pmt_fringe_raw_adaptive as v2  # noqa: E402
from pmt_fringe_raw_adaptive_v21 import clean_frame_v21  # noqa: E402
import stress_test_v2 as st  # noqa: E402

SANDBOX = Path(r"F:\bPACNewData2026\PreProcessing Optimization\Level3b copy")
OUT = SANDBOX / "defringe_runs" / "v21_packB_injection"
PACK_B = dict(
    max_alpha=0.85,
    max_alpha_high=1.0,
    high_gate=0.95,
    high_strength=0.18,
    strength_span=0.12,
    residual_pass=True,
    residual_strength_min=0.05,
    residual_alpha=0.95,
)
ALPHAS = (0.25, 0.5, 1.0, 1.5)
CHANNELS = ("ChanA", "ChanB")


def clean_packb(frame, families, preds):
    return clean_frame_v21(frame, families, preds, **PACK_B)


def tracking_from_stack(raw, families, trajectories):
    rows = []
    for i in range(raw.shape[0]):
        preds = [traj[i] for traj in trajectories]
        _, rem, tr = clean_packb(raw[i], families, preds)
        gates = [t["gate"] for t in tr] if tr else [0.0]
        qs = [t["q"] for t in tr] if tr else [families[0]["q"]]
        rows.append(
            {
                "frame": i,
                "removed_rms": float(np.sqrt(np.mean(rem.astype(float) ** 2))),
                "max_gate": float(max(gates)),
                "q": float(qs[0]),
            }
        )
    return rows


def run_injection(raw, families, trajectories, track, out_dir: Path):
    weak_frames, strong_frames = st.pick_frames(track, st.N_WEAK, st.N_STRONG)
    tight_mask = st.family_mask(
        raw.shape[1:], families, y_pad=2, fx_pad=0, x_weight_thresh=0.2
    )

    noise_bank = {}
    for s_idx in strong_frames:
        preds_s = [traj[s_idx] for traj in trajectories]
        cleaned_s, _, _ = clean_packb(raw[s_idx], families, preds_s)
        N_pack = raw[s_idx].astype(np.float32) - cleaned_s.astype(np.float32)
        N_broad, _ = st.extract_broad_fringe(raw[s_idx], families)
        noise_bank[s_idx] = {
            "packB_residual": N_pack,
            "broad_spectral": N_broad.astype(np.float32),
            "preds": preds_s,
            "N_pack_rms": st.rms(N_pack),
            "N_broad_rms": st.rms(N_broad),
        }

    rows = []
    examples = []
    for w_idx in weak_frames:
        I0 = raw[w_idx].astype(np.float32)
        preds_w = [traj[w_idx] for traj in trajectories]
        _, rem0, tr0 = clean_packb(I0, families, preds_w)
        baseline_gate = float(max(t["gate"] for t in tr0)) if tr0 else 0.0

        for s_idx, noises in noise_bank.items():
            preds_clean = noises["preds"]
            for noise_name, N in (
                ("packB_residual", noises["packB_residual"]),
                ("broad_spectral", noises["broad_spectral"]),
            ):
                for alpha in ALPHAS:
                    I_test = I0 + np.float32(alpha) * N
                    cleaned, _, tracking = clean_packb(
                        I_test.astype(np.float32), families, preds_clean
                    )
                    I_rec = cleaned.astype(np.float32)
                    E_recovery = st.rms(I_rec - I0)

                    inj = (alpha * N).astype(np.float64)
                    resid = (I_rec - I0).astype(np.float64)
                    F_res = np.fft.fftshift(np.fft.fft2(resid - np.median(resid)))
                    F_inj = np.fft.fftshift(np.fft.fft2(inj - np.mean(inj)))
                    p_res = float((np.abs(F_res)[tight_mask] ** 2).sum())
                    p_inj = float((np.abs(F_inj)[tight_mask] ** 2).sum())
                    remaining_frac = p_res / (p_inj + 1e-12)
                    gate = float(max(t["gate"] for t in tracking)) if tracking else 0.0

                    row = {
                        "weak_frame": w_idx,
                        "strong_frame": s_idx,
                        "noise": noise_name,
                        "alpha": alpha,
                        "baseline_gate_on_I0": baseline_gate,
                        "baseline_rms_on_I0": st.rms(rem0),
                        "E_recovery": E_recovery,
                        "E_remaining_total": st.rms(resid),
                        "remaining_fringe_band_frac": remaining_frac,
                        "injected_rms": st.rms(alpha * N),
                        "gate_on_test": gate,
                        "N_source_rms": float(
                            noises["N_pack_rms"]
                            if noise_name == "packB_residual"
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
                                "N": N,
                            }
                        )

    with open(out_dir / "injection_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    curves = {}
    for noise_name in ("packB_residual", "broad_spectral"):
        curves[noise_name] = []
        for alpha in ALPHAS:
            sub = [r for r in rows if r["noise"] == noise_name and r["alpha"] == alpha]
            curves[noise_name].append(
                {
                    "alpha": alpha,
                    "E_recovery_median": float(np.median([r["E_recovery"] for r in sub])),
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
    }


def save_curve_plot(path: Path, curves: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for noise_name, style in (("packB_residual", "-o"), ("broad_spectral", "--s")):
        c = curves[noise_name]
        a = [x["alpha"] for x in c]
        axes[0].plot(a, [x["E_recovery_median"] for x in c], style, label=noise_name)
        axes[1].plot(
            a,
            [x["remaining_fringe_band_frac_median"] for x in c],
            style,
            label=noise_name,
        )
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("median E_recovery = RMS(I_rec - I0)")
    axes[0].set_title("Biological distortion proxy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("median remaining fringe-band power frac")
    axes[1].set_title("Residual injected fringe")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_sum = {"params": PACK_B, "channels": {}}

    for chan in CHANNELS:
        print(f"\n=== {chan} pack_B injection ===", flush=True)
        raw_path = SANDBOX / "inputs" / "slices_500fr" / "raw" / f"{chan}_raw_500fr.tif"
        out_dir = OUT / chan
        out_dir.mkdir(parents=True, exist_ok=True)

        families, trajectories, _ = st.rebuild_families_from_stack(raw_path)
        raw = tifffile.imread(raw_path)
        print("  building tracking for weak/strong pick...", flush=True)
        track = tracking_from_stack(raw, families, trajectories)
        with open(out_dir / "tracking_for_injection.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(track[0].keys()))
            w.writeheader()
            w.writerows(track)

        cont = st.continuity_metrics(
            [
                {
                    **r,
                    "mean_gate": r["max_gate"],
                    "strength": np.nan,
                }
                for r in track
            ]
        )
        st.save_continuity_plots(out_dir / "continuity.png", track)
        (out_dir / "continuity_metrics.json").write_text(
            json.dumps(cont, indent=2), encoding="utf-8"
        )

        print("  running injection suite...", flush=True)
        inj = run_injection(raw, families, trajectories, track, out_dir)
        examples = inj.pop("examples")
        save_curve_plot(out_dir / "injection_curves.png", inj["curves"])
        st.save_example_panels(out_dir / "injection_examples.png", examples)

        summary = {
            "channel": chan,
            "families": [
                {"q": f["q"], "hi": f["hi"], "fx_ranges": f["fx_ranges"]} for f in families
            ],
            "continuity": cont,
            "injection": {
                "weak_frames": inj["weak_frames"],
                "strong_frames": inj["strong_frames"],
                "curves": inj["curves"],
                "n_rows": inj["n_rows"],
            },
            # highlight alpha=1 realistic operating point
            "alpha1": {
                name: next(c for c in inj["curves"][name] if c["alpha"] == 1.0)
                for name in inj["curves"]
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        all_sum["channels"][chan] = summary
        print(
            f"  alpha=1 packB_residual: E_rec={summary['alpha1']['packB_residual']['E_recovery_median']:.3f} "
            f"remain={100*summary['alpha1']['packB_residual']['remaining_fringe_band_frac_median']:.1f}%",
            flush=True,
        )
        print(
            f"  alpha=1 broad_spectral: E_rec={summary['alpha1']['broad_spectral']['E_recovery_median']:.3f} "
            f"remain={100*summary['alpha1']['broad_spectral']['remaining_fringe_band_frac_median']:.1f}%",
            flush=True,
        )

    (OUT / "results.json").write_text(json.dumps(all_sum, indent=2), encoding="utf-8")

    lines = [
        "# pack_B injection stress (500fr sandbox)",
        "",
        "I0 = weak frame; N from strong frame; I_test = I0 + alpha*N; clean with v2.1 pack_B.",
        "",
        "## Alpha = 1 (realistic)",
        "",
        "| Channel | noise | E_recovery (median) | remaining fringe-band frac |",
        "|---|---|---:|---:|",
    ]
    for chan, s in all_sum["channels"].items():
        for name, c in s["alpha1"].items():
            lines.append(
                f"| {chan} | {name} | {c['E_recovery_median']:.3f} | "
                f"{100*c['remaining_fringe_band_frac_median']:.1f}% |"
            )
    lines += [
        "",
        "Lower E_recovery = less spoilage of I0. Lower remaining frac = more fringe removed.",
        "",
        f"Outputs: `{OUT}`",
        "",
    ]
    text = "\n".join(lines)
    (OUT / "SUMMARY.md").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
