"""
v2.2 lever sweep vs pack_B (v2.1 accepted defaults).

ChanA-focused: residual pass, high_strength unlock, softer ratio_* .
No wider fx / full-row masks. ChanB used as regression check.

Writes: .../defringe_runs/v22_sweep_500fr/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reference" / "gpt"))

import sweep_v21_500fr as s21  # noqa: E402
from pmt_fringe_raw_adaptive_v21 import clean_frame_v21  # noqa: E402
import stress_test_v2 as st  # noqa: E402

SANDBOX = Path(r"F:\bPACNewData2026\PreProcessing Optimization\Level3b copy")
OUT = SANDBOX / "defringe_runs" / "v22_sweep_500fr"

# Current production defaults (= pack_B)
PACK_B = dict(
    max_alpha=0.85,
    max_alpha_high=1.0,
    high_gate=0.95,
    high_strength=0.18,
    strength_span=0.12,
    residual_pass=True,
    residual_strength_min=0.05,
    residual_alpha=0.95,
    ratio_start=1.6,
    ratio_full=4.0,
    frame_search=2,
    gate_low=0.10,
    gate_high=0.20,
    y_sigma=1.0,
    y_radius=2,
)

CONFIGS = [
    ("pack_B", dict(PACK_B)),
    ("resid_min04", {**PACK_B, "residual_strength_min": 0.04}),
    ("resid_min03", {**PACK_B, "residual_strength_min": 0.03}),
    ("resid_a98", {**PACK_B, "residual_alpha": 0.98}),
    ("resid_a100", {**PACK_B, "residual_alpha": 1.00}),
    ("hs15", {**PACK_B, "high_strength": 0.15}),
    ("ratio_soft", {**PACK_B, "ratio_start": 1.4, "ratio_full": 3.5}),
    (
        "pack_C",
        {
            **PACK_B,
            "residual_strength_min": 0.03,
            "residual_alpha": 1.00,
            "high_strength": 0.15,
        },
    ),
    (
        "pack_D",
        {
            **PACK_B,
            "residual_strength_min": 0.03,
            "residual_alpha": 1.00,
            "high_strength": 0.15,
            "ratio_start": 1.4,
            "ratio_full": 3.5,
        },
    ),
    (
        "pack_E",
        {
            **PACK_B,
            "residual_strength_min": 0.04,
            "residual_alpha": 0.98,
            "high_strength": 0.15,
            "ratio_start": 1.5,
            "ratio_full": 3.7,
        },
    ),
]


def run_config(raw, families, trajectories, params: dict):
    n = raw.shape[0]
    cleaned = np.empty_like(raw)
    gates = np.zeros(n, dtype=float)
    rms = np.zeros(n, dtype=float)
    qs_per = []
    resid_pass = np.zeros(n, dtype=int)
    for i in range(n):
        preds = [traj[i] for traj in trajectories]
        out, removed, tracking = clean_frame_v21(raw[i], families, preds, **params)
        cleaned[i] = out
        gates[i] = max(t["gate"] for t in tracking) if tracking else 0.0
        rms[i] = float(np.sqrt(np.mean(removed.astype(float) ** 2)))
        qs_per.append([t["q"] for t in tracking])
        resid_pass[i] = int(any(t.get("residual_pass", 0) for t in tracking))
        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"    cleaned {i+1}/{n}", flush=True)
    return cleaned, gates, rms, qs_per, resid_pass


def quick_injection(raw, families, trajectories, track, params, n_weak=2, n_strong=2):
    """Alpha=1 packB-style residual injection; return median E_rec and remain frac."""
    weak, strong = st.pick_frames(track, n_weak, n_strong)
    tight = st.family_mask(raw.shape[1:], families, y_pad=2, fx_pad=0, x_weight_thresh=0.2)
    e_list, r_list = [], []
    for s_idx in strong:
        preds_s = [traj[s_idx] for traj in trajectories]
        cleaned_s, _, _ = clean_frame_v21(raw[s_idx], families, preds_s, **params)
        N = raw[s_idx].astype(np.float32) - cleaned_s.astype(np.float32)
        for w_idx in weak:
            I0 = raw[w_idx].astype(np.float32)
            I_test = I0 + N
            I_rec, _, _ = clean_frame_v21(I_test, families, preds_s, **params)
            I_rec = I_rec.astype(np.float32)
            e_list.append(st.rms(I_rec - I0))
            resid = (I_rec - I0).astype(np.float64)
            inj = N.astype(np.float64)
            F_res = np.fft.fftshift(np.fft.fft2(resid - np.median(resid)))
            F_inj = np.fft.fftshift(np.fft.fft2(inj - np.mean(inj)))
            p_res = float((np.abs(F_res)[tight] ** 2).sum())
            p_inj = float((np.abs(F_inj)[tight] ** 2).sum())
            r_list.append(p_res / (p_inj + 1e-12))
    return {
        "E_recovery_median": float(np.median(e_list)),
        "remaining_frac_median": float(np.median(r_list)),
        "n": len(e_list),
    }


def track_rows(raw, families, trajectories, params):
    rows = []
    for i in range(raw.shape[0]):
        preds = [traj[i] for traj in trajectories]
        _, rem, tr = clean_frame_v21(raw[i], families, preds, **params)
        gates = [t["gate"] for t in tr] if tr else [0.0]
        rows.append(
            {
                "frame": i,
                "removed_rms": float(np.sqrt(np.mean(rem.astype(float) ** 2))),
                "max_gate": float(max(gates)),
                "q": float(tr[0]["q"]) if tr else float(families[0]["q"]),
            }
        )
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    _, raw_a, fams_a, traj_a = s21.prepare_channel("ChanA")
    results_a = []
    cache_a = {}

    for name, params in CONFIGS:
        print(f"\n=== ChanA {name} ===", flush=True)
        t1 = time.perf_counter()
        cleaned, gates, rms, qs_per, resid_pass = run_config(
            raw_a, fams_a, traj_a, params
        )
        metrics = s21.score(raw_a, cleaned, fams_a, gates, rms, qs_per, resid_pass)
        row = {
            "name": name,
            "params": {k: v for k, v in params.items() if k in (
                "max_alpha", "max_alpha_high", "high_gate", "high_strength",
                "strength_span", "residual_pass", "residual_strength_min",
                "residual_alpha", "ratio_start", "ratio_full",
            )},
            "metrics": s21.slim(metrics),
            "seconds": time.perf_counter() - t1,
        }
        results_a.append(row)
        cache_a[name] = (cleaned, metrics, gates)
        print(
            f"  gate>0.5={100*metrics['remaining_gate_gt_0.5_median']:.2f}% "
            f"strong25={100*metrics['remaining_strongest25_median']:.2f}% "
            f"gate0={metrics['gate0_removed_rms_median']:.4f} "
            f"resid_pass={100*metrics['residual_pass_frac']:.1f}%",
            flush=True,
        )

    base_m = next(r for r in results_a if r["name"] == "pack_B")["metrics"]

    def ok(r):
        m = r["metrics"]
        return (
            m["gate0_removed_rms_median"] < 1e-6
            and m["remaining_strongest25_median"]
            <= base_m["remaining_strongest25_median"] + 1e-6
        )

    ranked = sorted(
        results_a,
        key=lambda r: (
            0 if r["metrics"]["gate0_removed_rms_median"] < 1e-6 else 1,
            r["metrics"]["remaining_strongest25_median"],
            r["metrics"]["remaining_gate_gt_0.5_median"],
        ),
    )
    improved = [
        r
        for r in ranked
        if ok(r)
        and r["metrics"]["remaining_strongest25_median"]
        < base_m["remaining_strongest25_median"] - 0.005
    ]
    winner = improved[0] if improved else next(r for r in ranked if r["name"] == "pack_B")
    print(f"\nChanA winner candidate: {winner['name']}", flush=True)

    # Top few + pack_B on ChanB
    verify = []
    for r in ranked:
        if r["name"] not in verify:
            verify.append(r["name"])
        if len(verify) >= 4:
            break
    if "pack_B" not in verify:
        verify.append("pack_B")

    _, raw_b, fams_b, traj_b = s21.prepare_channel("ChanB")
    results_b = []
    cache_b = {}
    for name in verify:
        params = dict(next(c for c in CONFIGS if c[0] == name)[1])
        print(f"\n=== ChanB verify {name} ===", flush=True)
        cleaned, gates, rms, qs_per, resid_pass = run_config(
            raw_b, fams_b, traj_b, params
        )
        metrics = s21.score(raw_b, cleaned, fams_b, gates, rms, qs_per, resid_pass)
        results_b.append({"name": name, "params": params, "metrics": s21.slim(metrics)})
        cache_b[name] = (cleaned, metrics, gates)
        print(
            f"  gate>0.5={100*metrics['remaining_gate_gt_0.5_median']:.2f}% "
            f"strong25={100*metrics['remaining_strongest25_median']:.2f}% "
            f"gate0={metrics['gate0_removed_rms_median']:.4f}",
            flush=True,
        )

    b_base = next(r for r in results_b if r["name"] == "pack_B")["metrics"]
    accepted = winner
    if accepted["name"] in cache_b:
        b_w = cache_b[accepted["name"]][1]
        if (
            b_w["gate0_removed_rms_median"] > 1e-6
            or b_w["remaining_strongest25_median"] > b_base["remaining_strongest25_median"] + 0.01
        ):
            print(f"{accepted['name']} regresses ChanB; falling back", flush=True)
            safe = [
                r
                for r in ranked
                if r["name"] in cache_b
                and cache_b[r["name"]][1]["gate0_removed_rms_median"] < 1e-6
                and cache_b[r["name"]][1]["remaining_strongest25_median"]
                <= b_base["remaining_strongest25_median"] + 0.01
                and r["metrics"]["remaining_strongest25_median"]
                <= base_m["remaining_strongest25_median"] + 1e-6
            ]
            accepted = safe[0] if safe else next(r for r in ranked if r["name"] == "pack_B")

    # Ensure accepted carries params from CONFIGS / sweep row
    if not accepted.get("params"):
        accepted = dict(accepted)
        accepted["params"] = next(r for r in results_a if r["name"] == accepted["name"])["params"]

    # Injection on pack_B vs accepted (ChanA)
    print("\n=== ChanA injection alpha=1 ===", flush=True)
    track_b = track_rows(raw_a, fams_a, traj_a, PACK_B)
    params_acc = dict(next(c for c in CONFIGS if c[0] == accepted["name"])[1])
    inj_base = quick_injection(raw_a, fams_a, traj_a, track_b, PACK_B)
    inj_acc = quick_injection(raw_a, fams_a, traj_a, track_b, params_acc)
    print(f"  pack_B: E_rec={inj_base['E_recovery_median']:.3f} remain={100*inj_base['remaining_frac_median']:.1f}%", flush=True)
    print(f"  {accepted['name']}: E_rec={inj_acc['E_recovery_median']:.3f} remain={100*inj_acc['remaining_frac_median']:.1f}%", flush=True)

    # Reject if injection E_recovery jumps >15% relative
    if (
        accepted["name"] != "pack_B"
        and inj_acc["E_recovery_median"] > inj_base["E_recovery_median"] * 1.15
    ):
        print("Injection E_recovery rose >15%; keep pack_B", flush=True)
        accepted = next(r for r in ranked if r["name"] == "pack_B")
        params_acc = dict(PACK_B)
        inj_acc = inj_base

    promote_v22 = (
        accepted["name"] != "pack_B"
        and accepted["metrics"]["remaining_strongest25_median"]
        < base_m["remaining_strongest25_median"] - 0.005
        and inj_acc["E_recovery_median"] <= inj_base["E_recovery_median"] * 1.15
    )

    # Write accepted stacks
    acc_dir = OUT / "accepted" / accepted["name"]
    acc_dir.mkdir(parents=True, exist_ok=True)
    cln_a = cache_a[accepted["name"]][0]
    tifffile.imwrite(acc_dir / "ChanA_raw_500fr_v22.tif", cln_a, photometric="minisblack")
    s21.save_compare(
        acc_dir / "ChanA_strong_weak.png",
        raw_a,
        cln_a,
        cache_a[accepted["name"]][2],
    )
    if accepted["name"] in cache_b:
        cln_b = cache_b[accepted["name"]][0]
        tifffile.imwrite(acc_dir / "ChanB_raw_500fr_v22.tif", cln_b, photometric="minisblack")
        s21.save_compare(
            acc_dir / "ChanB_strong_weak.png",
            raw_b,
            cln_b,
            cache_b[accepted["name"]][2],
        )
    (acc_dir / "params.json").write_text(
        json.dumps(accepted["params"], indent=2), encoding="utf-8"
    )

    summary = {
        "baseline": "pack_B",
        "chanA_sweep": [
            {"name": r["name"], "params": r["params"], "metrics": r["metrics"]}
            for r in results_a
        ],
        "chanA_ranked": [r["name"] for r in ranked],
        "chanB_verify": results_b,
        "accepted": accepted["name"],
        "accepted_params": accepted["params"],
        "promote_v22": promote_v22,
        "injection_chanA_alpha1": {"pack_B": inj_base, accepted["name"]: inj_acc},
        "elapsed_sec": time.perf_counter() - t0,
    }
    (OUT / "sweep_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# v2.2 lever sweep vs pack_B (500fr)",
        "",
        f"**Accepted:** `{accepted['name']}`",
        f"**Promote as v2.2?** {'YES' if promote_v22 else 'NO — keep pack_B'}",
        "",
        "## ChanA residual",
        "",
        "| config | gate>0.5 | strong25 | all | gate0 | resid_pass |",
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
    lines += ["", "## ChanB verify", "", "| config | gate>0.5 | strong25 | gate0 |", "|---|---:|---:|---:|"]
    for r in results_b:
        m = r["metrics"]
        lines.append(
            f"| {r['name']} | {100*m['remaining_gate_gt_0.5_median']:.2f}% | "
            f"{100*m['remaining_strongest25_median']:.2f}% | "
            f"{m['gate0_removed_rms_median']:.4f} |"
        )
    lines += [
        "",
        "## Injection ChanA alpha=1",
        "",
        f"- pack_B: E_rec={inj_base['E_recovery_median']:.3f}, remain={100*inj_base['remaining_frac_median']:.1f}%",
        f"- {accepted['name']}: E_rec={inj_acc['E_recovery_median']:.3f}, remain={100*inj_acc['remaining_frac_median']:.1f}%",
        "",
        f"Params: `{json.dumps(accepted['params'])}`",
        "",
    ]
    text = "\n".join(lines)
    (OUT / "SUMMARY.md").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
