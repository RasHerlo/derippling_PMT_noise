"""Ground-truth validation of darkcurrent.confirm on synthetic stacks.

Builds a stack whose contents are known, then checks the battery assigns the
right verdict to each planted component. The point is to prove the test can
separate a *drifting* ridge from a *static* ridge near DC, which is the whole
question in DARKCURRENT.md §3.1.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from .confirm import confirm_channel

H = W = 512
T = 256
RATE = 29.595

# (q, fx, amplitude, temporal frequency in cycles/frame, phase jitter rad, expected verdict)
SCENARIOS = {
    "drifting near DC": [
        (6, 25, 18.0, 0.25, 0.0, "fringe_confirmed"),
        (54, 30, 14.0, 0.10, 0.0, "fringe_confirmed"),
        (20, 22, 18.0, 0.0, 0.0, "static_structure"),
        (100, 0, 0.0, 0.0, 0.0, "noise_like"),
    ],
    # The confound the q~6 claim has to survive: a ridge that sits near DC and
    # looks identical in the spectrum, but does not move.
    "static near DC": [
        (6, 25, 18.0, 0.0, 0.0, "static_structure"),
        (54, 30, 14.0, 0.10, 0.0, "fringe_confirmed"),
    ],
    # Phase that wanders, which is what §3.3 actually reports for real data.
    "jittery near DC": [
        (6, 25, 18.0, 0.25, 0.8, "fringe_confirmed"),
        (54, 30, 14.0, 0.10, 0.8, "fringe_confirmed"),
    ],
}


def build_stack(
    path: Path, planted: list[tuple], *, noise_sigma: float = 30.0, seed: int = 7
) -> None:
    rng = np.random.default_rng(seed)
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]

    # Static smooth shading plus a steep offset: the DC skirt this test must survive.
    shading = 400.0 * np.exp(-((yy - H / 2) ** 2) / (2 * (H / 2.5) ** 2)) * np.ones_like(xx)
    shading = shading + 150.0 * (xx / W)

    with tifffile.TiffWriter(path) as tw:
        for t in range(T):
            frame = 1200.0 + shading + rng.normal(0.0, noise_sigma, size=(H, W))
            for q, fx, amp, ftemp, jitter, _ in planted:
                if amp <= 0:
                    continue
                phase = 2.0 * np.pi * ftemp * t
                if jitter:
                    phase += rng.normal(0.0, jitter)
                frame += amp * np.cos(2.0 * np.pi * (q * yy / H + fx * xx / W) + phase)
            tw.write(
                np.clip(frame, 0, 65535).astype(np.uint16),
                contiguous=True,
                photometric="minisblack",
            )


def run_scenario(name: str, planted: list[tuple], out_dir: Path) -> bool:
    stack = out_dir / f"{name.replace(' ', '_')}.tif"
    build_stack(stack, planted)

    payload = confirm_channel(
        stack,
        "SYNTH",
        "ChanA",
        sample_n=64,
        temporal_frames=T,
        frame_rate=RATE,
        forced_qs=[p[0] for p in planted],
    )

    expected = {float(p[0]): p[5] for p in planted}
    truth = {float(p[0]): (p[2], p[3], p[4]) for p in planted}

    print(f"\n=== {name} ===")
    print("  q  planted                    verdict            f_peak      prom   SNR   gain  support")
    print("  " + "-" * 96)
    ok = True
    for c in payload["candidates"]:
        q = c["q"]
        if q not in expected:
            continue
        amp = c["amplitude"]
        tm = c["temporal"]
        sup = amp.get("support") or {}
        p_amp, p_f, p_j = truth[q]
        got = c["verdict"]
        good = got == expected[q]
        ok = ok and good
        print(
            f"  {q:>3.0f} amp={p_amp:>5.1f} f={p_f:>5.2f} jit={p_j:>3.1f}  "
            f"{'OK ' if good else 'BAD'} {got:<18} "
            f"{tm.get('f_peak_cycles_per_frame', 0):+.4f} "
            f"{tm.get('prominence', 0):>9.1f} {amp.get('snr', 0):>5.1f} "
            f"{amp.get('filter_gain', 0):>6.3f} {sup.get('support_bins')} bins "
            f"|fx|={sup.get('abs_fx_min')}-{sup.get('abs_fx_max')}"
        )
        if not good:
            print(f"        expected {expected[q]}; checks={c['checks']}")

    for c in payload["candidates"]:
        if c["role"] == "static_reference":
            tm = c["temporal"]
            print(
                f"  {c['role']:<18} q={c['q']:>5.0f} {c['verdict']:<18} "
                f"f_peak={tm.get('f_peak_cycles_per_frame', 0):+.4f} "
                f"prom={tm.get('prominence', 0):.1f}"
            )

    ns = payload.get("null_summary") or {}
    if ns.get("n_rows"):
        print(
            f"  null: {ns['n_false_positive']}/{ns['n_rows']} false positives, "
            f"SNR max={ns['snr_max']:.2f}, prominence max={ns['prominence_max']:.1f}, "
            f"support bins max={ns['support_bins_max']:.0f}"
        )
        ok = ok and ns["n_false_positive"] == 0

    (out_dir / f"{name.replace(' ', '_')}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return ok


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="confirm_validate_"))
    print(f"Scratch: {tmp}")
    results = [run_scenario(name, planted, tmp) for name, planted in SCENARIOS.items()]
    ok = all(results)
    print("\nVALIDATION", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
