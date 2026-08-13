#!/usr/bin/env python3
"""
v2.1 adaptive raw-only PMT fringe removal.

Builds on pmt_fringe_raw_adaptive.py with:
1) Confidence-scaled max_alpha: keep conservative attenuation for partial-confidence
   frames; allow higher removal only when gate is high and the paired PMT signature
   is strong.
2) Optional residual second pass: after the first clean, if the same paired ridge is
   still significantly above local spectral background, apply a small additional
   attenuation only at surviving fringe coefficients.

Default behavior on weak/absent fringe (gate=0) is unchanged: no filtering.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile

from pmt_fringe_raw_adaptive import (
    detect_families,
    family_score,
    learn_median_spectrum,
    save_diagnostics,
    search_q,
    track_family_blocks,
)


def _effective_max_alpha(
    gate: float,
    strength: float,
    paired: bool,
    *,
    max_alpha: float,
    max_alpha_high: float,
    high_gate: float,
    high_strength: float,
    strength_span: float = 0.15,
) -> float:
    """Blend conservative and high-confidence alpha ceilings.

    Full gate alone is not enough: strength must clear ``high_strength`` (just
    above the soft-gate region). The blend then ramps over ``strength_span`` so
    weaker-but-real PMT signatures (e.g. ChanA ~0.25–0.45) can still unlock
    higher attenuation, while partial-confidence frames stay at ``max_alpha``.
    """
    if (not paired) or gate < high_gate or strength < high_strength:
        return float(max_alpha)
    g = float(np.clip((gate - high_gate) / max(1e-9, 1.0 - high_gate), 0.0, 1.0))
    s = float(
        np.clip((strength - high_strength) / max(1e-9, strength_span), 0.0, 1.0)
    )
    conf = g * s
    return float(max_alpha + conf * (max_alpha_high - max_alpha))


def _attenuate_family_on_amp(
    src_amp: np.ndarray,
    dst_amp: np.ndarray,
    family: dict,
    q: float,
    gate: float,
    *,
    max_alpha: float,
    ratio_start: float,
    ratio_full: float,
    y_sigma: float,
    y_radius: int,
) -> None:
    """Attenuate one family: measure from src_amp, write into dst_amp."""
    h, w = src_amp.shape
    cy = h // 2

    dys = [q]
    if family["paired"]:
        dys.append(cy - q)

    yoffs = np.arange(-y_radius, y_radius + 1)
    yweights = np.exp(-0.5 * (yoffs / max(1e-6, y_sigma)) ** 2)
    xweight = family["x_weight"]

    for d in dys:
        for sgn in (-1, +1):
            yc = cy + sgn * int(round(d))
            for off, wy in zip(yoffs, yweights):
                y = yc + int(off)
                if not (0 <= y < h):
                    continue

                bgrows = []
                for boff in list(range(-9, -4)) + list(range(5, 10)):
                    yy = y + boff
                    if 0 <= yy < h:
                        bgrows.append(src_amp[yy, :])
                if not bgrows:
                    continue

                bg = np.median(np.stack(bgrows), axis=0)
                ratio = src_amp[y, :] / (bg + 1e-12)
                local_conf = np.clip(
                    (ratio - ratio_start) / max(1e-9, ratio_full - ratio_start),
                    0.0,
                    1.0,
                )
                alpha = max_alpha * gate * float(wy) * xweight * local_conf
                excess = np.maximum(src_amp[y, :] - bg, 0.0)
                candidate = src_amp[y, :] - alpha * excess
                dst_amp[y, :] = np.minimum(dst_amp[y, :], np.maximum(candidate, bg))


def ridge_excess_score(logamp: np.ndarray, family: dict, q: float, xvalid: np.ndarray) -> float:
    """Paired-ridge contrast used to decide whether a residual pass is warranted."""
    return float(family_score(logamp, int(round(q)), family["paired"], xvalid))


def clean_frame_v21(
    frame,
    families,
    predicted_qs,
    *,
    frame_search=2,
    gate_low=0.10,
    gate_high=0.20,
    ratio_start=1.6,
    ratio_full=4.0,
    max_alpha=0.85,
    max_alpha_high=0.97,
    high_gate=0.95,
    high_strength=0.22,
    strength_span=0.15,
    residual_pass=True,
    residual_strength_min=0.08,
    residual_alpha=0.70,
    y_sigma=1.0,
    y_radius=2,
):
    orig_dtype = frame.dtype
    x = frame.astype(np.float32)
    offset = float(np.median(x))
    x0 = x - offset

    F = np.fft.fftshift(np.fft.fft2(x0))
    amp = np.abs(F)
    phase = np.exp(1j * np.angle(F))
    logamp = np.log1p(amp)

    h, w = x.shape
    cy, cx = h // 2, w // 2
    fx = np.arange(w) - cx
    xvalid = (np.abs(fx) > 5) & (np.abs(fx) < cx - 10)

    newamp = amp.copy()
    tracking = []

    for family, q_pred in zip(families, predicted_qs):
        q, strength = search_q(logamp, q_pred, family["paired"], xvalid, frame_search)
        gate = float(
            np.clip((strength - gate_low) / max(1e-9, gate_high - gate_low), 0.0, 1.0)
        )
        eff_alpha = _effective_max_alpha(
            gate,
            strength,
            family["paired"],
            max_alpha=max_alpha,
            max_alpha_high=max_alpha_high,
            high_gate=high_gate,
            high_strength=high_strength,
            strength_span=strength_span,
        )

        entry = {
            "q": q,
            "strength": strength,
            "gate": gate,
            "eff_max_alpha": eff_alpha,
            "residual_pass": 0,
            "residual_strength": 0.0,
        }
        tracking.append(entry)
        if gate <= 0:
            continue

        # First pass: measure on original spectrum (same as v2)
        _attenuate_family_on_amp(
            amp,
            newamp,
            family,
            q,
            gate,
            max_alpha=eff_alpha,
            ratio_start=ratio_start,
            ratio_full=ratio_full,
            y_sigma=y_sigma,
            y_radius=y_radius,
        )

    # Residual second pass on still-strong paired ridges
    if residual_pass:
        log_after = np.log1p(newamp)
        after_src = newamp.copy()
        for family, entry in zip(families, tracking):
            if entry["gate"] < high_gate or not family["paired"]:
                continue
            q = entry["q"]
            r_strength = ridge_excess_score(log_after, family, q, xvalid)
            entry["residual_strength"] = float(r_strength)
            if r_strength < residual_strength_min:
                continue
            residual_gate = float(
                np.clip(
                    (r_strength - residual_strength_min)
                    / max(1e-9, residual_strength_min),
                    0.0,
                    1.0,
                )
            )
            _attenuate_family_on_amp(
                after_src,
                newamp,
                family,
                q,
                residual_gate,
                max_alpha=residual_alpha,
                ratio_start=ratio_start,
                ratio_full=max(ratio_full, ratio_start + 0.5),
                y_sigma=y_sigma,
                y_radius=y_radius,
            )
            entry["residual_pass"] = 1

    Fnew = newamp * phase
    cleaned = np.real(np.fft.ifft2(np.fft.ifftshift(Fnew))) + offset
    removed = x - cleaned

    if np.issubdtype(orig_dtype, np.integer):
        lim = np.iinfo(orig_dtype)
        cleaned_write = np.clip(np.rint(cleaned), lim.min, lim.max).astype(orig_dtype)
    else:
        cleaned_write = cleaned.astype(orig_dtype)

    return cleaned_write, removed.astype(np.float32), tracking


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Raw multipage TIFF stack")
    ap.add_argument("-o", "--output", help="Output defringed TIFF")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--diagnostics", help="Directory for JSON/PNG/CSV diagnostics")

    ap.add_argument("--sample-frames", type=int, default=80)
    ap.add_argument("--row-z", type=float, default=5.5)
    ap.add_argument("--pair-z", type=float, default=3.5)
    ap.add_argument("--x-z", type=float, default=3.5)
    ap.add_argument("--max-families", type=int, default=4)
    ap.add_argument("--allow-standalone", action="store_true")

    ap.add_argument("--block-size", type=int, default=50)
    ap.add_argument("--samples-per-block", type=int, default=8)
    ap.add_argument("--track-search", type=int, default=6)
    ap.add_argument("--track-update-min", type=float, default=0.10)
    ap.add_argument("--frame-search", type=int, default=2)

    ap.add_argument("--gate-low", type=float, default=0.10)
    ap.add_argument("--gate-high", type=float, default=0.20)
    ap.add_argument("--ratio-start", type=float, default=1.6)
    ap.add_argument("--ratio-full", type=float, default=4.0)
    ap.add_argument("--max-alpha", type=float, default=0.85, help="Conservative ceiling")
    ap.add_argument(
        "--max-alpha-high",
        type=float,
        default=0.97,
        help="High-confidence ceiling (gate high + strong paired peaks)",
    )
    ap.add_argument("--high-gate", type=float, default=0.95)
    ap.add_argument(
        "--high-strength",
        type=float,
        default=0.22,
        help="Min ridge strength to unlock high-alpha blend (just above soft gate)",
    )
    ap.add_argument(
        "--strength-span",
        type=float,
        default=0.15,
        help="Strength range above --high-strength over which alpha ramps to max-alpha-high",
    )
    ap.add_argument("--no-residual-pass", action="store_true")
    ap.add_argument("--residual-strength-min", type=float, default=0.08)
    ap.add_argument("--residual-alpha", type=float, default=0.70)

    args = ap.parse_args()
    if not args.analyze_only and not args.output:
        ap.error("--output is required unless --analyze-only is used")

    inpath = Path(args.input)
    with tifffile.TiffFile(inpath) as tf:
        shape = tf.series[0].shape
        if len(shape) != 3:
            raise ValueError(f"Expected TYX stack; got {shape}")
        n, h, w = shape
        print(f"Input: {inpath}")
        print(f"Shape: {shape}, dtype: {tf.pages[0].dtype}")
        print(
            f"v2.1: max_alpha={args.max_alpha} -> high={args.max_alpha_high}; "
            f"residual_pass={not args.no_residual_pass}"
        )

        medspec, _ = learn_median_spectrum(tf, args.sample_frames)
        families, row_profile, row_z = detect_families(
            medspec,
            row_z_thresh=args.row_z,
            pair_z_min=args.pair_z,
            x_z_thresh=args.x_z,
            max_families=args.max_families,
            allow_standalone=args.allow_standalone,
        )
        if not families:
            raise RuntimeError("No high-confidence paired fringe family detected.")

        print("Detected fringe families:")
        for i, f in enumerate(families, 1):
            print(
                f"  {i}: q={f['q']:.1f}, hi={f['hi']}, row_z={f['row_score']:.1f}, "
                f"paired={f['paired']}, fx={f['fx_ranges']}"
            )

        trajectories, all_blocks = [], []
        for i, f in enumerate(families):
            qtraj, blocks = track_family_blocks(
                tf,
                f,
                block_size=args.block_size,
                samples_per_block=args.samples_per_block,
                track_search=args.track_search,
                track_update_min=args.track_update_min,
            )
            trajectories.append(qtraj)
            for b in blocks:
                bb = dict(b)
                bb["family"] = i
                all_blocks.append(bb)

        if args.diagnostics:
            save_diagnostics(
                args.diagnostics, medspec, families, row_profile, row_z, all_blocks
            )
            meta = {
                "version": "v2.1",
                "max_alpha": args.max_alpha,
                "max_alpha_high": args.max_alpha_high,
                "high_gate": args.high_gate,
                "high_strength": args.high_strength,
                "strength_span": args.strength_span,
                "residual_pass": not args.no_residual_pass,
                "residual_strength_min": args.residual_strength_min,
                "residual_alpha": args.residual_alpha,
            }
            Path(args.diagnostics).mkdir(parents=True, exist_ok=True)
            with open(Path(args.diagnostics) / "v21_settings.json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)

        if args.analyze_only:
            print("Analyze-only mode: no output TIFF written.")
            return

        outpath = Path(args.output)
        est_bytes = n * h * w * np.dtype(tf.pages[0].dtype).itemsize
        bigtiff = est_bytes > 3_500_000_000

        csvfh = None
        diag_writer = None
        if args.diagnostics:
            csvfh = open(
                Path(args.diagnostics) / "temporal_tracking.csv",
                "w",
                newline="",
                encoding="utf-8",
            )
            fieldnames = ["frame", "removed_rms"]
            for i in range(len(families)):
                fieldnames += [
                    f"family{i}_q",
                    f"family{i}_strength",
                    f"family{i}_gate",
                    f"family{i}_eff_max_alpha",
                    f"family{i}_residual_pass",
                    f"family{i}_residual_strength",
                ]
            diag_writer = csv.DictWriter(csvfh, fieldnames=fieldnames)
            diag_writer.writeheader()

        with tifffile.TiffWriter(outpath, bigtiff=bigtiff) as tw:
            for fi in range(n):
                frame = tf.pages[fi].asarray()
                preds = [traj[fi] for traj in trajectories]
                cleaned, removed, tracking = clean_frame_v21(
                    frame,
                    families,
                    preds,
                    frame_search=args.frame_search,
                    gate_low=args.gate_low,
                    gate_high=args.gate_high,
                    ratio_start=args.ratio_start,
                    ratio_full=args.ratio_full,
                    max_alpha=args.max_alpha,
                    max_alpha_high=args.max_alpha_high,
                    high_gate=args.high_gate,
                    high_strength=args.high_strength,
                    strength_span=args.strength_span,
                    residual_pass=not args.no_residual_pass,
                    residual_strength_min=args.residual_strength_min,
                    residual_alpha=args.residual_alpha,
                )
                tw.write(cleaned, contiguous=True)

                if diag_writer:
                    row = {
                        "frame": fi,
                        "removed_rms": float(np.sqrt(np.mean(removed.astype(float) ** 2))),
                    }
                    for i, t in enumerate(tracking):
                        row[f"family{i}_q"] = t["q"]
                        row[f"family{i}_strength"] = t["strength"]
                        row[f"family{i}_gate"] = t["gate"]
                        row[f"family{i}_eff_max_alpha"] = t["eff_max_alpha"]
                        row[f"family{i}_residual_pass"] = t["residual_pass"]
                        row[f"family{i}_residual_strength"] = t["residual_strength"]
                    diag_writer.writerow(row)

                if (fi + 1) % 100 == 0 or fi == n - 1:
                    print(f"Processed {fi + 1}/{n} frames")

        if csvfh:
            csvfh.close()
        print(f"Wrote: {outpath}")
        if args.diagnostics:
            print(f"Diagnostics: {args.diagnostics}")


if __name__ == "__main__":
    main()
