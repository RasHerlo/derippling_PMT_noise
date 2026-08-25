"""
v2.2 adaptive raw-only PMT fringe removal (pack_D defaults).

Same architecture as v2.1 with stronger high-confidence residual attenuation
and slightly softer local excess thresholds (2026-08-19 sweep vs pack_B):

- residual_strength_min 0.05 → 0.03
- residual_alpha 0.95 → 1.00
- high_strength 0.18 → 0.15
- ratio_start/full 1.6/4.0 → 1.4/3.5

No full-row / widened fx masks. gate=0 frames unmodified.
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
    learn_median_spectrum,
    save_diagnostics,
    track_family_blocks,
)
from pmt_fringe_raw_adaptive_v21 import clean_frame_v21


def clean_frame_v22(frame, families, predicted_qs, **kwargs):
    """v2.2 = v2.1 cleaner with pack_D defaults."""
    defaults = dict(
        frame_search=2,
        gate_low=0.10,
        gate_high=0.20,
        ratio_start=1.4,
        ratio_full=3.5,
        max_alpha=0.85,
        max_alpha_high=1.00,
        high_gate=0.95,
        high_strength=0.15,
        strength_span=0.12,
        residual_pass=True,
        residual_strength_min=0.03,
        residual_alpha=1.00,
        y_sigma=1.0,
        y_radius=2,
    )
    defaults.update(kwargs)
    return clean_frame_v21(frame, families, predicted_qs, **defaults)


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
    ap.add_argument("--ratio-start", type=float, default=1.4)
    ap.add_argument("--ratio-full", type=float, default=3.5)
    ap.add_argument("--max-alpha", type=float, default=0.85)
    ap.add_argument("--max-alpha-high", type=float, default=1.00)
    ap.add_argument("--high-gate", type=float, default=0.95)
    ap.add_argument("--high-strength", type=float, default=0.15)
    ap.add_argument("--strength-span", type=float, default=0.12)
    ap.add_argument("--no-residual-pass", action="store_true")
    ap.add_argument("--residual-strength-min", type=float, default=0.03)
    ap.add_argument("--residual-alpha", type=float, default=1.00)

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
            f"v2.2 pack_D: max_alpha={args.max_alpha}->{args.max_alpha_high}; "
            f"resid_min={args.residual_strength_min} resid_alpha={args.residual_alpha}; "
            f"ratio={args.ratio_start}/{args.ratio_full}"
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
                "version": "v2.2",
                "config_id": "pack_D",
                "max_alpha": args.max_alpha,
                "max_alpha_high": args.max_alpha_high,
                "high_gate": args.high_gate,
                "high_strength": args.high_strength,
                "strength_span": args.strength_span,
                "residual_pass": not args.no_residual_pass,
                "residual_strength_min": args.residual_strength_min,
                "residual_alpha": args.residual_alpha,
                "ratio_start": args.ratio_start,
                "ratio_full": args.ratio_full,
            }
            Path(args.diagnostics).mkdir(parents=True, exist_ok=True)
            with open(Path(args.diagnostics) / "v22_settings.json", "w", encoding="utf-8") as fh:
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
                cleaned, removed, tracking = clean_frame_v22(
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
