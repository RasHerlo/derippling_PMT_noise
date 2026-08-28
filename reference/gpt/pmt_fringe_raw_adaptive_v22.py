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
import sys
from pathlib import Path

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
    ap = argparse.ArgumentParser(
        description=(
            "v2.2 pack_D single-stack defringe. Uses the same process_stack path as "
            "python -m batch_defringe (fringe-rich seed, priors, QC, defringe_v22/ report)."
        )
    )
    ap.add_argument("input", help="Raw multipage TIFF stack")
    ap.add_argument(
        "-o",
        "--output",
        help="Output defringed TIFF (default: <channel>/defringe_v22/*_defringed_v22.tif)",
    )
    ap.add_argument(
        "--analyze-only",
        action="store_true",
        help="Seed + report only; do not write a cleaned stack",
    )
    ap.add_argument(
        "--root",
        type=Path,
        help="Experiment root for priors/library (default: trial folder beside DATA)",
    )
    ap.add_argument("--diagnostics", help="Unused; report is always written to defringe_v22/")
    ap.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-defringe even if defringe_v22 output already exists",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from batch_defringe.discover import job_for_stack
    from batch_defringe.process import process_stack
    from batch_defringe.readout import cleaned_path_for

    inpath = Path(args.input)
    job = job_for_stack(inpath, root=args.root)
    batch_root = Path(args.root).resolve() if args.root else job.trial_dir
    out_tif = Path(args.output) if args.output else cleaned_path_for(inpath)

    print(f"Input: {inpath}")
    print(f"v2.2 pack_D via batch process_stack  computer={job.computer} channel={job.channel}")
    if job.missing_xml:
        print("  WARNING: no Experiment.xml — fresh seed, no microscope prior")

    result = process_stack(
        job.tif_path,
        batch_root=batch_root,
        computer=job.computer,
        channel=job.channel,
        fingerprint=job.fingerprint,
        out_tif=out_tif,
        skip_existing=not args.no_skip_existing,
        force_fresh_seed=job.missing_xml,
        update_prior_on_success=not job.missing_xml,
        recording_date=job.date_utc,
        write_clean=not args.analyze_only,
    )
    print(f"    {result.status.upper()}: {result.message}")
    if result.out_dir is not None:
        print(f"    readout: {result.out_dir}")
    if result.out_tif is not None:
        print(f"    cleaned: {result.out_tif}")
    if result.status == "needs_review":
        sys.exit(2)
    if result.status == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
