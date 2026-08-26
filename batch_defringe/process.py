"""Process one channel stack with soft microscope prior + v2.2 pack_D."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

_REPO = Path(__file__).resolve().parents[1]
_GPT = _REPO / "reference" / "gpt"
if str(_GPT) not in sys.path:
    sys.path.insert(0, str(_GPT))

from pmt_fringe_raw_adaptive_v22 import clean_frame_v22  # noqa: E402

from .experiment_xml import fingerprint_compatible
from .priors import load_prior, save_prior
from .readout import (
    cleaned_path_for,
    removed_path_for,
    tracking_row,
    write_readout,
)
from .seed import (
    detect_fringe_rich,
    hydrate_families,
    qc_tracking,
    sample_median_spectrum,
    track_all,
)

PACK_D = dict(
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
)


@dataclass
class ProcessResult:
    status: str  # ok | skipped | needs_review | error
    message: str
    out_tif: Path | None = None
    out_dir: Path | None = None
    removed_tif: Path | None = None
    overview_pdf: Path | None = None
    used_prior: bool = False
    reseeded: bool = False
    families_q: list[float] | None = None


def default_output_path(tif_path: Path) -> Path:
    return cleaned_path_for(tif_path)


def process_stack(
    tif_path: Path,
    *,
    batch_root: Path,
    computer: str,
    channel: str,
    fingerprint: dict,
    out_tif: Path | None = None,
    diag_dir: Path | None = None,
    skip_existing: bool = True,
    update_prior_on_success: bool = True,
    force_fresh_seed: bool = False,
) -> ProcessResult:
    tif_path = Path(tif_path)
    out_tif = Path(out_tif) if out_tif else default_output_path(tif_path)
    out_dir = out_tif.parent
    removed_tif = removed_path_for(tif_path, out_dir)

    if skip_existing and out_tif.is_file():
        return ProcessResult(
            status="skipped",
            message=f"exists: {out_dir.name}/{out_tif.name}",
            out_tif=out_tif,
            out_dir=out_dir,
            removed_tif=removed_tif if removed_tif.is_file() else None,
        )

    prior = None if force_fresh_seed else load_prior(batch_root, computer, channel)
    use_prior = bool(
        (not force_fresh_seed)
        and prior
        and prior.get("families")
        and fingerprint_compatible(prior.get("fingerprint"), fingerprint)
    )

    with tifffile.TiffFile(tif_path) as tf:
        shape = tf.series[0].shape
        if len(shape) != 3:
            return ProcessResult(status="error", message=f"bad shape {shape}")
        n, h, w = shape
        dtype = tf.pages[0].dtype

        families = None
        medspec = None
        seed_info: dict = {}
        reseeded = False
        had_usable_prior = False

        if use_prior:
            medspec = sample_median_spectrum(tf)
            families = hydrate_families(prior["families"], medspec)
            had_usable_prior = True
            seed_info = {
                "mode": "soft_prior",
                "prior_source": prior.get("source_tif"),
                "prior_qs": [float(f["q"]) for f in prior["families"]],
            }

        if families is None:
            try:
                families, medspec, seed_info = detect_fringe_rich(tf)
                seed_info["mode"] = "fresh_seed"
            except RuntimeError as exc:
                return ProcessResult(status="needs_review", message=str(exc))

        trajectories, all_blocks = track_all(tf, families)
        prior_qs = (
            [float(f["q"]) for f in prior["families"]]
            if had_usable_prior and prior
            else None
        )
        ok, qc_msg = qc_tracking(families, all_blocks, prior_qs=prior_qs)

        if not ok and had_usable_prior:
            # Soft prior failed — local reseed once.
            reseeded = True
            try:
                families, medspec, seed_info = detect_fringe_rich(tf)
                seed_info["mode"] = "reseed_after_qc_fail"
                seed_info["prior_qc"] = qc_msg
                trajectories, all_blocks = track_all(tf, families)
                ok, qc_msg = qc_tracking(families, all_blocks, prior_qs=None)
            except RuntimeError as exc:
                return ProcessResult(status="needs_review", message=str(exc))

        if not ok:
            return ProcessResult(
                status="needs_review",
                message=f"QC failed: {qc_msg}",
                families_q=[float(f["q"]) for f in families],
            )

        if diag_dir is not None:
            diag_dir = Path(diag_dir)
            diag_dir.mkdir(parents=True, exist_ok=True)
            with open(diag_dir / "seed_info.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "computer": computer,
                        "channel": channel,
                        "fingerprint": fingerprint,
                        "seed": seed_info,
                        "qc": qc_msg,
                        "used_prior": had_usable_prior and not reseeded,
                        "reseeded": reseeded,
                        "families": [
                            {
                                "q": float(f["q"]),
                                "hi": f.get("hi"),
                                "paired": f.get("paired"),
                                "row_score": f.get("row_score"),
                                "fx_ranges": f.get("fx_ranges"),
                            }
                            for f in families
                        ],
                    },
                    fh,
                    indent=2,
                )

        out_dir.mkdir(parents=True, exist_ok=True)
        est_bytes = n * h * w * np.dtype(dtype).itemsize
        rem_bytes = n * h * w * np.dtype(np.float32).itemsize
        bigtiff = est_bytes > 3_500_000_000
        bigtiff_rem = rem_bytes > 3_500_000_000

        csvfh = None
        writer = None
        if diag_dir is not None:
            csvfh = open(diag_dir / "temporal_tracking.csv", "w", newline="", encoding="utf-8")
            fields = ["frame", "removed_rms"]
            for i in range(len(families)):
                fields += [
                    f"family{i}_q",
                    f"family{i}_strength",
                    f"family{i}_gate",
                    f"family{i}_eff_max_alpha",
                    f"family{i}_residual_pass",
                    f"family{i}_residual_strength",
                ]
            writer = csv.DictWriter(csvfh, fieldnames=fields)
            writer.writeheader()

        rows: list[dict] = []
        acc_raw = np.zeros((h, w), dtype=np.float64)
        acc_clean = np.zeros((h, w), dtype=np.float64)
        acc_removed = np.zeros((h, w), dtype=np.float64)

        with (
            tifffile.TiffWriter(out_tif, bigtiff=bigtiff) as tw,
            tifffile.TiffWriter(removed_tif, bigtiff=bigtiff_rem) as tw_rem,
        ):
            for fi in range(n):
                frame = tf.pages[fi].asarray()
                preds = [traj[fi] for traj in trajectories]
                cleaned, removed, tracking = clean_frame_v22(
                    frame, families, preds, **PACK_D
                )
                tw.write(cleaned, contiguous=True)
                tw_rem.write(removed.astype(np.float32, copy=False), contiguous=True)
                acc_raw += np.asarray(frame, dtype=np.float64)
                acc_clean += np.asarray(cleaned, dtype=np.float64)
                acc_removed += np.asarray(removed, dtype=np.float64)
                row = tracking_row(fi, removed, tracking)
                rows.append(row)
                if writer is not None:
                    writer.writerow(
                        {
                            "frame": row["frame"],
                            "removed_rms": row["removed_rms"],
                            **{
                                k: row[k]
                                for i in range(len(families))
                                for k in (
                                    f"family{i}_q",
                                    f"family{i}_strength",
                                    f"family{i}_gate",
                                    f"family{i}_eff_max_alpha",
                                    f"family{i}_residual_pass",
                                    f"family{i}_residual_strength",
                                )
                            },
                        }
                    )
                if (fi + 1) % 200 == 0 or fi == n - 1:
                    print(f"      frames {fi + 1}/{n}", flush=True)

        if csvfh is not None:
            csvfh.close()

        inv_n = 1.0 / max(n, 1)
        readout_paths = write_readout(
            out_dir,
            tif_path=tif_path,
            families=families,
            params=PACK_D,
            seed_info=seed_info,
            medspec=medspec,
            n_frames=n,
            frame_hw=(h, w),
            source_shape=shape,
            mean_raw=acc_raw * inv_n,
            mean_cleaned=acc_clean * inv_n,
            mean_removed=acc_removed * inv_n,
            rows=rows,
            computer=computer,
            channel=channel,
            fingerprint=fingerprint,
            qc=qc_msg,
            used_prior=had_usable_prior and not reseeded,
            reseeded=reseeded,
        )

    if update_prior_on_success and not force_fresh_seed:
        save_prior(
            batch_root,
            computer,
            channel,
            families=families,
            fingerprint=fingerprint,
            source_tif=tif_path,
            notes=qc_msg,
        )

    pdf = readout_paths.get("overview_pdf") if readout_paths else None
    return ProcessResult(
        status="ok",
        message=qc_msg,
        out_tif=out_tif,
        out_dir=out_dir,
        removed_tif=removed_tif,
        overview_pdf=pdf if isinstance(pdf, Path) else None,
        used_prior=had_usable_prior and not reseeded,
        reseeded=reseeded,
        families_q=[float(f["q"]) for f in families],
    )
