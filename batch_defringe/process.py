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

from .eval_report import write_eval_report
from .experiment_xml import fingerprint_compatible
from .library import (
    append_catalog_record,
    append_local_record,
    catalog_status,
    lookup_prior,
    make_record,
)
from .priors import load_prior, save_prior
from .readout import (
    cleaned_path_for,
    removed_path_for,
    tracking_row,
    write_failure_readout,
    write_readout,
)
from .seed import (
    collect_block_spectra,
    detect_fringe_rich,
    hydrate_families,
    ladder_inspect,
    library_family_supported,
    qc_tracking,
    sample_mean_frame,
    sample_median_spectrum,
    track_all,
)
from .shutter_detect import (
    detect_shutter_windows,
    format_shutter_span,
    scan_frame_stats,
    shutter_public,
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
    prior_branch: str | None = None


def default_output_path(tif_path: Path) -> Path:
    return cleaned_path_for(tif_path)


def _pop_block_specs(seed_info: dict) -> list[dict]:
    return list(seed_info.pop("_block_specs", []) or [])


def _write_review(
    *,
    tif_path: Path,
    out_dir: Path,
    tf: tifffile.TiffFile,
    families: list[dict],
    seed_info: dict,
    medspec: np.ndarray | None,
    computer: str,
    channel: str,
    fingerprint: dict,
    qc: str,
    used_prior: bool,
    reseeded: bool,
    prior_branch: str | None,
    ladder: list[dict] | None,
    message: str,
    block_specs: list[dict] | None = None,
    catalog: dict | None = None,
) -> Path | None:
    n, h, w = tf.series[0].shape
    mean_raw = sample_mean_frame(tf)
    specs = list(block_specs or [])
    if not specs:
        specs = collect_block_spectra(tf)
    catalog = catalog or seed_info.get("catalog")
    try:
        paths = write_eval_report(
            out_dir,
            tf=tf,
            tif_path=tif_path,
            block_specs=specs,
            medspec=medspec,
            mean_raw=mean_raw,
            params=PACK_D,
            computer=computer,
            channel=channel,
            message=message,
            catalog=catalog,
            shutter=seed_info.get("shutter"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"      eval report failed ({exc}); writing basic needs_review page", flush=True)
        paths = write_failure_readout(
            out_dir,
            tif_path=tif_path,
            families=families,
            params=PACK_D,
            seed_info=seed_info,
            medspec=medspec,
            n_frames=n,
            frame_hw=(h, w),
            source_shape=tf.series[0].shape,
            mean_raw=mean_raw,
            computer=computer,
            channel=channel,
            fingerprint=fingerprint,
            qc=qc,
            used_prior=used_prior,
            reseeded=reseeded,
            prior_branch=prior_branch,
            ladder=ladder,
            failure_message=message,
            catalog=catalog,
        )
    pdf = paths.get("overview_pdf")
    return pdf if isinstance(pdf, Path) else None


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
    recording_date: str | None = None,
    write_clean: bool = True,
) -> ProcessResult:
    tif_path = Path(tif_path)
    out_tif = Path(out_tif) if out_tif else default_output_path(tif_path)
    out_dir = out_tif.parent
    removed_tif = removed_path_for(tif_path, out_dir)

    if skip_existing and write_clean and out_tif.is_file():
        return ProcessResult(
            status="skipped",
            message=f"exists: {out_dir.name}/{out_tif.name}",
            out_tif=out_tif,
            out_dir=out_dir,
            removed_tif=removed_tif if removed_tif.is_file() else None,
        )

    lib_hit = None
    if not force_fresh_seed:
        lib_hit = lookup_prior(
            computer=computer,
            channel=channel,
            fingerprint=fingerprint,
            recording_date=recording_date,
            batch_root=batch_root,
        )
    prior = None if force_fresh_seed else load_prior(batch_root, computer, channel)
    cache_prior_ok = bool(
        (not force_fresh_seed)
        and prior
        and prior.get("families")
        and fingerprint_compatible(prior.get("fingerprint"), fingerprint)
    )

    library_families = (lib_hit or {}).get("families") or []
    prior_branch = (lib_hit or {}).get("branch")
    # Library A/B beat a same-root cache prior when raster matches.
    use_library = bool(library_families) and lib_hit is not None
    use_cache_prior = cache_prior_ok and not use_library

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
        supported_qs: list[float] = []
        rejected_qs: list[float] = []

        shutter_det = detect_shutter_windows(scan_frame_stats(tf))
        shutter_pub = shutter_public(shutter_det)
        seed_info["shutter"] = shutter_pub
        print(f"      shutter auto: {format_shutter_span(shutter_det)}", flush=True)

        def _keep_shutter(info: dict) -> dict:
            info["shutter"] = shutter_pub
            return info

        def _note_catalog(*, used: bool, reseeded_flag: bool) -> dict:
            info = catalog_status(
                lib_hit,
                used=used,
                reseeded=reseeded_flag,
                cache_used=use_cache_prior and used,
                supported_qs=supported_qs,
                rejected_qs=rejected_qs,
            )
            seed_info["catalog"] = info
            return info

        if use_library or use_cache_prior:
            medspec = sample_median_spectrum(tf)
            src_fams = library_families if use_library else prior["families"]
            if use_library:
                kept, dropped = [], []
                for fam in src_fams:
                    q = float(fam["q"])
                    if library_family_supported(fam, medspec):
                        kept.append(fam)
                        supported_qs.append(q)
                    else:
                        dropped.append(fam)
                        rejected_qs.append(q)
                src_fams = kept
            if src_fams:
                families = hydrate_families(src_fams, medspec)
                had_usable_prior = True
                seed_info = _keep_shutter(
                    {
                        "mode": "soft_prior",
                        "prior_branch": prior_branch if use_library else "cache",
                        "prior_source": (lib_hit or {}).get("origin")
                        if use_library
                        else prior.get("source_tif"),
                        "prior_qs": [float(f["q"]) for f in src_fams],
                    }
                )
            else:
                families = None
                had_usable_prior = False

        if families is None:
            families, medspec, seed_info = detect_fringe_rich(
                tf, library_families=library_families or None
            )
            _keep_shutter(seed_info)
            seed_info["mode"] = "fresh_seed"
            if prior_branch:
                seed_info["prior_branch"] = prior_branch
            if library_families and medspec is not None and not supported_qs and not rejected_qs:
                for fam in library_families:
                    q = float(fam["q"])
                    if library_family_supported(fam, medspec):
                        supported_qs.append(q)
                    else:
                        rejected_qs.append(q)

        block_specs = _pop_block_specs(seed_info)

        if not families:
            ladder = ladder_inspect(
                block_specs, medspec, library_families=library_families or None
            )
            seed_info["ladder"] = ladder
            msg = "No high-confidence paired fringe family in fringe-rich scan."
            _note_catalog(used=False, reseeded_flag=False)
            pdf = _write_review(
                tif_path=tif_path,
                out_dir=out_dir,
                tf=tf,
                families=[],
                seed_info=seed_info,
                medspec=medspec,
                computer=computer,
                channel=channel,
                fingerprint=fingerprint,
                qc=msg,
                used_prior=False,
                reseeded=False,
                prior_branch=prior_branch,
                ladder=ladder,
                message=msg,
                block_specs=block_specs,
            )
            return ProcessResult(
                status="needs_review",
                message=msg,
                out_dir=out_dir,
                overview_pdf=pdf,
                prior_branch=prior_branch,
            )

        trajectories, all_blocks = track_all(tf, families)
        prior_qs = (
            [float(f["q"]) for f in (library_families if use_library else prior["families"])]
            if had_usable_prior
            else None
        )
        ok, qc_msg = qc_tracking(families, all_blocks, prior_qs=prior_qs)

        if not ok and had_usable_prior:
            reseeded = True
            families, medspec, seed_info = detect_fringe_rich(
                tf, library_families=library_families or None
            )
            _keep_shutter(seed_info)
            block_specs = _pop_block_specs(seed_info)
            seed_info["mode"] = "reseed_after_qc_fail"
            seed_info["prior_qc"] = qc_msg
            if not families:
                ladder = ladder_inspect(
                    block_specs, medspec, library_families=library_families or None
                )
                seed_info["ladder"] = ladder
                msg = f"QC failed then reseed empty: {qc_msg}"
                _note_catalog(used=True, reseeded_flag=True)
                pdf = _write_review(
                    tif_path=tif_path,
                    out_dir=out_dir,
                    tf=tf,
                    families=[],
                    seed_info=seed_info,
                    medspec=medspec,
                    computer=computer,
                    channel=channel,
                    fingerprint=fingerprint,
                    qc=msg,
                    used_prior=False,
                    reseeded=True,
                    prior_branch=prior_branch,
                    ladder=ladder,
                    message=msg,
                    block_specs=block_specs,
                )
                return ProcessResult(
                    status="needs_review",
                    message=msg,
                    out_dir=out_dir,
                    overview_pdf=pdf,
                    reseeded=True,
                    prior_branch=prior_branch,
                )
            trajectories, all_blocks = track_all(tf, families)
            ok, qc_msg = qc_tracking(families, all_blocks, prior_qs=None)

        if not ok:
            ladder = ladder_inspect(
                block_specs or [],
                medspec,
                library_families=library_families or None,
            )
            seed_info["ladder"] = ladder
            msg = f"QC failed: {qc_msg}"
            _note_catalog(used=had_usable_prior, reseeded_flag=reseeded)
            pdf = _write_review(
                tif_path=tif_path,
                out_dir=out_dir,
                tf=tf,
                families=families,
                seed_info=seed_info,
                medspec=medspec,
                computer=computer,
                channel=channel,
                fingerprint=fingerprint,
                qc=qc_msg,
                used_prior=had_usable_prior and not reseeded,
                reseeded=reseeded,
                prior_branch=prior_branch,
                ladder=ladder,
                message=msg,
                block_specs=block_specs,
            )
            return ProcessResult(
                status="needs_review",
                message=msg,
                out_dir=out_dir,
                overview_pdf=pdf,
                used_prior=had_usable_prior and not reseeded,
                reseeded=reseeded,
                families_q=[float(f["q"]) for f in families],
                prior_branch=prior_branch,
            )

        if not write_clean:
            _note_catalog(used=had_usable_prior, reseeded_flag=reseeded)
            mean_raw = sample_mean_frame(tf)
            paths = write_failure_readout(
                out_dir,
                tif_path=tif_path,
                families=families,
                params=PACK_D,
                seed_info=seed_info,
                medspec=medspec,
                n_frames=n,
                frame_hw=(h, w),
                source_shape=shape,
                mean_raw=mean_raw,
                computer=computer,
                channel=channel,
                fingerprint=fingerprint,
                qc=qc_msg + " (analyze-only, not applied)",
                used_prior=had_usable_prior and not reseeded,
                reseeded=reseeded,
                prior_branch=prior_branch,
                ladder=None,
                failure_message="analyze-only",
            )
            pdf = paths.get("overview_pdf")
            return ProcessResult(
                status="ok",
                message=qc_msg + " (analyze-only)",
                out_dir=out_dir,
                overview_pdf=pdf if isinstance(pdf, Path) else None,
                used_prior=had_usable_prior and not reseeded,
                reseeded=reseeded,
                families_q=[float(f["q"]) for f in families],
                prior_branch=prior_branch,
            )

        _note_catalog(used=had_usable_prior and not reseeded, reseeded_flag=reseeded)
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
                        "prior_branch": prior_branch,
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
            prior_branch=prior_branch,
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
        rec = make_record(
            source="live_clean",
            computer=computer,
            channel=channel,
            fingerprint=fingerprint,
            families=families,
            date_utc=recording_date,
            origin=str(tif_path),
            notes=qc_msg,
            complete=True,
        )
        try:
            append_local_record(batch_root, rec)
        except OSError as exc:
            print(f"      local library append skipped: {exc}", flush=True)
        try:
            append_catalog_record(rec)
        except OSError as exc:
            print(f"      repo catalog append skipped: {exc}", flush=True)

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
        prior_branch=prior_branch,
    )
