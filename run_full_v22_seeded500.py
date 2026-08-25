"""
Full-stack v2.2 (pack_D) defringe; families seeded from 500fr detection.

Writes:
  defringe_runs/v22_full_seeded500/
  inputs/defringed_v22/  (hardlinks; leaves defringed_v21 and mc_runs untouched)
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference" / "gpt"))
import pmt_fringe_raw_adaptive as v2  # noqa: E402
from pmt_fringe_raw_adaptive_v22 import clean_frame_v22  # noqa: E402

SANDBOX = Path(r"F:\bPACNewData2026\PreProcessing Optimization\Level3b copy")
OUT = SANDBOX / "defringe_runs" / "v22_full_seeded500"
PROMOTE = SANDBOX / "inputs" / "defringed_v22"

PACK_D = dict(
    max_alpha=0.85,
    max_alpha_high=1.0,
    high_gate=0.95,
    high_strength=0.15,
    strength_span=0.12,
    residual_pass=True,
    residual_strength_min=0.03,
    residual_alpha=1.0,
    ratio_start=1.4,
    ratio_full=3.5,
)


def detect_on_slice(slice_path: Path):
    with tifffile.TiffFile(slice_path) as tf:
        med, _ = v2.learn_median_spectrum(tf, sample_n=80)
        families, row_profile, row_z = v2.detect_families(
            med,
            row_z_thresh=5.5,
            pair_z_min=3.5,
            x_z_thresh=3.5,
            max_families=4,
            allow_standalone=False,
        )
    return families, med, row_profile, row_z


def process_channel(chan: str):
    slice_path = SANDBOX / "inputs" / "slices_500fr" / "raw" / f"{chan}_raw_500fr.tif"
    full_path = SANDBOX / "inputs" / "raw" / chan / f"{chan}_stk.tif"
    out_dir = OUT / chan
    diag = out_dir / "diagnostics"
    out_tif = out_dir / f"{chan}_stk_defringed_v22.tif"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {chan} v2.2 ===", flush=True)
    families, med, row_profile, row_z = detect_on_slice(slice_path)
    if not families:
        raise RuntimeError(f"No family on 500fr seed for {chan}")
    for i, f in enumerate(families, 1):
        print(
            f"  seed family {i}: q={f['q']:.1f}, hi={f['hi']}, "
            f"row_z={f['row_score']:.1f}, fx={f['fx_ranges']}",
            flush=True,
        )

    with tifffile.TiffFile(full_path) as tf:
        n, h, w = tf.series[0].shape
        dtype = tf.pages[0].dtype
        print(f"Full stack: shape={(n,h,w)} dtype={dtype}", flush=True)

        trajectories, all_blocks = [], []
        for i, fam in enumerate(families):
            qtraj, blocks = v2.track_family_blocks(tf, fam)
            trajectories.append(qtraj)
            for b in blocks:
                bb = dict(b)
                bb["family"] = i
                all_blocks.append(bb)

        v2.save_diagnostics(diag, med, families, row_profile, row_z, all_blocks)
        meta = {
            "version": "v2.2_pack_D_seeded500",
            "seed_slice": str(slice_path),
            "params": PACK_D,
            "families": [
                {
                    "q": f["q"],
                    "hi": f["hi"],
                    "paired": f["paired"],
                    "row_score": f["row_score"],
                    "fx_ranges": f["fx_ranges"],
                }
                for f in families
            ],
        }
        (diag / "v22_seeded_settings.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        est = n * h * w * np.dtype(dtype).itemsize
        bigtiff = est > 3_500_000_000
        with open(diag / "temporal_tracking.csv", "w", newline="", encoding="utf-8") as csvfh:
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

            with tifffile.TiffWriter(out_tif, bigtiff=bigtiff) as tw:
                for fi in range(n):
                    frame = tf.pages[fi].asarray()
                    preds = [traj[fi] for traj in trajectories]
                    cleaned, removed, tracking = clean_frame_v22(
                        frame, families, preds, **PACK_D
                    )
                    tw.write(cleaned, contiguous=True)
                    row = {
                        "frame": fi,
                        "removed_rms": float(
                            np.sqrt(np.mean(removed.astype(float) ** 2))
                        ),
                    }
                    for i, t in enumerate(tracking):
                        row[f"family{i}_q"] = t["q"]
                        row[f"family{i}_strength"] = t["strength"]
                        row[f"family{i}_gate"] = t["gate"]
                        row[f"family{i}_eff_max_alpha"] = t["eff_max_alpha"]
                        row[f"family{i}_residual_pass"] = t["residual_pass"]
                        row[f"family{i}_residual_strength"] = t["residual_strength"]
                    writer.writerow(row)
                    if (fi + 1) % 100 == 0 or fi == n - 1:
                        print(f"  Processed {fi+1}/{n}", flush=True)

    print(f"Wrote {out_tif}", flush=True)
    return out_tif, meta


def hardlink_promote(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    r = subprocess.run(
        ["cmd", "/c", "mklink", "/H", str(dst), str(src)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"hardlink failed: {r.stdout} {r.stderr}")
    print(f"Promoted: {dst}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metas = {}
    for chan in ("ChanA", "ChanB"):
        out_tif, meta = process_channel(chan)
        metas[chan] = meta
        hardlink_promote(out_tif, PROMOTE / chan / f"{chan}_stk_defringed_v22.tif")

    status = {
        "status": "completed",
        "method": "v2.2_pack_D_seeded500",
        "params": PACK_D,
        "outputs": {"runs": str(OUT), "promoted": str(PROMOTE)},
        "note": "Preferred over defringed_v21 for SUPPORT retrain / suite2p compares.",
        "channels": metas,
    }
    (OUT / "STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (OUT / "STATUS.md").write_text(
        "\n".join(
            [
                "# Full-stack v2.2 pack_D (seeded from 500fr)",
                "",
                "Status: **completed**",
                "",
                "ChanA strong25 residual on 500fr: pack_B 9.3% → **pack_D 6.9%**.",
                "gate0 RMS=0; ChanB also improved slightly.",
                "",
                "## Outputs",
                "",
                "- `inputs/defringed_v22/ChanA|B/*_stk_defringed_v22.tif`",
                "- `defringe_runs/v22_full_seeded500/`",
                "",
                "`inputs/defringed_v21` kept for A/B. Do not touch `mc_runs/`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
