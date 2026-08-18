"""
Full-stack v2.1 pack_B defringe with families seeded from 500fr detection.

Why: median spectrum over all 5400 frames can prefer a weak spurious family
(ChanA q=6) over the clear 500fr signature (q=14). Seed detection from the
500fr slices, then track+clean the full raw stacks.

Writes:
  defringe_runs/v21_full_seeded500/
  inputs/defringed_v21/  (hardlinks; does not touch inputs/defringed or mc_runs)
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
from pmt_fringe_raw_adaptive_v21 import clean_frame_v21  # noqa: E402

SANDBOX = Path(r"F:\bPACNewData2026\PreProcessing Optimization\Level3b copy")
OUT = SANDBOX / "defringe_runs" / "v21_full_seeded500"
PROMOTE = SANDBOX / "inputs" / "defringed_v21"

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
    out_tif = out_dir / f"{chan}_stk_defringed_v21.tif"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {chan} ===", flush=True)
    print(f"Seed detect from: {slice_path}", flush=True)
    families, med, row_profile, row_z = detect_on_slice(slice_path)
    if not families:
        raise RuntimeError(f"No family detected on 500fr seed for {chan}")
    for i, f in enumerate(families, 1):
        print(
            f"  seed family {i}: q={f['q']:.1f}, hi={f['hi']}, "
            f"row_z={f['row_score']:.1f}, fx={f['fx_ranges']}",
            flush=True,
        )

    with tifffile.TiffFile(full_path) as tf:
        n, h, w = tf.series[0].shape
        dtype = tf.pages[0].dtype
        print(f"Full stack: {full_path} shape={(n,h,w)} dtype={dtype}", flush=True)

        trajectories = []
        all_blocks = []
        for i, fam in enumerate(families):
            qtraj, blocks = v2.track_family_blocks(tf, fam)
            trajectories.append(qtraj)
            for b in blocks:
                bb = dict(b)
                bb["family"] = i
                all_blocks.append(bb)

        v2.save_diagnostics(diag, med, families, row_profile, row_z, all_blocks)
        meta = {
            "version": "v2.1_pack_B_seeded500",
            "seed_slice": str(slice_path),
            "full_input": str(full_path),
            "params": PACK_B,
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
        (diag / "v21_seeded_settings.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        est = n * h * w * np.dtype(dtype).itemsize
        bigtiff = est > 3_500_000_000
        csv_path = diag / "temporal_tracking.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfh:
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
                    cleaned, removed, tracking = clean_frame_v21(
                        frame, families, preds, **PACK_B
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
    # same volume hardlink
    r = subprocess.run(["cmd", "/c", "mklink", "/H", str(dst), str(src)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"hardlink failed: {r.stdout} {r.stderr}")
    print(f"Promoted hardlink: {dst}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # remove incomplete prior full run output if present
    bad = SANDBOX / "defringe_runs" / "v21_full" / "ChanA" / "ChanA_stk_defringed_v21.tif"
    if bad.exists():
        try:
            bad.unlink()
            print(f"Removed incomplete prior output: {bad}", flush=True)
        except Exception as e:
            print(f"Could not remove prior partial TIFF ({e}); continuing", flush=True)

    metas = {}
    for chan in ("ChanA", "ChanB"):
        out_tif, meta = process_channel(chan)
        metas[chan] = meta
        promote = PROMOTE / chan / f"{chan}_stk_defringed_v21.tif"
        hardlink_promote(out_tif, promote)

    status = {
        "status": "completed",
        "method": "v2.1_pack_B_seeded500",
        "note": "Families detected on 500fr slices, applied to full 5400-frame raw stacks.",
        "outputs": {
            "runs": str(OUT),
            "promoted": str(PROMOTE),
        },
        "do_not_use": "inputs/defringed (legacy v2)",
        "channels": metas,
    }
    (OUT / "STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (OUT / "STATUS.md").write_text(
        "\n".join(
            [
                "# Full-stack v2.1 pack_B (seeded from 500fr)",
                "",
                "Status: **completed**",
                "",
                "Detection seeded from `inputs/slices_500fr/raw/` so ChanA keeps q≈14",
                "(full-stack median alone prefers weak q≈6).",
                "",
                "## Outputs for suite2p",
                "",
                "- `inputs/defringed_v21/ChanA/ChanA_stk_defringed_v21.tif`",
                "- `inputs/defringed_v21/ChanB/ChanB_stk_defringed_v21.tif`",
                "",
                "Also under `defringe_runs/v21_full_seeded500/`.",
                "",
                "**Do not use** `inputs/defringed` (legacy v2).",
                "**Do not touch** `mc_runs/`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
