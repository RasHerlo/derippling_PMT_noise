"""Write mean projections for stacks: conserved .tif + contrast-enhanced LUT .png."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from matplotlib import cm
from PIL import Image

SANDBOX = Path(r"F:\bPACNewData2026\PreProcessing Optimization\Level3b copy")
PACK_B = SANDBOX / "defringe_runs" / "v21_sweep_500fr" / "accepted" / "pack_B"
RAW_500 = SANDBOX / "inputs" / "slices_500fr" / "raw"


def mean_proj(path: Path) -> np.ndarray:
    # streaming mean to avoid holding full float stack twice
    with tifffile.TiffFile(path) as tf:
        n = tf.series[0].shape[0]
        acc = None
        for i in range(n):
            fr = tf.pages[i].asarray().astype(np.float64)
            acc = fr if acc is None else acc + fr
        return (acc / n).astype(np.float32)


def save_avg(out_dir: Path, stem: str, mean: np.ndarray, source: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tif_path = out_dir / f"{stem}_avg.tif"
    png_path = out_dir / f"{stem}_avg_inferno.png"
    # conserved float32 mean (not rescaled)
    tifffile.imwrite(tif_path, mean, photometric="minisblack")
    lo, hi = np.percentile(mean, (1.0, 99.5))
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((mean - lo) / (hi - lo), 0, 1)
    rgb = (cm.inferno(norm)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(png_path)
    meta = {
        "source": source,
        "shape": list(mean.shape),
        "dtype_mean": "float32",
        "tif": str(tif_path),
        "png": str(png_path),
        "png_lut": "inferno",
        "png_percentile_stretch": [1.0, 99.5],
        "mean_min": float(mean.min()),
        "mean_max": float(mean.max()),
        "mean_median": float(np.median(mean)),
    }
    (out_dir / f"{stem}_avg.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {tif_path.name} + {png_path.name}", flush=True)


def main():
    jobs = [
        (RAW_500 / "ChanA_raw_500fr.tif", PACK_B, "ChanA_raw_500fr", "inputs/slices_500fr/raw"),
        (RAW_500 / "ChanB_raw_500fr.tif", PACK_B, "ChanB_raw_500fr", "inputs/slices_500fr/raw"),
        (PACK_B / "ChanA_raw_500fr_v21.tif", PACK_B, "ChanA_raw_500fr_v21", "accepted/pack_B"),
        (PACK_B / "ChanB_raw_500fr_v21.tif", PACK_B, "ChanB_raw_500fr_v21", "accepted/pack_B"),
    ]
    for src, out_dir, stem, label in jobs:
        if not src.exists():
            print(f"SKIP missing {src}")
            continue
        print(f"Averaging {src} ...", flush=True)
        save_avg(out_dir, stem, mean_proj(src), label)


if __name__ == "__main__":
    main()
