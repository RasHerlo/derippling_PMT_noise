"""Figure 2, Catalog-entry: one catalog entry applied by v4 to the recording it came from.

A catalog entry stores geometry only (band row ``q``, partner row ``hi``, fx
ranges). This figure loads a frame of the dark-current movie the entry was
measured on, marks the entry's rows in its Fourier picture, and runs v4's own
catalog step (``_catalog_line`` + ``apply_lines``) on that frame.

Six panels (2×3): raw frame, Fourier picture, mask; then IFFT of the mask
alone, the same mask applied to this frame (removed), and the cleaned frame.

``python -m demo_Sep2026.fig02_catalog_entry --index 3 --frame 667``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from batch_defringe.library import catalog_path, load_catalog
from batch_defringe.process_v4 import TRACK_SEARCH, _catalog_line, _xvalid, apply_lines

from .common import _percentile_limits, fft_log_amp, load_frame, new_figure, save_figure, search_q
from .paths import DATA_ROOT, OLD_DATA_ROOT

REPO = Path(__file__).resolve().parents[1]
MAX_ALPHA = 1.0


def source_stack(entry: dict) -> Path:
    """Dark-current stack behind a catalog entry, via darkcurrent/registry.json."""
    origin = str(entry.get("origin") or "")
    if not origin.startswith("darkcurrent:"):
        raise ValueError(f"entry is not from a dark-current run: {origin}")
    label = origin.split(":")[2]
    reg = json.loads((REPO / "darkcurrent" / "registry.json").read_text(encoding="utf-8"))
    for rec in reg["recordings"]:
        if rec.get("label") == label:
            p = str(rec["stacks"][entry["channel"]])
            return Path(p.replace(str(OLD_DATA_ROOT), str(DATA_ROOT), 1))
    raise KeyError(f"{label} not in darkcurrent/registry.json")


def _band_rows(fam: dict, h: int) -> list[int]:
    cy = h // 2
    offsets = [float(fam["q"])]
    if fam.get("paired", True):
        offsets.append(cy - float(fam["q"]))
    return sorted({cy + s * int(round(d)) for d in offsets for s in (-1, 1) if 0 <= cy + s * int(round(d)) < h})


def draw(index: int, frame_idx: int) -> dict:
    records = load_catalog()["records"]
    if not 0 <= index < len(records):
        raise IndexError(f"catalog index {index} outside 0..{len(records) - 1}")
    entry = records[index]
    fam = (entry.get("families") or [None])[0]
    if fam is None:
        raise ValueError("entry has no families")
    tif = source_stack(entry)
    raw, n_frames = load_frame(frame_idx, tif)
    h, w = raw.shape
    cy, cx = h // 2, w // 2
    logamp = fft_log_amp(raw)

    line = _catalog_line(fam, logamp)
    q_used = None
    fx_ranges_now = None
    if line is not None:
        q_used, _ = search_q(logamp, float(line.q), True, _xvalid(w), TRACK_SEARCH)
        fx_ranges_now = line.family.get("fx_ranges")
        trial = apply_lines(raw, [line], max_alpha=MAX_ALPHA, search=TRACK_SEARCH)
        applied = trial["applied"]
        removed = trial["removed"]
        cleaned = trial["cleaned"]
        used = True
    else:
        applied = np.zeros((h, w), dtype=np.float32)
        removed = np.zeros((h, w), dtype=np.float32)
        cleaned = np.asarray(raw)
        trial = {"removed_rms": 0.0, "n_active": 0}
        used = False

    # IFFT of the mask only: no amplitudes or phases from the frame.
    mask_alone = np.real(np.fft.ifft2(np.fft.ifftshift(np.asarray(applied, dtype=np.float64))))

    extent = (-cx - 0.5, w - cx - 0.5, h - cy - 0.5, -cy - 0.5)
    rows = _band_rows(fam, h)

    fig = new_figure()
    fig.suptitle(
        f"Catalog-entry  ·  {entry.get('channel')} dark current  ·  band row {fam['q']:.0f} "
        f"(partner {fam.get('hi', 0):.0f})  ·  frame {frame_idx}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    gs = fig.add_gridspec(2, 3, left=0.045, right=0.99, top=0.90, bottom=0.14, wspace=0.22, hspace=0.32)

    def _hide(ax):
        ax.set_xticks([])
        ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 0])
    lo, hi = _percentile_limits(raw)
    ax.imshow(raw, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title("1. Real space: dark-current frame", fontsize=10)
    _hide(ax)

    ax = fig.add_subplot(gs[0, 1])
    slo, shi = _percentile_limits(logamp, (3.0, 99.7))
    ax.imshow(logamp, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest", extent=extent)
    for y in rows:
        ax.axhline(y - cy, color="red", lw=0.6, alpha=0.7)
    ax.set_title("2. Fourier space (red = entry's band rows)", fontsize=10)
    ax.set_xlabel("fx")
    ax.set_ylabel("fy")

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(applied, cmap="gray", vmin=0, vmax=1, interpolation="nearest", extent=extent)
    ax.set_title(f"3. Mask v4 applies (strength {MAX_ALPHA:g})", fontsize=10)
    ax.set_xlabel("fx")
    ax.set_ylabel("fy")

    ax = fig.add_subplot(gs[1, 0])
    mlim = float(np.percentile(np.abs(mask_alone), 99.5)) or 1.0
    ax.imshow(mask_alone, cmap="gray", vmin=-mlim, vmax=mlim, interpolation="nearest")
    ax.set_title("4. Real space: IFFT of the mask alone", fontsize=10)
    _hide(ax)

    ax = fig.add_subplot(gs[1, 1])
    rlim = float(np.percentile(np.abs(removed), 99.5)) or 1.0
    ax.imshow(removed, cmap="gray", vmin=-rlim, vmax=rlim, interpolation="nearest")
    ax.set_title("5. Real space: mask applied to this frame (removed)", fontsize=10)
    _hide(ax)

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(cleaned, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title("6. Real space: cleaned frame", fontsize=10)
    _hide(ax)

    fp = entry.get("fingerprint") or {}
    fig.text(
        0.045,
        0.11,
        (
            f"Entry {index} of {catalog_path().name}: {entry.get('computer')}, recorded {str(entry.get('date_utc'))[:10]}, "
            f"{fp.get('pixelX')}×{fp.get('pixelY')} px, {fp.get('frameRate')} Hz, scan mode {fp.get('scanMode')}.  "
            f"Source: {tif.parent.parent.parent.name}\\{tif.parent.name}\\{tif.name}\n"
            "Panel 4 is the mask inverted by itself (no frame). Panel 5 is that mask applied to this frame. "
            "Panel 6 is the frame after that removal.\n"
            + (
                f"This frame: used, row {fam['q']:.0f} → {q_used:.0f}; left-right ranges stored {fam.get('fx_ranges')} "
                f"→ measured now {fx_ranges_now}."
                if used
                else "This frame: not used (nothing stands out on these rows)."
            )
        ),
        va="top",
        fontsize=8.5,
        color="0.25",
    )

    numbers = {
        "catalog_index": index,
        "catalog_file": str(catalog_path()),
        "entry": entry,
        "source_stack": str(tif),
        "frame": frame_idx,
        "n_frames": n_frames,
        "image_hw": [h, w],
        "band_rows_offset_from_centre": [int(y - cy) for y in rows],
        "used_on_frame": used,
        "q_stored": float(fam["q"]),
        "q_used": q_used,
        "fx_ranges_stored": fam.get("fx_ranges"),
        "fx_ranges_measured_on_frame": fx_ranges_now,
        "max_alpha": MAX_ALPHA,
        "search": TRACK_SEARCH,
        "removed_rms": float(trial["removed_rms"]),
        "mask_nonzero_pixels": int(np.count_nonzero(applied > 1e-3)),
        "mask_max": float(np.max(applied)),
        "mask_alone_rms": float(np.sqrt(np.mean(mask_alone * mask_alone))),
        "cleaned_median": float(np.median(np.asarray(cleaned, dtype=np.float64))),
    }
    written = save_figure(fig, f"fig02_catalog_entry_{index}_{entry.get('channel')}_q{int(fam['q'])}_frame{frame_idx}", numbers)
    numbers["written"] = [str(p) for p in written]
    return numbers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, default=3, help="Entry number in fringe_library/catalog.json (0-based)")
    ap.add_argument("--frame", type=int, default=667, help="Frame of the dark-current movie")
    args = ap.parse_args(argv)
    numbers = draw(args.index, args.frame)
    for p in numbers["written"]:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
