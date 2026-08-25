"""Fringe-rich seeding and soft-prior hydration for long stacks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, gaussian_filter1d

# Allow importing reference/gpt cleaners without installing a package.
_REPO = Path(__file__).resolve().parents[1]
_GPT = _REPO / "reference" / "gpt"
if str(_GPT) not in sys.path:
    sys.path.insert(0, str(_GPT))

from pmt_fringe_raw_adaptive import (  # noqa: E402
    contiguous_ranges,
    detect_families,
    fft_log_amp,
    ridge_z_at_row,
    track_family_blocks,
)


def sample_median_spectrum(
    tf: tifffile.TiffFile,
    sample_n: int = 48,
    *,
    prefer_mid: bool = True,
) -> np.ndarray:
    """Cheap spectrum for hydrating fx masks when a soft prior already exists."""
    n = tf.series[0].shape[0]
    if prefer_mid and n > sample_n * 2:
        lo = n // 4
        hi = (3 * n) // 4
        inds = np.linspace(lo, hi - 1, min(sample_n, hi - lo), dtype=int)
    else:
        inds = np.linspace(0, n - 1, min(sample_n, n), dtype=int)
    specs = [fft_log_amp(tf.pages[int(i)].asarray()) for i in inds]
    return np.median(np.stack(specs), axis=0)


def hydrate_families(families_like: list[dict], medspec: np.ndarray, x_z_thresh: float = 3.5) -> list[dict]:
    """Rebuild x_weight / fx support on the current stack spectrum from q/hi seeds."""
    h, w = medspec.shape
    cy, cx = h // 2, w // 2
    fx = np.arange(w) - cx
    xvalid = (np.abs(fx) > 5) & (np.abs(fx) < cx - 10)

    out = []
    for src in families_like:
        fam = {
            "q": float(src["q"]),
            "hi": None if src.get("hi") is None else float(src["hi"]),
            "paired": bool(src.get("paired", True)),
            "row_score": float(src.get("row_score", 0.0)),
            "pair_score": src.get("pair_score"),
        }
        components = [fam["q"]]
        if fam["hi"] is not None:
            components.append(fam["hi"])

        zx = []
        for d in components:
            zx.append(ridge_z_at_row(medspec, +int(round(d))))
            zx.append(ridge_z_at_row(medspec, -int(round(d))))
        zx = np.max(np.stack(zx), axis=0)

        support = (zx > x_z_thresh) & xvalid
        support = binary_dilation(support, iterations=1)
        weight = gaussian_filter1d(support.astype(float), sigma=1.0)
        if weight.max() > 0:
            weight /= weight.max()

        fam["x_weight"] = weight
        fam["x_z"] = zx
        fam["fx_ranges"] = contiguous_ranges(fx[weight > 0.20])
        out.append(fam)
    return out


def detect_fringe_rich(
    tf: tifffile.TiffFile,
    *,
    block_size: int = 50,
    samples_per_block: int = 8,
    max_blocks: int = 40,
    row_z_thresh: float = 5.5,
    pair_z_min: float = 3.5,
    x_z_thresh: float = 3.5,
) -> tuple[list[dict], np.ndarray, dict]:
    """
    Detect families from the strongest block spectrum (avoids weak full-stack median).

    Returns (families, medspec_used, info).
    """
    n, h, w = tf.series[0].shape
    best = None  # (score, families, medspec, start, stop)

    starts = list(range(0, n, block_size))
    if len(starts) > max_blocks:
        # Spread sampling across the stack when very long.
        idx = np.linspace(0, len(starts) - 1, max_blocks, dtype=int)
        starts = [starts[i] for i in idx]

    for start in starts:
        stop = min(n, start + block_size)
        inds = np.linspace(start, stop - 1, min(samples_per_block, stop - start), dtype=int)
        specs = [fft_log_amp(tf.pages[int(i)].asarray()) for i in inds]
        med = np.median(np.stack(specs), axis=0)
        families, _, _ = detect_families(
            med,
            row_z_thresh=row_z_thresh,
            pair_z_min=pair_z_min,
            x_z_thresh=x_z_thresh,
            max_families=4,
            allow_standalone=False,
        )
        if not families:
            continue
        score = float(max(f["row_score"] for f in families))
        if best is None or score > best[0]:
            best = (score, families, med, int(start), int(stop))

    if best is None:
        raise RuntimeError("No high-confidence paired fringe family in fringe-rich scan.")

    score, families, med, start, stop = best
    info = {
        "mode": "fringe_rich_block",
        "block_start": start,
        "block_stop": stop,
        "row_score_max": score,
        "n_frames": int(n),
    }
    return families, med, info


def qc_tracking(
    families: list[dict],
    all_blocks: list[dict],
    *,
    prior_qs: list[float] | None = None,
    min_update_frac: float = 0.15,
    max_q_drift: float = 12.0,
) -> tuple[bool, str]:
    """Lightweight sanity check after block tracking."""
    if not families or not all_blocks:
        return False, "no families/blocks"

    updated = [b for b in all_blocks if b.get("updated")]
    frac = len(updated) / max(1, len(all_blocks))
    if frac < min_update_frac:
        return False, f"low track update frac={frac:.2f} (<{min_update_frac})"

    if prior_qs:
        for i, pq in enumerate(prior_qs):
            qs = [b["q"] for b in all_blocks if b.get("family") == i]
            if not qs:
                continue
            med_q = float(np.median(qs))
            if abs(med_q - pq) > max_q_drift:
                return False, f"family{i} q drift {med_q:.1f} vs prior {pq:.1f}"

    return True, f"ok update_frac={frac:.2f}"


def track_all(tf, families, **kwargs):
    trajectories, all_blocks = [], []
    for i, fam in enumerate(families):
        qtraj, blocks = track_family_blocks(tf, fam, **kwargs)
        trajectories.append(qtraj)
        for b in blocks:
            bb = dict(b)
            bb["family"] = i
            all_blocks.append(bb)
    return trajectories, all_blocks
