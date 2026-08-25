"""Characterise the PMT fringe layer in dark-current recordings.

These stacks have no biology, so the fringe can be measured directly rather than
inferred against structured signal. Three questions drive the measurements:

1. Which ridge families exist, and how strong are they without biology competing?
2. Does the fringe change character across the field of view?
3. How do amplitude / phase / frequency move over time (what masking survives)?

Every amplitude here is an *excess* over a matched control mask placed on
fringe-free rows. Dark current is mostly white detector noise, so the raw power
inside a ridge mask mostly measures how many bins the mask has; only the excess
over an equivalent off-ridge mask is fringe.

Reuses the production detector from ``reference/gpt`` so findings are expressed
in the same terms the cleaner uses (``q``, ``hi``, ``fx`` support).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

_REPO = Path(__file__).resolve().parents[1]
_GPT = _REPO / "reference" / "gpt"
if str(_GPT) not in sys.path:
    sys.path.insert(0, str(_GPT))

from pmt_fringe_raw_adaptive import (  # noqa: E402
    detect_families,
    fft_log_amp,
    ridge_z_at_row,
    robust_local_z,
)

BG_OFFSETS = list(range(-9, -4)) + list(range(5, 10))


def sample_indices(n: int, sample_n: int) -> np.ndarray:
    if sample_n >= n:
        return np.arange(n)
    return np.linspace(0, n - 1, sample_n, dtype=int)


def linear_amp(frame: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.fftshift(np.fft.fft2(np.asarray(frame, dtype=float))))


def median_log_spectrum(tf: tifffile.TiffFile, inds: np.ndarray) -> np.ndarray:
    return np.median(
        np.stack([fft_log_amp(tf.pages[int(i)].asarray()) for i in inds]), axis=0
    )


def median_linear_amp(tf: tifffile.TiffFile, inds: np.ndarray) -> np.ndarray:
    return np.median(
        np.stack([linear_amp(tf.pages[int(i)].asarray()) for i in inds]), axis=0
    )


def family_report(medspec: np.ndarray) -> dict[str, Any]:
    """Detected families plus the raw row-z landscape behind the decision.

    Runs the production thresholds and a diagnostic-only relaxed pass so a
    'nothing detected' result can be distinguished from 'nothing present'.
    """
    h, w = medspec.shape
    cy, cx = h // 2, w // 2
    fx = np.arange(w) - cx
    xvalid = (np.abs(fx) > 5) & (np.abs(fx) < cx - 10)
    row_profile = np.percentile(medspec[:, xvalid], 95, axis=1)
    row_z, _ = robust_local_z(row_profile, radius=8, exclude=1)

    def summarize(fams: list[dict]) -> list[dict]:
        return [
            {
                "q": float(f["q"]),
                "hi": None if f.get("hi") is None else float(f["hi"]),
                "paired": bool(f["paired"]),
                "row_score": float(f["row_score"]),
                "pair_score": None
                if f.get("pair_score") is None
                else float(f["pair_score"]),
                "fx_ranges": f.get("fx_ranges", []),
                "period_px": float(h) / float(f["q"]) if f["q"] else None,
            }
            for f in fams
        ]

    production, _, _ = detect_families(
        medspec,
        row_z_thresh=5.5,
        pair_z_min=3.5,
        x_z_thresh=3.5,
        max_families=4,
        allow_standalone=False,
    )
    relaxed, _, _ = detect_families(
        medspec,
        row_z_thresh=4.0,
        pair_z_min=2.5,
        x_z_thresh=2.5,
        max_families=4,
        allow_standalone=True,
    )

    band = row_z[cy + 5 : h - 5]
    order = np.argsort(band)[::-1][:8]
    peaks = [
        {"q": int(i + 5), "row_z": float(band[i]), "period_px": float(h) / float(i + 5)}
        for i in order
    ]

    return {
        "production_families": summarize(production),
        "relaxed_families": summarize(relaxed),
        "top_row_peaks": peaks,
        "row_z_max": float(np.nanmax(band)),
    }


def ridge_support(medspec: np.ndarray, dy: float, x_z_thresh: float = 3.5) -> np.ndarray:
    h, w = medspec.shape
    cx = w // 2
    fx = np.arange(w) - cx
    xvalid = (np.abs(fx) > 5) & (np.abs(fx) < cx - 10)
    zx = ridge_z_at_row(medspec, int(round(dy)))
    return (zx > x_z_thresh) & xvalid


def row_excess_snr(
    med_amp: np.ndarray, dy: float, x_sel: np.ndarray
) -> dict[str, float]:
    """Excess linear amplitude at row ``dy`` over neighbouring rows.

    ``snr`` compares the excess at the candidate row with the same statistic
    evaluated on the neighbour rows themselves, which is the noise floor of the
    estimator. A real ridge sits well above 1.
    """
    h, w = med_amp.shape
    cy = h // 2
    if not x_sel.any():
        return {"excess": 0.0, "noise": 0.0, "snr": 0.0}

    def excess_at(y: int) -> float:
        bg = []
        for off in BG_OFFSETS:
            yy = y + off
            if 0 <= yy < h:
                bg.append(med_amp[yy, x_sel])
        if not bg:
            return 0.0
        base = np.median(np.stack(bg), axis=0)
        return float(np.mean(np.maximum(med_amp[y, x_sel] - base, 0.0)))

    vals = []
    for sgn in (-1, 1):
        y = cy + sgn * int(round(dy))
        if 0 <= y < h:
            vals.append(excess_at(y))
    ridge = float(np.mean(vals)) if vals else 0.0

    noise_vals = []
    for sgn in (-1, 1):
        y0 = cy + sgn * int(round(dy))
        for off in BG_OFFSETS:
            y = y0 + off
            if 0 <= y < h:
                noise_vals.append(excess_at(y))
    noise = float(np.mean(noise_vals)) if noise_vals else 0.0

    return {
        "excess": ridge,
        "noise": noise,
        "snr": float(ridge / (noise + 1e-12)),
    }


def build_masks(
    medspec: np.ndarray,
    families: list[dict],
    *,
    x_z_thresh: float = 3.5,
    y_radius: int = 2,
    y_sigma: float = 1.0,
    control_offset: int = 7,
) -> dict[str, Any]:
    """Ridge mask plus a matched off-ridge control mask.

    The control has the same horizontal support and vertical profile, shifted to
    rows that carry no fringe, so subtracting its power removes the white-noise
    contribution of the mask itself.
    """
    h, w = medspec.shape
    cy = h // 2

    ridge = np.zeros((h, w), dtype=float)
    ridge_rows: set[int] = set()
    entries: list[tuple[int, np.ndarray]] = []

    for fam in families:
        rows = [float(fam["q"])]
        if fam.get("hi") is not None:
            rows.append(float(fam["hi"]))
        for dy in rows:
            support = ridge_support(medspec, dy, x_z_thresh)
            if not support.any():
                continue
            weight = support.astype(float)
            for sgn in (-1, 1):
                y0 = cy + sgn * int(round(dy))
                for off in range(-y_radius, y_radius + 1):
                    y = y0 + off
                    if not (0 <= y < h):
                        continue
                    prof = float(np.exp(-0.5 * (off / y_sigma) ** 2))
                    ridge[y, :] = np.maximum(ridge[y, :], weight * prof)
                    ridge_rows.add(y)
                    entries.append((y, weight * prof))

    control = np.zeros((h, w), dtype=float)
    for y, wvec in entries:
        placed = False
        for off in (control_offset, -control_offset, control_offset + 4, -(control_offset + 4)):
            y2 = y + off
            if not (0 <= y2 < h):
                continue
            if abs(y2 - cy) < 5:
                continue
            if any(abs(y2 - ry) <= y_radius for ry in ridge_rows):
                continue
            control[y2, :] = np.maximum(control[y2, :], wvec)
            placed = True
            break
        if not placed:
            continue

    ref_bin = (cy, w // 2)
    best = -np.inf
    for fam in families:
        for dy in [fam["q"]] + ([fam["hi"]] if fam.get("hi") is not None else []):
            if dy is None:
                continue
            zx = ridge_z_at_row(medspec, int(round(float(dy))))
            support = ridge_support(medspec, float(dy), x_z_thresh)
            if not support.any():
                continue
            cand = np.where(support, zx, -np.inf)
            j = int(np.argmax(cand))
            if cand[j] > best:
                best = float(cand[j])
                ref_bin = (cy + int(round(float(dy))), j)

    return {
        "ridge": ridge,
        "control": control,
        "ref_bin": ref_bin,
        "ridge_power": float(np.sum(ridge**2)),
        "control_power": float(np.sum(control**2)),
        "ridge_bins": int((ridge > 0).sum()),
        "control_bins": int((control > 0).sum()),
    }


def spectral_excess(
    amp: np.ndarray,
    dy: float,
    x_sel: np.ndarray,
    *,
    y_radius: int = 2,
) -> float:
    """Mean excess linear amplitude on the ridge rows over neighbouring rows.

    Same estimator as the production residual score: excess inside the ridge
    support, measured against a local row background, so mask size and white
    noise do not inflate it.
    """
    h, _ = amp.shape
    cy = h // 2
    if not x_sel.any():
        return 0.0
    vals = []
    for sgn in (-1, 1):
        y0 = cy + sgn * int(round(dy))
        for off in range(-y_radius, y_radius + 1):
            y = y0 + off
            if not (0 <= y < h):
                continue
            bg = []
            for boff in BG_OFFSETS:
                yy = y + boff
                if 0 <= yy < h:
                    bg.append(amp[yy, x_sel])
            if not bg:
                continue
            base = np.median(np.stack(bg), axis=0)
            vals.append(float(np.mean(np.maximum(amp[y, x_sel] - base, 0.0))))
    return float(np.mean(vals)) if vals else 0.0


def fov_excess_map(
    tf: tifffile.TiffFile,
    medspec: np.ndarray,
    q: float,
    inds: np.ndarray,
    *,
    tiles: int = 4,
) -> dict[str, Any]:
    """Ridge excess per FOV tile, each tile measured in its own spectrum.

    Tiles have coarser frequency resolution, so ``q`` is rescaled to the tile
    height. The control column reports the same estimator on off-ridge rows,
    which is the noise floor for that tile.
    """
    n, h, w = tf.series[0].shape
    ty, tx = h // tiles, w // tiles
    q_tile = float(q) * ty / h
    if q_tile < 3:
        q_tile = 3.0

    acc: dict[tuple[int, int], list[np.ndarray]] = {}
    dc = np.zeros((tiles, tiles), dtype=float)
    for i in inds:
        frame = np.asarray(tf.pages[int(i)].asarray(), dtype=float)
        for r in range(tiles):
            for c in range(tiles):
                sub = frame[r * ty : (r + 1) * ty, c * tx : (c + 1) * tx]
                acc.setdefault((r, c), []).append(linear_amp(sub))
                dc[r, c] += float(sub.mean())
    dc /= float(len(inds))

    exc = np.zeros((tiles, tiles), dtype=float)
    ctrl = np.zeros((tiles, tiles), dtype=float)
    for (r, c), specs in acc.items():
        med = np.median(np.stack(specs), axis=0)
        th, tw = med.shape
        tcx = tw // 2
        tfx = np.arange(tw) - tcx
        valid = (np.abs(tfx) > 2) & (np.abs(tfx) < tcx - 3)
        zx = ridge_z_at_row(np.log1p(med), int(round(q_tile)))
        sel = (zx > 2.5) & valid
        if not sel.any():
            sel = valid
        exc[r, c] = spectral_excess(med, q_tile, sel)
        ctrl[r, c] = spectral_excess(med, q_tile + 7, sel)

    snr = exc / (ctrl + 1e-12)
    rel = exc / (dc + 1e-12)
    return {
        "tiles": tiles,
        "q_tile": q_tile,
        "excess": exc.tolist(),
        "control": ctrl.tolist(),
        "snr": snr.tolist(),
        "tile_dc": dc.tolist(),
        "excess_rel_to_dc": rel.tolist(),
        "excess_min": float(exc.min()),
        "excess_max": float(exc.max()),
        "excess_ratio_max_min": float(exc.max() / (exc.min() + 1e-12)),
        "snr_min": float(snr.min()),
        "snr_max": float(snr.max()),
        "row_means": exc.mean(axis=1).tolist(),
        "col_means": exc.mean(axis=0).tolist(),
    }


def x_profile(
    tf: tifffile.TiffFile,
    q: float,
    inds: np.ndarray,
    *,
    n_windows: int = 16,
) -> dict[str, Any]:
    """Ridge excess as a function of position along the fast (x) axis.

    Uses full-height column strips, so vertical frequency resolution is
    preserved and ``q`` needs no rescaling. A resonant scanner dwells longest at
    the turnarounds, so any x-dependence matters for how a mask should be
    weighted.
    """
    n, h, w = tf.series[0].shape
    ww = w // n_windows

    acc: list[list[np.ndarray]] = [[] for _ in range(n_windows)]
    dc = np.zeros(n_windows, dtype=float)
    for i in inds:
        frame = np.asarray(tf.pages[int(i)].asarray(), dtype=float)
        for k in range(n_windows):
            strip = frame[:, k * ww : (k + 1) * ww]
            acc[k].append(linear_amp(strip))
            dc[k] += float(strip.mean())
    dc /= float(len(inds))

    exc = np.zeros(n_windows, dtype=float)
    ctrl = np.zeros(n_windows, dtype=float)
    for k in range(n_windows):
        med = np.median(np.stack(acc[k]), axis=0)
        sw = med.shape[1]
        scx = sw // 2
        sfx = np.arange(sw) - scx
        valid = np.abs(sfx) > max(1, sw // 16)
        exc[k] = spectral_excess(med, q, valid)
        ctrl[k] = spectral_excess(med, q + 7, valid)

    snr = exc / (ctrl + 1e-12)
    centers = [(k + 0.5) * ww for k in range(n_windows)]
    half = n_windows // 4
    edge = float(np.mean(np.concatenate([exc[:half], exc[-half:]])))
    middle = float(np.mean(exc[half:-half])) if n_windows > 2 * half else float("nan")
    return {
        "n_windows": n_windows,
        "window_px": ww,
        "centers_px": centers,
        "excess": exc.tolist(),
        "control": ctrl.tolist(),
        "snr": snr.tolist(),
        "dc": dc.tolist(),
        "edge_mean": edge,
        "middle_mean": middle,
        "edge_over_middle": float(edge / (middle + 1e-12)),
        "max_over_min": float(exc.max() / (exc.min() + 1e-12)),
        "argmax_px": float(centers[int(np.argmax(exc))]),
    }


def tile_ridge_census(
    tf: tifffile.TiffFile,
    inds: np.ndarray,
    *,
    tiles: int = 4,
) -> dict[str, Any]:
    """Independent dominant-ridge detection per FOV tile.

    Makes no assumption about a global ``q``: each tile gets its own median
    spectrum, so a region with a different fringe character shows up as a
    different dominant vertical frequency, rescaled to cycles per full frame.
    """
    n, h, w = tf.series[0].shape
    ty, tx = h // tiles, w // tiles
    scale = h / ty

    acc: dict[tuple[int, int], list[np.ndarray]] = {}
    for i in inds:
        frame = np.asarray(tf.pages[int(i)].asarray(), dtype=float)
        for r in range(tiles):
            for c in range(tiles):
                sub = frame[r * ty : (r + 1) * ty, c * tx : (c + 1) * tx]
                acc.setdefault((r, c), []).append(fft_log_amp(sub))

    dom_q = np.zeros((tiles, tiles), dtype=float)
    dom_z = np.zeros((tiles, tiles), dtype=float)
    for (r, c), specs in acc.items():
        med = np.median(np.stack(specs), axis=0)
        th, tw = med.shape
        tcy, tcx = th // 2, tw // 2
        tfx = np.arange(tw) - tcx
        valid = (np.abs(tfx) > 2) & (np.abs(tfx) < tcx - 3)
        prof = np.percentile(med[:, valid], 95, axis=1)
        z, _ = robust_local_z(prof, radius=4, exclude=1)
        band = z[tcy + 3 : th - 3]
        j = int(np.argmax(band))
        dom_q[r, c] = float((j + 3) * scale)
        dom_z[r, c] = float(band[j])

    return {
        "tiles": tiles,
        "dominant_q_scaled": dom_q.tolist(),
        "dominant_row_z": dom_z.tolist(),
        "q_unique_count": int(len(np.unique(np.round(dom_q)))),
        "q_spread": float(dom_q.max() - dom_q.min()),
        "z_min": float(dom_z.min()),
        "z_max": float(dom_z.max()),
    }


def temporal_excess_trace(
    tf: tifffile.TiffFile,
    q: float,
    x_sel: np.ndarray,
    *,
    every: int = 4,
) -> dict[str, Any]:
    """Per-frame ridge excess and its off-ridge control over the recording."""
    n = tf.series[0].shape[0]
    frames = list(range(0, n, every))

    exc, ctrl, dcs = [], [], []
    for i in frames:
        frame = np.asarray(tf.pages[int(i)].asarray(), dtype=float)
        amp = np.abs(np.fft.fftshift(np.fft.fft2(frame)))
        exc.append(spectral_excess(amp, q, x_sel))
        ctrl.append(spectral_excess(amp, q + 7, x_sel))
        dcs.append(float(frame.mean()))

    arr = np.asarray(exc)
    ctrl_arr = np.asarray(ctrl)
    return {
        "stride": every,
        "frames": frames,
        "excess": arr.tolist(),
        "control": ctrl_arr.tolist(),
        "dc": dcs,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "cv": float(arr.std() / (arr.mean() + 1e-12)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "control_mean": float(ctrl_arr.mean()),
        "snr_mean": float(arr.mean() / (ctrl_arr.mean() + 1e-12)),
        "frac_below_control": float(np.mean(arr <= ctrl_arr)),
    }


def phase_continuity(
    tf: tifffile.TiffFile,
    ref_bin: tuple[int, int],
    *,
    start: int = 0,
    count: int = 300,
) -> dict[str, Any]:
    """Frame-to-frame phase behaviour at the strongest ridge bin.

    Consecutive frames (stride 1) are required: any coarser sampling aliases the
    phase and cannot distinguish a slow drift from frame-to-frame randomness.
    """
    n = tf.series[0].shape[0]
    ry, rx = ref_bin
    stop = min(n, start + count)
    idx = list(range(start, stop))

    phases, mags = [], []
    for i in idx:
        spec = np.fft.fftshift(np.fft.fft2(np.asarray(tf.pages[i].asarray(), dtype=float)))
        val = spec[ry, rx]
        phases.append(float(np.angle(val)))
        mags.append(float(np.abs(val)))

    ph = np.asarray(phases)
    unwrapped = np.unwrap(ph)
    dphi = np.diff(unwrapped)
    wrapped_steps = (dphi + np.pi) % (2 * np.pi) - np.pi

    # Coherence of the step distribution: 1 = deterministic drift, 0 = random.
    step_coherence = (
        float(np.abs(np.mean(np.exp(1j * wrapped_steps)))) if wrapped_steps.size else 0.0
    )
    lag1 = 0.0
    if len(mags) > 2:
        m = np.asarray(mags)
        m = m - m.mean()
        denom = float(np.dot(m, m))
        if denom > 0:
            lag1 = float(np.dot(m[:-1], m[1:]) / denom)

    return {
        "ref_bin": [int(ry), int(rx)],
        "start": start,
        "count": len(idx),
        "phase": ph.tolist(),
        "magnitude": mags,
        "step_median_rad": float(np.median(wrapped_steps)) if wrapped_steps.size else None,
        "step_abs_median_rad": float(np.median(np.abs(wrapped_steps)))
        if wrapped_steps.size
        else None,
        "step_coherence": step_coherence,
        "magnitude_lag1_autocorr": lag1,
    }


def q_trace_blocks(
    tf: tifffile.TiffFile,
    q0: float,
    *,
    block_size: int = 50,
    samples_per_block: int = 8,
    search: int = 8,
) -> dict[str, Any]:
    """Track the ridge row per block by maximising row-z near ``q0``."""
    n, h, w = tf.series[0].shape
    cy, cx = h // 2, w // 2
    fx = np.arange(w) - cx
    xvalid = (np.abs(fx) > 5) & (np.abs(fx) < cx - 10)

    qs, zs, starts = [], [], []
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        inds = np.linspace(start, stop - 1, min(samples_per_block, stop - start), dtype=int)
        med = np.median(
            np.stack([fft_log_amp(tf.pages[int(i)].asarray()) for i in inds]), axis=0
        )
        row_profile = np.percentile(med[:, xvalid], 95, axis=1)
        row_z, _ = robust_local_z(row_profile, radius=8, exclude=1)
        lo = max(5, int(round(q0)) - search)
        hi = min(cy - 5, int(round(q0)) + search)
        cand = np.arange(lo, hi + 1)
        best = cand[int(np.argmax(row_z[cy + cand]))]
        qs.append(int(best))
        zs.append(float(row_z[cy + best]))
        starts.append(int(start))

    return {
        "block_size": block_size,
        "block_starts": starts,
        "q": qs,
        "row_z": zs,
        "q_median": float(np.median(qs)),
        "q_min": int(min(qs)),
        "q_max": int(max(qs)),
        "q_span": int(max(qs) - min(qs)),
        "row_z_median": float(np.median(zs)),
    }


def fx_profile(medspec: np.ndarray, q: float) -> dict[str, Any]:
    """Horizontal structure of the ridge: is it a segment or a full row?"""
    h, w = medspec.shape
    cx = w // 2
    fx = np.arange(w) - cx
    zx = ridge_z_at_row(medspec, int(round(q)))
    xvalid = (np.abs(fx) > 5) & (np.abs(fx) < cx - 10)
    support = {}
    for thresh in (2.5, 3.5, 5.0):
        sel = (zx > thresh) & xvalid
        support[str(thresh)] = {
            "n_bins": int(sel.sum()),
            "frac_of_valid": float(sel.sum() / max(1, xvalid.sum())),
            "abs_fx_min": int(np.abs(fx[sel]).min()) if sel.any() else None,
            "abs_fx_max": int(np.abs(fx[sel]).max()) if sel.any() else None,
        }
    return {
        "q": float(q),
        "zx_max": float(zx[xvalid].max()),
        "support_by_threshold": support,
    }


@dataclass
class ChannelResult:
    label: str
    channel: str
    tif: Path
    shape: tuple[int, int, int]
    dtype: str
    q_used: float | None = None
    intensity: dict[str, Any] = field(default_factory=dict)
    families: dict[str, Any] = field(default_factory=dict)
    family_snr: list[dict] = field(default_factory=list)
    fx: dict[str, Any] = field(default_factory=dict)
    masks: dict[str, Any] = field(default_factory=dict)
    fov: dict[str, Any] = field(default_factory=dict)
    x_prof: dict[str, Any] = field(default_factory=dict)
    tile_census: dict[str, Any] = field(default_factory=dict)
    temporal: dict[str, Any] = field(default_factory=dict)
    phase: dict[str, Any] = field(default_factory=dict)
    q_track: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "channel": self.channel,
            "tif": str(self.tif),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "q_used": self.q_used,
            "intensity": self.intensity,
            "families": self.families,
            "family_snr": self.family_snr,
            "fx": self.fx,
            "masks": self.masks,
            "fov": self.fov,
            "x_profile": self.x_prof,
            "tile_census": self.tile_census,
            "temporal": self.temporal,
            "phase": self.phase,
            "q_track": self.q_track,
        }


def analyze_channel(
    tif_path: Path,
    label: str,
    channel: str,
    *,
    sample_n: int = 160,
    tiles: int = 4,
    temporal_every: int = 4,
    phase_count: int = 300,
    do_q_track: bool = True,
) -> ChannelResult:
    tif_path = Path(tif_path)
    with tifffile.TiffFile(tif_path) as tf:
        n, h, w = tf.series[0].shape
        res = ChannelResult(
            label=label,
            channel=channel,
            tif=tif_path,
            shape=(n, h, w),
            dtype=str(tf.pages[0].dtype),
        )

        inds = sample_indices(n, sample_n)
        stat_frames = np.stack(
            [tf.pages[int(i)].asarray() for i in inds[:: max(1, len(inds) // 24)]]
        )
        res.intensity = {
            "mean": float(stat_frames.mean()),
            "std": float(stat_frames.std()),
            "p01": float(np.percentile(stat_frames, 1)),
            "p50": float(np.percentile(stat_frames, 50)),
            "p99": float(np.percentile(stat_frames, 99)),
            "min": float(stat_frames.min()),
            "max": float(stat_frames.max()),
            "frames_used": int(stat_frames.shape[0]),
        }

        medspec = median_log_spectrum(tf, inds)
        med_amp = median_linear_amp(tf, inds)
        res.families = family_report(medspec)

        fams_json = (
            res.families["production_families"] or res.families["relaxed_families"]
        )
        candidates = fams_json or [
            {"q": float(res.families["top_row_peaks"][0]["q"]), "hi": None}
        ]

        # Independent evidence check per candidate, including the near-DC rows
        # the production detector is deliberately suspicious of.
        seen: set[float] = set()
        for cand in candidates + [
            {"q": float(p["q"]), "hi": None} for p in res.families["top_row_peaks"][:4]
        ]:
            q = float(cand["q"])
            if q in seen:
                continue
            seen.add(q)
            x_sel = ridge_support(medspec, q, 3.5)
            if not x_sel.any():
                x_sel = ridge_support(medspec, q, 2.0)
            snr = row_excess_snr(med_amp, q, x_sel)
            res.family_snr.append(
                {
                    "q": q,
                    "hi": cand.get("hi"),
                    "support_bins": int(x_sel.sum()),
                    "period_px": float(h) / q if q else None,
                    **snr,
                }
            )

        # Strongest evidence wins, so a near-DC candidate cannot claim the
        # analysis just because the detector listed it first.
        best = max(res.family_snr, key=lambda s: s["snr"]) if res.family_snr else None
        q = float(best["q"]) if best else float(candidates[0]["q"])
        res.q_used = q
        res.fx = fx_profile(medspec, q)

        masks = build_masks(medspec, candidates)
        if masks["ridge_bins"] == 0:
            masks = build_masks(medspec, candidates, x_z_thresh=2.0)
        res.masks = {
            "ridge_bins": masks["ridge_bins"],
            "control_bins": masks["control_bins"],
            "ref_bin": list(masks["ref_bin"]),
        }

        x_sel = ridge_support(medspec, q, 3.5)
        if not x_sel.any():
            x_sel = ridge_support(medspec, q, 2.0)

        fov_inds = sample_indices(n, min(sample_n, 48))
        res.fov = fov_excess_map(tf, medspec, q, fov_inds, tiles=tiles)
        res.x_prof = x_profile(tf, q, fov_inds)
        res.tile_census = tile_ridge_census(tf, fov_inds, tiles=tiles)
        res.temporal = temporal_excess_trace(tf, q, x_sel, every=temporal_every)
        res.phase = phase_continuity(
            tf, masks["ref_bin"], start=n // 3, count=phase_count
        )
        if do_q_track:
            res.q_track = q_trace_blocks(tf, q)

    return res
