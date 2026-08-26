"""Independently confirm near-DC ridge candidates (DARKCURRENT.md §9, item 1).

The §3.1 claim that ChanA carries a real family at ``q≈5-7`` was measured with an
estimator that borrows its background from rows ±5-9 away. For ``q=6`` those rows
are ``cy-3 … cy+1`` — they *include the DC row itself*. The defect reaches the
``fx`` support too: ``ridge_z_at_row`` takes its background from offsets ±4-10, so
for a near-DC row the support ``|fx|≈11-41`` is DC-contaminated as well. Neither
the amplitude nor its horizontal extent was established independently.

Near DC the hard part is not which background rows to use but the **DC skirt**:
smooth large-scale structure plus window leakage puts a steep pedestal around DC,
and six bins away that pedestal is still large enough that its *curvature* can
imitate a ridge. Four ideas do the work here, and each exists because a simpler
version of this analysis failed a control:

1. **Skirt suppression.** Frames are high-passed along y (per-column boxcar
   removal) and apodised with a separable Hann window before the FFT, removing
   the offset and ``q≲1`` structure that feeds the pedestal while leaving ``q≥5``
   intact. The filter's gain at each tested ``q`` is calibrated by pushing a
   synthetic sinusoid of known amplitude through the identical pipeline, so a
   reported excess is gain-corrected rather than merely post-filter.

2. **In-row background.** A row is compared against a wide median filter along
   ``fx`` within that same row, so no other row — least of all a DC-adjacent one
   — is consulted, and the ``fx`` support becomes independently measurable.

3. **Split-half support and measurement.** The support is chosen on one half of
   the sampled frames and the excess measured on the other. Without this, any row
   gets credit for its own best-looking bins: on these stacks a noise row scored
   3-15x the noise floor instead of the 0.4 that pure noise should give, which
   made the null useless and buried the real families in it.

4. **Temporal character.** For a run of consecutive frames the complex FFT
   coefficient at each support bin is tracked and its two-sided temporal power
   spectrum averaged over bins. This needs no spatial background model and is
   what separates a fringe that moves from structure that is merely fixed:
   static shading (and anything that is really DC leakage) peaks at ``f=0``,
   whereas a drifting fringe peaks at a non-zero frequency.

Thresholds are not constants: they are calibrated per channel against an
empirical null built from every row not under test, with robust rejection of the
rows that carry structure. The fraction of those empty rows that the battery
nevertheless calls a fringe is reported as ``false_positive_rate`` and is what
licenses the verdicts.

One caveat is built into the taxonomy rather than hidden. §3.3 found the fringe
phase moves 1.1-1.8 rad between frames with only partial coherence, so its
temporal power is *broad*, and demanding a sharp temporal peak would reject real
families. A candidate with solid spatial evidence but no resolvable temporal peak
is therefore reported as ``structure_uncharacterized`` — real, character unknown
— rather than being discarded.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy.ndimage import median_filter, uniform_filter1d

_REPO = Path(__file__).resolve().parents[1]
_GPT = _REPO / "reference" / "gpt"
if str(_GPT) not in sys.path:
    sys.path.insert(0, str(_GPT))

from pmt_fringe_raw_adaptive import detect_families, robust_local_z  # noqa: E402

LEGACY_OFFSETS = list(range(-9, -4)) + list(range(5, 10))
DC_GUARD = 12
FX_MEDFILT = 81
HP_KERNEL = 161
Y_RADIUS = 2
N_BG_ROWS = 10
MAX_PROBE_BINS = 64
EDGE_GUARD = 20

# Absolute floors. The operative thresholds come from the per-channel null.
MIN_AMP_SNR = 3.0
MIN_INROW_Z = 3.5
MIN_SUPPORT_BINS = 4
MAX_SUPPORT_FRAC = 0.5
MIN_GAIN = 0.20
MIN_PROMINENCE = 5.0
STATIC_BINS = 3
NULL_SIGMA = 5.0


def sample_indices(n: int, sample_n: int) -> np.ndarray:
    if sample_n >= n:
        return np.arange(n)
    return np.linspace(0, n - 1, sample_n, dtype=int)


def robust_core(values: np.ndarray, *, sigma: float = NULL_SIGMA) -> dict[str, Any]:
    """Split a null sample into its bulk and the rows that carry structure.

    Both statistics used here are heavy-tailed, so ``median + k*MAD`` overshoots
    as a threshold. The largest value that is *not* itself an outlier is a
    stabler and more legible bound: "beat every row that does not already look
    like it contains something".
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0, "median": 0.0, "sigma": 0.0, "core_max": 0.0, "max": 0.0}
    med = float(np.median(arr))
    mad_sigma = float(1.4826 * np.median(np.abs(arr - med))) or 1e-12
    outlier = arr > med + sigma * mad_sigma
    core = arr[~outlier]
    return {
        "n": int(arr.size),
        "median": med,
        "sigma": mad_sigma,
        "core_max": float(core.max()) if core.size else med,
        "max": float(arr.max()),
        "n_outliers": int(outlier.sum()),
        "outlier_mask": outlier,
    }


def nyquist_partner(q: float, cy: int) -> float:
    """The mirror row a bidirectional scan puts every family at.

    ``detect_families`` already reports these as ``hi`` partners (DARKCURRENT
    §3.5): a family at ``q`` also appears at ``cy - q``. They are one family, so
    a control must never be drawn from a candidate's partner — at ``cy=256`` a
    "control" at ``q=250`` is the ``q=6`` candidate seen from the other side.
    """
    return float(cy) - float(q)


def canonical_q(q: float, cy: int) -> float:
    """Representative of the ``{q, cy-q}`` pair, chosen away from the Nyquist edge."""
    return float(min(float(q), nyquist_partner(q, cy)))


def hann2d(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    return np.hanning(h), np.hanning(w)


def preprocess(
    frame: np.ndarray,
    wy: np.ndarray,
    wx: np.ndarray,
    *,
    hp_kernel: int = HP_KERNEL,
) -> np.ndarray:
    """High-pass along y and apodise, so the DC skirt cannot masquerade as a ridge."""
    x = np.asarray(frame, dtype=np.float64)
    x = x - x.mean()
    x = x - uniform_filter1d(x, size=int(hp_kernel) | 1, axis=0, mode="reflect")
    return x * wy[:, None] * wx[None, :]


def _amp(frame: np.ndarray) -> np.ndarray:
    """Linear FFT amplitude with the spatial median removed, as the cleaner does."""
    x = np.asarray(frame, dtype=np.float64)
    x = x - np.median(x)
    return np.abs(np.fft.fftshift(np.fft.fft2(x)))


def median_spectra(
    tf: tifffile.TiffFile,
    inds: np.ndarray,
    *,
    hp_kernel: int = HP_KERNEL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw median spectrum, plus skirt-suppressed medians of two disjoint halves.

    The halves interleave sampled frames, so both cover the whole recording and
    differ only in noise. One picks the support, the other measures it.
    """
    n, h, w = tf.series[0].shape
    wy, wx = hann2d(h, w)
    raw = np.empty((len(inds), h, w), dtype=np.float32)
    filt = np.empty((len(inds), h, w), dtype=np.float32)
    for k, i in enumerate(inds):
        frame = np.asarray(tf.pages[int(i)].asarray(), dtype=np.float64)
        raw[k] = _amp(frame).astype(np.float32)
        filt[k] = np.abs(
            np.fft.fftshift(np.fft.fft2(preprocess(frame, wy, wx, hp_kernel=hp_kernel)))
        ).astype(np.float32)
    med_raw = np.median(raw, axis=0).astype(np.float64)
    filt_a = np.median(filt[0::2], axis=0).astype(np.float64)
    filt_b = np.median(filt[1::2], axis=0).astype(np.float64)
    del raw, filt
    return med_raw, filt_a, filt_b


def filter_gain(
    h: int, w: int, q: float, f0: int, *, hp_kernel: int = HP_KERNEL
) -> float:
    """Recovered fraction of a unit-amplitude sinusoid at ``(q, f0)``.

    Calibrates the high-pass and Hann window together, so a gain-corrected
    excess is comparable across candidates instead of silently favouring the
    ones inside the passband.
    """
    wy, wx = hann2d(h, w)
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    patt = np.cos(2.0 * np.pi * (q * yy / h + f0 * xx / w))
    spec = np.abs(
        np.fft.fftshift(np.fft.fft2(preprocess(patt, wy, wx, hp_kernel=hp_kernel)))
    )
    cy, cx = h // 2, w // 2
    y, x = cy + int(round(q)), cx + int(f0)
    if not (0 <= y < h and 0 <= x < w):
        return 0.0
    # Hann spreads a tone over a few bins; take the local maximum.
    peak = float(
        spec[max(0, y - 3) : min(h, y + 4), max(0, x - 3) : min(w, x + 4)].max()
    )
    return peak / (0.5 * h * w)


def ridge_rows(
    dy: float, cy: int, h: int, *, y_radius: int = Y_RADIUS, dc_guard: int = 0
) -> list[int]:
    """Rows carrying the ridge, optionally refusing rows too close to DC."""
    rows = []
    for sgn in (-1, 1):
        y0 = cy + sgn * int(round(dy))
        for off in range(-y_radius, y_radius + 1):
            y = y0 + off
            if 0 <= y < h and abs(y - cy) >= dc_guard:
                rows.append(y)
    return sorted(set(rows))


def forbidden_rows(
    candidates: list[float], cy: int, h: int, *, pad: int = Y_RADIUS + 1
) -> set[int]:
    out: set[int] = set()
    for q in candidates:
        for sgn in (-1, 1):
            y0 = cy + sgn * int(round(q))
            for off in range(-pad, pad + 1):
                y = y0 + off
                if 0 <= y < h:
                    out.add(y)
    return out


def background_rows(
    y: int,
    cy: int,
    h: int,
    *,
    policy: str,
    forbidden: set[int],
    dc_guard: int = DC_GUARD,
    n_bg: int = N_BG_ROWS,
) -> list[int]:
    """Background rows under a named policy (``legacy`` ignores DC; ``dc_safe`` does not)."""
    if policy == "legacy":
        return [y + o for o in LEGACY_OFFSETS if 0 <= y + o < h]
    if policy != "dc_safe":
        raise ValueError(f"unknown background policy: {policy}")

    picked: list[int] = []
    for mag in range(5, h):
        for sgn in (1, -1):
            yy = y + sgn * mag
            if not (0 <= yy < h) or abs(yy - cy) < dc_guard or yy in forbidden:
                continue
            picked.append(yy)
            if len(picked) >= n_bg:
                return picked
    return picked


def _excess_at_row(amp: np.ndarray, y: int, x_sel: np.ndarray, bg: list[int]) -> float:
    if not bg or not x_sel.any():
        return 0.0
    base = np.median(amp[np.asarray(bg), :][:, x_sel], axis=0)
    return float(np.mean(np.maximum(amp[y, x_sel] - base, 0.0)))


def background_policy_probe(
    amp: np.ndarray,
    dy: float,
    x_sel: np.ndarray,
    *,
    forbidden: set[int],
    dc_guard: int = DC_GUARD,
) -> dict[str, Any]:
    """Document what the §3.1 estimator used as background, and what changes.

    The interesting output is not the amplitude but ``nearest_bg_dy``, which
    shows how close to DC the legacy policy reached.
    """
    h, _ = amp.shape
    cy = h // 2
    out: dict[str, Any] = {}
    for policy in ("legacy", "dc_safe"):
        vals, used = [], []
        for y in ridge_rows(dy, cy, h):
            bg = background_rows(
                y, cy, h, policy=policy, forbidden=forbidden, dc_guard=dc_guard
            )
            used.extend(bg)
            vals.append(_excess_at_row(amp, y, x_sel, bg))
        arr = np.asarray(sorted(set(used))) if used else np.zeros(0, dtype=int)
        out[policy] = {
            "excess": float(np.mean(vals)) if vals else 0.0,
            "n_background_rows": int(arr.size),
            "nearest_bg_dy": int(np.abs(arr - cy).min()) if arr.size else None,
            "uses_dc_row": bool((arr == cy).any()) if arr.size else False,
            "touches_dc_guard": bool((np.abs(arr - cy) < dc_guard).any())
            if arr.size
            else False,
        }
    return out


def _row_residual(row: np.ndarray, medfilt: int) -> np.ndarray:
    """Row minus its own wide median filter along fx; no other row is used."""
    return row - median_filter(row, size=int(medfilt) | 1, mode="nearest")


def in_row_support(
    amp: np.ndarray,
    dy: float,
    xvalid: np.ndarray,
    *,
    medfilt: int = FX_MEDFILT,
    z_thresh: float = MIN_INROW_Z,
    dc_guard: int = 0,
) -> dict[str, Any]:
    """fx support and its significance, derived from the ridge rows themselves."""
    h, w = amp.shape
    cy, cx = h // 2, w // 2
    fx = np.arange(w) - cx

    zs = []
    for sgn in (-1, 1):
        y = cy + sgn * int(round(dy))
        if not (0 <= y < h) or abs(y - cy) < dc_guard:
            continue
        resid = _row_residual(amp[y, :], medfilt)
        med = float(np.median(resid[xvalid]))
        mad = float(np.median(np.abs(resid[xvalid] - med))) + 1e-12
        zs.append((resid - med) / (1.4826 * mad))
    if not zs:
        return {"support": np.zeros(w, dtype=bool), "z_max": 0.0, "support_bins": 0}

    z = np.mean(np.stack(zs), axis=0)
    support = (z > z_thresh) & xvalid
    return {
        "support": support,
        "z_max": float(z[xvalid].max()),
        "support_bins": int(support.sum()),
        "support_frac_of_valid": float(support.sum() / max(1, int(xvalid.sum()))),
        "abs_fx_min": int(np.abs(fx[support]).min()) if support.any() else None,
        "abs_fx_max": int(np.abs(fx[support]).max()) if support.any() else None,
    }


def in_row_excess(
    amp: np.ndarray,
    dy: float,
    x_sel: np.ndarray,
    *,
    medfilt: int = FX_MEDFILT,
    y_radius: int = Y_RADIUS,
    dc_guard: int = 0,
) -> float:
    h, _ = amp.shape
    cy = h // 2
    rows = ridge_rows(dy, cy, h, y_radius=y_radius, dc_guard=dc_guard)
    if not x_sel.any() or not rows:
        return 0.0
    vals = [
        float(np.mean(np.maximum(_row_residual(amp[y, :], medfilt)[x_sel], 0.0)))
        for y in rows
    ]
    return float(np.mean(vals))


def residual_noise_scale(
    med_filt: np.ndarray,
    xvalid: np.ndarray,
    *,
    forbidden: set[int],
    medfilt: int = FX_MEDFILT,
    dc_guard: int = DC_GUARD,
    n_rows: int = 24,
) -> float:
    """Robust per-bin scale of the in-row residual on rows carrying no candidate.

    Used instead of a single control row's excess, which can be exactly zero
    (when the candidate's support bins sit below their local median at that row)
    and would then make an excess/control ratio meaningless.
    """
    h, _ = med_filt.shape
    cy = h // 2
    rows = [y for y in range(h) if abs(y - cy) >= dc_guard and y not in forbidden]
    if not rows:
        return 0.0
    picked = np.asarray(rows)[
        np.linspace(0, len(rows) - 1, min(n_rows, len(rows)), dtype=int)
    ]
    vals = np.concatenate(
        [_row_residual(med_filt[int(y), :], medfilt)[xvalid] for y in picked]
    )
    med = float(np.median(vals))
    return float(1.4826 * np.median(np.abs(vals - med)))


def row_snr(
    filt_a: np.ndarray,
    filt_b: np.ndarray,
    dy: float,
    xvalid: np.ndarray,
    *,
    noise_scale: float,
    medfilt: int = FX_MEDFILT,
) -> tuple[float, dict[str, Any]]:
    """Split-half in-row SNR for one row: support from ``filt_a``, excess from ``filt_b``."""
    sup = in_row_support(filt_a, dy, xvalid, medfilt=medfilt, z_thresh=MIN_INROW_Z)
    x_sel = sup["support"] if sup["support_bins"] else xvalid
    exc = in_row_excess(filt_b, dy, x_sel, medfilt=medfilt)
    return (float(exc / noise_scale) if noise_scale > 0 else 0.0), sup


def amplitude_test(
    filt_a: np.ndarray,
    filt_b: np.ndarray,
    dy: float,
    xvalid: np.ndarray,
    *,
    forbidden: set[int],
    noise_scale: float,
    medfilt: int = FX_MEDFILT,
    hp_kernel: int = HP_KERNEL,
    dc_guard: int = DC_GUARD,
) -> dict[str, Any]:
    """Gain-corrected, skirt-suppressed excess with an in-row fx background."""
    h, w = filt_b.shape
    cy = h // 2

    sup = in_row_support(filt_a, dy, xvalid, medfilt=medfilt, z_thresh=MIN_INROW_Z)
    x_sel = sup["support"] if sup["support_bins"] else xvalid
    excess = in_row_excess(filt_b, dy, x_sel, medfilt=medfilt)

    # Secondary diagnostic: the same statistic on one off-ridge row clear of DC.
    ctrl = []
    for sgn in (-1, 1):
        for step in (17, 23, 29, 37):
            y = cy + sgn * (int(round(dy)) + step)
            if not (0 <= y < h) or abs(y - cy) < dc_guard or y in forbidden:
                continue
            ctrl.append(
                float(np.mean(np.maximum(_row_residual(filt_b[y, :], medfilt)[x_sel], 0.0)))
            )
            break
    control = float(np.mean(ctrl)) if ctrl else 0.0

    fx = np.arange(w) - (w // 2)
    f0 = 0
    if sup["support_bins"]:
        pos = fx[sup["support"] & (fx > 0)]
        f0 = int(pos[len(pos) // 2]) if pos.size else int(np.abs(fx[sup["support"]]).max())
    gain = filter_gain(h, w, dy, f0 or 20, hp_kernel=hp_kernel)

    return {
        "excess": excess,
        "control_row_excess": control,
        "noise_scale": noise_scale,
        # Pure noise gives mean(max(resid,0)) ~ 0.4 * scale, so a real segment
        # sits far above 1 and the ratio can never blow up.
        "snr": float(excess / noise_scale) if noise_scale > 0 else 0.0,
        "filter_gain": gain,
        "gain_calib_fx": int(f0 or 20),
        "excess_gain_corrected": float(excess / gain) if gain > 0 else None,
        "support": {k: v for k, v in sup.items() if k != "support"},
        "_support_mask": sup["support"] if sup["support_bins"] else xvalid,
    }


def amplitude_null(
    filt_a: np.ndarray,
    filt_b: np.ndarray,
    xvalid: np.ndarray,
    *,
    noise_scale: float,
    candidate_qs: list[float],
    medfilt: int = FX_MEDFILT,
    edge_guard: int = EDGE_GUARD,
    outlier_sigma: float = NULL_SIGMA,
) -> dict[str, Any]:
    """Distribution of the amplitude statistic over every row not under test.

    Deliberately does *not* use the row-z detector to decide which rows are
    empty: that detector keys on a 95th percentile across ``fx`` and so misses
    ridges narrow enough to matter (a 6-bin segment out of ~480 valid bins does
    not move it). Robust statistics find the structure-carrying rows instead, so
    the null is defined by the same measurement the candidates are judged by.
    """
    h, _ = filt_b.shape
    cy = h // 2
    blocked = set()
    for q in candidate_qs:
        for qq in (canonical_q(q, cy), nyquist_partner(canonical_q(q, cy), cy), q):
            blocked.update(int(round(qq)) + d for d in range(-5, 6))

    qs = [q for q in range(edge_guard, cy - edge_guard) if q not in blocked]
    snrs = []
    for q in qs:
        s, _ = row_snr(
            filt_a, filt_b, float(q), xvalid, noise_scale=noise_scale, medfilt=medfilt
        )
        snrs.append(s)

    arr = np.asarray(snrs, dtype=float)
    if arr.size == 0:
        return {"n_rows": 0}
    core = robust_core(arr, sigma=outlier_sigma)
    is_outlier = core["outlier_mask"]

    return {
        "n_rows": int(arr.size),
        "snr_median": core["median"],
        "snr_sigma": core["sigma"],
        "snr_core_max": core["core_max"],
        "snr_max": core["max"],
        "n_structured_rows": int(is_outlier.sum()),
        "structured_q": [int(qs[i]) for i in np.flatnonzero(is_outlier)][:12],
        "threshold": float(max(MIN_AMP_SNR, core["core_max"])),
        "clean_q": [int(qs[i]) for i in np.flatnonzero(~is_outlier)],
    }


def probe_bins(
    support: np.ndarray, dy: float, h: int, w: int, *, max_bins: int = MAX_PROBE_BINS
) -> list[tuple[int, int]]:
    """Upper-half-plane bins for a candidate row.

    Only one half plane is used: for a real image ``(cy-q, cx-f)`` is the complex
    conjugate of ``(cy+q, cx+f)`` and would add no independent information.
    """
    cy = h // 2
    y = cy + int(round(dy))
    if not (0 <= y < h):
        return []
    idx = np.flatnonzero(support)
    if idx.size == 0:
        return []
    if idx.size > max_bins:
        idx = idx[np.linspace(0, idx.size - 1, max_bins, dtype=int)]
    return [(y, int(x)) for x in idx]


def temporal_bin_series(
    tf: tifffile.TiffFile,
    bins: list[tuple[int, int]],
    *,
    start: int,
    count: int,
) -> np.ndarray:
    """Complex FFT coefficient at each probe bin over consecutive frames.

    Consecutive frames (stride 1) are required: coarser sampling aliases the
    phase advance that the whole test rests on.
    """
    n, h, w = tf.series[0].shape
    stop = min(n, start + count)
    rows = np.array([b[0] for b in bins], dtype=int)
    cols = np.array([b[1] for b in bins], dtype=int)
    out = np.empty((stop - start, len(bins)), dtype=np.complex128)
    for k, i in enumerate(range(start, stop)):
        frame = np.asarray(tf.pages[int(i)].asarray(), dtype=np.float64)
        spec = np.fft.fftshift(np.fft.fft2(frame - np.median(frame)))
        out[k] = spec[rows, cols]
    return out


def temporal_character(
    series: np.ndarray,
    *,
    frame_rate: float | None = None,
    static_bins: int = STATIC_BINS,
) -> dict[str, Any]:
    """Two-sided temporal power spectrum of the bin series, averaged over bins.

    A real image's coefficient series is complex, so the spectrum is two-sided
    and a drifting fringe appears at a signed non-zero frequency. Static
    structure sits at ``f=0``; white noise is flat.
    """
    if series.size == 0:
        return {"n_frames": 0, "n_bins": 0, "prominence": 0.0}

    t, nb = series.shape
    # The f=0 component is deliberately kept: it is what identifies static
    # structure, which is one of the outcomes this test has to separate.
    spec = np.fft.fftshift(np.fft.fft(series * np.hanning(t)[:, None], axis=0), axes=0)
    power = (np.abs(spec) ** 2).mean(axis=1)
    freqs = np.fft.fftshift(np.fft.fftfreq(t))

    med = float(np.median(power))
    j = int(np.argmax(power))
    f_peak = float(freqs[j])

    static = np.abs(freqs) <= (static_bins / t)
    nonstatic = ~static
    jn = (
        int(np.flatnonzero(nonstatic)[np.argmax(power[nonstatic])])
        if nonstatic.any()
        else j
    )
    return {
        "n_frames": int(t),
        "n_bins": int(nb),
        "f_peak_cycles_per_frame": f_peak,
        "f_peak_hz": float(f_peak * frame_rate) if frame_rate else None,
        "prominence": float(power[j] / (med + 1e-30)),
        "peak_is_static": bool(abs(f_peak) <= static_bins / t),
        "static_power_fraction": float(power[static].sum() / (power.sum() + 1e-30)),
        "f_nonstatic_peak_cycles_per_frame": float(freqs[jn]),
        "nonstatic_prominence": float(power[jn] / (med + 1e-30)),
        "nonstatic_peak_hz": float(freqs[jn] * frame_rate) if frame_rate else None,
    }


def pick_candidates(
    amp: np.ndarray,
    xvalid: np.ndarray,
    *,
    near_dc_max: int = 12,
    dc_guard: int = DC_GUARD,
    edge_guard: int = EDGE_GUARD,
) -> dict[str, Any]:
    """Near-DC candidates to test, plus a positive control that is a *different* family."""
    h, _ = amp.shape
    cy = h // 2
    logamp = np.log1p(amp)
    row_profile = np.percentile(logamp[:, xvalid], 95, axis=1)
    row_z, _ = robust_local_z(row_profile, radius=8, exclude=1)

    detected: list[float] = []
    for thresholds in (
        dict(row_z_thresh=5.5, pair_z_min=3.5, x_z_thresh=3.5, allow_standalone=False),
        dict(row_z_thresh=4.0, pair_z_min=2.5, x_z_thresh=2.5, allow_standalone=True),
    ):
        fams, _, _ = detect_families(logamp, max_families=4, **thresholds)
        for f in fams:
            q = float(f["q"])
            if all(abs(q - d) > 2 for d in detected):
                detected.append(q)

    band = row_z[cy + 3 : h - 5]
    peaks = [(int(i + 3), float(band[i])) for i in np.argsort(band)[::-1][:12]]

    near_dc = [q for q in detected if q <= near_dc_max]
    for q, _z in peaks:
        if q <= near_dc_max and all(abs(q - d) > 2 for d in near_dc):
            near_dc.append(float(q))
    near_dc = sorted(near_dc)[:3]

    # Far from DC in *both* representations, so it cannot be a near-DC candidate
    # wearing its Nyquist mirror.
    positive = None
    for q, _z in peaks:
        canon = canonical_q(q, cy)
        if canon < edge_guard:
            continue
        if any(abs(canon - canonical_q(c, cy)) < 3 for c in near_dc):
            continue
        positive = canon
        break

    return {
        "near_dc": near_dc,
        "near_dc_partners": [nyquist_partner(q, cy) for q in near_dc],
        "positive_control": positive,
        "dc_guard": int(dc_guard),
        "edge_guard": int(edge_guard),
        "row_z_peaks": [
            {"q": q, "row_z": z, "canonical_q": canonical_q(q, cy)} for q, z in peaks[:8]
        ],
    }


@dataclass
class CandidateResult:
    q: float
    role: str
    period_px: float
    background_probe: dict[str, Any] = field(default_factory=dict)
    amplitude: dict[str, Any] = field(default_factory=dict)
    temporal: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)
    verdict: str = "not_tested"

    def to_json(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "role": self.role,
            "period_px": self.period_px,
            "background_probe": self.background_probe,
            "amplitude": {
                k: v for k, v in self.amplitude.items() if not k.startswith("_")
            },
            "temporal": self.temporal,
            "checks": self.checks,
            "verdict": self.verdict,
        }


def _verdict(checks: dict[str, bool]) -> str:
    """Separate "is there structure here" from "does that structure move".

    ``structure_uncharacterized`` exists because §3.3 measured a phase that
    wanders, which spreads temporal power: requiring a sharp temporal peak would
    throw away families whose spatial evidence is overwhelming.
    """
    real = checks.get("localized_in_fx", False) and checks.get(
        "amplitude_above_null", False
    )
    prominent = checks.get("temporal_peak_prominent", False)
    moving = checks.get("temporal_peak_moving", False)

    # A peak pinned at f=0 identifies static structure on its own, whether or not
    # the spatial checks pass, so it is tested first.
    if prominent and not moving:
        return "static_structure"
    if real and prominent:
        return "fringe_confirmed"
    if real:
        return "structure_uncharacterized"
    if prominent:
        return "moving_not_localized"
    return "noise_like"


def confirm_channel(
    tif_path: Path,
    label: str,
    channel: str,
    *,
    sample_n: int = 160,
    temporal_frames: int = 512,
    dc_guard: int = DC_GUARD,
    medfilt: int = FX_MEDFILT,
    hp_kernel: int = HP_KERNEL,
    frame_rate: float | None = None,
    forced_qs: list[float] | None = None,
    n_null_rows: int = 24,
) -> dict[str, Any]:
    """Run the confirmation battery on one channel stack."""
    tif_path = Path(tif_path)
    with tifffile.TiffFile(tif_path) as tf:
        n, h, w = tf.series[0].shape
        cy, cx = h // 2, w // 2
        fx = np.arange(w) - cx
        xvalid = (np.abs(fx) > 5) & (np.abs(fx) < cx - 10)

        inds = sample_indices(n, sample_n)
        med_raw, filt_a, filt_b = median_spectra(tf, inds, hp_kernel=hp_kernel)

        picked = pick_candidates(med_raw, xvalid, dc_guard=dc_guard)
        roles: list[tuple[float, str]] = []
        if forced_qs:
            roles = [(float(q), "forced") for q in forced_qs]
        else:
            roles += [(q, "near_dc_candidate") for q in picked["near_dc"]]
            if picked["positive_control"] is not None:
                roles.append((picked["positive_control"], "positive_control"))
        roles.append((0.0, "static_reference"))

        candidate_qs = [q for q, _ in roles if q > 0]
        forbidden = forbidden_rows(
            candidate_qs + [nyquist_partner(q, cy) for q in candidate_qs], cy, h
        )
        noise_scale = residual_noise_scale(
            filt_b, xvalid, forbidden=forbidden, medfilt=medfilt, dc_guard=dc_guard
        )

        # Calibrate on rows that are empty by the same measurement, then draw the
        # temporal null from those rows, so both thresholds come from this
        # channel's own behaviour rather than from a guessed constant.
        amp_null = amplitude_null(
            filt_a,
            filt_b,
            xvalid,
            noise_scale=noise_scale,
            candidate_qs=candidate_qs,
            medfilt=medfilt,
        )
        clean = amp_null.get("clean_q") or []
        if len(clean) > n_null_rows:
            clean = [
                clean[i] for i in np.linspace(0, len(clean) - 1, n_null_rows, dtype=int)
            ]
        roles += [(float(q), "null_row") for q in clean]

        start = max(0, n // 3)

        # Spatial phase first: the temporal pass needs every row's support before
        # it runs, so all probe bins are collected in a single sweep over the
        # frames instead of re-reading the stack per candidate.
        results: list[CandidateResult] = []
        for q, role in roles:
            res = CandidateResult(
                q=float(q),
                role=role,
                period_px=float(h) / float(q) if q else float("inf"),
            )
            if role == "static_reference":
                # fy=0 row: structure constant in y, so a genuine static
                # reference. Support borrowed from the strongest candidate so the
                # comparison is on the same fx bins.
                donor = next(
                    (r for r in results if r.role == "near_dc_candidate"), None
                ) or next((r for r in results if r.role == "positive_control"), None)
                res.amplitude = {
                    "note": "fy=0 reference; amplitude not meaningful",
                    "_support_mask": donor.amplitude["_support_mask"]
                    if donor is not None
                    else xvalid,
                    "support": {},
                }
            else:
                if role != "null_row":
                    res.background_probe = background_policy_probe(
                        med_raw, q, xvalid, forbidden=forbidden, dc_guard=dc_guard
                    )
                res.amplitude = amplitude_test(
                    filt_a,
                    filt_b,
                    q,
                    xvalid,
                    forbidden=forbidden,
                    noise_scale=noise_scale,
                    medfilt=medfilt,
                    hp_kernel=hp_kernel,
                    dc_guard=dc_guard if q > 0 else 0,
                )
            results.append(res)

        bins_all: list[tuple[int, int]] = []
        spans: list[tuple[int, int]] = []
        for res in results:
            bins = probe_bins(res.amplitude["_support_mask"], res.q, h, w)
            spans.append((len(bins_all), len(bins_all) + len(bins)))
            bins_all.extend(bins)

        series_all = (
            temporal_bin_series(tf, bins_all, start=start, count=temporal_frames)
            if bins_all
            else np.zeros((0, 0), dtype=np.complex128)
        )
        for res, (lo, hi) in zip(results, spans):
            res.temporal = (
                temporal_character(series_all[:, lo:hi], frame_rate=frame_rate)
                if hi > lo
                else {"n_frames": 0, "n_bins": 0, "prominence": 0.0}
            )

        null_prom = np.asarray(
            [r.temporal.get("prominence", 0.0) for r in results if r.role == "null_row"],
            dtype=float,
        )
        prom_core = robust_core(null_prom)
        prom_threshold = float(max(MIN_PROMINENCE, prom_core["core_max"]))
        amp_threshold = float(amp_null.get("threshold", MIN_AMP_SNR))

        for res in results:
            sup = res.amplitude.get("support", {}) or {}
            gain = res.amplitude.get("filter_gain")
            res.checks = {
                # A ridge is a *segment*: one or two isolated bins scraping past
                # the z threshold is not the thing being claimed.
                "localized_in_fx": bool(
                    sup.get("support_bins", 0) >= MIN_SUPPORT_BINS
                    and sup.get("z_max", 0.0) >= MIN_INROW_Z
                    and sup.get("support_frac_of_valid", 1.0) <= MAX_SUPPORT_FRAC
                ),
                "amplitude_above_null": bool(
                    res.amplitude.get("snr", 0.0) >= amp_threshold
                ),
                "inside_filter_passband": bool(gain is not None and gain >= MIN_GAIN),
                "temporal_peak_prominent": bool(
                    res.temporal.get("prominence", 0.0) >= prom_threshold
                ),
                "temporal_peak_moving": bool(
                    not res.temporal.get("peak_is_static", True)
                ),
            }
            res.verdict = _verdict(res.checks)

        nulls = [r for r in results if r.role == "null_row"]
        null_summary = summarize_null(nulls)
        null_summary.update(
            {
                "prominence_threshold": prom_threshold,
                "prominence_core_max": prom_core["core_max"],
                "prominence_outliers": prom_core.get("n_outliers", 0),
                "amplitude_threshold": amp_threshold,
                "amplitude": {
                    k: v
                    for k, v in amp_null.items()
                    if k not in ("clean_q", "outlier_mask")
                },
            }
        )

    return {
        "label": label,
        "channel": channel,
        "tif": str(tif_path),
        "shape": [int(n), int(h), int(w)],
        "frames_sampled": int(len(inds)),
        "frame_rate": frame_rate,
        "settings": {
            "dc_guard": int(dc_guard),
            "fx_medfilt": int(medfilt),
            "hp_kernel": int(hp_kernel),
            "y_radius": Y_RADIUS,
            "temporal_frames": int(temporal_frames),
            "temporal_start": int(start),
            "residual_noise_scale": noise_scale,
            "thresholds": {
                "min_amp_snr": MIN_AMP_SNR,
                "min_in_row_z": MIN_INROW_Z,
                "min_support_bins": MIN_SUPPORT_BINS,
                "max_support_frac": MAX_SUPPORT_FRAC,
                "min_gain": MIN_GAIN,
                "min_prominence": MIN_PROMINENCE,
                "static_bins": STATIC_BINS,
                "null_sigma": NULL_SIGMA,
            },
        },
        "picked": picked,
        "null_summary": null_summary,
        "candidates": [r.to_json() for r in results if r.role != "null_row"],
        "null_rows": [r.to_json() for r in results if r.role == "null_row"],
    }


def summarize_null(nulls: list[CandidateResult]) -> dict[str, Any]:
    """Spread of the statistics over rows known to carry no family.

    ``false_positive_rate`` is the headline number: the fraction of rows with
    nothing in them that the battery nevertheless calls a confirmed fringe. It is
    what licenses (or refuses) the verdicts on the real candidates.
    """
    if not nulls:
        return {"n_rows": 0}
    snr = np.asarray([r.amplitude.get("snr", 0.0) for r in nulls], dtype=float)
    prom = np.asarray([r.temporal.get("prominence", 0.0) for r in nulls], dtype=float)
    bins = np.asarray(
        [(r.amplitude.get("support") or {}).get("support_bins", 0) for r in nulls],
        dtype=float,
    )
    passed = [r for r in nulls if r.verdict == "fringe_confirmed"]
    return {
        "n_rows": len(nulls),
        "q_values": [r.q for r in nulls],
        "snr_median": float(np.median(snr)),
        "snr_max": float(snr.max()),
        "prominence_median": float(np.median(prom)),
        "prominence_max": float(prom.max()),
        "support_bins_median": float(np.median(bins)),
        "support_bins_max": float(bins.max()),
        "n_false_positive": len(passed),
        "false_positive_q": [r.q for r in passed],
        "false_positive_rate": float(len(passed) / len(nulls)),
        "verdicts": {
            v: sum(1 for r in nulls if r.verdict == v) for v in {r.verdict for r in nulls}
        },
    }
