"""Fringe-rich seeding, recurrent families, and locked tracking for long stacks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, gaussian_filter1d

_REPO = Path(__file__).resolve().parents[1]
_GPT = _REPO / "reference" / "gpt"
if str(_GPT) not in sys.path:
    sys.path.insert(0, str(_GPT))

from pmt_fringe_raw_adaptive import (  # noqa: E402
    contiguous_ranges,
    detect_families,
    fft_log_amp,
    ridge_z_at_row,
    search_q,
)

SAFE_ROW_Z = 5.5
SAFE_PAIR_Z = 3.5
SAFE_X_Z = 3.5
TRACK_SEARCH = 10
FORBIDDEN_Q_RADIUS = 3
Q_CLUSTER_TOL = 3.0
MAX_FAMILIES = 4

LADDER_RUNGS = [
    {
        "name": "safe_paired",
        "row_z": SAFE_ROW_Z,
        "pair_z": SAFE_PAIR_Z,
        "allow_standalone": False,
    },
    {
        "name": "standalone",
        "row_z": SAFE_ROW_Z,
        "pair_z": SAFE_PAIR_Z,
        "allow_standalone": True,
    },
    {
        "name": "lower_z",
        "row_z": 4.0,
        "pair_z": 2.5,
        "allow_standalone": True,
    },
]


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


def sample_mean_frame(tf: tifffile.TiffFile, sample_n: int = 32) -> np.ndarray:
    n, h, w = tf.series[0].shape
    inds = np.linspace(0, n - 1, min(sample_n, n), dtype=int)
    acc = np.zeros((h, w), dtype=np.float64)
    for i in inds:
        acc += np.asarray(tf.pages[int(i)].asarray(), dtype=np.float64)
    return acc / max(len(inds), 1)


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


def library_family_supported(fam: dict, medspec: np.ndarray, x_z_thresh: float = SAFE_X_Z) -> bool:
    """True if a library q still has fx support on this stack (do not copy blindly)."""
    hydrated = hydrate_families([fam], medspec, x_z_thresh=x_z_thresh)
    if not hydrated:
        return False
    weight = hydrated[0].get("x_weight")
    ranges = hydrated[0].get("fx_ranges") or []
    if weight is None:
        return False
    return float(np.max(weight)) > 0.20 and len(ranges) > 0


def cluster_recurrent_families(
    block_hits: list[dict],
    *,
    q_tol: float = Q_CLUSTER_TOL,
    max_families: int = MAX_FAMILIES,
) -> tuple[list[dict], np.ndarray | None, dict]:
    """Merge families whose q matches across blocks; prefer recurrence then score."""
    clusters: list[dict] = []
    for hit in block_hits:
        for fam in hit.get("families") or []:
            q = float(fam["q"])
            found = None
            for c in clusters:
                if abs(c["q"] - q) < q_tol:
                    found = c
                    break
            if found is None:
                clusters.append(
                    {
                        "q": q,
                        "n_blocks": 1,
                        "row_score": float(fam.get("row_score", 0.0)),
                        "fam": fam,
                        "medspec": hit["medspec"],
                        "block_start": hit["start"],
                        "block_stop": hit["stop"],
                    }
                )
            else:
                found["n_blocks"] += 1
                score = float(fam.get("row_score", 0.0))
                if score > found["row_score"]:
                    found["q"] = q
                    found["row_score"] = score
                    found["fam"] = fam
                    found["medspec"] = hit["medspec"]
                    found["block_start"] = hit["start"]
                    found["block_stop"] = hit["stop"]

    if not clusters:
        return [], None, {"mode": "recurrent_blocks", "n_clusters": 0, "chosen": []}

    clusters.sort(key=lambda c: (c["n_blocks"], c["row_score"]), reverse=True)
    chosen = clusters[:max_families]
    best_spec = max(chosen, key=lambda c: c["row_score"])["medspec"]
    families = []
    for c in chosen:
        fam = dict(c["fam"])
        fam["n_blocks"] = c["n_blocks"]
        families.append(fam)
    info = {
        "mode": "recurrent_blocks",
        "n_clusters": len(clusters),
        "block_start": chosen[0]["block_start"],
        "block_stop": chosen[0]["block_stop"],
        "row_score_max": float(max(c["row_score"] for c in chosen)),
        "chosen": [
            {
                "q": float(c["q"]),
                "n_blocks": int(c["n_blocks"]),
                "row_score": float(c["row_score"]),
            }
            for c in chosen
        ],
    }
    return families, best_spec, info


def _block_starts(n: int, block_size: int, max_blocks: int) -> list[int]:
    starts = list(range(0, n, block_size))
    if len(starts) > max_blocks:
        idx = np.linspace(0, len(starts) - 1, max_blocks, dtype=int)
        starts = [starts[i] for i in idx]
    return starts


def collect_block_spectra(
    tf: tifffile.TiffFile,
    *,
    block_size: int = 50,
    samples_per_block: int = 8,
    max_blocks: int = 40,
) -> list[dict]:
    n = tf.series[0].shape[0]
    out = []
    for start in _block_starts(n, block_size, max_blocks):
        stop = min(n, start + block_size)
        inds = np.linspace(start, stop - 1, min(samples_per_block, stop - start), dtype=int)
        specs = [fft_log_amp(tf.pages[int(i)].asarray()) for i in inds]
        out.append(
            {
                "start": int(start),
                "stop": int(stop),
                "medspec": np.median(np.stack(specs), axis=0),
            }
        )
    return out


def _hits_for_rung(block_specs: list[dict], rung: dict) -> list[dict]:
    hits = []
    for block in block_specs:
        families, _, _ = detect_families(
            block["medspec"],
            row_z_thresh=float(rung["row_z"]),
            pair_z_min=float(rung["pair_z"]),
            x_z_thresh=SAFE_X_Z,
            max_families=MAX_FAMILIES,
            allow_standalone=bool(rung["allow_standalone"]),
        )
        if families:
            hits.append({**block, "families": families})
    return hits


def detect_fringe_rich(
    tf: tifffile.TiffFile,
    *,
    block_size: int = 50,
    samples_per_block: int = 8,
    max_blocks: int = 40,
    row_z_thresh: float = SAFE_ROW_Z,
    pair_z_min: float = SAFE_PAIR_Z,
    x_z_thresh: float = SAFE_X_Z,
    library_families: list[dict] | None = None,
) -> tuple[list[dict], np.ndarray, dict]:
    """
    Detect recurrent paired families across fringe-rich blocks.

    Returns (families, medspec_used, info). Empty families if the safe rung finds nothing
    (caller may run the inspect-only ladder).
    """
    del x_z_thresh
    n = tf.series[0].shape[0]
    block_specs = collect_block_spectra(
        tf,
        block_size=block_size,
        samples_per_block=samples_per_block,
        max_blocks=max_blocks,
    )
    safe_rung = {
        "name": "safe_paired",
        "row_z": row_z_thresh,
        "pair_z": pair_z_min,
        "allow_standalone": False,
    }
    hits = _hits_for_rung(block_specs, safe_rung)
    families, medspec, info = cluster_recurrent_families(hits)
    info["n_frames"] = int(n)
    info["n_blocks_scanned"] = len(block_specs)
    info["n_blocks_with_safe_family"] = len(hits)
    if medspec is None:
        medspec = sample_median_spectrum(tf)

    if families and library_families:
        existing_q = [float(f["q"]) for f in families]
        added = []
        for src in library_families:
            q = float(src["q"])
            if any(abs(q - eq) < Q_CLUSTER_TOL for eq in existing_q):
                continue
            if not library_family_supported(src, medspec):
                continue
            extra = hydrate_families([src], medspec)[0]
            extra["from_library"] = True
            families.append(extra)
            existing_q.append(q)
            added.append(q)
            if len(families) >= MAX_FAMILIES:
                break
        info["library_added_q"] = added

    info["block_specs_kept"] = False
    return families, medspec, {**info, "_block_specs": block_specs}


def ladder_inspect(
    block_specs: list[dict],
    medspec: np.ndarray,
    *,
    library_families: list[dict] | None = None,
) -> list[dict]:
    """Inspect-only extra rungs after the safe paired seed failed."""
    report = []
    for rung in LADDER_RUNGS[1:]:
        hits = _hits_for_rung(block_specs, rung)
        families, _, info = cluster_recurrent_families(hits)
        report.append(
            {
                "rung": rung["name"],
                "row_z": rung["row_z"],
                "pair_z": rung["pair_z"],
                "allow_standalone": rung["allow_standalone"],
                "n_hits": len(hits),
                "families": [
                    {
                        "q": float(f["q"]),
                        "hi": f.get("hi"),
                        "paired": bool(f.get("paired", True)),
                        "row_score": float(f.get("row_score", 0.0)),
                        "n_blocks": int(f.get("n_blocks", 1)),
                    }
                    for f in families
                ],
                "chosen": info.get("chosen") or [],
            }
        )
    lib_ok = []
    lib_fail = []
    for src in library_families or []:
        q = float(src["q"])
        ok = library_family_supported(src, medspec)
        item = {"q": q, "hi": src.get("hi"), "supported_on_stack": ok}
        if ok:
            lib_ok.append(item)
        else:
            lib_fail.append(item)
    report.append(
        {
            "rung": "library_q",
            "families": lib_ok,
            "rejected": lib_fail,
        }
    )
    return report


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


def track_all(
    tf,
    families,
    *,
    block_size: int = 50,
    samples_per_block: int = 8,
    track_search: int = TRACK_SEARCH,
    track_update_min: float = 0.08,
    **_kwargs,
):
    """Lockstep tracking: each family searches ±track_search but cannot hop onto another's q."""
    n, h, w = tf.series[0].shape
    cx = w // 2
    fx = np.arange(w) - cx
    xvalid = (np.abs(fx) > 5) & (np.abs(fx) < cx - 10)

    q_states = [float(f["q"]) for f in families]
    paired_flags = [bool(f.get("paired", True)) for f in families]
    per_family_blocks: list[list[dict]] = [[] for _ in families]

    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        inds = np.linspace(start, stop - 1, min(samples_per_block, stop - start), dtype=int)
        specs = [fft_log_amp(tf.pages[int(i)].asarray()) for i in inds]
        block_spec = np.median(np.stack(specs), axis=0)
        new_states = list(q_states)
        for i, fam in enumerate(families):
            forbidden = [q_states[j] for j in range(len(families)) if j != i]
            q_found, score = search_q(
                block_spec,
                q_states[i],
                paired_flags[i],
                xvalid,
                track_search,
                forbidden_qs=forbidden,
                forbidden_radius=FORBIDDEN_Q_RADIUS,
            )
            updated = bool(score >= track_update_min) and np.isfinite(score)
            if updated:
                new_states[i] = q_found
            per_family_blocks[i].append(
                {
                    "start": int(start),
                    "stop": int(stop),
                    "mid": 0.5 * (start + stop - 1),
                    "q": float(new_states[i]),
                    "score": float(score) if np.isfinite(score) else 0.0,
                    "updated": updated,
                    "family": i,
                }
            )
        q_states = new_states

    trajectories = []
    all_blocks = []
    frame_idx = np.arange(n, dtype=float)
    for i, blocks in enumerate(per_family_blocks):
        mids = np.array([b["mid"] for b in blocks], dtype=float)
        qs = np.array([b["q"] for b in blocks], dtype=float)
        frame_q = np.interp(frame_idx, mids, qs, left=qs[0], right=qs[-1])
        trajectories.append(frame_q)
        all_blocks.extend(blocks)
    return trajectories, all_blocks
