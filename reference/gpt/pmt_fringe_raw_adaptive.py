#!/usr/bin/env python3
"""
Adaptive raw-only PMT fringe removal for TIFF stacks.

Design goals
------------
1) Learn recurrent electronic fringe bands directly from RAW data.
2) Track slow frequency drift block-by-block and refine per frame.
3) Treat Fourier rows only as candidate regions; attenuate only recurrent
   ridge SEGMENTS (fx support), not entire rows.
4) Attenuate only spectral EXCESS above a local background estimate.
5) Smooth/taper all attenuation and skip weak/absent-fringe frames.
6) Preserve Fourier phase and enforce conservative defaults.

Dependencies:
    numpy, tifffile, scipy
Optional diagnostics:
    matplotlib

Example:
    python pmt_fringe_raw_adaptive.py input.tif -o input_defringed.tif --diagnostics diagnostics

Characterize only:
    python pmt_fringe_raw_adaptive.py input.tif --analyze-only --diagnostics diagnostics
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter1d, binary_dilation


def robust_local_z(profile, radius=8, exclude=1):
    profile = np.asarray(profile, dtype=float)
    n = len(profile)
    z = np.zeros(n, dtype=float)
    diff = np.zeros(n, dtype=float)
    for i in range(n):
        lo, hi = max(0, i-radius), min(n, i+radius+1)
        idx = [j for j in range(lo, hi) if abs(j-i) > exclude]
        if len(idx) < 3:
            continue
        vals = profile[idx]
        med = np.median(vals)
        mad = np.median(np.abs(vals-med)) + 1e-9
        diff[i] = profile[i] - med
        z[i] = diff[i] / (1.4826*mad)
    return z, diff


def fft_log_amp(frame):
    x = frame.astype(np.float32)
    x -= np.median(x)
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(x)))).astype(np.float32)


def learn_median_spectrum(tf, sample_n=80):
    n, h, w = tf.series[0].shape
    inds = np.linspace(0, n-1, min(sample_n, n), dtype=int)
    specs = [fft_log_amp(tf.pages[int(i)].asarray()) for i in inds]
    return np.median(np.stack(specs), axis=0), inds


def ridge_z_at_row(medspec, dy, neighbor_inner=4, neighbor_outer=10):
    h, w = medspec.shape
    cy, cx = h//2, w//2
    y = cy + int(dy)
    if not (0 <= y < h):
        return np.zeros(w, dtype=float)
    neigh = []
    for off in range(-neighbor_outer, neighbor_outer+1):
        if abs(off) < neighbor_inner:
            continue
        yy = y + off
        if 0 <= yy < h:
            neigh.append(yy)
    bg = np.median(medspec[neigh, :], axis=0)
    excess = medspec[y, :] - bg
    fx = np.arange(w) - cx
    valid = (np.abs(fx) > 5) & (np.abs(fx) < cx-10)
    med = np.median(excess[valid])
    mad = np.median(np.abs(excess[valid]-med)) + 1e-9
    return (excess-med)/(1.4826*mad)


def contiguous_ranges(values):
    values = sorted(set(int(v) for v in values))
    if not values:
        return []
    out = []
    start = prev = values[0]
    for v in values[1:]:
        if v == prev + 1:
            prev = v
        else:
            out.append([start, prev])
            start = prev = v
    out.append([start, prev])
    return out


def detect_families(
    medspec,
    row_z_thresh=5.5,
    pair_z_min=3.5,
    x_z_thresh=2.5,
    max_families=4,
    allow_standalone=False,
):
    h, w = medspec.shape
    cy, cx = h//2, w//2
    fx = np.arange(w) - cx
    xvalid = (np.abs(fx) > 5) & (np.abs(fx) < cx-10)

    row_profile = np.percentile(medspec[:, xvalid], 95, axis=1)
    row_z, row_diff = robust_local_z(row_profile, radius=8, exclude=1)

    candidates = []
    for dy in range(5, cy-5):
        y = cy + dy
        if row_z[y] < row_z_thresh:
            continue
        lo, hi = max(cy+5, y-2), min(h-5, y+3)
        if row_z[y] >= np.max(row_z[lo:hi]):
            candidates.append((dy, float(row_z[y]), float(row_diff[y])))

    candidates.sort(key=lambda t: t[1], reverse=True)
    families, used = [], set()

    for dy, score, _ in candidates:
        if dy in used:
            continue

        comp = cy - dy
        comp_range = range(max(5, comp-2), min(cy-5, comp+2)+1)
        if comp_range:
            comp2 = max(comp_range, key=lambda d: row_z[cy+d])
            comp_score = float(row_z[cy+comp2])
        else:
            comp2, comp_score = comp, -np.inf

        paired = comp_score >= pair_z_min
        if paired:
            q, hi = sorted([dy, comp2])
            fam = {
                "q": float(q),
                "hi": float(hi),
                "row_score": float(max(score, comp_score)),
                "pair_score": float(min(score, comp_score)),
                "paired": True,
            }
            used.add(dy)
            used.add(comp2)
        elif allow_standalone and score >= max(9.0, row_z_thresh+2):
            fam = {
                "q": float(dy),
                "hi": None,
                "row_score": float(score),
                "pair_score": None,
                "paired": False,
            }
            used.add(dy)
        else:
            continue

        # De-duplicate close families.
        if any(abs(fam["q"] - f["q"]) < 3 for f in families):
            continue

        # Learn recurrent fx support from all conjugate/components.
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

        families.append(fam)
        if len(families) >= max_families:
            break

    return families, row_profile, row_z


def row_contrast(logamp, dy, xvalid, inner=5, outer=9):
    h = logamp.shape[0]
    cy = h//2
    vals = []
    for sgn in (-1, +1):
        y = cy + sgn*int(dy)
        if not (0 <= y < h):
            continue
        rp = np.percentile(logamp[y, xvalid], 95)
        bgrows = []
        for off in list(range(-outer, -inner+1)) + list(range(inner, outer+1)):
            yy = y + off
            if 0 <= yy < h:
                bgrows.append(np.percentile(logamp[yy, xvalid], 95))
        if bgrows:
            vals.append(rp - np.median(bgrows))
    return float(np.median(vals)) if vals else -np.inf


def family_score(logamp, q, paired, xvalid):
    h = logamp.shape[0]
    cy = h//2
    scores = [row_contrast(logamp, q, xvalid)]
    if paired:
        scores.append(row_contrast(logamp, cy-q, xvalid))
    return float(np.median(scores))


def search_q(logamp, q0, paired, xvalid, radius, forbidden_qs=None, forbidden_radius=3):
    cy = logamp.shape[0]//2
    lo = max(5, int(round(q0))-radius)
    hi = min(cy-5, int(round(q0))+radius)
    qs = np.arange(lo, hi+1)
    if len(qs) == 0:
        return float(q0), float("-inf")
    scores = np.array([family_score(logamp, int(q), paired, xvalid) for q in qs])
    if forbidden_qs:
        for i, q in enumerate(qs):
            if any(abs(int(q) - int(fq)) < int(forbidden_radius) for fq in forbidden_qs):
                scores[i] = -np.inf
        if not np.isfinite(scores).any():
            return float(q0), float("-inf")
    j = int(np.argmax(scores))
    return float(qs[j]), float(scores[j])


def track_family_blocks(
    tf,
    family,
    block_size=50,
    samples_per_block=8,
    track_search=6,
    track_update_min=0.08,
):
    n, h, w = tf.series[0].shape
    cx = w//2
    fx = np.arange(w) - cx
    xvalid = (np.abs(fx) > 5) & (np.abs(fx) < cx-10)

    q_state = float(family["q"])
    blocks = []

    for start in range(0, n, block_size):
        stop = min(n, start+block_size)
        inds = np.linspace(start, stop-1, min(samples_per_block, stop-start), dtype=int)
        specs = [fft_log_amp(tf.pages[int(i)].asarray()) for i in inds]
        block_spec = np.median(np.stack(specs), axis=0)

        q_found, score = search_q(
            block_spec, q_state, family["paired"], xvalid, track_search
        )
        if score >= track_update_min:
            q_state = q_found

        blocks.append({
            "start": int(start),
            "stop": int(stop),
            "mid": 0.5*(start+stop-1),
            "q": float(q_state),
            "score": float(score),
            "updated": bool(score >= track_update_min),
        })

    mids = np.array([b["mid"] for b in blocks], dtype=float)
    qs = np.array([b["q"] for b in blocks], dtype=float)
    frame_q = np.interp(np.arange(n, dtype=float), mids, qs, left=qs[0], right=qs[-1])
    return frame_q, blocks


def clean_frame(
    frame,
    families,
    predicted_qs,
    frame_search=2,
    gate_low=0.08,
    gate_high=0.18,
    ratio_start=1.5,
    ratio_full=4.0,
    max_alpha=0.85,
    y_sigma=1.0,
    y_radius=2,
):
    orig_dtype = frame.dtype
    x = frame.astype(np.float32)
    offset = float(np.median(x))
    x0 = x - offset

    F = np.fft.fftshift(np.fft.fft2(x0))
    amp = np.abs(F)
    phase = np.exp(1j*np.angle(F))
    logamp = np.log1p(amp)

    h, w = x.shape
    cy, cx = h//2, w//2
    fx = np.arange(w) - cx
    xvalid = (np.abs(fx) > 5) & (np.abs(fx) < cx-10)

    newamp = amp.copy()
    tracking = []

    for family, q_pred in zip(families, predicted_qs):
        q, strength = search_q(
            logamp, q_pred, family["paired"], xvalid, frame_search
        )
        gate = float(np.clip(
            (strength-gate_low) / max(1e-9, gate_high-gate_low), 0.0, 1.0
        ))

        tracking.append({"q": q, "strength": strength, "gate": gate})
        if gate <= 0:
            continue

        dys = [q]
        if family["paired"]:
            dys.append(cy-q)

        yoffs = np.arange(-y_radius, y_radius+1)
        yweights = np.exp(-0.5*(yoffs/max(1e-6, y_sigma))**2)
        xweight = family["x_weight"]

        for d in dys:
            for sgn in (-1, +1):
                yc = cy + sgn*int(round(d))
                for off, wy in zip(yoffs, yweights):
                    y = yc + int(off)
                    if not (0 <= y < h):
                        continue

                    bgrows = []
                    for boff in list(range(-9, -4)) + list(range(5, 10)):
                        yy = y + boff
                        if 0 <= yy < h:
                            bgrows.append(amp[yy, :])
                    if not bgrows:
                        continue

                    bg = np.median(np.stack(bgrows), axis=0)
                    ratio = amp[y, :] / (bg + 1e-12)

                    local_conf = np.clip(
                        (ratio-ratio_start) / max(1e-9, ratio_full-ratio_start),
                        0.0, 1.0
                    )
                    alpha = max_alpha * gate * float(wy) * xweight * local_conf

                    # Remove only excess above local spectral background.
                    excess = np.maximum(amp[y, :] - bg, 0.0)
                    candidate = amp[y, :] - alpha*excess

                    # If multiple families touch the same coefficient, keep the
                    # most attenuated result, but never go below local background.
                    newamp[y, :] = np.minimum(newamp[y, :], np.maximum(candidate, bg))

    Fnew = newamp * phase
    cleaned = np.real(np.fft.ifft2(np.fft.ifftshift(Fnew))) + offset
    removed = x - cleaned

    if np.issubdtype(orig_dtype, np.integer):
        lim = np.iinfo(orig_dtype)
        cleaned_write = np.clip(np.rint(cleaned), lim.min, lim.max).astype(orig_dtype)
    else:
        cleaned_write = cleaned.astype(orig_dtype)

    return cleaned_write, removed.astype(np.float32), tracking


def save_diagnostics(diag_dir, medspec, families, row_profile, row_z, track_blocks):
    diag_dir = Path(diag_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "families": [],
        "tracking_blocks": track_blocks,
    }
    for f in families:
        summary["families"].append({
            "q": f["q"],
            "hi": f["hi"],
            "paired": f["paired"],
            "row_score": f["row_score"],
            "pair_score": f["pair_score"],
            "fx_ranges_weight_gt_0.20": f["fx_ranges"],
        })

    with open(diag_dir/"signature.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    try:
        import matplotlib.pyplot as plt

        h, w = medspec.shape
        cy, cx = h//2, w//2

        plt.figure(figsize=(7, 6))
        lo, hi = np.percentile(medspec, [3, 99.7])
        plt.imshow(medspec, cmap="gray", vmin=lo, vmax=hi)
        for f in families:
            for d in [f["q"]] + ([f["hi"]] if f["hi"] is not None else []):
                plt.axhline(cy+d, linewidth=0.8)
                plt.axhline(cy-d, linewidth=0.8)
        plt.title("Median raw Fourier spectrum + detected fringe rows")
        plt.tight_layout()
        plt.savefig(diag_dir/"detected_spectrum.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 3.5))
        dy = np.arange(h) - cy
        plt.plot(dy, row_z)
        plt.xlabel("fy offset (Fourier bins)")
        plt.ylabel("robust row anomaly z")
        plt.title("Recurrent row/ridge anomaly score")
        plt.tight_layout()
        plt.savefig(diag_dir/"row_anomaly_score.png", dpi=160)
        plt.close()

    except Exception as e:
        print(f"Diagnostics plots skipped: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Raw multipage TIFF stack")
    ap.add_argument("-o", "--output", help="Output defringed TIFF")
    ap.add_argument("--analyze-only", action="store_true",
                    help="Detect/track fringe without writing cleaned TIFF")
    ap.add_argument("--diagnostics", help="Directory for JSON/PNG/CSV diagnostics")

    ap.add_argument("--sample-frames", type=int, default=80)
    ap.add_argument("--row-z", type=float, default=5.5)
    ap.add_argument("--pair-z", type=float, default=3.5)
    ap.add_argument("--x-z", type=float, default=3.5)
    ap.add_argument("--max-families", type=int, default=4)
    ap.add_argument("--allow-standalone", action="store_true")

    ap.add_argument("--block-size", type=int, default=50)
    ap.add_argument("--samples-per-block", type=int, default=8)
    ap.add_argument("--track-search", type=int, default=6)
    ap.add_argument("--track-update-min", type=float, default=0.10)
    ap.add_argument("--frame-search", type=int, default=2)

    ap.add_argument("--gate-low", type=float, default=0.10)
    ap.add_argument("--gate-high", type=float, default=0.20)
    ap.add_argument("--ratio-start", type=float, default=1.6)
    ap.add_argument("--ratio-full", type=float, default=4.0)
    ap.add_argument("--max-alpha", type=float, default=0.85)

    args = ap.parse_args()

    if not args.analyze_only and not args.output:
        ap.error("--output is required unless --analyze-only is used")

    inpath = Path(args.input)

    with tifffile.TiffFile(inpath) as tf:
        shape = tf.series[0].shape
        if len(shape) != 3:
            raise ValueError(f"Expected a 3-D TIFF stack (frames,y,x); got {shape}")
        n, h, w = shape
        if h % 2 or w % 2:
            print("Warning: odd image dimensions are less tested.")

        print(f"Input: {inpath}")
        print(f"Shape: {shape}, dtype: {tf.pages[0].dtype}")

        medspec, sampled = learn_median_spectrum(tf, args.sample_frames)
        families, row_profile, row_z = detect_families(
            medspec,
            row_z_thresh=args.row_z,
            pair_z_min=args.pair_z,
            x_z_thresh=args.x_z,
            max_families=args.max_families,
            allow_standalone=args.allow_standalone,
        )

        if not families:
            raise RuntimeError(
                "No high-confidence recurrent paired fringe family was detected. "
                "For safety, no filtering was performed. Inspect diagnostics or "
                "retry with a slightly lower --row-z / --pair-z."
            )

        print("Detected fringe families:")
        for i, f in enumerate(families, 1):
            print(
                f"  {i}: q={f['q']:.1f}, hi={f['hi']}, "
                f"row_z={f['row_score']:.1f}, paired={f['paired']}, "
                f"fx support={f['fx_ranges']}"
            )

        trajectories = []
        all_blocks = []
        for i, f in enumerate(families):
            qtraj, blocks = track_family_blocks(
                tf, f,
                block_size=args.block_size,
                samples_per_block=args.samples_per_block,
                track_search=args.track_search,
                track_update_min=args.track_update_min,
            )
            trajectories.append(qtraj)
            for b in blocks:
                bb = dict(b)
                bb["family"] = i
                all_blocks.append(bb)

        if args.diagnostics:
            save_diagnostics(
                args.diagnostics, medspec, families, row_profile, row_z, all_blocks
            )

        if args.analyze_only:
            print("Analyze-only mode: no output TIFF written.")
            return

        outpath = Path(args.output)
        est_bytes = n*h*w*np.dtype(tf.pages[0].dtype).itemsize
        bigtiff = est_bytes > 3_500_000_000

        diag_writer = None
        csvfh = None
        if args.diagnostics:
            csvfh = open(Path(args.diagnostics)/"temporal_tracking.csv",
                         "w", newline="", encoding="utf-8")
            fieldnames = ["frame", "removed_rms"]
            for i in range(len(families)):
                fieldnames += [f"family{i}_q", f"family{i}_strength", f"family{i}_gate"]
            diag_writer = csv.DictWriter(csvfh, fieldnames=fieldnames)
            diag_writer.writeheader()

        with tifffile.TiffWriter(outpath, bigtiff=bigtiff) as tw:
            for fi in range(n):
                frame = tf.pages[fi].asarray()
                preds = [traj[fi] for traj in trajectories]
                cleaned, removed, tracking = clean_frame(
                    frame, families, preds,
                    frame_search=args.frame_search,
                    gate_low=args.gate_low,
                    gate_high=args.gate_high,
                    ratio_start=args.ratio_start,
                    ratio_full=args.ratio_full,
                    max_alpha=args.max_alpha,
                )
                tw.write(cleaned, contiguous=True)

                if diag_writer:
                    row = {
                        "frame": fi,
                        "removed_rms": float(np.sqrt(np.mean(removed.astype(float)**2))),
                    }
                    for i, t in enumerate(tracking):
                        row[f"family{i}_q"] = t["q"]
                        row[f"family{i}_strength"] = t["strength"]
                        row[f"family{i}_gate"] = t["gate"]
                    diag_writer.writerow(row)

                if (fi+1) % 100 == 0 or fi == n-1:
                    print(f"Processed {fi+1}/{n} frames")

        if csvfh:
            csvfh.close()

        print(f"Wrote: {outpath}")
        if args.diagnostics:
            print(f"Diagnostics: {args.diagnostics}")


if __name__ == "__main__":
    main()
