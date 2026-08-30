"""Visual trial-clean report for needs_review stacks (not applied to the full TIFF)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile

_REPO = Path(__file__).resolve().parents[1]
_GPT = _REPO / "reference" / "gpt"
if str(_GPT) not in sys.path:
    sys.path.insert(0, str(_GPT))

from pmt_fringe_raw_adaptive import family_score, fft_log_amp, search_q  # noqa: E402
from pmt_fringe_raw_adaptive_v22 import clean_frame_v22  # noqa: E402

from .library import format_catalog_line
from .readout import (
    FRAMES_PER_PAGE,
    _jsonable,
    _percentile_limits,
    draw_frame_inspection_page,
    fft_mask_image,
    write_mean_tif,
)
from .seed import (
    EVAL_ANCHOR_FRAMES,
    EVAL_Z_STEPS,
    TRACK_SEARCH,
    collect_block_spectra,
    families_at_rung,
)

N_STRONG = 2
N_WEAK = 2
SCORE_STRIDE = 10


def _xvalid(width: int) -> np.ndarray:
    cx = width // 2
    fx = np.arange(width) - cx
    return (np.abs(fx) > 5) & (np.abs(fx) < cx - 10)


def _frame_score(frame: np.ndarray, families: list[dict]) -> float:
    if not families:
        return 0.0
    logamp = fft_log_amp(frame)
    xvalid = _xvalid(logamp.shape[1])
    scores = [
        family_score(logamp, int(round(float(f["q"]))), bool(f.get("paired", True)), xvalid)
        for f in families
    ]
    return float(max(scores)) if scores else 0.0


def choose_example_frames(
    tf: tifffile.TiffFile,
    families: list[dict],
    *,
    anchors: tuple[int, ...] = EVAL_ANCHOR_FRAMES,
    n_strong: int = N_STRONG,
    n_weak: int = N_WEAK,
    stride: int = SCORE_STRIDE,
) -> tuple[list[dict], list[dict]]:
    """Pick human anchors plus auto strong/weak frames. Returns (chosen, score_trace)."""
    n = int(tf.series[0].shape[0])
    sample_idx = list(range(0, n, max(1, stride)))
    for a in anchors:
        if 0 <= int(a) < n and int(a) not in sample_idx:
            sample_idx.append(int(a))
    sample_idx = sorted(set(sample_idx))

    trace = []
    for i in sample_idx:
        frame = tf.pages[int(i)].asarray()
        sc = _frame_score(frame, families) if families else 0.0
        trace.append({"frame": int(i), "score": sc})

    ranked = sorted(trace, key=lambda t: t["score"], reverse=True)
    chosen: list[dict] = []
    seen: set[int] = set()

    def _add(idx: int, role: str, score: float | None = None) -> None:
        if idx in seen or not (0 <= idx < n):
            return
        seen.add(idx)
        chosen.append({"frame": idx, "role": role, "score": float(score if score is not None else 0.0)})

    for a in anchors:
        rec = next((t for t in trace if t["frame"] == int(a)), None)
        _add(int(a), "anchor", rec["score"] if rec else 0.0)

    for rec in ranked:
        if len([c for c in chosen if c["role"] == "strong"]) >= n_strong:
            break
        if rec["frame"] in seen:
            continue
        _add(rec["frame"], "strong", rec["score"])

    for rec in reversed(ranked):
        if len([c for c in chosen if c["role"] == "weak"]) >= n_weak:
            break
        if rec["frame"] in seen:
            continue
        _add(rec["frame"], "weak", rec["score"])

    chosen.sort(key=lambda c: (0 if c["role"] == "anchor" else 1 if c["role"] == "strong" else 2, c["frame"]))
    return chosen, trace


def _trial_clean_frame(frame: np.ndarray, families: list[dict], params: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    logamp = fft_log_amp(frame)
    xvalid = _xvalid(logamp.shape[1])
    preds = []
    for fam in families:
        q, _ = search_q(
            logamp,
            float(fam["q"]),
            bool(fam.get("paired", True)),
            xvalid,
            TRACK_SEARCH,
        )
        preds.append(q)
    return clean_frame_v22(frame, families, preds, **params)


def _draw_cover(
    fig,
    *,
    mean_raw: np.ndarray,
    medspec: np.ndarray | None,
    chosen: list[dict],
    trace: list[dict],
    title: str,
    subtitle: str,
    message: str,
    catalog: dict | None = None,
) -> None:
    from matplotlib.gridspec import GridSpec

    fig.clear()
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 2, figure=fig, left=0.06, right=0.98, top=0.82, bottom=0.08, hspace=0.32, wspace=0.22)
    fig.text(0.06, 0.97, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.06, 0.935, subtitle, fontsize=8.5, va="top", color="0.25")
    fig.text(0.06, 0.905, format_catalog_line(catalog), fontsize=8, va="top", color="0.2")
    fig.text(0.06, 0.875, message, fontsize=8, va="top", color="0.35")

    ax0 = fig.add_subplot(gs[0, 0])
    vmin, vmax = _percentile_limits(mean_raw)
    ax0.imshow(mean_raw, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax0.set_title("Mean raw (sampled)", fontsize=9)
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[0, 1])
    if medspec is not None:
        spec = np.asarray(medspec)
        slo, shi = _percentile_limits(spec, (3.0, 99.7))
        ax1.imshow(spec, cmap="gray", vmin=slo, vmax=shi, interpolation="nearest")
        ax1.set_title("Seed FFT (safe scan found no pair)", fontsize=9)
    else:
        ax1.text(0.5, 0.5, "No spectrum", ha="center", va="center")
        ax1.set_axis_off()
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[1, 0])
    if trace:
        xs = [t["frame"] for t in trace]
        ys = [t["score"] for t in trace]
        ax2.plot(xs, ys, color="0.2", lw=0.8)
        colors = {"anchor": "C3", "strong": "C1", "weak": "C0"}
        for c in chosen:
            ax2.axvline(c["frame"], color=colors.get(c["role"], "0.5"), lw=0.8, alpha=0.85)
        ax2.set_title("Ridge score vs frame (higher = stronger candidate)", fontsize=9)
        ax2.set_xlabel("frame")
        ax2.set_ylabel("score")
    else:
        ax2.set_axis_off()

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    lines = [
        "Example frames (same set at every threshold):",
        "  red = you flagged as obvious fringe",
        "  orange = auto strongest  |  blue = auto weakest",
        "",
    ]
    for c in chosen:
        lines.append(f"  frame {c['frame']:<5}  {c['role']:<7}  score={c['score']:.3g}")
    lines += [
        "",
        "Following pages: original | trial-cleaned | removed.",
        "Full stack is NOT overwritten. Look for biology in 'removed'.",
    ]
    ax3.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8, family="monospace", transform=ax3.transAxes)


def write_eval_report(
    out_dir: Path,
    *,
    tf: tifffile.TiffFile,
    tif_path: Path,
    block_specs: list[dict] | None,
    medspec: np.ndarray | None,
    mean_raw: np.ndarray,
    params: dict,
    computer: str,
    channel: str,
    message: str,
    anchors: tuple[int, ...] = EVAL_ANCHOR_FRAMES,
    catalog: dict | None = None,
) -> dict[str, Path | None]:
    """Multi-page trial-clean PDF. Does not write a full-stack cleaned TIFF."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n, h, w = tf.series[0].shape
    if not block_specs:
        block_specs = collect_block_spectra(tf)

    scoring_families: list[dict] = []
    scoring_rung = None
    for rung in EVAL_Z_STEPS:
        fams, spec, _info = families_at_rung(block_specs, rung, medspec)
        if fams:
            scoring_families = fams
            scoring_rung = rung
            if spec is not None:
                medspec = spec
            break

    chosen, trace = choose_example_frames(tf, scoring_families, anchors=anchors)

    pdf_path = out_dir / "overview.pdf"
    png_path = out_dir / "overview.png"
    fig = plt.figure(figsize=(11.69, 8.27))
    ladder_out: list[dict] = []

    with PdfPages(pdf_path) as pdf:
        _draw_cover(
            fig,
            mean_raw=mean_raw,
            medspec=medspec,
            chosen=chosen,
            trace=trace,
            title=f"v2.2 trial-clean evaluation — NEEDS REVIEW — {channel}",
            subtitle=f"{tif_path.name}  ·  {computer} / {channel}  ·  stack not overwritten",
            message=message,
            catalog=catalog,
        )
        fig.savefig(png_path, dpi=130)
        pdf.savefig(fig, dpi=140)

        for rung in EVAL_Z_STEPS:
            families, spec, info = families_at_rung(block_specs, rung, medspec)
            qs = [float(f["q"]) for f in families]
            rung_rec = {
                "rung": rung["name"],
                "row_z": rung["row_z"],
                "pair_z": rung["pair_z"],
                "allow_standalone": bool(rung["allow_standalone"]),
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
            }
            triples: list[dict] = []
            if families:
                for c in chosen:
                    frame = np.asarray(tf.pages[int(c["frame"])].asarray())
                    cleaned, removed, tracking = _trial_clean_frame(frame, families, params)
                    triples.append(
                        {
                            "frame": c["frame"],
                            "role": c["role"],
                            "raw": frame,
                            "cleaned": cleaned,
                            "removed": removed,
                            "qs": [float(t.get("q", 0.0)) for t in tracking],
                            "removed_rms": float(np.sqrt(np.mean(np.asarray(removed, dtype=np.float64) ** 2))),
                        }
                    )
                rung_rec["per_frame"] = [
                    {"frame": t["frame"], "role": t["role"], "removed_rms": t["removed_rms"], "qs": t["qs"]}
                    for t in triples
                ]
            else:
                rung_rec["per_frame"] = []
            ladder_out.append(rung_rec)

            subtitle = (
                f"row_z={rung['row_z']}  pair_z={rung['pair_z']}  "
                f"standalone={rung['allow_standalone']}  "
                f"q={qs if qs else '—'}   (trial only)"
            )
            if not triples:
                fig.clear()
                fig.patch.set_facecolor("white")
                fig.text(0.06, 0.97, f"Threshold {rung['name']} — nothing detected", fontsize=12, fontweight="bold")
                fig.text(0.06, 0.90, subtitle, fontsize=8, color="0.35")
                fig.text(
                    0.06,
                    0.75,
                    "No family passed this cut. Next pages lower the threshold.",
                    fontsize=10,
                )
                pdf.savefig(fig, dpi=140)
                continue

            for start in range(0, len(triples), FRAMES_PER_PAGE):
                chunk = triples[start : start + FRAMES_PER_PAGE]
                draw_frame_inspection_page(
                    fig,
                    title=f"Trial clean at {rung['name']}",
                    subtitle=subtitle,
                    triples=chunk,
                    cleaned_label="trial cleaned",
                )
                pdf.savefig(fig, dpi=140)

    plt.close(fig)

    eval_path = out_dir / "eval.json"
    eval_path.write_text(
        json.dumps(
            _jsonable(
                {
                    "message": message,
                    "catalog": catalog,
                    "scoring_rung": None if scoring_rung is None else scoring_rung["name"],
                    "example_frames": chosen,
                    "score_trace": trace,
                    "ladder": ladder_out,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    ladder_path = out_dir / "ladder.json"
    ladder_path.write_text(json.dumps(_jsonable({"message": message, "ladder": ladder_out}), indent=2), encoding="utf-8")
    write_mean_tif(out_dir / "mean_raw.tif", mean_raw)
    if scoring_families and medspec is not None:
        mask = fft_mask_image((h, w), scoring_families)
        tifffile.imwrite(out_dir / "mask_fft.tif", mask, photometric="minisblack")

    return {
        "overview_pdf": pdf_path,
        "overview_png": png_path,
        "eval_json": eval_path,
        "ladder_json": ladder_path,
    }
