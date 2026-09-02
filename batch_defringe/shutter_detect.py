"""Detect in-stack shutter quiet windows from the FOV z-profile.

Live frames have cells (high contrast). When the shutter closes, FOV signal
plummets: std / (p90−p10) collapse. The PMT offset keeps the *mean* high, so a
mean-fraction cut misses the event.

``python -m batch_defringe.shutter_detect`` writes a method PDF. ``process_stack``
runs the same detector at the start of every clean and draws it on overview.pdf.
Detected frames are recorded, not used as a hard seed (image-check veto first).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

LIVE_PERCENTILE = 75.0
STD_FRAC = 0.40
CLIFF_FRAC = 0.35
MIN_LEN = 3
OUTPUT_SUBDIR = "shutter_detect"
DEFAULT_TIFS = (
    Path(r"F:\bPACNewData2026\Haj Grant Example\DATA\ChanA\ChanA_stk.tif"),
    Path(r"F:\bPACNewData2026\Haj Grant Example\DATA\ChanB\ChanB_stk.tif"),
)


def scan_frame_stats(tf: tifffile.TiffFile) -> list[dict]:
    """Cheap z-profile: mean and std per frame (FOV contrast, not a mean-offset)."""
    n = int(tf.series[0].shape[0])
    out: list[dict] = []
    for i in range(n):
        f = np.asarray(tf.pages[i].asarray(), dtype=np.float32)
        out.append({"frame": int(i), "mean": float(f.mean()), "std": float(f.std())})
    return out


def detect_shutter_windows(
    stats: list[dict],
    *,
    live_percentile: float = LIVE_PERCENTILE,
    std_frac: float = STD_FRAC,
    cliff_frac: float = CLIFF_FRAC,
    min_len: int = MIN_LEN,
) -> dict:
    """Find consecutive quiet frames after a contrast cliff.

    Quiet: std ≤ std_frac × (live_percentile of std).
    Cliff: entering the run, std drops by at least cliff_frac × live_std
    (or the run starts at frame 0). Mean is recorded for the plot; it is not
    the cut, because DC offset stays ~850 ADU on Haj Grant ChanA.
    """
    if not stats:
        return {
            "frames": [],
            "runs": [],
            "n_frames": 0,
            "live_std": None,
            "live_mean": None,
            "std_thresh": None,
            "method": "contrast_cliff",
            "std_frac": float(std_frac),
            "cliff_frac": float(cliff_frac),
            "live_percentile": float(live_percentile),
            "min_len": int(min_len),
        }
    std = np.array([float(s["std"]) for s in stats], dtype=np.float64)
    mean = np.array([float(s["mean"]) for s in stats], dtype=np.float64)
    n = int(std.size)
    live_std = float(np.percentile(std, live_percentile))
    live_mean = float(np.percentile(mean, live_percentile))
    thresh = float(std_frac) * live_std
    quiet = std <= thresh
    cliff_need = float(cliff_frac) * live_std
    runs: list[dict] = []
    i = 0
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i + 1
        while j < n and quiet[j]:
            j += 1
        length = j - i
        if length >= int(min_len):
            if i == 0:
                cliff = True
                drop = None
            else:
                drop = float(std[i - 1] - std[i])
                cliff = drop >= cliff_need
            if cliff:
                frames = list(range(i, j))
                runs.append(
                    {
                        "start": int(i),
                        "stop": int(j - 1),
                        "n": int(length),
                        "frames": frames,
                        "drop_std": drop,
                        "mean_std": float(np.mean(std[i:j])),
                        "mean_mean": float(np.mean(mean[i:j])),
                    }
                )
        i = j
    frames = [f for r in runs for f in r["frames"]]
    return {
        "frames": frames,
        "runs": runs,
        "n_frames": n,
        "live_std": live_std,
        "live_mean": live_mean,
        "std_thresh": thresh,
        "method": "contrast_cliff",
        "std_frac": float(std_frac),
        "cliff_frac": float(cliff_frac),
        "live_percentile": float(live_percentile),
        "min_len": int(min_len),
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
    }


def format_shutter_span(det: dict | None) -> str:
    frames = list((det or {}).get("frames") or [])
    if not frames:
        return "none"
    runs = det.get("runs") or []
    bits = []
    for r in runs:
        bits.append(f"{r['start']}–{r['stop']} ({r['n']} frames)")
    return "; ".join(bits) if bits else "none"


def shutter_public(det: dict | None) -> dict:
    """JSON-safe record without the full z-profile arrays (those stay for plots)."""
    if not det:
        return {"frames": [], "runs": [], "method": "contrast_cliff"}
    return {
        "frames": [int(i) for i in det.get("frames") or []],
        "runs": [
            {
                "start": int(r["start"]),
                "stop": int(r["stop"]),
                "n": int(r["n"]),
                "frames": [int(i) for i in r["frames"]],
                "drop_std": r.get("drop_std"),
                "mean_std": r.get("mean_std"),
                "mean_mean": r.get("mean_mean"),
            }
            for r in (det.get("runs") or [])
        ],
        "n_frames": int(det.get("n_frames") or 0),
        "live_std": det.get("live_std"),
        "live_mean": det.get("live_mean"),
        "std_thresh": det.get("std_thresh"),
        "method": det.get("method") or "contrast_cliff",
        "std_frac": det.get("std_frac"),
        "cliff_frac": det.get("cliff_frac"),
        "live_percentile": det.get("live_percentile"),
        "min_len": det.get("min_len"),
        "mean": [float(x) for x in (det.get("mean") or [])],
        "std": [float(x) for x in (det.get("std") or [])],
    }


def low_std_runs(stats: list[dict], *, ratio: float = STD_FRAC, min_len: int = MIN_LEN) -> list[list[int]]:
    """Back-compat wrapper: quiet runs as lists of frame indices."""
    det = detect_shutter_windows(stats, std_frac=ratio, min_len=min_len)
    return [list(r["frames"]) for r in det.get("runs") or []]


def draw_shutter_page(
    fig,
    *,
    title: str,
    subtitle: str,
    det: dict,
    live_frame: np.ndarray | None = None,
    shutter_frame: np.ndarray | None = None,
    live_idx: int | None = None,
    shutter_idx: int | None = None,
) -> None:
    """Z-profile + optional live vs shutter stills (for overview and the method PDF)."""
    from .readout import _percentile_limits

    fig.clear()
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.97, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.06, 0.935, subtitle, fontsize=8.5, va="top", color="0.35")

    mean = np.asarray(det.get("mean") or [], dtype=float)
    std = np.asarray(det.get("std") or [], dtype=float)
    n = int(det.get("n_frames") or max(mean.size, std.size, 1))
    x = np.arange(n)
    thresh = det.get("std_thresh")
    live_std = det.get("live_std")
    span = format_shutter_span(det)

    fig.text(
        0.06,
        0.90,
        f"Detected: {span}   ·   quiet if std ≤ {det.get('std_frac', STD_FRAC):.2f} × "
        f"p{det.get('live_percentile', LIVE_PERCENTILE):.0f}(std)"
        + (f" = {thresh:.2f}" if thresh is not None else "")
        + f"   ·   cliff ≥ {det.get('cliff_frac', CLIFF_FRAC):.2f} × live std"
        + (f" ({live_std:.1f})" if live_std is not None else ""),
        fontsize=8,
        va="top",
        family="monospace",
    )
    fig.text(
        0.06,
        0.875,
        "Cut is contrast (std), not mean. PMT offset stays high when the shutter closes; "
        "cells (p90 / std) fall. Mean is plotted so a false mean-fraction cut is visible.",
        fontsize=7.5,
        va="top",
        color="0.35",
    )

    show_stills = live_frame is not None or shutter_frame is not None
    if show_stills:
        gs = fig.add_gridspec(
            2, 2, left=0.07, right=0.98, top=0.84, bottom=0.07, hspace=0.32, wspace=0.18, height_ratios=[1.15, 1.0]
        )
        ax_m = fig.add_subplot(gs[0, 0])
        ax_s = fig.add_subplot(gs[0, 1], sharex=ax_m)
        ax_live = fig.add_subplot(gs[1, 0])
        ax_sh = fig.add_subplot(gs[1, 1])
    else:
        gs = fig.add_gridspec(2, 1, left=0.07, right=0.98, top=0.84, bottom=0.08, hspace=0.28)
        ax_m = fig.add_subplot(gs[0])
        ax_s = fig.add_subplot(gs[1], sharex=ax_m)
        ax_live = ax_sh = None

    def _shade(ax) -> None:
        for r in det.get("runs") or []:
            ax.axvspan(r["start"] - 0.5, r["stop"] + 0.5, color="C1", alpha=0.22, zorder=0)

    if mean.size:
        ax_m.plot(x, mean, color="0.2", lw=0.8)
        _shade(ax_m)
        ax_m.set_ylabel("FOV mean (ADU)")
        ax_m.set_title("Mean stays high (offset) — not the shutter cut", fontsize=9)
        ax_m.tick_params(labelbottom=False)
    else:
        ax_m.set_axis_off()

    if std.size:
        ax_s.plot(x, std, color="0.15", lw=0.8, label="std")
        _shade(ax_s)
        if thresh is not None:
            ax_s.axhline(thresh, color="C3", lw=1.0, label=f"quiet thresh {thresh:.1f}")
        if live_std is not None:
            ax_s.axhline(live_std, color="C0", lw=0.8, ls="--", label=f"live p75 std {live_std:.1f}")
        ax_s.set_ylabel("FOV std (ADU)")
        ax_s.set_xlabel("frame")
        ax_s.set_title("Std plummets when the shutter closes", fontsize=9)
        ax_s.legend(fontsize=7, loc="upper right", frameon=False)
    else:
        ax_s.set_axis_off()

    if ax_live is not None:
        if live_frame is None:
            ax_live.set_axis_off()
        else:
            vmin, vmax = _percentile_limits(np.asarray(live_frame, dtype=float))
            ax_live.imshow(live_frame, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax_live.set_title(f"Live  frame {live_idx}", fontsize=9)
            ax_live.set_xticks([])
            ax_live.set_yticks([])
        if shutter_frame is None:
            ax_sh.set_axis_off()
        else:
            vmin, vmax = _percentile_limits(np.asarray(shutter_frame, dtype=float))
            ax_sh.imshow(shutter_frame, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax_sh.set_title(f"Shutter  frame {shutter_idx}", fontsize=9)
            ax_sh.set_xticks([])
            ax_sh.set_yticks([])


def write_method_pdf(path: Path, channels: list[dict]) -> Path:
    """Short method PDF: one page per channel z-profile + live/shutter stills."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.69, 8.27))
    with PdfPages(path) as pdf:
        fig.clear()
        fig.patch.set_facecolor("white")
        fig.text(0.08, 0.92, "In-stack shutter detect", fontsize=16, fontweight="bold", va="top")
        lines = [
            "Live FOV has cells → high std. Shutter closed → cells gone, only PMT offset + ripples.",
            "The mean barely moves (ChanA ~962 → 851) because the dark offset is huge. A cut like",
            "mean < 0.55 × median(mean) finds nothing. Contrast does: std 203 → 31 on ChanA,",
            "109 → 35 on ChanB, both exactly frames 756–760.",
            "",
            "Rule (same on every stack):",
            f"  live_std = p{LIVE_PERCENTILE:.0f}(std across frames)",
            f"  quiet if std ≤ {STD_FRAC:.2f} × live_std, in a run of ≥ {MIN_LEN} frames",
            f"  and a cliff into the run of ≥ {CLIFF_FRAC:.2f} × live_std (or the run starts at 0).",
            "",
            "Detected windows are written on overview.pdf / families.json. They are not a hard",
            "seed until an image-domain check can veto a bad transfer onto live frames.",
        ]
        fig.text(0.08, 0.84, "\n".join(lines), fontsize=10, va="top", family="sans-serif")
        pdf.savefig(fig, dpi=140)
        for ch in channels:
            draw_shutter_page(
                fig,
                title=f"{ch['name']}  ·  shutter z-profile",
                subtitle=str(ch.get("tif") or ""),
                det=ch["det"],
                live_frame=ch.get("live_frame"),
                shutter_frame=ch.get("shutter_frame"),
                live_idx=ch.get("live_idx"),
                shutter_idx=ch.get("shutter_idx"),
            )
            pdf.savefig(fig, dpi=140)
    plt.close(fig)
    return path


def _load_example(tf: tifffile.TiffFile, det: dict, live_hint: int = 160) -> tuple:
    n = int(tf.series[0].shape[0])
    frames = det.get("frames") or []
    sh_idx = int(frames[len(frames) // 2]) if frames else None
    live_idx = int(live_hint) if 0 <= live_hint < n and live_hint not in frames else 0
    if live_idx in (frames or []):
        live_idx = 0
    live = np.asarray(tf.pages[live_idx].asarray())
    shutter = None if sh_idx is None else np.asarray(tf.pages[sh_idx].asarray())
    return live_idx, live, sh_idx, shutter


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tif", type=Path, action="append", default=None)
    args = ap.parse_args(argv)
    tifs = list(args.tif) if args.tif else [p for p in DEFAULT_TIFS if p.is_file()]
    if not tifs:
        print("MISSING stacks")
        return 1
    channels = []
    for tif in tifs:
        if not tif.is_file():
            print(f"MISSING {tif}")
            return 1
        with tifffile.TiffFile(tif) as tf:
            stats = scan_frame_stats(tf)
            det = detect_shutter_windows(stats)
            live_idx, live, sh_idx, shutter = _load_example(tf, det)
            print(f"{tif.name}: shutter {format_shutter_span(det)}", flush=True)
            channels.append(
                {
                    "name": tif.stem,
                    "tif": str(tif),
                    "det": det,
                    "live_idx": live_idx,
                    "live_frame": live,
                    "shutter_idx": sh_idx,
                    "shutter_frame": shutter,
                }
            )
        out_dir = tif.parent / "defringe_v22" / OUTPUT_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "shutter.json").write_text(
            json.dumps(shutter_public(det), indent=2), encoding="utf-8"
        )
    out = tifs[0].parent / "defringe_v22" / OUTPUT_SUBDIR / "overview.pdf"
    if len(tifs) > 1:
        out = tifs[0].parent.parent / "shutter_detect_overview.pdf"
        # keep it next to DATA if both ChanA and ChanB
        data = tifs[0].parent.parent
        if data.name.upper() == "DATA" or data.exists():
            out = data / "shutter_detect_overview.pdf"
    pdf = write_method_pdf(out, channels)
    print(f"wrote {pdf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
