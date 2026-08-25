"""Figures for dark-current fringe characterisation (matplotlib optional)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] matplotlib unavailable, skipping figures: {exc}")
        return None


def plot_channel(result: dict[str, Any], out_dir: Path) -> list[Path]:
    plt = _mpl()
    if plt is None:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    tag = f"{result['label']}_{result['channel']}"

    fov = result.get("fov") or {}
    temporal = result.get("temporal") or {}
    qtrack = result.get("q_track") or {}

    census = result.get("tile_census") or {}

    fig, axes = plt.subplots(2, 4, figsize=(21, 8.5))
    fig.suptitle(f"Dark-current fringe: {tag}  (q={result.get('q_used')})")

    ax = axes[0][0]
    if fov.get("excess"):
        amp = np.asarray(fov["excess"])
        im = ax.imshow(amp, cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Fringe excess by FOV tile")
        ax.set_xlabel("tile x")
        ax.set_ylabel("tile y")
        for r in range(amp.shape[0]):
            for c in range(amp.shape[1]):
                ax.text(
                    c, r, f"{amp[r, c]:.0f}", ha="center", va="center",
                    color="w", fontsize=7,
                )

    ax = axes[0][1]
    if fov.get("snr"):
        rel = np.asarray(fov["snr"])
        im = ax.imshow(rel, cmap="cividis")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Tile fringe SNR vs off-ridge control")
        ax.set_xlabel("tile x")
        ax.set_ylabel("tile y")

    xprof = result.get("x_profile") or {}
    ax = axes[0][2]
    if xprof.get("excess"):
        ax.plot(xprof["centers_px"], xprof["excess"], marker="o", ms=3, label="ridge")
        ax.plot(
            xprof["centers_px"], xprof["control"], marker=".", ms=3,
            color="crimson", alpha=0.7, label="control",
        )
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.set_title(
            f"Excess vs x (edge/middle={xprof.get('edge_over_middle', float('nan')):.2f})"
        )
        ax.set_xlabel("x position (px)")
        ax.set_ylabel("ridge excess")

    ax = axes[0][3]
    if census.get("dominant_q_scaled"):
        dq = np.asarray(census["dominant_q_scaled"])
        im = ax.imshow(dq, cmap="plasma")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Per-tile dominant q (assumption-free)")
        ax.set_xlabel("tile x")
        ax.set_ylabel("tile y")
        for r in range(dq.shape[0]):
            for c in range(dq.shape[1]):
                ax.text(
                    c, r, f"{dq[r, c]:.0f}", ha="center", va="center",
                    color="w", fontsize=7,
                )

    ax = axes[1][0]
    if temporal.get("excess"):
        ax.plot(temporal["frames"], temporal["excess"], lw=0.8, label="ridge")
        if temporal.get("control"):
            ax.plot(
                temporal["frames"], temporal["control"], lw=0.8,
                color="crimson", alpha=0.7, label="control",
            )
        ax.legend(fontsize=8)
        ax.set_title(
            f"Fringe excess over time (CV={temporal.get('cv', float('nan')):.2f})"
        )
        ax.set_xlabel("frame")
        ax.set_ylabel("excess amplitude")

    ax = axes[1][1]
    phase = result.get("phase") or {}
    if phase.get("phase"):
        idx = np.arange(phase["start"], phase["start"] + phase["count"])
        ax.plot(idx, np.unwrap(np.asarray(phase["phase"])), lw=0.8)
        ax.set_title(
            f"Ridge phase, consecutive frames "
            f"(coherence={phase.get('step_coherence', float('nan')):.2f})"
        )
        ax.set_xlabel("frame")
        ax.set_ylabel("unwrapped phase (rad)")

    ax = axes[1][2]
    if qtrack.get("q"):
        ax.plot(qtrack["block_starts"], qtrack["q"], marker="o", ms=3, lw=0.8)
        ax.set_title(
            f"Tracked ridge q per {qtrack.get('block_size')}-frame block "
            f"(span={qtrack.get('q_span')})"
        )
        ax.set_xlabel("frame")
        ax.set_ylabel("q (cycles/frame)")

    ax = axes[1][3]
    snr_rows = result.get("family_snr") or []
    if snr_rows:
        qs = [s["q"] for s in snr_rows]
        snrs = [s["snr"] for s in snr_rows]
        order = np.argsort(qs)
        ax.bar([str(int(qs[i])) for i in order], [snrs[i] for i in order])
        ax.set_yscale("log")
        ax.set_title("Candidate evidence (excess SNR)")
        ax.set_xlabel("q")
        ax.set_ylabel("SNR vs control rows")

    fig.tight_layout()
    path = out_dir / f"{tag}_summary.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)
    return written


def plot_condition_compare(results: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    """Amplitude and q across conditions, per channel."""
    plt = _mpl()
    if plt is None:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = sorted({r["channel"] for r in results})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Dark-current fringe across Pockels settings")

    def sort_key(r: dict[str, Any]):
        pc = r.get("pockels_setting")
        return (pc if pc is not None else 10**9, r["label"])

    for ch in channels:
        rows = [r for r in results if r["channel"] == ch]
        rows.sort(key=sort_key)
        labels = [
            f"PC{r['pockels_setting']}" if r.get("pockels_setting") is not None
            else r["label"]
            for r in rows
        ]
        amps = [(r.get("temporal") or {}).get("mean", np.nan) for r in rows]
        qs = [(r.get("q_track") or {}).get("q_median", np.nan) for r in rows]
        axes[0].plot(labels, amps, marker="o", label=ch)
        axes[1].plot(labels, qs, marker="o", label=ch)

    axes[0].set_title("Mean ridge field RMS")
    axes[0].set_ylabel("ADU")
    axes[1].set_title("Median tracked q")
    axes[1].set_ylabel("cycles/frame")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path = out_dir / "conditions_compare.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path]
