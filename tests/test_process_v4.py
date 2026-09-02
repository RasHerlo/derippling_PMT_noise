"""v4 grow_frame: one mask, fx+fy, predicted close to removed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "reference" / "gpt"))

from batch_defringe.process_v4 import (  # noqa: E402
    Line,
    apply_lines,
    collect_lines,
    grow_frame,
    pick_inspect_frames,
    unique_lines,
)

H, W = 128, 128


def _vstripes(q: float = 8.0) -> np.ndarray:
    x = np.arange(W)[None, :]
    return np.broadcast_to(80.0 + 40.0 * np.sin(2.0 * np.pi * q * x / W), (H, W)).copy()


def test_unique_keeps_fy_and_fx():
    dummy = {"q": 10.0}
    lines = [
        Line("fy", 10.0, dummy, "fft"),
        Line("fx", 10.0, dummy, "linescan"),
        Line("fy", 10.2, dummy, "fft"),
    ]
    out = unique_lines(lines)
    assert {ln.axis for ln in out} == {"fy", "fx"}


def test_collect_vertical_stripes_uses_peak_mask():
    frame = _vstripes()
    lines, seed = collect_lines(frame)
    peaks = [ln for ln in lines if ln.kind == "peak"]
    assert seed.get("ok")
    assert peaks
    assert peaks[0].axis == "fx"
    assert peaks[0].source == "linescan"
    assert peaks[0].peak_mask is not None
    assert float(np.max(peaks[0].peak_mask)) > 0


def test_grow_vertical_stripes_cleans():
    frame = _vstripes()
    rec = grow_frame(frame, role="live_anchor")
    assert rec["removed_rms"] > 1.0
    assert rec["agree"] > 0.5
    assert rec["n_lines"] >= 1
    axes = {a["axis"] for a in rec["accepted"]}
    assert "fx" in axes
    kinds = {a.get("kind") for a in rec["accepted"]}
    assert "peak" in kinds


def test_apply_identity_when_no_lines():
    frame = _vstripes()
    rec = apply_lines(frame, [], max_alpha=0.28, search=4)
    assert rec["removed_rms"] == 0.0
    assert rec["agree"] == 1.0


def test_pick_inspect_uses_this_channel_shutter_not_chana_anchors():
    rows = [
        {"frame": 10, "role": "live", "removed_rms": 1.0, "empty": False, "brake": False, "n_lines": 1},
        {"frame": 20, "role": "live", "removed_rms": 9.0, "empty": False, "brake": False, "n_lines": 4},
        {"frame": 30, "role": "live", "removed_rms": 0.0, "empty": True, "brake": False, "n_lines": 0},
        {"frame": 40, "role": "live", "removed_rms": 2.0, "empty": False, "brake": True, "n_lines": 2},
        {"frame": 50, "role": "shutter", "removed_rms": 20.0, "empty": False, "brake": False, "n_lines": 3},
        {"frame": 51, "role": "shutter", "removed_rms": 18.0, "empty": False, "brake": False, "n_lines": 3},
        {"frame": 52, "role": "shutter", "removed_rms": 19.0, "empty": False, "brake": False, "n_lines": 3},
    ]
    chosen = pick_inspect_frames(rows, {"frames": [50, 51, 52]})
    idxs = [i for i, _ in chosen]
    assert 51 in idxs  # this channel's shutter mid
    assert 20 in idxs  # strongest live
    assert 160 not in idxs
    assert 1061 not in idxs


if __name__ == "__main__":
    test_unique_keeps_fy_and_fx()
    test_apply_identity_when_no_lines()
    test_collect_vertical_stripes_uses_peak_mask()
    test_grow_vertical_stripes_cleans()
    test_pick_inspect_uses_this_channel_shutter_not_chana_anchors()
    print("ok")
