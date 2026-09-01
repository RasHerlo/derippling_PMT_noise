"""Synthetic tests for linescan vs original vs combo seeding."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "reference" / "gpt"))

from batch_defringe.seed_compare import (  # noqa: E402
    ANCHOR_LIVE,
    N_RANDOM,
    SHUTTER_PAIR,
    compare_frame,
    eval_combo,
    eval_linescan,
    eval_original,
    pick_frames,
)

H, W = 128, 128


def _vstripes(q: float = 16.0, amp: float = 40.0) -> np.ndarray:
    x = np.arange(W)[None, :]
    return np.broadcast_to(80.0 + amp * np.sin(2.0 * np.pi * q * x / W), (H, W)).copy()


def _hstripes(q: float = 8.0, amp: float = 40.0) -> np.ndarray:
    y = np.arange(H)[:, None]
    return np.broadcast_to(80.0 + amp * np.sin(2.0 * np.pi * q * y / H), (H, W)).copy()


def test_pick_frames_includes_required_and_five_random():
    n = 2000
    picked = pick_frames(n, rng_seed=20260901)
    idxs = [i for i, _ in picked]
    roles = {i: r for i, r in picked}
    assert len(picked) == len(ANCHOR_LIVE) + len(SHUTTER_PAIR) + N_RANDOM
    for i in ANCHOR_LIVE:
        assert roles[i] == "live_anchor"
    for i in SHUTTER_PAIR:
        assert roles[i] == "shutter"
    assert len([r for r in roles.values() if r == "random"]) == N_RANDOM
    assert pick_frames(n, rng_seed=20260901) == picked
    assert len(set(idxs)) == len(idxs)


def test_vertical_stripes_linescan_and_combo_not_original():
    frame = _vstripes()
    rec = compare_frame(frame, index=0, role="live_anchor")
    assert rec["linescan"]["seed"]["winner"] == "fx"
    assert rec["linescan"]["notch"]["status"] in ("PASS", "FAIL")
    assert rec["linescan"]["notch"]["axis"] == "fx"
    assert rec["original"]["detect_q"] == []
    assert rec["original"]["notch"]["status"] in ("none", "off")
    assert rec["combo"]["source"] == "linescan_fx"
    assert rec["combo"]["notch"]["axis"] == "fx"


def test_horizontal_stripes_linescan_proposes_fy():
    frame = _hstripes()
    original = eval_original(frame, search=4)
    linescan = eval_linescan(frame, search=4)
    assert linescan["seed"]["winner"] == "fy"
    assert linescan["seed"]["qy"] is not None
    assert abs(float(linescan["seed"]["qy"]) - 8.0) <= 2.0
    combo = eval_combo(frame, search=4, linescan=linescan, original=original)
    assert combo["source"] == "linescan_fy"
    assert combo["n_tried"] >= 1


if __name__ == "__main__":
    test_pick_frames_includes_required_and_five_random()
    test_vertical_stripes_linescan_and_combo_not_original()
    test_horizontal_stripes_linescan_proposes_fy()
    print("ok")
