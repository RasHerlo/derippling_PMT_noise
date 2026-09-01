"""Congruent (qy, qx) from H, V, and both diagonals."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from batch_defringe.congruence import (  # noqa: E402
    choose_hypothesis,
    congruent_seed,
    predicted_diag_period,
    seed_peak_mask,
)

H, W = 128, 128


def _vstripes(q: float = 16.0, amp: float = 40.0) -> np.ndarray:
    x = np.arange(W)[None, :]
    return np.broadcast_to(80.0 + amp * np.sin(2.0 * np.pi * q * x / W), (H, W)).copy()


def _hstripes(q: float = 8.0, amp: float = 40.0) -> np.ndarray:
    y = np.arange(H)[:, None]
    return np.broadcast_to(80.0 + amp * np.sin(2.0 * np.pi * q * y / H), (H, W)).copy()


def test_diag_period_square_axis_aligned():
    p = predicted_diag_period(0.0, 16.0, H, W, "main")
    assert p is not None and abs(p - W / 16.0) < 0.2
    p2 = predicted_diag_period(0.0, 16.0, H, W, "anti")
    assert p2 is not None and abs(p2 - W / 16.0) < 0.2
    p3 = predicted_diag_period(8.0, 16.0, H, W, "main")
    assert p3 is not None and abs(p3 - H / (8.0 + 16.0)) < 0.2
    p4 = predicted_diag_period(8.0, 16.0, H, W, "anti")
    assert p4 is not None and abs(p4 - H / abs(8.0 - 16.0)) < 0.2


def test_vertical_stripes_are_fx_only():
    r = congruent_seed(_vstripes())
    assert r["winner"] == "fx", r
    assert r["qy"] is None
    assert r["qx"] is not None and abs(r["qx"] - 16.0) <= 3.0


def test_horizontal_stripes_are_fy_only():
    r = congruent_seed(_hstripes())
    assert r["winner"] == "fy", r
    assert r["qx"] is None
    assert r["qy"] is not None and abs(r["qy"] - 8.0) <= 3.0


def test_vote_fx_when_diags_match_qx_not_qy():
    """Frame-160 geometry: H and both diagonals agree; V is a slow fake qy."""
    h = w = 512
    qx, qy_fake = 15.0, 6.3
    p_h = w / qx
    chosen = choose_hypothesis(
        h,
        w,
        qx_h=qx,
        qy_v=qy_fake,
        p_main=p_h,
        p_anti=p_h,
    )
    win = chosen["winner"]
    assert win is not None and win["name"] == "fx"
    names = {d["name"]: d["score"] for d in chosen["hypotheses"]}
    assert names["fx"] < names["tilted"]
    assert names["fx"] < names["fy"]


def test_vote_tilted_when_diags_match_sum_and_diff():
    h = w = 256
    qy, qx = 10.0, 16.0
    p_main = predicted_diag_period(qy, qx, h, w, "main")
    p_anti = predicted_diag_period(qy, qx, h, w, "anti")
    chosen = choose_hypothesis(h, w, qx_h=qx, qy_v=qy, p_main=p_main, p_anti=p_anti)
    win = chosen["winner"]
    assert win is not None and win["name"] == "tilted", chosen


def test_fx_mask_is_on_fx_axis_not_fy():
    r = {"winner": "fx", "qy": None, "qx": 16.0}
    mask = seed_peak_mask(H, W, r)
    cy, cx = H // 2, W // 2
    assert mask[cy, cx + 16] > 0.3
    assert mask[cy + 16, cx] < 0.05
    assert mask[cy, cx] == 0.0
    assert float(mask.mean()) < 0.02


def test_tilted_mask_is_off_axis():
    r = {"winner": "tilted", "qy": 8.0, "qx": 16.0}
    mask = seed_peak_mask(H, W, r)
    cy, cx = H // 2, W // 2
    assert mask[cy + 8, cx + 16] > 0.3
    assert mask[cy, cx + 16] < 0.05
    assert mask[cy + 8, cx] < 0.05


if __name__ == "__main__":
    test_diag_period_square_axis_aligned()
    test_vertical_stripes_are_fx_only()
    test_horizontal_stripes_are_fy_only()
    test_vote_fx_when_diags_match_qx_not_qy()
    test_vote_tilted_when_diags_match_sum_and_diff()
    test_fx_mask_is_on_fx_axis_not_fy()
    test_tilted_mask_is_off_axis()
    print("ok")
