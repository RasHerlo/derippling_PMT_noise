"""Synthetic tests for the recursive image-domain check."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "reference" / "gpt"))

from batch_defringe.image_check import (  # noqa: E402
    Candidate,
    compact_blob_mask,
    line_scan_summary,
    notch_one,
    run_image_check,
    score_removed,
    write_image_check_pdf,
)
from pmt_fringe_raw_adaptive import contiguous_ranges  # noqa: E402

H, W = 128, 128
Q_GRATE = 8
FX_GRATE = 16
Q_GRATE2 = 22
FX_GRATE2 = 24


def _family(q: float, fx: float, *, paired: bool = True, fx_halfwidth: int = 5) -> dict:
    xw = np.zeros(W, dtype=float)
    cx = W // 2
    for fxc in (fx, -fx):
        for i in range(W):
            xw[i] += np.exp(-0.5 * ((i - cx - fxc) / max(fx_halfwidth / 1.5, 1.0)) ** 2)
    xw /= max(float(xw.max()), 1e-12)
    fx_ax = np.arange(W) - cx
    hi = float(H // 2 - q) if paired else None
    return {
        "q": float(q),
        "hi": hi,
        "paired": paired,
        "row_score": 12.0,
        "x_weight": xw,
        "fx_ranges": contiguous_ranges(fx_ax[xw > 0.20]),
    }


def _grating(q: float, fx: float, amp: float = 50.0, n_fx: int = 1, seed: int = 0) -> np.ndarray:
    """Nearly-horizontal bands; n_fx>1 lights several fx bins so the notch can gate."""
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), dtype=np.float64)
    half = max(n_fx // 2, 0)
    used = 0
    for df in range(-half, half + 1):
        fxc = int(fx) + df
        if abs(fxc) < 6:
            continue
        phase = float(rng.uniform(0.0, 2.0 * np.pi)) if n_fx > 1 else 0.0
        img += np.sin(2.0 * np.pi * q * y / H + 2.0 * np.pi * fxc * x / W + phase)
        used += 1
    img *= amp / max(used, 1)
    return img


def _cells(rng: np.random.Generator, n: int = 7, amp: float = 90.0) -> np.ndarray:
    img = np.zeros((H, W), dtype=np.float64)
    yy, xx = np.ogrid[:H, :W]
    for _ in range(n):
        cy = int(rng.integers(18, H - 18))
        cx = int(rng.integers(18, W - 18))
        sig = float(rng.uniform(3.0, 6.0))
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sig**2))
    return img


def _onesided_grating() -> np.ndarray:
    g = _grating(Q_GRATE, FX_GRATE)
    g[:, W // 2 :] = 0.0
    return g


def test_grating_removed_passes():
    score = score_removed(_grating(Q_GRATE, FX_GRATE, n_fx=1))
    names = {t["name"]: t["passed"] for t in score["traits"]}
    assert names["coverage"], score
    assert names["even"], score
    assert names["ridges"], score
    assert names["blob"], score
    assert score["passed"]


def test_cells_removed_fails():
    rng = np.random.default_rng(1)
    score = score_removed(_cells(rng))
    names = {t["name"]: t["passed"] for t in score["traits"]}
    assert not names["coverage"] or not names["blob"] or not names["ridges"], score
    assert not score["passed"]


def test_onesided_fails_even():
    score = score_removed(_onesided_grating())
    even = score["by_name"]["even"]
    assert not even["passed"], score
    assert not score["passed"]


def test_loop_accepts_grating_then_stops():
    base = 80.0 + _grating(Q_GRATE, FX_GRATE, n_fx=7)
    frames = {0: base, 1: base}
    roles = {0: "shutter", 1: "live"}
    fam = _family(Q_GRATE, FX_GRATE)
    result = run_image_check(
        frames,
        roles=roles,
        extra_candidates=[Candidate("test", fam)],
        detect_on_leftover=False,
        max_rounds=3,
    )
    assert result.rounds, result
    assert result.rounds[0].verdict == "accept", (
        result.rounds[0].verdict,
        result.rounds[0].reason,
        [(e.frame, e.active, e.gate, e.passed, e.score) for e in result.rounds[0].frames],
    )
    assert len(result.accepted) == 1
    assert abs(result.accepted[0]["q"] - Q_GRATE) < 1


def test_loop_rejects_onesided_and_stops():
    img = 80.0 + _onesided_grating()
    frames = {0: img, 1: img}
    roles = {0: "shutter", 1: "live"}
    fams = [_family(Q_GRATE, FX_GRATE), _family(Q_GRATE2, FX_GRATE2)]
    result = run_image_check(
        frames,
        roles=roles,
        extra_candidates=[Candidate("test", f) for f in fams],
        detect_on_leftover=False,
        max_rounds=4,
    )
    verdicts = [r.verdict for r in result.rounds]
    assert "accept" not in verdicts, (
        verdicts,
        [r.reason for r in result.rounds],
        [(e.passed, e.active, e.score["by_name"]["even"]) for r in result.rounds for e in r.frames],
    )
    assert result.accepted == []
    assert any(r.verdict == "reject" for r in result.rounds), verdicts


def test_loop_accepts_two_gratings():
    img = (
        80.0
        + _grating(Q_GRATE, FX_GRATE, amp=50.0, n_fx=7, seed=0)
        + _grating(Q_GRATE2, FX_GRATE2, amp=50.0, n_fx=7, seed=1)
    )
    frames = {0: img, 1: img}
    roles = {0: "shutter", 1: "live"}
    fams = [_family(Q_GRATE, FX_GRATE), _family(Q_GRATE2, FX_GRATE2)]
    result = run_image_check(
        frames,
        roles=roles,
        extra_candidates=[Candidate("test", f) for f in fams],
        detect_on_leftover=False,
        max_rounds=4,
    )
    accepts = [r for r in result.rounds if r.verdict == "accept"]
    assert len(accepts) == 2, (
        [(r.verdict, r.q, r.reason) for r in result.rounds],
        [(e.frame, e.active, e.gate, e.passed) for r in result.rounds for e in r.frames],
    )


def test_line_scan_inactive_is_unchanged():
    g = 80.0 + _grating(Q_GRATE, FX_GRATE, n_fx=1)
    s = line_scan_summary(g, g, np.zeros_like(g))
    assert s["frac_left"] > 0.99
    assert s["main"]["frac_left"] > 0.99
    assert s["anti"]["frac_left"] > 0.99
    assert s["main"]["rms_removed"] < 1e-9
    assert s["anti"]["rms_removed"] < 1e-9
    assert s["written_rms"] < 1e-9


def test_period_on_uniform_grating():
    from batch_defringe.image_check import _period_by_x_third

    g = _grating(Q_GRATE, FX_GRATE, n_fx=1)
    per = _period_by_x_third(g)
    expected = H / Q_GRATE
    for key in ("left", "center", "right"):
        assert per[key] is not None, per
        assert abs(per[key] - expected) <= 2.0, (key, per, expected)


def test_ridge_edge_compact_does_not_reject():
    g = _grating(Q_GRATE, FX_GRATE, n_fx=1)
    yy, xx = np.ogrid[:H, :W]
    extra = np.zeros_like(g)
    for cy, cx in ((20, 20), (40, 90), (90, 40), (70, 70), (50, 30)):
        extra += 80.0 * (((yy - cy) ** 2 + (xx - cx) ** 2) <= 6 ** 2)
    score = score_removed(g + extra)
    assert score["by_name"]["ridges"]["passed"], score
    assert score["by_name"]["blob"]["passed"], score


def test_compact_mask_marks_cells_not_grating():
    rng = np.random.default_rng(0)
    img = rng.normal(0.0, 1.0, (H, W))
    yy, xx = np.ogrid[:H, :W]
    for cy, cx in ((32, 32), (32, 96), (96, 32), (96, 96), (64, 64)):
        img = np.where((yy - cy) ** 2 + (xx - cx) ** 2 <= 7 ** 2, 80.0, img)
    mask = compact_blob_mask(img)
    assert int(mask.sum()) > 0
    blob = score_removed(img)["by_name"]["blob"]
    assert blob["n_compact"] >= 3, blob
    grate = _grating(Q_GRATE, FX_GRATE, n_fx=1)
    gmask = compact_blob_mask(grate)
    assert int(gmask.sum()) == 0
    assert score_removed(grate)["by_name"]["blob"]["passed"]


def test_applied_map_is_localized_not_full_row():
    img = 80.0 + _grating(Q_GRATE, FX_GRATE, n_fx=7)
    fam = _family(Q_GRATE, FX_GRATE)
    _cleaned, _removed, tracking, applied, seed = notch_one(img, fam, search=2)
    assert tracking[0]["gate"] > 0, tracking[0]
    assert float(applied.max()) > 0.05
    row_frac = (applied > 0.05).mean(axis=1)
    assert float(row_frac.max()) < 0.5, float(row_frac.max())
    assert float(seed.max()) > 0.2
    # four thin fy bands, not the whole spectrum
    n_fy = int(np.sum((applied > 0.05).any(axis=1)))
    assert n_fy < H // 2, n_fy


def test_gate0_applied_empty_seed_present():
    img = np.full((H, W), 80.0)
    fam = _family(Q_GRATE, FX_GRATE)
    _cleaned, _removed, tracking, applied, seed = notch_one(img, fam, search=2)
    assert float(tracking[0]["gate"]) == 0.0
    assert float(applied.max()) < 1e-6
    assert float(seed.max()) > 0.2


def test_pdf_writes(tmp_path: Path | None = None):
    base = 80.0 + _grating(Q_GRATE, FX_GRATE, n_fx=7)
    frames = {4: base, 10: base}
    roles = {4: "shutter", 10: "live"}
    result = run_image_check(
        frames,
        roles=roles,
        extra_candidates=[Candidate("test", _family(Q_GRATE, FX_GRATE))],
        detect_on_leftover=False,
        max_rounds=1,
    )
    result.source_tif = "synthetic.tif"
    result.channel = "ChanA"
    result.computer = "test"
    out = Path(tmp_path) if tmp_path is not None else _REPO / "tests" / "_tmp_image_check.pdf"
    if tmp_path is None:
        out.parent.mkdir(parents=True, exist_ok=True)
    write_image_check_pdf(out, result)
    assert out.is_file()
    if tmp_path is None:
        out.unlink(missing_ok=True)
        out.with_suffix(".png").unlink(missing_ok=True)


if __name__ == "__main__":
    test_grating_removed_passes()
    test_cells_removed_fails()
    test_onesided_fails_even()
    test_loop_accepts_grating_then_stops()
    test_loop_rejects_onesided_and_stops()
    test_loop_accepts_two_gratings()
    test_line_scan_inactive_is_unchanged()
    test_period_on_uniform_grating()
    test_compact_mask_marks_cells_not_grating()
    test_ridge_edge_compact_does_not_reject()
    test_applied_map_is_localized_not_full_row()
    test_gate0_applied_empty_seed_present()
    test_pdf_writes()
    print("ok")
