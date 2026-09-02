"""v3 union-apply and candidate uniqueness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "reference" / "gpt"))

from batch_defringe.image_check import FrameEval  # noqa: E402
from batch_defringe.process_v3 import (  # noqa: E402
    KeptFamily,
    V3Cand,
    _soften_leftover_votes,
    apply_frame,
    unique_cands,
)
from batch_defringe.spatial_seed import fx_family_from_q  # noqa: E402
from pmt_fringe_raw_adaptive import fft_log_amp  # noqa: E402

H, W = 64, 64


def _vstripes(q: float = 8.0) -> np.ndarray:
    x = np.arange(W)[None, :]
    return np.broadcast_to(80.0 + 40.0 * np.sin(2.0 * np.pi * q * x / W), (H, W)).copy()


def test_unique_keeps_fy_and_fx_same_q():
    dummy = {"q": 10.0}
    cands = [
        V3Cand("fft_detect", "fy", dummy),
        V3Cand("linescan_fx", "fx", dummy),
        V3Cand("catalog", "fy", {"q": 10.2}),
    ]
    out = unique_cands(cands, [])
    axes = {c.axis for c in out}
    assert axes == {"fy", "fx"}


def test_apply_fx_on_vertical_stripes_removes_energy():
    frame = _vstripes()
    logamp = fft_log_amp(frame)
    fam = fx_family_from_q(logamp, 8.0)
    kept = [KeptFamily(axis="fx", source="linescan_fx", family=fam, q_seed=8.0, eval_round=1)]
    rec = apply_frame(frame, kept, [8.0], search=4)
    assert rec["removed_rms"] > 1.0
    assert rec["tracking"][0]["axis"] == "fx"


def test_leftover_shutter_fail_does_not_veto_live_pass():
    dummy = np.zeros((8, 8), dtype=np.float32)

    def ev(frame, role, rms, passed, gate=1.0):
        return FrameEval(
            frame=frame,
            role=role,
            gate=gate,
            q=15.0,
            removed_rms=rms,
            active=gate > 0,
            score={"passed": passed},
            passed=passed,
            raw=dummy,
            cleaned=dummy,
            removed=dummy,
        )

    evals = [
        ev(160, "live_anchor", 1.28, True),
        ev(700, "live_anchor", 1.04, False),
        ev(756, "shutter", 0.47, False),
        ev(1061, "live_anchor", 0.29, False),
    ]
    cand = V3Cand("linescan_fx", "fx", {"q": 15.0})
    kept = [KeptFamily(axis="fy", source="shutter_hint", family={"q": 10.0}, q_seed=10.0, eval_round=1)]
    _soften_leftover_votes(evals, kept=kept, cand=cand)
    by_frame = {e.frame: e for e in evals}
    assert by_frame[756].active is False
    assert by_frame[1061].active is False
    assert by_frame[160].passed and by_frame[160].active
    from batch_defringe.image_check import _aggregate_verdict

    verdict, _ = _aggregate_verdict(evals)
    assert verdict == "accept"


def test_schematic_pdf(tmp_path=None):
    from batch_defringe.v3_report import write_schematic_pdf

    dest = Path(tmp_path) / "v3_pipeline_schematic.pdf" if tmp_path is not None else _REPO / "v3_pipeline_schematic.pdf"
    out = write_schematic_pdf(dest)
    assert out.is_file()
    assert out.stat().st_size > 1000
    return out


if __name__ == "__main__":
    test_unique_keeps_fy_and_fx_same_q()
    test_apply_fx_on_vertical_stripes_removes_energy()
    test_leftover_shutter_fail_does_not_veto_live_pass()
    test_schematic_pdf()
    print("ok")
